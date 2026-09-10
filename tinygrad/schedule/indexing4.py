#NOTE: check for dead imports

from tinygrad.schedule.indexing import apply_movement_op, BufferizeOpts
from tinygrad.uop.ops import AxisType, PatternMatcher, UOp, UPat, GroupOp, Ops, graph_rewrite, remove_all_tags, ParamArg, to_max_shape, KernelInfo, AddrSpace
from tinygrad.device import MultiBuffer, Buffer
from tinygrad.helpers import prod, VIZ
from dataclasses import dataclass, replace, field

import itertools 

def new_ranges(shp, rid=itertools.count(0), ty=AxisType.WEAK): return tuple(UOp.range(sz, next(rid), ty) for i,sz in enumerate(shp))

pm_insert_expands = PatternMatcher([
  (UPat(GroupOp.Binary|GroupOp.Ternary|{Ops.STORE}, name="x"), 
  lambda x: x.replace(src=tuple(u.expand(x.shape) if u.shape != x.shape else u for u in x.src)))
])

def stage_in(s): return s.src[1:]
def stage_out(s): return s.src[0].src[1:]

def compose_ranges(f, g):
  out_rngs = UOp.sink(*stage_out(g)).substitute(dict(zip(stage_in(g), f.src[1:])), walk=True).src
  return g.src[0].src[0].index(*out_rngs)

def push_ranges(s):
  if (x:=s.src[0].src[0]).op is Ops.REDUCE: 
    return x.replace(src=(x.src[0].stage((rr:=x.src[1:])+stage_in(s), rr+stage_out(s)), *rr))
  elif x.op in GroupOp.Elementwise: 
    return x.replace(src=tuple(u.stage(stage_in(s), stage_out(s)) for u in x.src))
  return None

pm_add_ranges = PatternMatcher([
  (UPat(Ops.REDUCE, src=(UPat(),), name="r"), lambda r: r.replace(src=((x:=r.src[0]), *new_ranges(x.shape[:r.arg[1]], ty=AxisType.REDUCE)))),
  (UPat.var("dst").store(UPat.var("src")), lambda dst, src: dst.stage(rngs:=new_ranges(dst.shape), rngs).store(src.stage(rngs, rngs))),
])

pm_fold_ranges = PatternMatcher([
  (UPat(GroupOp.Movement-{Ops.PAD}, name="m", src=(UPat(name="x"),), allow_any_len=True),
  lambda x, m: x.stage(in_rngs:=new_ranges(m.shape), apply_movement_op(m.op, x.shape, m.marg, in_rngs))),
  (UPat(Ops.STAGE, name="g", allow_any_len=True).index(name="f", allow_any_len=True), compose_ranges),
  (UPat(Ops.STAGE, name="s"), push_ranges)
])

#TODO: incorporate into fold_ranges, or keep separate?

pm_push_multi = PatternMatcher([
  (UPat((Ops.MSTACK, Ops.MSELECT), src=(UPat(Ops.STAGE, name="s"),), allow_any_len=True, name="m"),
  lambda m, s: m.replace(src=(s.src[0].src[0], *m.src[1:])).stage(stage_in(s), stage_out(s)))
])

def count_consumes(tsink):
  realize, consumes = {}, {tsink:0}
  for x in reversed(tsink.toposort(enter_calls=False)):
    assert x in consumes, f"{x.op} not in consumes"
    #TODO: can it just use the sink's device if device is None, like in current rangeify?
    #TODO: figure out what to do with contiguous
    if x.op is Ops.CONTIGUOUS or (x.op in GroupOp.ALU|{Ops.REDUCE} and x.device is not None and x.shape != () and consumes[x] > 1):
      bx = x.src[0] if x.op is Ops.CONTIGUOUS else x
      realize[x] = (buf:=mint(bx)).after(buf.view_as(bx.shape, bx.axis).store(bx.rtag())).view_as(bx.shape, bx.axis)
      consumes[x] = 1
    if x.op is Ops.STORE: consumes[x] = 1
    if x.op is Ops.EXPAND: consumes[x] *= x.max_numel() // x.src[0].max_numel()
    for i,s in enumerate(x.src):
      consumes[s] = consumes.get(s,0) + (consumes[x] if x.op is not Ops.STORE or i > 0 else 0)
  return realize, consumes

debug_counts = PatternMatcher([
  (UPat(GroupOp.All, name="x"), lambda ctx, x: x.rtag(tag=ctx[1][x] if x not in ctx[0] else "REAL") if x in ctx[1] else None)
])

def convert_stack_to_where(s, x):
  req = stage_out(s)[0]
  acc = x.src[-1].stage(in_rngs:=stage_in(s), out_rngs:=stage_out(s)[1:])
  for i in range(len(x.src)-2, -1, -1): acc = req.eq(i).where(x.src[i].stage(in_rngs, out_rngs), acc)
  return acc

def convert_pad_to_where(s, x):
  pad_rngs = apply_movement_op(x.op, x.src[0].shape, x.marg, stage_out(s))
  valid = UOp.const(True).uprod(*(r.get_valid() for r in pad_rngs))
  return valid.where(x.src[0].stage(stage_in(s), pad_rngs), UOp.const(x.dtype.const(0)))

pm_convert_ranges = PatternMatcher([
  (UPat(Ops.STAGE, name="s", src=(UPat(Ops.STACK, name="x").index(allow_any_len=True),), allow_any_len=True), convert_stack_to_where),
  (UPat(Ops.STAGE, name="s", src=(UPat(Ops.PAD, name="x").index(allow_any_len=True),), allow_any_len=True), convert_pad_to_where)
])

def bind_buffer(x):
  return x.replace(arg=replace((p:=x.arg), buffer=(MultiBuffer if isinstance(p.device, tuple) else Buffer)(p.device, p.size, p.dtype)))

pm_presplit = PatternMatcher([
  (UPat(Ops.BUFFER, name="x"), lambda x: bind_buffer(x) if x.is_unbound else None),
  (UPat(Ops.REDUCE, name="r"), lambda r: r.replace(arg=(r.arg[0], 0)) if r.arg[1] != 0 else None),
  (UPat(Ops.STAGE, name="s"), lambda s: s.src[0]),
  (UPat(Ops.INDEX, name="i"), lambda i: None if i.src[0].shape else i.src[0])
])

def add_arg(ctx, x):
  if x.op is Ops.PARAM and x.addrspace is AddrSpace.ALU: return x.replace(arg=replace(x.arg, slot=-1)).rtag()
  if not ((x.has_buffer_identity(after_ok=True) or x.op in {Ops.MSTACK, Ops.MSELECT}) and x.tag is None): return None
  ctx[1].append(x)
  return x.param_like(slot=len(ctx[1])-1).rtag()

pm_kernel_arg = PatternMatcher([
  (UPat(GroupOp.All, name="x"), add_arg),
  (UPat(Ops.RANGE, name="r"), lambda ctx, r: r.replace(arg=(next(ctx[0]), r.arg[1])).rtag() if r.tag is None else None)
])

def split_kernels(s):
  s = graph_rewrite(s, pm_kernel_arg, ctx=(kernel_ctx:=(itertools.count(0), [])), bottom_up=True)
  return s.end(*s.ranges).sink(arg=KernelInfo()).call(*kernel_ctx[1])

pm_split_kernels = PatternMatcher([
  (UPat(Ops.STORE, name="s"), split_kernels)
])

def run_rangeify(tsink, b):
  tsink = graph_rewrite(tsink, pm_insert_expands, name="insert expands")
  realize, consumes = count_consumes(tsink)

  '''
  if VIZ:
    counts = graph_rewrite(tsink, debug_counts, ctx=(realize, consumes), bottom_up=True)
    graph_rewrite(counts, PatternMatcher([]), name="view counts")
    tsink = graph_rewrite(tsink, remove_all_tags, name="remove tags")
  '''

  tsink = graph_rewrite(tsink.substitute(realize), remove_all_tags, walk=True, name="bufferize")
  tsink = graph_rewrite(tsink, pm_add_ranges, walk=True, name="add ranges")
  tsink = graph_rewrite(tsink, pm_fold_ranges+pm_push_multi, bottom_up=True, name="fold ranges")
  tsink = graph_rewrite(tsink, pm_convert_ranges+pm_fold_ranges, bottom_up=True, name="convert ranges")

  tsink = graph_rewrite(tsink, pm_presplit, walk=True, name="prepare to split kernels")
  tsink = graph_rewrite(tsink, pm_split_kernels, bottom_up=True, name="split kernels")
  return tsink

#TODO: integrate with ops.py

def mint(x):
  size = prod(to_max_shape(x.shard_shape)) if x.shard_shape else None
  return UOp(Ops.BUFFER, arg=ParamArg(next(UOp.unique_num), dtype=x.dtype, size=size, device=x.device))






