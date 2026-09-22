#NOTE: check for dead imports

from tinygrad.schedule.indexing import apply_movement_op, BufferizeOpts
from tinygrad.uop.ops import AxisType, PatternMatcher, UOp, UPat, GroupOp, Ops, graph_rewrite, remove_all_tags, ParamArg, to_max_shape, KernelInfo, AddrSpace, BottomUpGate, _substitute
from tinygrad.device import MultiBuffer, Buffer
from tinygrad.helpers import prod, VIZ, dedup
from tinygrad.dtype import dtypes
from dataclasses import dataclass, replace, field

import itertools 

def new_ranges(shp, rid=itertools.count(0), ty=AxisType.WEAK): return tuple(UOp.range(sz, next(rid), ty) for i,sz in enumerate(shp))

pm_insert_expands = PatternMatcher([
  (UPat(GroupOp.Binary|GroupOp.Ternary|{Ops.STORE}, name="x"), 
  lambda x: x.replace(src=tuple(u.expand(x.shape) if u.shape != x.shape else u for u in x.src)))
])

def substitute_ranges(ctx, x):
  if x.op is Ops.STAGE: raise BottomUpGate()
  return ctx.get(x)

pm_substitute_ranges = PatternMatcher([(UPat(GroupOp.All, name="x"), substitute_ranges)])

pm_add_ranges = PatternMatcher([
  (UPat(Ops.REDUCE, src=(UPat(),), name="r"), lambda r: r.replace(src=((x:=r.src[0]), *new_ranges(x.shape[:r.arg[1]], ty=AxisType.REDUCE)))),
  (UPat.var("dst").store(UPat.var("src")), 
  lambda dst, src: UOp(Ops.STAGE, (dst.index(*(rngs:=new_ranges(dst.shape))), *rngs)).store(UOp(Ops.STAGE, (src.index(*rngs), *rngs))))
])

def compose_ranges(ind, st): return graph_rewrite(st.src[0], pm_substitute_ranges, ctx=dict(zip(st.src[1:], ind.src[1:])), bottom_up=True)

def push_ranges(i):
  if (x:=i.src[0]).op is Ops.REDUCE:
    return x.replace(src=(x.src[0].index(*(rr:=x.src[1:]), *i.src[1:]), *rr), arg=(x.arg[0], 0))
  elif x.op in GroupOp.Elementwise:
    rngs = i.src[1:]
    return x.replace(src=tuple(u.index(*rngs) for u in x.src))

pm_fold_ranges = PatternMatcher([
  (UPat(GroupOp.Movement-{Ops.PAD}, name="m", src=(UPat.var("x"),), allow_any_len=True).index(name="ind", allow_any_len=True),
  lambda ind, m, x: x.index(*apply_movement_op(m.op, x.shape, m.marg, ind.src[1:]))),
  (UPat(Ops.STAGE, name="st").index(allow_any_len=True, name="ind"), compose_ranges),
  (UPat(Ops.INDEX, name="i"), push_ranges)
])

def realize(tsink):
  realized, consumes = {}, {tsink:0}
  for x in reversed(tsink.toposort(enter_calls=False)):
    # if x.op is Ops.CONTIGUOUS or (x.op in GroupOp.Elementwise|{Ops.REDUCE} and x.ndim > 0 and consumes[x] > 1):
    if x.op is Ops.CONTIGUOUS or (x.op in GroupOp.Elementwise|{Ops.REDUCE} and x.ndim > 0 and (not x.dtype in dtypes.weaks) and consumes[x] > 1):
      buf = UOp.new_buffer(x.device if x.device is not None else tsink.device, prod(to_max_shape(x.shape)), x.dtype, bound=x.op is Ops.CONTIGUOUS)
      realized[x] = buf.after(buf.view_as(x.shape, x.axis).store((x.src[0] if x.op is Ops.CONTIGUOUS else x).rtag())).view_as(x.shape, x.axis)
      consumes[x] = 1
    if x.op is Ops.STORE: consumes[x] = 1
    if x.op is Ops.EXPAND: consumes[x] *= x.max_numel() // x.src[0].max_numel()
    for i,s in enumerate(x.src):
      consumes[s] = consumes.get(s,0) + (consumes[x] if x.op is not Ops.STORE or i > 0 else 0)
  return realized, consumes

debug_counts = PatternMatcher([
  (UPat(GroupOp.All, name="x"), lambda ctx, x: x.rtag(tag=ctx[1][x] if x not in ctx[0] else "REAL") if x in ctx[1] else None)
])

def derealize(tsink):
  info, derealized = {}, {}
  for x in tsink.toposort(enter_calls=False):
    if x in info: continue
    sbufs, sred = zip(*(info[u] for u in x.src)) if x.src else ([], [])
    bufs, red = dedup(sum(sbufs, [])), any(sred)
    if x.op is Ops.REDUCE and bufs: red = True
    elif x.op is Ops.AFTER and (bx:=x.src[0]).is_unbound:
      if len(bufs) > 3 or red: derealized[bx] = bind_buffer(bx)
      else: derealized[x] = x.src[1].src[1].reshape(x.shape)
    if x.has_buffer_identity(after_ok=True) or (x.op is Ops.AFTER and x.src[0] in derealized):
      bufs, red = [x], False
    info[x] = (bufs, red)
  return derealized

def convert_stack_to_where(i, x):
  req, rngs = i.src[1], i.src[2:]
  acc = x.src[-1].index(*rngs)
  for j in range(len(x.src)-2, -1, -1): acc = req.eq(j).where(x.src[j].index(*rngs), acc)
  return acc

def convert_pad_to_where(ind, x):
  pad_rngs = apply_movement_op(x.op, x.src[0].shape, x.marg, ind.src[1:])
  valid = UOp.const(True).uprod(*(r.get_valid() for r in pad_rngs))
  return valid.where(x.src[0].index(*pad_rngs), UOp.const(x.dtype.const(0)))

pm_convert_ranges = PatternMatcher([
  (UPat(Ops.STACK, name="x").index(allow_any_len=True, name="i"), convert_stack_to_where),
  (UPat(Ops.PAD, name="x").index(allow_any_len=True, name="ind"), convert_pad_to_where)
])

def bind_buffer(x): 
  return x.replace(arg=replace((p:=x.arg), buffer=(MultiBuffer if isinstance(p.device, tuple) else Buffer)(p.device, p.size, p.dtype)))

pm_presplit = PatternMatcher([
  (UPat(Ops.STAGE, name="s"), lambda s: s.src[0]),
  (UPat(Ops.INDEX, name="i"), lambda i: None if i.src[0].shape else i.src[0])
])

def add_arg(ctx, x):
  if x.op is Ops.PARAM and x.addrspace is AddrSpace.ALU: return x.replace(arg=replace(x.arg, slot=-1)).rtag()
  if not (x.has_buffer_identity(after_ok=True) and x.tag is None): return None
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
  realized, consumes = realize(tsink)

  '''
  if VIZ:
    counts = graph_rewrite(tsink, debug_counts, ctx=(realized, consumes), bottom_up=True)
    graph_rewrite(counts, PatternMatcher([]), name="view counts")
    tsink = graph_rewrite(tsink, remove_all_tags, name="remove tags")
  '''

  tsink = graph_rewrite(tsink, _substitute, ctx=realized, bottom_up=True, name="realize")
  tsink = graph_rewrite(tsink, remove_all_tags, walk=True)
  tsink = graph_rewrite(tsink, pm_add_ranges, walk=True, name="add ranges")
  tsink = graph_rewrite(tsink, pm_fold_ranges+pm_convert_ranges, bottom_up=True, name="fold ranges")

  derealized = derealize(tsink)
  tsink = graph_rewrite(tsink, _substitute+pm_fold_ranges, ctx=derealized, bottom_up=True, name="derealize")

  tsink = graph_rewrite(tsink, pm_presplit, walk=True, name="prepare to split kernels")
  tsink = graph_rewrite(tsink, pm_split_kernels, bottom_up=True, name="split kernels")
  return tsink

