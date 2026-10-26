#TODO: consistent upat arg order?
import itertools, functools
from enum import Enum
from tinygrad.uop.ops import AxisType, PatternMatcher, UOp, UPat, GroupOp, Ops, graph_rewrite, KernelInfo, BottomUpGate, resolve
from tinygrad.uop.symbolic import symbolic, pm_simplify_valid, pm_drop_and_clauses
from tinygrad.codegen.simplify import pm_reduce_simplify
from tinygrad.helpers import dedup, argsort, VIZ
from tinygrad.schedule.prepare import pm_gated_index

@functools.cache
def apply_reshape(in_shape:tuple[sint,...], out_shape:tuple[sint, ...], urngs:UOp) -> UOp:
  acc:sint = 1
  axes_in:list[UOp] = []
  for sz,r in reversed(tuple(zip(out_shape, urngs.src))):
    axes_in.append(acc*r)
    acc *= sz
  combined_axes = UOp.const(0).usum(axes_in)
  axes_out:list[UOp] = []
  for sz in reversed(in_shape):
    axes_out.append(combined_axes % sz)
    combined_axes //= sz
  # This simplify is doing a lot of heavy lifting. This is the replacement for the reshape view merging code.
  return graph_rewrite(UOp.sink(*reversed(axes_out)), symbolic+pm_simplify_valid+pm_drop_and_clauses, name="reshape")

@functools.cache
def apply_movement_op(op:Ops, in_shape:tuple[sint,...], arg:tuple, rngs:tuple[UOp, ...]) -> tuple[UOp, ...]:
  match op:
    case Ops.PERMUTE: return tuple(rngs[p] for p in argsort(arg))
    case Ops.EXPAND:  return rngs[len(arg):]
    case Ops.PAD:
      #TODO: comment?
      return tuple(r if (sz == sh and off == 0) else (r-off).valid(graph_rewrite((r >= off) & (r < (sh+off)),
        symbolic+pm_simplify_valid, name="pad")) for r,sh,(off,sz) in zip(rngs, in_shape, arg))
    case Ops.RESHAPE:
      sink = UOp.sink(*rngs).simplify() # This applies any commutative flips to the ranges early.
      subs = {r:r.replace(arg=(AxisType.PLACEHOLDER, i)) for i,r in enumerate(sink.ranges)}
      return apply_reshape(in_shape, arg, sink.substitute(subs)).substitute({v:k for k,v in subs.items()}).src
    case Ops.SHRINK:  return tuple(a if off == 0 else a+off for a,(off,_) in zip(rngs, arg))
    case _: raise RuntimeError(f"{op} is not a MovementOp")

def new_ranges(shape, rid=itertools.count(0)): return tuple(UOp.range(sz, next(rid)) for sz in shape)

def broadcast_ranges(x:UOp, s:UOp, rngs:tuple[UOp, ...]) -> tuple[UOp, ...]:
  if x.op not in GroupOp.Broadcastable: return rngs
  return tuple(UOp.const(0) if resolve(sz == 1, default=False) else r for sz,r in zip(s.shape, rngs[len(x.shape)-len(s.shape):]))

pm_add_ranges = PatternMatcher([
  (UPat.var("dst").store(UPat.var("src")), lambda dst, src: dst.index(*(r:=new_ranges(dst.shape))).store(src.index(*r)).end(*r)),
  (UPat.var("x").stage(name="st"), lambda st, x: x.index(*(r:=new_ranges(x.shape))).stage(*r)),
  (UPat.var("x").reduce(name="r"), lambda r, x: x.reduce(*new_ranges(x.shape[:r.arg[1]]), arg=r.arg))
])

def push_ranges(ind, x):
  match x.op:
    case Ops.CONST: return x
    case Ops.REDUCE: return x.src[0].index(*(rr:=x.src[1:]), *ind.src[1:]).reduce(*rr, arg=(x.arg[0], 0))
    case op if op in GroupOp.Elementwise:
      rngs = ind.src[1:]
      return x.replace(src=tuple(s.index(*broadcast_ranges(x, s, rngs)) for s in x.src))

pm_push_ranges = PatternMatcher([
  (UPat.var("x").index(allow_any_len=True, name="ind"), push_ranges),
  (UPat(GroupOp.Movement, name="m", src=(UPat.var("x"),), allow_any_len=True).index(name="ind", allow_any_len=True),
  lambda ind, m, x: x.index(*apply_movement_op(m.op, x.shape, m.marg, ind.src[1:]))),
])

def stack_select(rcmp, srcs):
  # MSB-first bit tree
  if len(srcs) == 1: return srcs[0]
  half = 1 << ((len(srcs)-1).bit_length()-1)
  return (rcmp % (2*half) < half).where(stack_select(rcmp, srcs[:half]), stack_select(rcmp, srcs[half:]))

def convert_stack_to_where(ind, x): return stack_select(ind.src[1], [s.index(*ind.src[2:]) for s in x.src])

def convert_pad_to_where(ind, x):
  rngs = apply_movement_op(x.op, x.src[0].shape, x.marg, ind.src[1:])
  valid = UOp.const(True).uprod(*(r.get_valid() for r in rngs))
  return valid.where(x.src[0].index(*rngs), UOp.const(x.dtype.const(0)))

pm_convert_ranges = PatternMatcher([
  (UPat(Ops.STACK, name="x").index(allow_any_len=True, name="ind"), convert_stack_to_where),
  (UPat(Ops.PAD, name="x").index(allow_any_len=True, name="ind"), convert_pad_to_where)
])

Kernelize = Enum('Kernelize', [('CANDIDATE', 0), ('RANGES', 1), ('FORCE', 2)])

def count_uses(tsink):
  uses: dict[UOp, UOp|Kernelize] = {tsink:tsink}
  kernvals: dict[UOp, Kernelize] = {}
  for x in reversed(tsink.toposort()):
    if (is_k:=isinstance(use:=uses[x], Kernelize)) and x.op in GroupOp.Elementwise|{Ops.REDUCE,Ops.SHRINK}: kernvals.setdefault(x, use)
    if x in kernvals or x.has_buffer_identity() or (x.op in GroupOp.Movement|{Ops.REDUCE} and not is_k): use = x
    for s in x.src:
      src_use = Kernelize.CANDIDATE if x.op in GroupOp.Broadcastable|{Ops.EXPAND} and x.max_numel() >= 2*s.max_numel() else use
      if (curr:=uses.setdefault(s, src_use)) is not src_use: uses[s] = Kernelize.RANGES
    '''
    if x.op is Ops.MSTACK or x.op is Ops.MSELECT or ((is_call:=x.op is Ops.CALL) and isinstance(x.body.arg, KernelInfo)):
      for s in x.src[is_call:]:
        if not((sb:=s if is_call else s.base).has_buffer_identity() or sb.is_variable): kernvals[sb] = Kernelize.FORCE
    '''
    if x.op is Ops.MSTACK or x.op is Ops.MSELECT:
      for s in x.src:
        if not (sb:=s.base).has_buffer_identity(): kernvals[sb] = Kernelize.FORCE
    elif x.op is Ops.CALL and isinstance(x.body.arg, KernelInfo):
      for s in x.src[1:]:
        if not (s.has_buffer_identity() or s.is_variable): kernvals[s] = Kernelize.FORCE
  return kernvals

pm_insert_stages = PatternMatcher([
  (UPat(GroupOp.All-{Ops.STAGE}, name="x"), 
  lambda ctx, x: x.replace(src=tuple(s if (k:=ctx.get(s)) is None else s.stage(arg=k) for s in x.src)))
])

def kernelize(ctx, x):
  info, dev, ret = *ctx, x
  if x.op is Ops.MSTACK or x.op is Ops.MSELECT: 
    info[ret] = ([ret:=x.replace(src=tuple(s.base for s in x.src)).view_as(x.shape)], False)
  elif x.has_buffer_identity() or x.op is Ops.CALL: 
    info[ret] = ([ret], False)
  else:
    src_bufs, src_red = zip(*(info[s] for s in x.src)) if x.src else ([], [])
    bufs, red = dedup(sum(src_bufs, [])), any(src_red)
    if x.op is Ops.REDUCE and bufs: red = True
    elif x.op is Ops.STAGE:
      if (k:=x.arg) is Kernelize.FORCE or len(bufs) > 3 or red:
        ret = (a:=UOp.alloc(x.shape, x.dtype, device=dev if x.device is None else x.device)).base.after(a.store(x.src[0])).view_as(x.shape)
        bufs, red = [ret], False
      elif k is not Kernelize.RANGES or x.max_numel() == 1: ret = x.src[0]
    info[ret] = (bufs, red)
  return ret

pm_kernelize = PatternMatcher([(UPat(GroupOp.All, name="x"), kernelize)])

def substitute_ranges(ctx, x):
  if not any(r in ctx for r in x.ranges): raise BottomUpGate()
  return ctx.get(x)

pm_substitute_ranges = PatternMatcher([(UPat(GroupOp.All, name="x"), substitute_ranges)])

def compose_ranges(st, ind): return graph_rewrite(st.src[0], pm_substitute_ranges, ctx=dict(zip(st.src[1:], ind.src[1:])), bottom_up=True)

pm_finalize_index = PatternMatcher([
  (UPat(Ops.STAGE, name="st").index(allow_any_len=True, name="ind"), compose_ranges),
  (UPat.var("x").index(), lambda x: x.index(UOp.const(0)) if x.has_buffer_identity() else x)
])

def add_arg(ctx, x):
  if not x.has_buffer_identity(): return None
  (args:=ctx[0]).append(x)
  return x.param_like(slot=len(args)-1)

def add_range(ctx, r): return r.replace(arg=(r.arg[0], ctx[1][r.arg]))

pm_kernel = PatternMatcher([(UPat(GroupOp.All, name="x"), add_arg), (UPat(Ops.RANGE, name="r"), add_range)])

def split_kernels(s):
  rargs = (x.arg for x in s.toposort(gate=lambda x: not x.has_buffer_identity()) if x.op is Ops.RANGE)
  s = graph_rewrite(s, pm_kernel, ctx=((args:=[]), {arg:i for i,arg in enumerate(sorted(rargs))}), bottom_up=True, walk=True, name="kernel")
  return s.end(*sorted(s.ranges, key=lambda r: r.arg)).sink(arg=KernelInfo()).call(*args)

pm_split_kernels = PatternMatcher([(UPat((Ops.STORE, Ops.END), name="s"), lambda s: split_kernels(s.src[0] if s.op is Ops.END else s))])

def get_kernel_graph(tsink):
  tsink = graph_rewrite(tsink, pm_insert_stages, ctx=count_uses(tsink), bottom_up=True, name="insert stages")
  tsink = graph_rewrite(tsink, pm_kernelize, ctx=({}, tsink.device), walk=True, name="kernelize")
  tsink = graph_rewrite(tsink, pm_add_ranges, walk=True, name="add ranges")
  tsink = graph_rewrite(tsink, pm_convert_ranges+pm_push_ranges, bottom_up=True, name="push ranges")
  tsink = graph_rewrite(tsink, pm_finalize_index+pm_gated_index+symbolic+pm_reduce_simplify, name="finalize graph")
  tsink = graph_rewrite(tsink, pm_split_kernels, bottom_up=True, name="split kernels")

  if VIZ: graph_rewrite(tsink, PatternMatcher([]), name="view kernel graph")
  return tsink

