import itertools, functools
from enum import Enum
from dataclasses import dataclass, field
from tinygrad.uop.ops import AxisType, PatternMatcher, UOp, UPat, GroupOp, Ops, graph_rewrite, KernelInfo, BottomUpGate, _substitute, broadcast_axes
from tinygrad.uop.symbolic import symbolic, pm_simplify_valid, pm_drop_and_clauses
from tinygrad.codegen.simplify import pm_reduce_simplify
from tinygrad.helpers import dedup, argsort, VIZ
from tinygrad.schedule.prepare import pm_gated_index

@functools.cache
def _apply_reshape(in_shape:tuple[sint,...], out_shape:tuple[sint, ...], urngs:UOp) -> UOp:
  acc:sint = 1
  axes_in:list[UOp] = []
  for s,src in list(zip(out_shape, urngs.src))[::-1]:
    axes_in.append(acc*src)
    acc *= s
  combined_axes = UOp.const(0).usum(axes_in)
  axes_out:list[UOp] = []
  for s in in_shape[::-1]:
    axes_out.append(combined_axes % s)
    combined_axes //= s
  # this simplify is doing a lot of heavy lifting. this is the replacement for the reshape view merging code
  return graph_rewrite(UOp.sink(*axes_out[::-1]), symbolic+pm_simplify_valid+pm_drop_and_clauses, name="reshape")

@functools.cache
def apply_movement_op(op:Ops, in_shape:tuple[sint,...], arg:tuple, rngs:tuple[UOp, ...]) -> tuple[UOp, ...]:
  match op:
    case Ops.PERMUTE: return tuple(rngs[p] for p in argsort(arg))
    case Ops.EXPAND:  return rngs[len(arg):]
    case Ops.PAD:
      # Pad conversion should solely use the new valid, so existing ranges (which might already be gated) aren't included in the rewrite.
      return tuple(r if (sz == sh and off == 0) else (r-off).valid(graph_rewrite((r >= off) & (r < (sh+off)),
        symbolic+pm_simplify_valid, name="pad")) for r,sh,(off,sz) in zip(rngs, in_shape, arg))
    case Ops.RESHAPE:
      sink = UOp.sink(*rngs).simplify() # NOTE: this applies any commutative flips to the rngs early
      sub_array = {r:r.replace(arg=(AxisType.PLACEHOLDER, i)) for i,r in enumerate(sink.ranges)}
      return _apply_reshape(in_shape, arg, sink.substitute(sub_array)).substitute({v:k for k,v in sub_array.items()}).src
    case Ops.SHRINK:  return tuple(a if off == 0 else a+off for a,(off,_) in zip(rngs, arg))
    case _: raise RuntimeError(f"{op} is not a MovementOp")

def new_ranges(shape, rid=itertools.count(0)): return tuple(UOp.range(sz, next(rid)) for sz in shape)

def broadcast_ranges(x:UOp, src:UOp, rngs:tuple[UOp, ...]) -> tuple[UOp, ...]:
  if x.op not in GroupOp.Broadcastable: return rngs
  baxes, nleft = broadcast_axes(src.shape, x.shape), len(x.shape)-len(src.shape)
  return tuple(r.const_like(0) if j in baxes else r for j,r in enumerate(rngs) if j >= nleft)

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
  broadcastable: set[UOp] = GroupOp.Broadcastable|{Ops.EXPAND}
  for x in reversed(tsink.toposort(enter_calls=False)):
    # Kernels, movement ops, and REDUCE are all distinct uses.  >1 distinct uses of an elementwise op or a REDUCE => candidacy.
    if x.has_buffer_identity(after_ok=True): use = x
    elif isinstance(use:=uses[x], Kernelize) and (x.op in GroupOp.Elementwise or x.op is Ops.REDUCE or use is Kernelize.FORCE):
      kernvals[x] = use if not (use is Kernelize.RANGES and x.max_numel() == 1) else Kernelize.CANDIDATE
      use = x
    elif x.op in GroupOp.Movement|{Ops.REDUCE} and not isinstance(use, Kernelize): use = x
    for s in x.src:
      # Broadcasting => candidacy.  TODO: need a better comment, it only overrides K=R if the other uses are all K=C as well. 
      broadcasted = x.op in broadcastable and use is not Kernelize.FORCE and x.max_numel() >= 2*s.base.max_numel()
      src_use = use if not broadcasted else Kernelize.CANDIDATE
      if (curr:=uses.setdefault(s, src_use)) is not src_use and curr is not Kernelize.FORCE: uses[s] = Kernelize.RANGES
    # MSTACK, MSELECT, and custom kernel srcs must be kernelized.  MSTACK and MSELECT srcs drop their views.
    if x.op is Ops.MSTACK or x.op is Ops.MSELECT:
      for s in x.src:
        if not (sb:=s.base).has_buffer_identity(after_ok=True): uses[sb] = Kernelize.FORCE
    elif x.op is Ops.CALL and isinstance(x.body.arg, KernelInfo):
      for s in x.src[1:]:
        if not (s.has_buffer_identity(after_ok=True) or s.is_variable): uses[s] = Kernelize.FORCE
  return kernvals

def count_buffers(tsink, kernvals):
  info: dict[UOp, tuple[list[UOp], bool]] = {}
  forced: set[UOp] = {Ops.MSTACK, Ops.MSELECT, Ops.CALL}
  allocs: dict[UOp, UOp] = {}
  for x in tsink.toposort():
    if x.has_buffer_identity(after_ok=True) or x.op in forced: 
      info[x] = ([x], False)
      continue
    src_bufs, src_red = zip(*(info[s] for s in x.src)) if x.src else ([], [])
    bufs, red = dedup(sum(src_bufs, [])), any(src_red)
    if x.op is Ops.REDUCE and bufs: red = True
    if (k:=kernvals.get(x)) is not None:
      if k is Kernelize.FORCE or len(bufs) > 3 or red: 
        bufs, red = [x], False
        allocs[x] = (a:=UOp.alloc(x.shape, x.dtype, device=x.device if x.device else tsink.device)).base.after(a.store(x)).view_as(x.shape)
      elif k is not Kernelize.RANGES: del kernvals[x]
    info[x] = (bufs, red)
  return kernvals, allocs

def kernelize(ctx, x):
  kernvals, allocs = ctx
  src = (s.base for s in x.src) if (multi:=x.op is Ops.MSTACK or x.op is Ops.MSELECT) else x.src
  ret = x.replace(src=tuple((s.stage() if s in kernvals else s) if (a:=allocs.get(s)) is None else a for s in src))
  return ret.view_as(x.shape) if multi else ret

pm_kernelize = PatternMatcher([(UPat(GroupOp.All-{Ops.STORE,Ops.STAGE}, name="x"), kernelize)])

def substitute_ranges(ctx, x):
  if not any(r in ctx for r in x.ranges): raise BottomUpGate()
  return ctx.get(x)

pm_substitute_ranges = PatternMatcher([(UPat(GroupOp.All, name="x"), substitute_ranges)])

def compose_ranges(st, ind): return graph_rewrite(st.src[0], pm_substitute_ranges, ctx=dict(zip(st.src[1:], ind.src[1:])), bottom_up=True)

pm_canonicalize_indexing = PatternMatcher([
  (UPat(Ops.STAGE, name="st").index(allow_any_len=True, name="ind"), compose_ranges),
  (UPat.var("x").index(), lambda x: x.index(UOp.const(0)) if x.has_buffer_identity(after_ok=True) else x)
])

def add_arg(ctx, x):
  if not x.has_buffer_identity(after_ok=True): return None
  (args:=ctx[0]).append(x)
  return x.param_like(slot=len(args)-1)

def add_range(ctx, r): return r.replace(arg=(r.arg[0], ctx[1][r.arg]))

pm_kernel = PatternMatcher([(UPat(GroupOp.All, name="x"), add_arg), (UPat(Ops.RANGE, name="r"), add_range)])

def split_kernels(s):
  rargs = (x.arg for x in s.toposort(gate=lambda x: not x.has_buffer_identity(after_ok=True)) if x.op is Ops.RANGE)
  s = graph_rewrite(s, pm_kernel, ctx=((args:=[]), {arg:i for i,arg in enumerate(sorted(rargs))}), bottom_up=True, walk=True, name="kernel")
  return s.end(*sorted(s.ranges, key=lambda r: r.arg)).sink(arg=KernelInfo()).call(*args)

pm_split_kernels = PatternMatcher([(UPat((Ops.STORE, Ops.END), name="s"), lambda s: split_kernels(s.src[0] if s.op is Ops.END else s))])

def get_kernel_graph(tsink):
  tsink = graph_rewrite(tsink, pm_kernelize, ctx=count_buffers(tsink, count_uses(tsink)), bottom_up=True, name="kernelize")
  tsink = graph_rewrite(tsink, pm_add_ranges, walk=True, name="add ranges")
  tsink = graph_rewrite(tsink, pm_convert_ranges+pm_push_ranges, bottom_up=True, name="push ranges")
  tsink = graph_rewrite(tsink, pm_canonicalize_indexing+pm_gated_index+symbolic+pm_reduce_simplify, name="finalize graph")
  tsink = graph_rewrite(tsink, pm_split_kernels, bottom_up=True, name="split kernels")

  if VIZ: graph_rewrite(tsink, PatternMatcher([]), name="view kernel graph")
  return tsink

