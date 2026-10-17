import itertools, functools
from enum import Enum
from tinygrad.uop.ops import AxisType, PatternMatcher, UOp, UPat, GroupOp, Ops, graph_rewrite, remove_all_tags, KernelInfo, BottomUpGate, _substitute, broadcast_axes
#TODO: revert symbolic, dont need fold_nested_where anymore
from tinygrad.uop.symbolic import symbolic, pm_simplify_valid, symbolic_simple, fold_nested_where, pm_drop_and_clauses
from tinygrad.codegen.simplify import pm_reduce_simplify
from tinygrad.helpers import dedup, argsort
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

def broadcast_ranges(x:UOp, src:UOp, rngs:tuple[UOp, ...]) -> tuple[UOp, ...]:
  if x.op not in GroupOp.Broadcastable: return rngs
  baxes, nleft = broadcast_axes(src.shape, x.shape), len(x.shape)-len(src.shape)
  return tuple(r.const_like(0) if j in baxes else r for j,r in enumerate(rngs) if j >= nleft)

def new_ranges(shape, rid=itertools.count(0)): return tuple(UOp.range(sz, next(rid)) for sz in shape)

pm_add_ranges = PatternMatcher([
  (UPat.var("x").reduce(name="r"), lambda r, x: x.reduce(*new_ranges(x.shape[:r.arg[1]]), arg=r.arg)),
  (UPat.var("dst").store(UPat.var("src")), lambda dst, src: dst.index(*(r:=new_ranges(dst.shape))).store(src.index(*r)).end(*r)),
])

def push_ranges(ind, x):
  match x.op:
    case Ops.CONST: return x
    case Ops.REDUCE: return x.src[0].index(*(rr:=x.src[1:]), *ind.src[1:]).reduce(*rr, arg=(x.arg[0], 0))
    case op if op in GroupOp.Elementwise:
      rngs = ind.src[1:]
      return x.replace(src=tuple(u.index(*broadcast_ranges(x, u, rngs)) for u in x.src))
    case _: return None

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

def tag_with_use(tsink):
  uses: dict[UOp, UOp|Kernelize] = {tsink:tsink}
  tags: dict[UOp, UOp] = {}
  broadcastable: set[UOp] = GroupOp.Broadcastable|{Ops.EXPAND}
  for x in reversed(tsink.toposort(enter_calls=False)):
    # Kernels, movement ops, and REDUCE are all distinct uses.  >1 distinct use => candidacy.
    if x.has_buffer_identity(after_ok=True): use = x
    elif isinstance(use:=uses[x], Kernelize) and (x.op in GroupOp.Elementwise or x.op is Ops.REDUCE or use is Kernelize.FORCE):
      tags[x] = x.rtag(use if x.max_numel() > 1 else Kernelize.CANDIDATE) 
      use = x
    elif x.op in GroupOp.Movement|{Ops.REDUCE} and not isinstance(use, Kernelize): use = x
    for s in x.src:
      # Broadcasting => candidacy.  This rule overrides Kernelize.RANGES because broadcasted ranges are dropped by srcs anyways.  
      su = Kernelize.CANDIDATE if x.op in broadcastable and use is not Kernelize.FORCE and x.max_numel() >= 2*s.base.max_numel() else use
      if (curr:=uses.setdefault(s, su)) is not su and curr is not Kernelize.FORCE: uses[s] = Kernelize.RANGES
    # MSTACK, MSELECT, and custom kernel srcs must be kernelized.  MSTACK and MSELECT srcs drop their views.
    if x.op is Ops.MSTACK or x.op is Ops.MSELECT:
      for s in x.src:
        if not (sb:=s.base).has_buffer_identity(after_ok=True): uses[sb] = Kernelize.FORCE
    elif x.op is Ops.CALL and isinstance(x.body.arg, KernelInfo):
      for s in x.src[1:]:
        if not (s.has_buffer_identity(after_ok=True) or s.is_variable): uses[s] = Kernelize.FORCE
  return graph_rewrite(tsink, _substitute, ctx=tags, bottom_up=True, name="tag with use")

def kernelize(ctx, x):
  info, dev, ret = *ctx, x.rtag(None)
  if ret.op is Ops.MSTACK or ret.op is Ops.MSELECT:
    info[ret] = ([ret:=ret.replace(src=(tuple(s.base for s in ret.src))).view_as(ret.shape)], False)
  elif ret.has_buffer_identity(after_ok=True) or ret.op is Ops.CALL:
    info[ret] = ([ret], False)
  else:
    src_bufs, src_red = zip(*(info[s] for s in ret.src)) if ret.src else ([], [])
    bufs, red = dedup(sum(src_bufs, [])), any(src_red)
    if ret.op is Ops.REDUCE and bufs: red = True
    if isinstance(use:=x.tag, Kernelize) and (use is Kernelize.FORCE or len(bufs) > 3 or red):
      a = UOp.alloc(ret.shape, ret.dtype, device=dev if ret.device is None else ret.device)
      bufs, red = [ret:=a.base.after(a.store(ret)).view_as(ret.shape)], False
    elif use is Kernelize.RANGES: ret = ret.index(*(r:=new_ranges(ret.shape))).stage(*r)
    info[ret] = (bufs, red)
  return ret

pm_kernelize = PatternMatcher([(UPat(GroupOp.All, name="x"), kernelize)])

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
  if not (x.has_buffer_identity(after_ok=True) and x.tag is None): return None
  ctx[1].append(x)
  return x.param_like(slot=len(ctx[1])-1).rtag()

pm_kernel = PatternMatcher([
  (UPat(GroupOp.All, name="x"), add_arg),
  (UPat(Ops.RANGE, name="r"), lambda ctx, r: r.replace(arg=(r.arg[0], ctx[0][r.arg])).rtag() if r.tag is None else None)
])

def split_kernels(s):
  rargs = (x.arg for x in s.toposort(gate=lambda x: not x.has_buffer_identity(after_ok=True)) if x.op is Ops.RANGE)
  s = graph_rewrite(s, pm_kernel, ctx=(split_ctx:=({a:i for i,a in enumerate(sorted(rargs))}, [])), bottom_up=True, name="kernel")
  return s.end(*sorted(s.ranges, key=lambda r: r.arg)).sink(arg=KernelInfo()).call(*split_ctx[1])

pm_split_kernels = PatternMatcher([(UPat((Ops.STORE, Ops.END), name="s"), lambda s: split_kernels(s.src[0] if s.op is Ops.END else s))])

def get_kernel_graph(tsink):
  tsink = graph_rewrite(tag_with_use(tsink), pm_kernelize, ctx=({}, tsink.device), walk=True, name="kernelize")
  tsink = graph_rewrite(tsink, pm_add_ranges, walk=True, name="add ranges")
  tsink = graph_rewrite(tsink, pm_convert_ranges+pm_push_ranges, bottom_up=True, name="push ranges")

  tsink = graph_rewrite(tsink, pm_canonicalize_indexing+pm_gated_index+symbolic+pm_reduce_simplify, name="finalize graph")
  tsink = graph_rewrite(tsink, pm_split_kernels, bottom_up=True, name="split kernels")
  return graph_rewrite(tsink, remove_all_tags, walk=True, enter_calls=True)

