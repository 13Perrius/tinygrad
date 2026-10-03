from tinygrad.uop.ops import AxisType, PatternMatcher, UOp, UPat, GroupOp, Ops, graph_rewrite, remove_all_tags, to_max_shape, KernelInfo, AddrSpace, BottomUpGate, _substitute
from tinygrad.uop.symbolic import symbolic, pm_simplify_valid, symbolic_simple, fold_nested_where, pm_drop_and_clauses
from tinygrad.codegen.simplify import pm_reduce_simplify
from tinygrad.helpers import prod, dedup, argsort, getenv
import itertools, functools

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
def apply_movement_op(op:Ops, in_shape:tuple, arg:tuple, rngs:tuple[UOp, ...]) -> tuple[UOp, ...]:
  match op:
    case Ops.SHRINK:  rngs = tuple(a if off == 0 else a+off for a,(off,_) in zip(rngs, arg))
    case Ops.PERMUTE: return tuple(rngs[p] for p in argsort(arg))
    case Ops.FLIP:    rngs = tuple((-a+(s-1)) if f else a for a,s,f in zip(rngs, in_shape, arg))
    case Ops.EXPAND:  return rngs[len(arg):]
    case Ops.PAD:
      rngs = tuple(r if (sz == sh and off == 0) else (r-off).valid(graph_rewrite((r >= off) & (r < (sh+off)),
        symbolic+pm_simplify_valid, name="pad")) for r,sh,(off,sz) in zip(rngs, in_shape, arg))
    case Ops.RESHAPE:
      sink = UOp.sink(*rngs).simplify()
      sub_array = {r:r.replace(src=r.src[:1], arg=(i, AxisType.PLACEHOLDER)) for i,r in enumerate(sink.ranges)}
      return _apply_reshape(in_shape, arg, sink.substitute(sub_array)).substitute({v:k for k,v in sub_array.items()}).src
    case _: raise RuntimeError(f"{op} is not a MovementOp")
  return graph_rewrite(UOp.sink(*rngs), symbolic_simple+fold_nested_where, name="simplify ranges").src

def new_ranges(shape, rid=itertools.count(0), ty=AxisType.WEAK): return tuple(UOp.range(sz, next(rid), ty) for i,sz in enumerate(shape))

pm_insert_expands = PatternMatcher([
  (UPat(GroupOp.Binary|GroupOp.Ternary|{Ops.STORE}, name="x"), 
  lambda x: x.replace(src=tuple(u.expand(x.shape) if u.shape != x.shape else u for u in x.src)))
])

pm_add_ranges = PatternMatcher([
  (UPat(Ops.REDUCE, src=(UPat.var("x"),), name="r"), lambda r, x: r.replace(src=(x, *new_ranges(x.shape[:r.arg[1]], ty=AxisType.REDUCE)))),
  (UPat.var("dst").store(UPat.var("src")), lambda dst, src: dst.index(*(rngs:=new_ranges(dst.shape))).store(src.index(*rngs)).end(*rngs))
])

def push_ranges(ind):
  if (x:=ind.src[0]).op is Ops.CONST: return x
  elif x.op is Ops.REDUCE: return x.replace(src=(x.src[0].index(*(rr:=x.src[1:]), *ind.src[1:]), *rr), arg=(x.arg[0], 0))
  elif x.op in GroupOp.Elementwise:
    rngs = ind.src[1:]
    return x.replace(src=tuple(u.index(*rngs) for u in x.src))

pm_push_ranges = PatternMatcher([
  (UPat(Ops.INDEX, name="ind"), push_ranges),
  (UPat(GroupOp.Movement-{Ops.PAD}, name="m", src=(UPat.var("x"),), allow_any_len=True).index(name="ind", allow_any_len=True),
  lambda ind, m, x: x.index(*apply_movement_op(m.op, x.shape, m.marg, ind.src[1:]))),
])

def convert_stack_to_where(ind, x):
  req, rngs = ind.src[1], ind.src[2:]
  acc = x.src[-1].index(*rngs)
  for j in range(len(x.src)-2, -1, -1): acc = req.eq(j).where(x.src[j].index(*rngs), acc)
  return acc

def convert_pad_to_where(ind, x):
  pad_rngs = apply_movement_op(x.op, x.src[0].shape, x.marg, ind.src[1:])
  valid = UOp.const(True).uprod(*(r.get_valid() for r in pad_rngs))
  return valid.where(x.src[0].index(*pad_rngs), UOp.const(x.dtype.const(0)))

pm_convert_ranges = PatternMatcher([
  (UPat(Ops.STACK, name="x").index(allow_any_len=True, name="ind"), convert_stack_to_where),
  (UPat(Ops.PAD, name="x").index(allow_any_len=True, name="ind"), convert_pad_to_where)
])

REALIZE_OP_SRCS = {Ops.MSELECT, Ops.MSTACK}

def count_consumes(tsink):
  candidates, consumes = {}, {tsink:0}
  for x in reversed(tsink.toposort(enter_calls=False)):
    if x.op in GroupOp.Elementwise|{Ops.REDUCE} and not x.is_virtual and consumes[x] > 1:
      candidates[x] = candidates.get(x,False) 
      consumes[x] = 1
    if x.op is Ops.STORE: consumes[x] = 1
    if x.op is Ops.EXPAND: consumes[x] *= x.max_numel() // x.src[0].max_numel()
    for i,s in enumerate(x.src): consumes[s] = consumes.get(s,0) + (consumes[x] if x.op is not Ops.STORE or i > 0 else 0)
    if x.op in REALIZE_OP_SRCS:
      for s in x.src: 
        if not (sb:=s.base).has_buffer_identity(after_ok=True) and not sb.is_virtual: candidates[sb] = True
  return candidates

def realize(ctx, x):
  info, dev = ctx
  if x.op in REALIZE_OP_SRCS: 
    info[ret] = ([ret:=x.replace(src=tuple(s.base for s in x.src)).view_as(x.shape)], False) 
    return ret
  if x.has_buffer_identity(after_ok=True) or x.op is Ops.CALL:
    info[x] = ([x], False)
    return None
  src_bufs, src_red = zip(*(info[s] for s in x.src)) if x.src else ([], [])
  bufs, red = dedup(sum(src_bufs, [])), any(src_red)
  if x.op is Ops.REDUCE and bufs: red = True
  if x.tag is not None and (x.tag or len(bufs) > 3 or red):
    b = UOp.new_buffer(dev if x.device is None else x.device, prod(to_max_shape(x.shape)), x.dtype)
    info[ret] = ([ret:=b.after(b.view_as(x.shape).store(x.rtag())).view_as(x.shape)], False)
    return ret
  info[x] = (bufs, red)

pm_realize = PatternMatcher([(UPat(GroupOp.All, name="x"), realize)])

pm_canonicalize_index = PatternMatcher([
  (UPat.var("x").index(), lambda x: x.index(UOp.const(0)) if x.has_buffer_identity(after_ok=True) else x)
])

def add_arg(ctx, x):
  if x.op in {Ops.PARAM, Ops.BUFFER} and x.arg.addrspace is AddrSpace.ALU: return x.replace(op=Ops.PARAM)
  if not (x.has_buffer_identity(after_ok=True) and x.tag is None): return None
  ctx[1].append(x)
  return x.param_like(slot=len(ctx[1])-1).rtag()

pm_kernel = PatternMatcher([
  (UPat(GroupOp.All, name="x"), add_arg),
  (UPat(Ops.RANGE, name="r"), 
  lambda ctx, r: r.replace(arg=(-1 if r.arg[1] is AxisType.DEVICE else next(ctx[0]), r.arg[1])).rtag(r.arg[0]) if r.tag is None else None)
])

def split_kernels(s):
  s = graph_rewrite(s, pm_kernel, ctx=(split_ctx:=(itertools.count(0), [])), bottom_up=True, name="kernel")
  return s.end(*sorted(s.ranges, key=lambda r: r.tag)).sink(arg=KernelInfo()).call(*split_ctx[1])

pm_split_kernels = PatternMatcher([(UPat((Ops.STORE, Ops.END), name="s"), lambda s: split_kernels(s.src[0] if s.op is Ops.END else s))])

def get_kernel_graph(tsink):
  tsink = graph_rewrite(tsink, pm_insert_expands, name="insert expands")
  candidates = count_consumes(tsink)

  tsink = graph_rewrite(tsink, _substitute, ctx={x:x.rtag(t) for x,t in candidates.items()}, bottom_up=True, name="tag candidates")
  tsink = graph_rewrite(tsink, pm_realize, ctx=({}, tsink.device), walk=True, name="realize")
  tsink = graph_rewrite(tsink, remove_all_tags, walk=True)

  tsink = graph_rewrite(tsink, pm_add_ranges, walk=True, name="add ranges")
  tsink = graph_rewrite(tsink, pm_push_ranges+pm_convert_ranges, bottom_up=True, name="push ranges")

  tsink = graph_rewrite(tsink, symbolic+pm_reduce_simplify+pm_canonicalize_index, name="simplify graph")
  tsink = graph_rewrite(tsink, pm_split_kernels, bottom_up=True, name="split kernels")
  tsink = graph_rewrite(tsink, remove_all_tags, walk=True, enter_calls=True)
  return tsink

