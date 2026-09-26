from tinygrad.schedule.indexing import apply_movement_op
from tinygrad.uop.ops import AxisType, PatternMatcher, UOp, UPat, GroupOp, Ops, graph_rewrite, remove_all_tags, to_max_shape, KernelInfo, AddrSpace, BottomUpGate, _substitute
from tinygrad.helpers import prod, dedup
import itertools 

def new_ranges(shp, rid=itertools.count(0), ty=AxisType.WEAK): return tuple(UOp.range(sz, next(rid), ty) for i,sz in enumerate(shp))

pm_insert_expands = PatternMatcher([
  (UPat(GroupOp.Binary|GroupOp.Ternary|{Ops.STORE}, name="x"), 
  lambda x: x.replace(src=tuple(u.expand(x.shape) if u.shape != x.shape else u for u in x.src)))
])

pm_add_ranges = PatternMatcher([
  (UPat(Ops.REDUCE, src=(UPat.var("x"),), name="r"), lambda r, x: r.replace(src=(x, *new_ranges(x.shape[:r.arg[1]], ty=AxisType.REDUCE)))),
  (UPat.var("dst").store(UPat.var("src")), 
  lambda dst, src: UOp(Ops.STAGE, (dst.index(*(rngs:=new_ranges(dst.shape))), *rngs)).store(UOp(Ops.STAGE, (src.index(*rngs), *rngs))))
])

def substitute_ranges(ctx, x):
  if x.op is Ops.STAGE: raise BottomUpGate()
  return ctx.get(x)

pm_substitute_ranges = PatternMatcher([(UPat(GroupOp.All, name="x"), substitute_ranges)])

def compose_ranges(ind, st): return graph_rewrite(st.src[0], pm_substitute_ranges, ctx=dict(zip(st.src[1:], ind.src[1:])), bottom_up=True)

def push_ranges(ind):
  if (x:=ind.src[0]).op is Ops.REDUCE:
    return x.replace(src=(x.src[0].index(*(rr:=x.src[1:]), *ind.src[1:]), *rr), arg=(x.arg[0], 0))
  elif x.op in GroupOp.Elementwise:
    rngs = ind.src[1:]
    return x.replace(src=tuple(u.index(*rngs) for u in x.src))

pm_fold_ranges = PatternMatcher([
  (UPat(Ops.INDEX, name="ind"), push_ranges),
  (UPat(Ops.STAGE, name="st").index(allow_any_len=True, name="ind"), compose_ranges),
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
  unremovable, consumes = {}, {tsink:0}
  for x in reversed(tsink.toposort(enter_calls=False)):
    if (contig:=x.op is Ops.CONTIGUOUS) or (x.op in GroupOp.Elementwise|{Ops.REDUCE} and not x.is_virtual and consumes[x] > 1):
      unremovable[x] = unremovable.get(x,False) or contig
      consumes[x] = 1
    if x.op is Ops.STORE: consumes[x] = 1
    if x.op is Ops.EXPAND: consumes[x] *= x.max_numel() // x.src[0].max_numel()
    for i,s in enumerate(x.src): consumes[s] = consumes.get(s,0) + (consumes[x] if x.op is not Ops.STORE or i > 0 else 0)
    if x.op in REALIZE_OP_SRCS:
      for s in x.src: 
        if not (sb:=s.base).has_buffer_identity(after_ok=True) and not sb.is_virtual: unremovable[sb] = True
  return unremovable

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
    info[ret] = ([ret:=b.after(b.view_as(x.shape).store(x.src[0] if x.op is Ops.CONTIGUOUS else x.rtag())).view_as(x.shape)], False)
    return ret
  info[x] = (bufs, red)

pm_realize = PatternMatcher([(UPat(GroupOp.All, name="x"), realize)])

def canonicalize_index(ind, x): return x.index(UOp.const(0)) if x.has_buffer_identity(after_ok=True) else x

pm_presplit = PatternMatcher([
  (UPat(Ops.STAGE, name="st"), lambda st: st.src[0]),
  (UPat(Ops.INDEX, src=(UPat.var("x"),), name="ind", allow_any_len=True), lambda ind, x: None if x.ndim > 0 else canonicalize_index(ind, x))
])

def add_arg(ctx, x):
  if x.op is Ops.PARAM and x.addrspace is AddrSpace.ALU: return x.rtag()
  if not (x.has_buffer_identity(after_ok=True) and x.tag is None): return None
  ctx[1].append(x)
  return x.param_like(slot=len(ctx[1])-1).rtag()

pm_kernel_arg = PatternMatcher([
  (UPat(GroupOp.All, name="x"), add_arg),
  (UPat(Ops.RANGE, name="r"), 
  lambda ctx, r: r.replace(arg=(-1 if r.arg[1] is AxisType.DEVICE else next(ctx[0]), r.arg[1])).rtag(r.arg[0]) if r.tag is None else None)
])

def split_kernels(s):
  s = graph_rewrite(s, pm_kernel_arg, ctx=(kernel_ctx:=(itertools.count(0), [])), bottom_up=True)
  return s.end(*sorted(s.ranges, key=lambda r: r.tag)).sink(arg=KernelInfo()).call(*kernel_ctx[1])

pm_split_kernels = PatternMatcher([(UPat(Ops.STORE, name="s"), split_kernels)])

def run_rangeify(tsink, b):
  tsink = graph_rewrite(tsink, pm_insert_expands, name="insert expands")
  unremovable = count_consumes(tsink)

  tsink = graph_rewrite(tsink, _substitute, ctx={x:x.rtag(t) for x,t in unremovable.items()}, bottom_up=True, name="tag candidates")
  tsink = graph_rewrite(tsink, pm_realize, ctx=({}, tsink.device), walk=True, name="realize")
  tsink = graph_rewrite(tsink, remove_all_tags, walk=True)

  tsink = graph_rewrite(tsink, pm_add_ranges, walk=True, name="add ranges")
  tsink = graph_rewrite(tsink, pm_fold_ranges+pm_convert_ranges, bottom_up=True, name="fold ranges")

  tsink = graph_rewrite(tsink, pm_presplit, walk=True, name="prepare to split kernels")
  tsink = graph_rewrite(tsink, pm_split_kernels, bottom_up=True, name="split kernels")
  #TODO: remove tags from split in post-split pass analogous to the one in r=0 rangeify, with reduce_simplify etc.?
  # something like: tsink = graph_rewrite(tsink, remove_all_tags, enter_calls=True, walk=True) works 
  return tsink

