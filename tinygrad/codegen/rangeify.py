import itertools
from dataclasses import dataclass
from enum import Enum
from tinygrad.uop.ops import PatternMatcher, UOp, UPat, GroupOp, Ops, graph_rewrite, KernelInfo, BottomUpGate, resolve, sint, AddrSpace, StageOpts
from tinygrad.uop.symbolic import symbolic
from tinygrad.codegen.simplify import pm_reduce_simplify
from tinygrad.helpers import dedup, VIZ, SPEC
from tinygrad.schedule.prepare import pm_gated_index
from tinygrad.uop.movement import mop_cleanup
from tinygrad.schedule.indexing import apply_reshape, apply_movement_op

def new_ranges(shape:tuple[sint, ...], rid=itertools.count(0)) -> tuple[UOp, ...]: return tuple(UOp.range(sz, next(rid)) for sz in shape)

def broadcast_ranges(x:UOp, s:UOp, rngs:tuple[UOp, ...]) -> tuple[UOp, ...]:
  if x.op not in GroupOp.Broadcastable: return rngs
  return tuple(UOp.const(0) if resolve(sz == 1, default=False) else r for sz,r in zip(s.shape, rngs[len(x.shape)-len(s.shape):]))

pm_add_ranges = PatternMatcher([
  (UPat.var("dst").store(UPat.var("src")), lambda dst, src: dst.index(*(r:=new_ranges(dst.shape))).store(src.index(*r)).end(*r)),
  (UPat.var("x").stage(name="stg"), lambda stg, x: x.index(*(r:=new_ranges(x.shape))).stage(*r) if stg.arg is None else None),
  (UPat.var("x").reduce(name="r", allow_any_len=True), lambda r, x: x.reduce(*new_ranges(x.shape[:r.arg[1]]), *r.src[1:], arg=r.arg))
])

def push_ranges(ind:UOp, x:UOp) -> UOp|None:
  if x.op is Ops.REDUCE:
    return x.src[0].index(*(rr:=x.src[1:])[:x.arg[1]], *ind.src[1:]).reduce(*rr, arg=(x.arg[0], 0))
  elif x.op in GroupOp.Elementwise:
    rngs = ind.src[1:]
    return x.replace(src=tuple(s if s.op is Ops.CONST else s.index(*broadcast_ranges(x, s, rngs)) for s in x.src))
  return None

pm_push_ranges = PatternMatcher([
  (UPat.var("x").index(name="j", allow_any_len=True).index(name="i", allow_any_len=True), lambda i, j, x: x.index(*j.src[1:], *i.src[1:])),
  (UPat.var("x").index(name="ind", allow_any_len=True), push_ranges),
  (UPat(GroupOp.Movement, name="m", src=(UPat.var("x"),), allow_any_len=True).index(name="ind", allow_any_len=True),
  lambda ind, m, x: x.index(*apply_movement_op(m.op, x.shape, m.marg, ind.src[1:])))
])

def stack_select(rcmp, srcs):
  # MSB-first bit tree.
  if len(srcs) == 1: return srcs[0]
  half = 1 << ((len(srcs)-1).bit_length()-1)
  return (rcmp % (2*half) < half).where(stack_select(rcmp, srcs[:half]), stack_select(rcmp, srcs[half:]))

def convert_stack_to_where(ind:UOp, x:UOp) -> UOp: return stack_select(ind.src[1], [s.index(*ind.src[2:]) for s in x.src])

def convert_pad_to_where(ind:UOp, x:UOp) -> UOp:
  rngs = apply_movement_op(x.op, x.src[0].shape, x.marg, ind.src[1:])
  valid = UOp.const(True).uprod(*(r.get_valid() for r in rngs))
  return valid.where(x.src[0].index(*rngs), UOp.const(x.dtype.const(0)))

pm_convert_ranges = PatternMatcher([
  (UPat(Ops.STACK, name="x").index(name="ind", allow_any_len=True), convert_stack_to_where),
  (UPat(Ops.PAD, name="x").index(name="ind", allow_any_len=True), convert_pad_to_where)
])

Stageable = Enum('Stageable', [('CANDIDATE', 0), ('RANGES', 1), ('FORCE', 2)])

def kernel_boundary(x:UOp) -> bool:
  return x.has_buffer_identity(after_ok=True) or (x.op is Ops.STAGE and x.arg is not None and x.arg.addrspace is AddrSpace.GLOBAL)

def count_uses(tsink:UOp) -> dict[UOp, Stageable]:
  uses: dict[UOp, UOp|Stageable] = {}
  stageable: dict[UOp, Stageable] = {}
  for x in reversed(tsink.toposort()):
    if x is tsink: continue
    if (is_stageable:=isinstance(use:=uses.get(x,x), Stageable)) and x.op in GroupOp.Elementwise|{Ops.REDUCE,Ops.SHRINK}:
      stageable.setdefault(x, use)
    # A kernel, a movement op, or a REDUCE are all distinct uses. >=2 distinct uses => stageable.
    if x in stageable or kernel_boundary(x) or (x.op in GroupOp.Movement|{Ops.REDUCE} and not is_stageable): use = x
    for s in x.src:
      # A use of a broadcasted src is always CANDIDATE because it drops leading ranges, so distinct uses might still index into it uniformly.
      src_use = Stageable.CANDIDATE if x.op in GroupOp.Broadcastable|{Ops.EXPAND} and x.max_numel() >= 2*s.max_numel() else use
      if uses.setdefault(s, src_use) is not src_use: uses[s] = Stageable.RANGES
    # MSTACK, MSELECT, and custom kernel or program srcs are always materialized. MSTACK and MSELECT srcs drop their views.
    if x.op is Ops.MSTACK or x.op is Ops.MSELECT or (x.op is Ops.CALL and (x.body.op is Ops.PROGRAM or isinstance(x.body.arg, KernelInfo))):
      for s in x.src_without_body:
        if not (kernel_boundary(sb:=s if x.op is Ops.CALL else s.base) or sb.is_variable): stageable[sb] = Stageable.FORCE
  return stageable

def count_buffers(tsink:UOp) -> dict[UOp, Stageable]:
  info: dict[UOp, tuple[list[UOp], bool]] = {}
  stageable = count_uses(tsink)
  for x in tsink.toposort():
    if kernel_boundary(x) or x.op in {Ops.MSTACK,Ops.MSELECT,Ops.CALL}:
      info[x] = ([x], False)
      continue
    src_bufs, src_red = zip(*(info[s] for s in x.src)) if x.src else ([], [])
    bufs, red = dedup(sum(src_bufs, [])), any(src_red)
    if x.op is Ops.REDUCE and bufs: red = True
    if (v:=stageable.get(x)) is not None:
      if (len(bufs) > 3 or red) and x.ndim > 0: v = stageable[x] = Stageable.FORCE
      if v is Stageable.FORCE: bufs, red = [x], False
      elif v is not Stageable.RANGES or x.max_numel() == 1: del stageable[x]
    info[x] = (bufs, red)
  return stageable

@dataclass
class StageCtx:
  stageable: dict[UOp, Stageable]
  device: str|tuple[str, ...]

def insert_stage(ctx, s):
  if (v:=ctx.stageable.get(s)) is None: return s
  return s.stage(arg=StageOpts(device=ctx.device if s.device is None else s.device) if v is Stageable.FORCE else None)

pm_insert_stages = PatternMatcher([
  (UPat(GroupOp.All-{Ops.STAGE,Ops.SINK}, name="x"), lambda ctx, x: x.replace(src=tuple(insert_stage(ctx, s) for s in x.src)))
])

def stage_to_anon_store(stg:UOp):
  # The buffer created here is inside the call and is not persisted, like the buffers created for copies.
  if stg.arg is None or stg.arg.addrspace is not AddrSpace.GLOBAL: return None
  alloc = UOp.alloc(stg.shape, stg.dtype, device=stg.device)
  return alloc.base.after(alloc.index(*r).store(stg.src[0]).end(*r) if (r:=stg.src[1:]) else alloc.store(stg.src[0])).view_as(stg.shape)

pm_stage_to_anon_store = PatternMatcher([
  (UPat(Ops.STAGE, name="stg"), stage_to_anon_store),
  (UPat((Ops.MSELECT, Ops.MSTACK), name="m"), lambda m: m.replace(src=tuple(s.base for s in m.src)).view_as(m.shape))
])

def substitute_ranges(ctx, x:UOp) -> UOp|None:
  if not any(r in ctx for r in x.ranges): raise BottomUpGate()
  return ctx.get(x)

pm_substitute_ranges = PatternMatcher([(UPat(GroupOp.All, name="x"), substitute_ranges)])

def compose_ranges(ind:UOp, stg:UOp) -> UOp:
  return graph_rewrite(stg.src[0], pm_substitute_ranges, ctx=dict(zip(stg.src[1:], ind.src[1:])), bottom_up=True)

pm_compose_ranges = PatternMatcher([
  # This is where a stage inserted via Stageable.RANGES is folded away.
  (UPat(Ops.STAGE, name="stg").index(name="ind", allow_any_len=True), lambda ind, stg: compose_ranges(ind, stg) if stg.arg is None else None),
])

def fix_store_hazard(ind:UOp, dst:UOp, src:UOp) -> UOp|None:
  reads = [r for r in src.toposort(gate=lambda x: not kernel_boundary(x)) if r.op is Ops.INDEX and r.buf_uop is dst.buf_uop]
  if any(r.src[1].get_idx() is not ind.src[1].get_idx() for r in reads):
    frngs = apply_reshape(dst.shape, (st:=src.stage(*src.ranges)).shape, UOp.sink(*src.ranges)).src
    store = UOp.alloc(dst.shape, src.dtype, device=src.device).index(*frngs).store(src)
    end = graph_rewrite(store, pm_substitute_ranges, ctx=dict(zip(src.ranges, srngs:=new_ranges(st.shape))), bottom_up=True).end(*srngs)
    return ind.store(end.buf_uop.after(end).index(*frngs))
  return None

pm_finalize_graph = PatternMatcher([
  (UPat.var("x").index(), lambda x: x.index(UOp.const(0)) if x.has_buffer_identity(after_ok=True) else x),
  (UPat(GroupOp.Movement|{Ops.UNSHARD}, name="m", src=(UPat.var("x"),), allow_any_len=True), lambda m, x: x),
  (UPat.var("dst").index(name="ind", allow_any_len=True).store(UPat.var("src")), fix_store_hazard),
  (UPat.var("x").store(UPat.var("x")), lambda x: UOp(Ops.NOOP)),
  (UPat(Ops.NOOP, name="n").end(allow_any_len=True), lambda n: n),
  (UPat(Ops.REDUCE, name="r"), lambda r: r.replace(src=(r.src[0], *(s for s in r.src[1:] if s.op is not Ops.CONST))))
])

def add_arg(ctx, x:UOp) -> UOp|None:
  if not kernel_boundary(x): return None
  (args:=ctx[0]).append(x)
  return x.param_like(slot=len(args)-1)

def add_range(ctx, r:UOp) -> UOp: return r.replace(arg=(r.arg[0], ctx[1][r.arg]))

pm_kernel = PatternMatcher([(UPat(GroupOp.All, name="x"), add_arg), (UPat(Ops.RANGE, name="r"), add_range)])

def split_kernels(s:UOp) -> UOp:
  args: list[UOp] = []
  # Codegen prefers that ranges are ordered and renumbered according to creation order.
  rargs = [x.arg for x in s.toposort(gate=lambda x: not kernel_boundary(x)) if x.op is Ops.RANGE]
  s = graph_rewrite(s, pm_kernel, ctx=(args, {arg:i for i,arg in enumerate(sorted(rargs))}), bottom_up=True, walk=True, name="kernel")
  return s.end(*sorted(s.ranges, key=lambda r: r.arg)).sink(arg=KernelInfo()).call(*args)

pm_split_kernels = PatternMatcher([(UPat((Ops.STORE, Ops.END), name="s"), lambda s: split_kernels(s.src[0] if s.op is Ops.END else s))])

def get_kernel_graph(tsink:UOp) -> UOp:
  tsink = graph_rewrite(tsink, pm_insert_stages, ctx=StageCtx(count_buffers(tsink), tsink.device), bottom_up=True, name="insert stages")
  tsink = graph_rewrite(tsink, mop_cleanup+pm_stage_to_anon_store, walk=True, name="stage to anon store")
  tsink = graph_rewrite(tsink, pm_add_ranges, walk=True, name="add ranges")
  tsink = graph_rewrite(tsink, pm_convert_ranges+pm_push_ranges, bottom_up=True, name="push ranges")
  tsink = graph_rewrite(tsink, pm_compose_ranges+pm_finalize_graph+pm_gated_index+symbolic+pm_reduce_simplify, name="finalize graph")
  tsink = graph_rewrite(tsink, pm_split_kernels, bottom_up=True, name="split kernels")

  if VIZ: graph_rewrite(tsink, PatternMatcher([]), name="View Kernel Graph")
  if SPEC:
    from tinygrad.uop.spec import type_verify, spec_kernel_graph
    type_verify(tsink, spec_kernel_graph)
  return tsink

