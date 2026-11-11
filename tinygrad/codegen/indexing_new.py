from dataclasses import dataclass
import functools
from tinygrad.uop.ops import Ops, UOp, graph_rewrite, sint, AxisType, AddrSpace
from tinygrad.uop.symbolic import symbolic, pm_simplify_valid, pm_drop_and_clauses
from tinygrad.helpers import argsort

@functools.cache
def apply_reshape(in_shape:tuple[sint, ...], out_shape:tuple[sint, ...], urngs:UOp) -> UOp:
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
def apply_movement_op(op:Ops, in_shape:tuple[sint, ...], arg:tuple, rngs:tuple[UOp, ...]) -> tuple[UOp, ...]:
  match op:
    case Ops.PERMUTE: return tuple(rngs[p] for p in argsort(arg))
    case Ops.EXPAND: return rngs[len(arg):]
    case Ops.PAD:
      # Pad conversion should solely use the new valid, so only the condition is rewritten.
      return tuple(r if (sz == sh and off == 0) else (r-off).valid(graph_rewrite((r >= off) & (r < (sh+off)),
        symbolic+pm_simplify_valid, name="pad")) for r,sh,(off,sz) in zip(rngs, in_shape, arg))
    case Ops.RESHAPE:
      sink = UOp.sink(*rngs).simplify() # This applies any commutative flips to the ranges early.
      subs = {r:r.replace(arg=(AxisType.PLACEHOLDER, i)) for i,r in enumerate(sink.ranges)}
      return apply_reshape(in_shape, arg, sink.substitute(subs)).substitute({v:k for k,v in subs.items()}).src
    case Ops.SHRINK: return tuple(a if off == 0 else a+off for a,(off,_) in zip(rngs, arg))
    case _: raise RuntimeError(f"{op} is not a MovementOp")

@dataclass(frozen=True)
class StageOpts:
  device: str|tuple[str, ...]|None = None
  addrspace: AddrSpace = AddrSpace.GLOBAL

