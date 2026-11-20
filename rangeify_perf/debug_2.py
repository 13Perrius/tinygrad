from tinygrad import Tensor
from tinygrad.helpers import Context, GlobalCounters, disable_gc

from perf import Stop, from_key, pickles, benchmarks

import contextlib
import io
import os
import pickle
import re

realize_unpatched = Tensor.realize

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TM = re.compile(r"\btm\s.*?/\s*([0-9.]+)ms")

def patch_debug_2_runtime_fxn(self, *lst, **kwargs): raise Stop((self.uop, *(t.uop for t in lst)))

def patch_debug_2_runtime(): Tensor.realize = patch_debug_2_runtime_fxn

def restore_debug_2_runtime(): Tensor.realize = realize_unpatched

@disable_gc()
def run_debug_2_runtime(key):
  uops = from_key(key, pickles)

  def realize() -> str:
    GlobalCounters.reset()
    out = io.StringIO()
    # fresh Tensors per realize, so each one runs the full schedule
    with Context(DEBUG=2), contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()): Tensor.realize(*(Tensor(u) for u in uops))
    return out.getvalue()

  # warm-up: compile and first launch of every kernel happen outside of the timed run
  realize()
  warm_kernel_count = GlobalCounters.kernel_count
  out = realize()
  assert GlobalCounters.kernel_count == warm_kernel_count, f"warm-up ran {warm_kernel_count} kernels, timed run {GlobalCounters.kernel_count}"

  tms = [float(m.group(1)) for line in ANSI.sub("", out).splitlines() if (m:=TM.search(line))]
  assert len(tms), f"no kernel timings found in DEBUG=2 output for {key}"
  assert abs(tms[-1] - GlobalCounters.time_sum_s*1e3) < 0.01, f"{tms[-1]} ms != {GlobalCounters.time_sum_s*1e3} ms"

  # estimated ops and bytes are tinygrad's static per-kernel estimates, summed over the timed run
  res = {"run_debug_2_runtime": GlobalCounters.time_sum_s*1e9, "estimated_ops": GlobalCounters.global_ops,
         "estimated_mem": GlobalCounters.global_mem}
  with open(os.path.join(benchmarks, key), mode="wb") as f: pickle.dump(res, f)
