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
  out = io.StringIO()
  with Context(DEBUG=2), contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()): Tensor.realize(*(Tensor(u) for u in uops))

  tms = [float(m.group(1)) for line in ANSI.sub("", out.getvalue()).splitlines() if (m:=TM.search(line))]
  assert len(tms), f"no kernel timings found in DEBUG=2 output for {key}"
  assert abs(tms[-1] - GlobalCounters.time_sum_s*1e3) < 0.01, f"{tms[-1]} ms != {GlobalCounters.time_sum_s*1e3} ms"

  res = GlobalCounters.time_sum_s*1e9
  with open(os.path.join(benchmarks, key), mode="wb") as f: pickle.dump(res, f)
