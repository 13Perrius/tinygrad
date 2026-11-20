from tinygrad.helpers import disable_gc
from tinygrad.uop.ops import graph_rewrite, Ops
from tinygrad.codegen import pm_schedule
from tinygrad.codegen import do_to_program, to_program_key
from tinygrad.engine.realize import _get_call_to_compile

from perf import from_key, pickles, benchmarks
from debug_2 import run_debug_2_runtime

import os
import pickle
import time

@disable_gc()
def run_scheduler(key):
  call, = from_key(key, pickles)
  start = time.perf_counter_ns()
  linear = graph_rewrite(call, pm_schedule, name="schedule to linear", enter_calls=True)
  end = time.perf_counter_ns()

  res = end-start
  with open(os.path.join(benchmarks, key), mode="wb") as f: pickle.dump(res, f)

@disable_gc()
def run_codegen(key):
  linear, = from_key(key, pickles)
  todo = {to_program_key(*a): a for x in linear.toposort() if x.op is Ops.CALL and (a:=_get_call_to_compile(x)) is not None}
  for _, renderer in todo.values(): renderer.compiler.compile_cached = lambda src: b""

  start = time.perf_counter_ns()
  for ast, renderer in todo.values(): do_to_program(ast, renderer)
  end = time.perf_counter_ns()

  res = end-start
  with open(os.path.join(benchmarks, key), mode="wb") as f: pickle.dump(res, f)

PATCH_FXNS = {"patch_scheduler": run_scheduler, "patch_codegen": run_codegen, "patch_debug_2_runtime": run_debug_2_runtime}

