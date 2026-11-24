import tinygrad
from tinygrad import Tensor
from tinygrad.uop.ops import UOp, graph_rewrite
import tinygrad.codegen
from tinygrad.codegen import pm_schedule
from tinygrad.engine.realize import run_linear

from tinygrad.engine.realize import lower_and_compile as lower_and_compile_unpatched

import subprocess
import os
from dataclasses import dataclass
import functools
import pickle
import time
from typing import Any

perf_dir = os.path.dirname(__file__)
pickles = os.path.join(perf_dir, "pickles")
benchmarks = os.path.join(perf_dir, "benchmarks")
dumps = os.path.join(perf_dir, "dumps")
save_keys = os.path.join(perf_dir, "save_keys.tmp")
python = os.path.normpath(os.path.join(perf_dir, "../venv/bin/python"))
repo_root = os.path.normpath(os.path.join(perf_dir, ".."))
# tree label -> source root. tinygrad-base is a submodule pinned to the commit the fork rebases on
trees = {"tinygrad-base": os.path.join(repo_root, "tinygrad-base"), "tinygrad-fork": repo_root}

# the label is passed explicitly, so keys don't depend on the name of the directory the repo is cloned into
def tree_name(): return os.environ["PERF_TREE"]

def variant_env(base, tree, r):
  env = {k:v for k,v in base.items() if k != "R"} | {"PYTHONPATH": trees[tree], "PERF_TREE": tree}
  return env if r is None else env | {"R": r}

def path_from_key(key, subdir): return os.path.join(subdir, key)

def from_key(key, subdir): 
  assert os.path.exists(path:=path_from_key(key, subdir)), f"{path}"
  with open(path, mode="rb") as f: return pickle.load(f)

def run(cmd, env, data):
  proc = subprocess.run(cmd, env=env, cwd=perf_dir, capture_output=True, input=data)
  if len(err:=proc.stderr.decode().strip()):
    print(err)
    raise Exception("subprocess error")
  elif (rc:=proc.returncode) != 0:
    print(rc)
    raise Exception(f"subprocess failed")
  return proc

@dataclass
class Stop(Exception): 
  uops: tuple[UOp, ...]

# scheduler

def patch_scheduler_fxn(sink, pm, *args, **kwargs):
  if pm is pm_schedule: raise Stop((sink,))
  return graph_rewrite(sink, pm, *args, **kwargs)

def patch_scheduler(): tinygrad.codegen.graph_rewrite = patch_scheduler_fxn

def restore_scheduler(): tinygrad.codegen.graph_rewrite = graph_rewrite

# codegen

def patch_codegen_fxn(linear, verbose=True): raise Stop((linear,))

def patch_codegen(): tinygrad.engine.realize.lower_and_compile = patch_codegen_fxn

def restore_codegen(): tinygrad.engine.realize.lower_and_compile = lower_and_compile_unpatched

def save_uop(patches):
  def decorator(fxn):
    @functools.wraps(fxn)
    def decorated(*args, **kwargs):
      ret = fxn(*args, **kwargs)
      assert isinstance(ret, tuple) and all(isinstance(t, Tensor) for t in ret)
      uops = tuple(t.uop for t in ret)
      keys = []
      for patch, restore in patches:
        try:
          patch()
          # Fresh tensors have to get created per patch in order to force realization.
          Tensor.realize(*(Tensor(u) for u in uops))
        except Stop as e:
          stopped = e.uops
        else:
          raise Exception("patch must raise stop exception")
        finally:
          restore()
        # pickle after restore: pickling a realized Buffer copies it out, which can run (and so hit) the patched codegen
        # function name e.g. mnist, patch name, initial uops key, stopped uops key, tree name, R value
        key = ':'.join((fxn.__name__, patch.__name__, UOp.sink(*uops).key.hex(), UOp.sink(*stopped).key.hex(), tree_name(), str(os.environ.get("R"))))
        keys.append(key)
        if not os.path.exists(path:=os.path.join(pickles, key)):
          with open(path, mode="xb") as f: pickle.dump(stopped, f)
      with open(save_keys, mode="x") as f: f.write("\n".join(keys))
    return decorated
  return decorator








