import os
import sys

# the parent process imports the fork's tinygrad from source, so ./venv needs no install. each variant runs in a subprocess with its own PYTHONPATH
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from tinygrad import getenv

from uops import CASES
from benchmark import PATCH_FXNS
from perf import run, python, from_key, benchmarks, save_keys, path_from_key, variant_env

import pickle
from collections import defaultdict

# benchmark functions (and the metrics they report) that only run with RT=1
RUNTIME = {"run_debug_2_runtime", "estimated_ops", "estimated_mem"}

def average_rounds_interleaved(fxn, name, pool, rounds, cfg=None):
  all_res = [[] for _ in range(len(pool))]
  for j,_ in enumerate(range(rounds+1)):
    for i,((tree,r),k) in enumerate(pool.items()):
      pickled_call = pickle.dumps((fxn, (k,), {}))
      # delete pre-existing file if it exists, so a new run that fails never reads a stale file
      # in practice, however, the run function should raise if the process fails
      if os.path.exists(path:=path_from_key(k, benchmarks)): os.remove(path)
      run(cmd, env=variant_env({**os.environ, **cfg}, tree, None if r == "None" else r), data=pickled_call)
      # one warm-up round per variant
      if j > 0: all_res[i].append(from_key(k, benchmarks))
  # each benchmark function reports {metric: value}. times are in ns, ops and bytes are raw counts: all are printed scaled by 1e-6
  for metric in all_res[0][0]:
    for i,(tree,r) in enumerate(pool):
      avg = sum(float(res[metric]) for res in all_res[i])/rounds
      print(f"{metric} {name} in {avg*1e-6} with {tree}"+("" if r == "None" else f":{r}"))

if __name__ == "__main__":
  cfg = {"CACHELEVEL":"0", "SCACHE":"0"}
  cmd = [python, "new_proc.py"]
  variants = (("tinygrad-base", None), ("tinygrad-fork", None))

  assert (case:=os.getenv("CASE")) is not None
  rounds = getenv("ROUNDS", 1)
  rt = getenv("RT", 0)

  # save uops

  keys = []
  for u in CASES[case]:
    for tree,r in variants:
      try:
        run(cmd, env=variant_env({**os.environ, **cfg}, tree, r), data=pickle.dumps((u, (), {})))
        with open(save_keys) as f: keys += f.read().split("\n")
      finally:
        if os.path.exists(save_keys): os.remove(save_keys)

  # benchmark uops

  pools = defaultdict(dict)

  for key in keys:
    ks = key.split(":")
    name, patch, tree, r = ks[0], ks[1], ks[-2], ks[-1]
    if (fxn:=PATCH_FXNS.get(patch)) is None: raise RuntimeError(f"unknown patch {patch}")
    if rt ^ (fxn.__name__ in RUNTIME): continue
    assert (tree,r) not in (d:=pools[(name,patch)])
    d[(tree,r)] = key

  for (name,patch),pool in pools.items():
    patch_fxn = PATCH_FXNS[patch]
    average_rounds_interleaved(patch_fxn, name, pool, rounds, cfg=cfg)

