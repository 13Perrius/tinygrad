import os
import sys
import operator

from nd import ND
from run_benchmark import RUNTIME

BASE = os.getenv("BASE", "tinygrad-base")

def vs_base(x, fxn):
  base = x.kmap(lambda k, v: v if k[0] == BASE else 0).reduce((0,), operator.add).expand(x.axis(0), 0)
  return x.elem(base, fxn)

def mean(x, axis): return x.reduce((axis,), operator.add).map(lambda v: v/len(x.axis(axis)))

def print_cases(wins_table, title):
  # we want to left-align labels and right-align numbers
  fmt = lambda label, cases: f"\t\t{label:<10}{len(cases):>10}  {', '.join(cases)}"
  # wins_table: r, case
  rs = wins_table.axis(0)
  print(title, end="\n\n")
  for r in rs:
    if r == BASE: continue
    wins, losses = [], []
    for c in wins_table[r]:
      (wins if wins_table[r][c] else losses).append(c)
    print(f"\t{r} vs {BASE}")
    print(fmt("wins", wins))
    print(fmt("losses", losses), end="\n\n")

if __name__ == "__main__":
  assert len(sys.argv) == 2
  data = {}
  with open(sys.argv[1]) as f:
    for line in f:
      # parse "<patch> <case> in <time> with <variant>"
      patch, case, _, time, _, r = line.split()
      data[(case, patch, r.removeprefix("R="))] = float(time)

  # r, case, patch
  table = ND(data, 3).permute((2, 0, 1))

  # r, case
  total = table.kmap(lambda k, v: 0 if k[2] in RUNTIME else v).reduce((2,), operator.add)

  # 0 => loss, 1 => win
  total_wins = vs_base(total, operator.le)
  all_pass_wins = vs_base(table, operator.le)

  patches = table.axis(2)
  if (ct_patches:=tuple(p for p in patches if p not in RUNTIME)):
    total_title = "total compile time (" + " + ".join(ct_patches) + ")"
    print_cases(total_wins, total_title)

  for patch in patches:
    pass_wins = all_pass_wins.permute((2,0,1))[patch]
    print_cases(pass_wins, patch)

  # r, case, patch
  patch_pct = vs_base(table, lambda v, b: (v-b)/b)

  # r, patch
  patch_means = mean(patch_pct, 1)

  if ct_patches:
    # r, case
    total_pct = vs_base(total, lambda v, b: (v-b)/b)

    # r
    total_means = mean(total_pct, 1)

  print(f"mean percent difference relative to {BASE}", end="\n\n")
  for r in patch_means:
    if r == BASE: continue
    print(f"{r:<15}", end="")
    if ct_patches: print(f"{'total compile':<15} {total_means[r]:^20.3%}", end="")
    for patch in patch_means[r]:
      print(f"{patch:<15} {patch_means[(r,patch)]:^20.3%}", end="")
    print(end="\n\n")
