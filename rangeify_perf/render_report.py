import sys
from collections import defaultdict

if __name__ == "__main__":
  assert len(sys.argv) == 2
  target = sys.argv[1]

  # r, time
  fstr = "{:>20} {:>10.2f}"

  data = defaultdict(lambda: defaultdict(list))

  with open(target) as f:
    for line in f:
      patch, case, _, time, _, r = line.split()
      data[case][patch].append((r, float(time)))

    for case in data:
      _case = "  "+case+"  "
      print(f"{_case:-^60}", end="\n\n")
      for patch in data[case]:
        print(f"{patch}", end="\n\n")
        for r,time in data[case][patch]:
          print(fstr.format(r, time))
        print()
