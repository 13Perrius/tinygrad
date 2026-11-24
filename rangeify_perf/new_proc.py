import os
import sys
import pickle

if __name__ == "__main__":
  if (pp:=os.environ.get("PYTHONPATH")) is not None:
    import tinygrad
    assert os.path.commonpath([tinygrad.__file__, pp]) == os.path.normpath(pp), f"tinygrad from {tinygrad.__file__}, expected {pp}"
  fxn, args, kwargs = pickle.loads(sys.stdin.buffer.read())
  fxn(*args, **kwargs)
