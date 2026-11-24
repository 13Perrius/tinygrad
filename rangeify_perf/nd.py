import functools
import operator
from collections import defaultdict
from copy import deepcopy

def build_nd(data, n):
  if n == 1: return data
  collect = {}
  for key in data:
    head, *tail = key
    if head not in collect: collect[head] = {}
    tail = tail[0] if len(tail) == 1 else tuple(tail)
    collect[head][tail] = data[key]
  return {head:build_nd(collect[head], n-1) for head in collect}

def coords_nd(nd, n, prefix=()):
  if n == 1:
    for k in nd: yield prefix+(k,)
  else:
    for k in nd: yield from coords_nd(nd[k], n-1, prefix=prefix+(k,))

def index_nd(nd, n, key): 
  return functools.reduce(lambda acc, k: acc[k], key, nd)

def set_nd(nd, n, key, value):
  d = functools.reduce(lambda acc, k: acc[k], key[:-1], nd)
  d[key[-1]] = value

class ND:
  def __init__(self, data, n):
    self.nd = build_nd(data, n)
    self.n = n
  @classmethod
  def from_nd(self, nd, n): 
    obj = ND.__new__(ND)
    obj.nd, obj.n = nd, n
    return obj
  def coords(self): return coords_nd(self.nd, self.n)
  def kmap(self, fxn):
    data = {}
    for x in self.coords():
      data[x if len(x) > 1 else x[0]] = fxn(x, self[x])
    return ND(data, self.n)
  def map(self, fxn): return self.kmap(fxn= lambda k, v: fxn(v))
  def elem(self, other, fxn):
    data = {}
    for x,y in zip(self.coords(), other.coords()):
      if x != y: raise RuntimeError(f"key mismatch on elementwise op: {x} != {y}")
      data[x if len(x) > 1 else x[0]] = fxn(self[x], other[x]) 
    return ND(data, self.n)
  def reduce_left(self, r, fxn):
    nd, n = self.nd, self.n
    nds = (ND.from_nd(d, n-1) for d in nd.values())
    res = functools.reduce(lambda acc, nd: acc.elem(nd, fxn), nds)
    return res if r == 1 else res.reduce_left(r-1, fxn)
  def expand_left(self, ks): return ND.from_nd({k:deepcopy(self.nd) for k in ks}, self.n+1)
  def expand(self, ks, i): 
    phi = tuple(j+1 if j < i else (0 if j == i else j) for j in range(self.n+1))
    return self.expand_left(ks).permute(phi)
  def permute(self, phi):
    data = {}
    for x in self.coords():
      _,p = zip(*sorted(enumerate(x), key=lambda k: phi.index(k[0])))
      data[p] = self[x]
    return ND(data, self.n)
  def reduce(self, axes, fxn):
    rem = tuple(i for i in range(self.n) if i not in axes)
    phi = axes+rem
    return self.permute(phi).reduce_left(len(axes), fxn)
  def axis(self, i): 
    k, d, count = next(iter(self)), self.nd, i
    while count > 0: 
      d = d[k]
      k = next(iter(d))
      count -= 1
    return tuple(d.keys())
  def __add__(self, other): return self.elem(other, operator.add)
  def __sub__(self, other): return self.elem(other, operator.sub)
  def __mul__(self, other): return self.elem(other, operator.mul)
  def __iter__(self): return iter(self.nd)
  def __getitem__(self, key): 
    if isinstance(key, tuple): return index_nd(self.nd, self.n, key)
    return ND.from_nd(deepcopy(self.nd[key]), self.n-1) if self.n > 1 else self.nd[key]
  def __setitem__(self, key, value): set_nd(self.nd, self.n, key, value)
  def __repr__(self): return repr(self.nd)

def compare_leading(x):
  mins = x.reduce((0,), min)
  cmp = x.elem(mins.expand(x.axis(0), 0), lambda x, y: x <= y)
  return cmp
