from tinygrad import Tensor, nn, getenv, TinyJit, Variable
from tinygrad.helpers import Context
from tinygrad.llm.model import TransformerConfig, SSMConfig, GatedDeltaNetBlock

from perf import patch_scheduler, restore_scheduler, patch_codegen, restore_codegen, save_uop
from debug_2 import patch_debug_2_runtime, restore_debug_2_runtime

patches = ((patch_scheduler, restore_scheduler), (patch_codegen, restore_codegen), (patch_debug_2_runtime, restore_debug_2_runtime))

@save_uop(patches=patches)
def mnist():
  layers = [
    nn.Conv2d(1, 32, 5), Tensor.relu,
    nn.Conv2d(32, 32, 5), Tensor.relu,
    nn.BatchNorm(32), Tensor.max_pool2d,
    nn.Conv2d(32, 64, 3), Tensor.relu,
    nn.Conv2d(64, 64, 3), Tensor.relu,
    nn.BatchNorm(64), Tensor.max_pool2d,
    lambda x: x.flatten(1), nn.Linear(576, 10)]

  probs = Tensor.rand(1, 1, 28, 28).sequential(layers)
  return (probs,)

@save_uop(patches=patches)
def conv_pool_bwd():
  x = Tensor.rand(128, 1, 28, 28).realize()
  w = Tensor.rand(8, 1, 3, 3).realize()
  return (x.conv2d(w).max_pool2d().sum().gradient(w)[0],)

@save_uop(patches=patches)
def tinyconv_bwd():
  x, y = Tensor.rand(128, 1, 28, 28).realize(), Tensor.randint(128, high=10).realize()
  c1, c2, l1 = Tensor.rand(8, 1, 3, 3).realize(), Tensor.rand(16, 8, 3, 3).realize(), Tensor.rand(400, 10).realize()
  out = x.conv2d(c1).relu().max_pool2d().conv2d(c2).relu().max_pool2d().flatten(1).dot(l1)
  return (out.sparse_categorical_crossentropy(y).gradient(c1)[0],)

@save_uop(patches=patches)
def gdb():
  cfg_dict = {
    "num_blocks":1,
    "dim":8,
    "hidden_dim":16,
    "n_heads":1,
    "n_kv_heads":1,
    "norm_eps":1e-5,
    "vocab_size":32,
    "head_dim":8,
    "rope_theta":10000.0,
    "rope_dim":8,
    "v_head_dim":8,
    "max_context":3,
    "ssm_layers":(True,),
    "ssm":SSMConfig(conv_kernel=2, state_size=4, group_count=1, time_step_rank=1, inner_size=4)
  }

  cfg = TransformerConfig(**cfg_dict)
  block = GatedDeltaNetBlock(cfg, cfg.ssm)

  x = Tensor.ones((3*cfg.dim,)).reshape((1, 3, cfg.dim))
  x_norm = block.attn_norm(x)
  block._init_state(x_norm)

  start_pos = 0
  return (block._attention(x_norm, (start_pos,)),)

@save_uop(patches=patches)
def newton_schulz():
  b = Tensor.randn(2, 2, 2).newton_schulz(steps=20, params=(2.0, -1.4, 0.2, 0.2), eps=0.0)
  return (b @ b.transpose(-2, -1),)

@save_uop(patches=patches)
def last_row_softmax():
  logits = Tensor.empty(4096, 1024) @ Tensor.empty(1024, 1024)
  return (logits[-1].softmax(),)


# kernel count exceptions
# each builder transcribes the named test up to its check_schedule/assert_jit_cache_len call, and returns what that call checks.
# the builder's defaults are the test's sizes (KCE). KCE_LARGE calls the same builders with sizes that drown out per-kernel overhead
# in RT=1 runtime numbers, and otherwise leaves the test as is.
# NOTE: TestSchedule.setUp sets SPLIT_REDUCEOP=0, which is read at schedule time (codegen/prepare.py split_reduceop), so a case can't
#       set it. none of the KCE cases below have a reduce big enough to split (the linears are identical with SPLIT_REDUCEOP=0/1),
#       except preserve_multistage_reduce, which sets SPLIT_REDUCEOP=1 itself (the default). KCE_LARGE cases do split their reduces.

def _realize_weights(m):
  for p in nn.state.get_parameters(m): p.realize()

# test/null/test_schedule.py::TestSchedule::test_preserve_multistage_reduce
def preserve_multistage_reduce_step(big_enough:int):
  x = Tensor.empty(big_enough).realize()
  with Context(SPLIT_REDUCEOP=1):
    out = (x - x.max(keepdim=True)).max()
    return (out,)

# test/null/test_schedule.py::TestSchedule::test_image_conv_fusion_minimal
def image_conv_fusion_minimal_step(c=16, px=32):
  b1 = Tensor.empty(c)
  b2 = Tensor.empty(c)
  def p(x): return x.permute(1,0).contiguous().reshape(px,c,1).expand(px,c,16).sum(axis=2).permute(1,0)

  x = Tensor.empty(c, px)
  x = base = p(x) + b1.reshape(c,1)
  x = p(x)
  x = x + b2.reshape(c,1)
  x = x + base
  del base
  x = p(x)
  return (x,)

# test/null/test_schedule.py::TestSchedule::test_fold_conv_batchnorm_optim(_adam)
def fold_conv_batchnorm_optim_step(adam=False, bs=1, hw=4, ch1=32):
  optim = nn.optim.Adam if adam else nn.optim.SGD
  with Context(TRAINING=1):
    img = Tensor.ones(bs,3,hw,hw)
    c1 = nn.Conv2d(3,ch1,3)
    bn = nn.BatchNorm2d(ch1, track_running_stats=False)
    _realize_weights([c1, bn])
    opt = optim(nn.state.get_parameters([c1, bn]))
    Tensor.realize(img, *nn.state.get_parameters(opt))
    img_bn = bn(c1(img)).elu().sum()
    opt.zero_grad()
    img_bn.backward()
    return tuple(opt.schedule_step())

# test/null/test_schedule.py::TestSchedule::test_adam_step_fusion
def adam_step_fusion_step(bs=4, seq=64, dim=32):
  with Context(TRAINING=1):
    x = Tensor.empty(bs, seq, dim)
    layer = nn.Linear(dim, dim*4)
    _realize_weights(layer)
    opt = nn.optim.Adam(nn.state.get_parameters(layer), lr=1e-4)
    Tensor.realize(*nn.state.get_parameters(opt))
    layer(x).relu().sum().backward()
    return tuple(opt.schedule_step())

# test/null/test_schedule.py::TestSchedule::test_adam_conv_fuse
def adam_conv_fuse_step(bs=2, hw=4, ch1=32):
  with Context(TRAINING=1):
    img = Tensor.empty(bs,3,hw,hw)
    c1 = nn.Conv2d(3,ch1,3)
    _realize_weights(c1)
    opt = nn.optim.Adam(nn.state.get_parameters(c1), lr=1e-4)
    Tensor.realize(*nn.state.get_parameters(opt))
    opt.zero_grad()
    c1(img).relu().sum().backward()
    return tuple(opt.schedule_step())

# test/null/test_schedule.py::TestSchedule::test_adam_2convs_fuse
def adam_2convs_fuse_step(bs=2, hw=4, ch1=16, ch2=32):
  with Context(TRAINING=1):
    img = Tensor.empty(bs,3,hw,hw)
    c1 = nn.Conv2d(3,ch1,3,bias=False)
    c2 = nn.Conv2d(ch1,ch2,2,bias=False)
    _realize_weights([c1, c2])
    opt = nn.optim.Adam(nn.state.get_parameters([c1, c2]), lr=1e-4)
    Tensor.realize(*nn.state.get_parameters(opt))
    opt.zero_grad()
    c2(c1(img).relu()).relu().sum().backward()
    return tuple(opt.schedule_step())

# test/null/test_schedule.py::TestSchedule::test_sgd_conv_fuse
def sgd_conv_fuse_step(bs=2, hw=4, ch1=32):
  with Context(TRAINING=1):
    img = Tensor.empty(bs,3,hw,hw)
    c1 = nn.Conv2d(3,ch1,3)
    _realize_weights(c1)
    opt = nn.optim.SGD(nn.state.get_parameters(c1))
    opt.zero_grad()
    c1(img).relu().sum().backward()
    return tuple(opt.schedule_step())

# test/null/test_schedule.py::TestSchedule::test_sgd_2convs_fuse
def sgd_2convs_fuse_step(bs=2, hw=4, ch1=16, ch2=32):
  with Context(TRAINING=1):
    img = Tensor.empty(bs,3,hw,hw)
    c1 = nn.Conv2d(3,ch1,3,bias=False)
    c2 = nn.Conv2d(ch1,ch2,2,bias=False)
    _realize_weights([c1, c2])
    opt = nn.optim.SGD(nn.state.get_parameters([c1, c2]))
    opt.zero_grad()
    c2(c1(img).relu()).relu().sum().backward()
    return tuple(opt.schedule_step())

# test/null/test_schedule.py::TestSchedule::test_fold_2convs_sgd_nesterov_momentum_wd
def fold_2convs_sgd_nesterov_momentum_wd_step(bs=2, hw=4, ch1=16, ch2=32):
  with Context(TRAINING=1):
    img = Tensor.empty(bs,3,hw,hw)
    c1 = nn.Conv2d(3,ch1,3,bias=False)
    c2 = nn.Conv2d(ch1,ch2,2,bias=False)
    _realize_weights([c1, c2])
    opt = nn.optim.SGD(nn.state.get_parameters([c1, c2]), nesterov=True, momentum=0.9, weight_decay=0.1)
    Tensor.realize(*nn.state.get_parameters(opt))
    opt.zero_grad()
    c2(c1(img).relu()).relu().sum().backward()
    return tuple(opt.schedule_step())

# test/null/test_schedule.py::TestSchedule::test_sgd_4convs_fuse
# NOTE: test_sgd_4convs_fuse_conv_bw has an identical body, CASES maps it here
def sgd_4convs_fuse_step(bs=2, hw=16, ch=(4, 8, 16, 32)):
  with Context(TRAINING=1):
    img = Tensor.empty(bs,3,hw,hw)
    c1 = nn.Conv2d(3,ch[0],3,bias=False)
    c2 = nn.Conv2d(ch[0],ch[1],3,bias=False)
    c3 = nn.Conv2d(ch[1],ch[2],3,bias=False)
    c4 = nn.Conv2d(ch[2],ch[3],3,bias=False)
    _realize_weights([c1, c2, c3, c4])
    opt = nn.optim.SGD(nn.state.get_parameters([c1, c2, c3, c4]))
    opt.zero_grad()
    c4(c3(c2(c1(img).relu()).relu()).relu()).relu().sum().backward()
    return tuple(opt.schedule_step())

# test/runtime/test_symbolic_jit.py::TestSymbolicJit::test_attention
# the jit captures on its second call (i=2), assert_jit_cache_len counts the kernels realized in f during that call.
# NOTE: realize happens outside of a capturing jit, so create_linear runs with plan=True. memory planning is after pm_schedule and
#       doesn't change the kernel ASTs, so the scheduler and codegen benchmarks see the same work.
def symbolic_jit_attention_step(bs=2, heads=4, dim=8, maxlen=10, capture_len=2):
  from tinygrad.engine.jit import _prepare_jit_inputs
  def sdpa(q, k, v): return Tensor.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
  def f(q, k, v): return sdpa(q, k, v).realize()
  jf = TinyJit(f)
  q = Tensor.rand(bs, 1, heads, dim)
  k = Tensor.rand(bs, maxlen, heads, dim)
  v = Tensor.rand(bs, maxlen, heads, dim)
  # i=1: jit ignore
  vi = Variable("i", 1, maxlen).bind(1)
  jf(q, k[:, :vi], v[:, :vi])[:bs, :heads, :1, :dim].numpy()
  f(q, k[:, :1], v[:, :1]).numpy()
  # jit capture, the jit realizes its inputs before calling f
  vi = Variable("i", 1, maxlen).bind(capture_len)
  args = (q, k[:, :vi], v[:, :vi])
  _prepare_jit_inputs(args, {})
  return (sdpa(*args),)

# KCE: test sizes

@save_uop(patches=patches)
def preserve_multistage_reduce(): return preserve_multistage_reduce_step(getenv("REDUCEOP_SPLIT_THRESHOLD", 32768))
@save_uop(patches=patches)
def image_conv_fusion_minimal(): return image_conv_fusion_minimal_step()
@save_uop(patches=patches)
def fold_conv_batchnorm_optim(): return fold_conv_batchnorm_optim_step(adam=False)
@save_uop(patches=patches)
def fold_conv_batchnorm_optim_adam(): return fold_conv_batchnorm_optim_step(adam=True)
@save_uop(patches=patches)
def adam_step_fusion(): return adam_step_fusion_step()
@save_uop(patches=patches)
def adam_conv_fuse(): return adam_conv_fuse_step()
@save_uop(patches=patches)
def adam_2convs_fuse(): return adam_2convs_fuse_step()
@save_uop(patches=patches)
def sgd_conv_fuse(): return sgd_conv_fuse_step()
@save_uop(patches=patches)
def sgd_2convs_fuse(): return sgd_2convs_fuse_step()
@save_uop(patches=patches)
def fold_2convs_sgd_nesterov_momentum_wd(): return fold_2convs_sgd_nesterov_momentum_wd_step()
@save_uop(patches=patches)
def sgd_4convs_fuse(): return sgd_4convs_fuse_step()
@save_uop(patches=patches)
def symbolic_jit_attention(): return symbolic_jit_attention_step()

# KCE_LARGE: same cases, scaled up

CONV = {"bs":64, "hw":128}

@save_uop(patches=patches)
def preserve_multistage_reduce_large(): return preserve_multistage_reduce_step(2**28)
@save_uop(patches=patches)
def image_conv_fusion_minimal_large(): return image_conv_fusion_minimal_step(c=4096, px=16384)
@save_uop(patches=patches)
def fold_conv_batchnorm_optim_large(): return fold_conv_batchnorm_optim_step(adam=False, **CONV, ch1=128)
@save_uop(patches=patches)
def fold_conv_batchnorm_optim_adam_large(): return fold_conv_batchnorm_optim_step(adam=True, **CONV, ch1=128)
@save_uop(patches=patches)
def adam_step_fusion_large(): return adam_step_fusion_step(bs=4, seq=512, dim=2048)
@save_uop(patches=patches)
def adam_conv_fuse_large(): return adam_conv_fuse_step(**CONV, ch1=128)
@save_uop(patches=patches)
def adam_2convs_fuse_large(): return adam_2convs_fuse_step(**CONV, ch1=64, ch2=128)
@save_uop(patches=patches)
def sgd_conv_fuse_large(): return sgd_conv_fuse_step(**CONV, ch1=128)
@save_uop(patches=patches)
def sgd_2convs_fuse_large(): return sgd_2convs_fuse_step(**CONV, ch1=64, ch2=128)
@save_uop(patches=patches)
def fold_2convs_sgd_nesterov_momentum_wd_large(): return fold_2convs_sgd_nesterov_momentum_wd_step(**CONV, ch1=64, ch2=128)
@save_uop(patches=patches)
def sgd_4convs_fuse_large(): return sgd_4convs_fuse_step(**CONV, ch=(16, 32, 64, 128))
@save_uop(patches=patches)
def symbolic_jit_attention_large(): return symbolic_jit_attention_step(bs=8, heads=32, dim=128, maxlen=8192, capture_len=4096)

# should be tuples

CORE = [mnist, conv_pool_bwd, tinyconv_bwd, gdb, newton_schulz, last_row_softmax]

KCE = [adam_2convs_fuse, adam_conv_fuse, adam_step_fusion, fold_2convs_sgd_nesterov_momentum_wd, fold_conv_batchnorm_optim,
       fold_conv_batchnorm_optim_adam, preserve_multistage_reduce, image_conv_fusion_minimal, sgd_2convs_fuse, sgd_4convs_fuse,
       sgd_conv_fuse, symbolic_jit_attention]

KCE_LARGE = [adam_2convs_fuse_large, adam_conv_fuse_large, adam_step_fusion_large, fold_2convs_sgd_nesterov_momentum_wd_large,
             fold_conv_batchnorm_optim_large, fold_conv_batchnorm_optim_adam_large, preserve_multistage_reduce_large,
             image_conv_fusion_minimal_large, sgd_2convs_fuse_large, sgd_4convs_fuse_large, sgd_conv_fuse_large,
             symbolic_jit_attention_large]

ALL = list(dict.fromkeys(CORE + KCE + KCE_LARGE))

CASES = {fxn.__name__:(fxn,) for fxn in ALL}
CASES.update(sgd_4convs_fuse_conv_bw=(sgd_4convs_fuse,), sgd_4convs_fuse_conv_bw_large=(sgd_4convs_fuse_large,))
CASES.update(CORE=CORE, KCE=KCE, KCE_LARGE=KCE_LARGE, ALL=ALL)
