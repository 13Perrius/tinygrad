import os
import sys

from tinygrad import Tensor, nn, dtypes, Device, UOp, Context
from tinygrad.helpers import prod
from tinygrad.uop.ops import KernelInfo
from tinygrad.llm.model import TransformerConfig, SSMConfig, GatedDeltaNetBlock, Transformer, precompute_freqs_cis, apply_rope
from tinygrad.llm.kernels.amd import Linear
from tinygrad.llm.kernels.amd_gguf import QUANT_SIZES
from tinygrad.llm.gguf import ggml_data_to_tensor

from perf import patch_scheduler, restore_scheduler, patch_codegen, restore_codegen, save_uop
from debug_2 import patch_debug_2_runtime, restore_debug_2_runtime

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from extra.llama_kernels import local_abs_max

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

def custom_add_one_kernel(B:UOp, A:UOp) -> UOp:
  A, B = A.flatten(), B.flatten()
  i = UOp.range(A.numel(), 0)
  return B[i].store(A[i] + 1).end(i).sink(arg=KernelInfo(name=f"add_one_{A.numel()}"))

def custom_add_one(x:Tensor) -> Tensor: return Tensor.custom_kernel(Tensor.empty_like(x), x, fxn=custom_add_one_kernel)[0]

@save_uop(patches=patches)
def apply_rope_jit_prune():
  x, start_pos = Tensor.randn(1, 2, 4, 8, dtype=dtypes.float32), UOp.variable("start_pos", 0, 100).bind(1)
  precompute_freqs_cis.cache_clear()
  return (apply_rope(x, precompute_freqs_cis(x.shape[-1], start_pos+x.shape[2])[start_pos:start_pos+x.shape[2]]),)

@save_uop(patches=patches)
def image_conv_fusion_minimal():
  b1, b2 = Tensor.empty(16), Tensor.empty(16)
  def p(x): return x.permute(1,0).contiguous().reshape(32,16,1).expand(32,16,16).sum(axis=2).permute(1,0)
  x = base = p(Tensor.empty(16, 32)) + b1.reshape(16,1)
  x = p(x) + b2.reshape(16,1) + base
  return (p(x),)

@save_uop(patches=patches)
def preserve_multistage_reduce():
  x = Tensor.empty(32768)
  return ((x - x.max(keepdim=True)).max(),)

@save_uop(patches=patches)
def multireduce_pad_reduce_unsafe():
  a, b = Tensor.empty(3, 4, 5), Tensor.empty(3, 4, 5)
  return ((a.log2().pad(((0, 1), (0, 1), (0, 1)), value=1.0).sum()+b).abs().log2().pad(((0, 1), (0, 1), (0, 1)), value=1.0).sum().contiguous(),)

@save_uop(patches=patches)
def custom_kernel_2d_shrink(): return (custom_add_one(Tensor.empty(32, dtype=dtypes.int32).reshape(4, 8)[:, 2:6]),)

@save_uop(patches=patches)
def custom_kernel_double_permute(): return (custom_add_one(Tensor.empty(32, dtype=dtypes.int32).reshape(4, 8).T.T),)

@save_uop(patches=patches)
def custom_kernel_expand(): return (custom_add_one(Tensor.empty(32, dtype=dtypes.int32).reshape(16, 2)[:, :1].expand(16, 2)),)

@save_uop(patches=patches)
def custom_kernel_flip(): return (custom_add_one(Tensor.empty(32, dtype=dtypes.int32).flip(0)),)

@save_uop(patches=patches)
def custom_kernel_offset_shrink(): return (custom_add_one(Tensor.empty(32, dtype=dtypes.int32)[4:8]),)

@save_uop(patches=patches)
def custom_kernel_pad(): return (custom_add_one(Tensor.empty(32, dtype=dtypes.int32)[:4].pad(((0, 4),))),)

@save_uop(patches=patches)
def custom_kernel_permute(): return (custom_add_one(Tensor.empty(32, dtype=dtypes.int32).reshape(4, 8).T),)

@save_uop(patches=patches)
def custom_kernel_reshape(): return (custom_add_one(Tensor.empty(32, dtype=dtypes.int32).reshape(16, 2)),)

@save_uop(patches=patches)
def custom_kernel_shrink(): return (custom_add_one(Tensor.empty(32, dtype=dtypes.int32)[:4]),)

@save_uop(patches=patches)
def multi_tensor_local_shard_amax():
  devs = ("CPU:0", "CPU:1")
  x = Tensor(Tensor.empty(2, 4, device=devs).uop.unshard(0), device=devs)
  return ((x * local_abs_max(x)).clone(),)

@save_uop(patches=patches)
def cache_reduce_multiple_children():
  x, y = Tensor.empty(32), Tensor.empty(4, 4)
  r0 = x.mean(axis=0, keepdim=True)
  r1 = (x - r0).sum(axis=0).div(2)
  return (r0 + y, r1 + y)

@save_uop(patches=patches)
def arange_fuse_grouped_children():
  r = (Tensor.empty(4, 4)+Tensor.arange(16).reshape(4, 4)).sum()
  return (r+2, r+3)

@save_uop(patches=patches)
def partial_fuse1():
  a, b = Tensor.empty(16, 16), Tensor.empty(16, 16)
  return (a.sum() + 2, (a.sum() - b.sum()) * 4)

@save_uop(patches=patches)
def partial_fuse3():
  a, b = Tensor.empty(16, 16), Tensor.empty(16, 16)
  c, d = a.sum() + 2, a.sum() * 2
  e = c * d
  return (c, d, e, b.sum() - e)

@save_uop(patches=patches)
def partial_fuse4():
  a, b = Tensor.empty(16, 16), Tensor.empty(16, 16)
  c, d = a.sum() + 2, a.sum() * 2
  e = c * d
  return (c, d, e, (b - d).sum() - e)

@save_uop(patches=patches)
def reduce_expand_child():
  a, b = Tensor.empty(32, 32, 32), Tensor.empty(1, 16)
  return (a.sum() + 2, a.sum() + b)

@save_uop(patches=patches)
def reduce_ext_reduce_child():
  a, b = Tensor.empty(4, 4), Tensor.empty(4, 4)
  return (a.sum() + b.sum() + 2, a.sum() + b.sum() + 4)

@save_uop(patches=patches)
def reduce_multiple_paths():
  a = Tensor.empty(4, 4)
  out0 = a.sum().exp2()
  return (out0, a.sum() + out0)

@save_uop(patches=patches)
def multireduce_reduce_multiple_paths():
  a = Tensor.empty(4, 4)
  out0 = a.sum().exp2()
  out1 = a.sum() + out0
  b = a + out0 + out1
  out2 = b.sum().exp2()
  return (out0, out1, out2, b.sum() + out2)

@save_uop(patches=patches)
def reduce_multiple_paths_midreduce_fused():
  a, b = Tensor.empty(4, 4), Tensor.empty(4, 4)
  out0 = a.sum() + 4
  out1 = b.max() + out0*2
  return (out0, out1, a.sum() + out1)

@save_uop(patches=patches)
def reduce_same_size():
  a = Tensor.empty(4, 4)
  out0, out1 = a.sum() + 2, a.sum() + 4
  return (out0, out1, out0 * out1)

@save_uop(patches=patches)
def reduce_shrink_child():
  a, b = Tensor.empty(100, 100), Tensor.empty(10,)
  return (a.sum() + b[0], a.sum() + 2)

def realized(*ts:Tensor):
  for t in ts: t.replace(Tensor.empty(*t.shape, dtype=t.dtype, device=t.device))

def adam_step(params:list[Tensor], loss_fxn) -> tuple[Tensor, ...]:
  realized(*params)
  opt = nn.optim.Adam(params, lr=1e-4)
  realized(*nn.state.get_parameters(opt))
  opt.zero_grad()
  loss_fxn().backward()
  return tuple(opt.schedule_step())

@save_uop(patches=patches)
def adam_2convs_fuse():
  with Context(TRAINING=1):
    img, c1, c2 = Tensor.empty(2, 3, 4, 4), nn.Conv2d(3, 16, 3, bias=False), nn.Conv2d(16, 32, 2, bias=False)
    return adam_step(nn.state.get_parameters([c1, c2]), lambda: c2(c1(img).relu()).relu().sum())

@save_uop(patches=patches)
def adam_conv_fuse():
  with Context(TRAINING=1):
    img, c1 = Tensor.empty(2, 3, 4, 4), nn.Conv2d(3, 32, 3)
    return adam_step(nn.state.get_parameters(c1), lambda: c1(img).relu().sum())

@save_uop(patches=patches)
def adam_step_fusion():
  with Context(TRAINING=1):
    x, layer = Tensor.empty(4, 64, 32), nn.Linear(32, 32*4)
    return adam_step(nn.state.get_parameters(layer), lambda: layer(x).relu().sum())

@save_uop(patches=patches)
def fold_conv_batchnorm_optim_adam():
  with Context(TRAINING=1):
    img, c1, bn = Tensor.empty(1, 3, 4, 4), nn.Conv2d(3, 32, 3), nn.BatchNorm2d(32, track_running_stats=False)
    return adam_step(nn.state.get_parameters([c1, bn]), lambda: bn(c1(img)).elu().sum())

@save_uop(patches=patches)
def symbolic_attention():
  q, k, v = Tensor.rand(2, 1, 4, 8), Tensor.rand(2, 10, 4, 8), Tensor.rand(2, 10, 4, 8)
  i = UOp.variable("i", 1, 10).bind(4)
  return (Tensor.scaled_dot_product_attention(q.transpose(1, 2), k[:, :i].transpose(1, 2), v[:, :i].transpose(1, 2)),)

# slowest tests

def gdn_config(**kwargs):
  return TransformerConfig(**({"num_blocks":1, "dim":8, "hidden_dim":16, "n_heads":1, "n_kv_heads":1, "norm_eps":1e-5, "vocab_size":32,
                               "head_dim":8, "rope_theta":10000.0, "rope_dim":8, "v_head_dim":8, "max_context":4, "ssm_layers":(True,),
                               "ssm":SSMConfig(conv_kernel=2, state_size=4, group_count=1, time_step_rank=1, inner_size=4)} | kwargs))

def linspace(start:float, stop:float, shape:tuple[int, ...]) -> Tensor:
  return Tensor.linspace(start, stop, prod(shape), dtype=dtypes.float32).reshape(*shape)

def gdn_prefill(block:GatedDeltaNetBlock, x:Tensor) -> Tensor:
  x_norm = block.attn_norm(x)
  block._init_state(x_norm)
  return block._attention(x_norm, (0,))

@save_uop(patches=patches)
def gatedeltanet_reference_prefill():
  config = gdn_config(max_context=3)
  block = GatedDeltaNetBlock(config, config.ssm)
  block.attn_norm.weight = linspace(0.8, 1.2, (config.dim,))
  block.attn_qkv.weight = linspace(-0.15, 0.2, (block.conv_channels, config.dim))
  block.attn_gate.weight = linspace(-0.1, 0.15, (config.ssm.inner_size, config.dim))
  block.ssm_alpha.weight = linspace(-0.08, 0.12, (block.num_v_heads, config.dim))
  block.ssm_beta.weight = linspace(-0.12, 0.07, (block.num_v_heads, config.dim))
  block.ssm_conv1d["weight"] = linspace(-0.05, 0.05, (block.conv_channels, block.ssm_conv_kernel))
  block.ssm_dt["bias"] = linspace(-0.1, 0.1, (block.num_v_heads,))
  block.ssm_a = linspace(-0.1, -0.05, (block.num_v_heads,))
  block.ssm_norm.weight = linspace(0.9, 1.1, (block.head_v_dim,))
  block.ssm_out.weight = linspace(-0.2, 0.18, (config.dim, config.ssm.inner_size))
  return (gdn_prefill(block, Tensor.linspace(-1.0, 1.0, 3 * config.dim, dtype=dtypes.float32).reshape(1, 3, config.dim)),)

@save_uop(patches=patches)
def kda_prefill():
  config = gdn_config(ssm=SSMConfig(conv_kernel=2, state_size=4, group_count=1, time_step_rank=1, inner_size=4, kda=True))
  block = GatedDeltaNetBlock(config, config.ssm)
  for p in nn.state.get_parameters(block): p.replace(linspace(-0.05, 0.05, p.shape) if len(p.shape) > 1 else linspace(0.05, 0.1, p.shape))
  return (gdn_prefill(block, linspace(-0.5, 0.5, (1, 3, config.dim))),)

@save_uop(patches=patches)
def keccak_variable_bs():
  data = Tensor([b"abc", b"abc", b"def"], dtype=dtypes.uint8).repeat(2048, 1)
  return (data.shrink_to(UOp.variable("bs", 1, 4096).bind(3), data.shape[-1]).keccak().shrink_to(3, 32),)

@save_uop(patches=patches)
def keccak_referenced(): return (Tensor(b"abc").keccak(),)

def fancy_index(x:Tensor) -> Tensor:
  b, c, d, e = (Tensor.empty(*s, dtype=dtypes.int32) for s in ((1,3,1,1,1,1), (1,1,4,1,1,1), (2,1,1,5,1,1), (1,1,1,1,6,1)))
  return x[None, b, c, d, e]

@save_uop(patches=patches)
def slice_fancy_indexing_dim_inject_none(): return (fancy_index(Tensor.empty(2, 5, 6, 5, 3, 4)),)

@save_uop(patches=patches)
def slice_fancy_indexing_dim_inject_none_bwd():
  x = Tensor.empty(2, 5, 6, 5, 3, 4)
  return (fancy_index(x).sum().gradient(x)[0],)

@save_uop(patches=patches)
def two_overlapping_binops():
  return (Tensor.randn(1, 3, 8, 16).conv2d(Tensor.randn(4, 3, 3, 3), stride=(2, 2), padding=(1, 1)).relu(),)

@save_uop(patches=patches)
def broadcast_full(): return (Tensor.empty(1, 3, 1, 7, 1).pow(Tensor.empty(2, 1, 5, 1, 8)),)

@save_uop(patches=patches)
def broadcast_partial(): return (Tensor.empty(5, 13, 24, 16, 2).pow(Tensor.empty(1, 13, 24, 1, 1)),)

@save_uop(patches=patches)
def randperm():
  Tensor.manual_seed(0)
  return (Tensor.randperm(1000),)

@save_uop(patches=patches)
def logcumsumexp(): return (Tensor.empty(45, 65).logcumsumexp(0),)

@save_uop(patches=patches)
def logcumsumexp_bwd():
  x = Tensor.empty(45, 65)
  return (x.logcumsumexp(0).sum().gradient(x)[0],)

@save_uop(patches=patches)
def scaled_dot_product_attention():
  return (Tensor.empty(32, 8, 16, 64).scaled_dot_product_attention(Tensor.empty(32, 8, 16, 64), Tensor.empty(32, 8, 16, 64)),)

@save_uop(patches=patches)
def scaled_dot_product_attention_mismatch_ls():
  return (Tensor.empty(32, 8, 4, 64).scaled_dot_product_attention(Tensor.empty(32, 8, 16, 64), Tensor.empty(32, 8, 16, 64)),)

@save_uop(patches=patches)
def multinomial_replacement(): return (Tensor([[0.453, 0., 1., 0.81], [0.1, 0.8, 0., 0.1]]).multinomial(300, replacement=True),)

@save_uop(patches=patches)
def multinomial_no_replacement(): return (Tensor(list(range(32))).multinomial(5, replacement=False),)

@save_uop(patches=patches)
def backprop_conv_wino_forward():
  devs = (f"{Device.DEFAULT}:1", f"{Device.DEFAULT}:2")
  conv = nn.Conv2d(3, 8, 3)
  conv.weight, conv.bias = Tensor.empty(8, 3, 3, 3, device=devs), Tensor.empty(8, device=devs)
  with Context(WINO=1): return (conv(Tensor(Tensor.empty(1, 3, 16, 16, device=devs).uop.unshard(0), device=devs)),)

@save_uop(patches=patches)
def padded_conv3d(): return (Tensor.empty(1, 4, 5, 5, 5).conv2d(Tensor.empty(4, 4, 3, 3, 3), padding=[1, 1, 1, 1, 1, 1]),)

@save_uop(patches=patches)
def cumprod(): return (Tensor.empty(20, 30, 40).cumprod(axis=2),)

@save_uop(patches=patches)
def cumprod_bwd():
  x = Tensor.empty(20, 30, 40)
  return (x.cumprod(axis=2).sum().gradient(x)[0],)

@save_uop(patches=patches)
def quant_linear_fallback():
  ggml_type, in_features, out_features = 12, 256, 16
  raw = Tensor.empty(out_features*in_features//256*QUANT_SIZES[ggml_type] + 4, dtype=dtypes.uint8)[4:]
  linear = Linear(in_features, out_features, bias=False)
  linear.weight = ggml_data_to_tensor(raw, out_features*in_features, ggml_type).reshape(out_features, in_features)
  linear.bias = Tensor.empty(out_features)
  return (linear(Tensor.empty(3, in_features))[:3],)

@save_uop(patches=patches)
def dilated_conv_transpose2d(): return (Tensor.empty(2, 4, 9, 9).conv_transpose2d(Tensor.empty(4, 4, 3, 3), dilation=(1, 2)),)

@save_uop(patches=patches)
def kv_cache_prefill():
  config = TransformerConfig(num_blocks=1, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2, norm_eps=1e-5, vocab_size=100, head_dim=32,
                             rope_theta=10000.0, rope_dim=32, v_head_dim=32, max_context=32)
  model, prompt = Transformer(config), list(range(1, 6))
  tokens = Tensor(prompt + [0] * (config.max_context - len(prompt)), dtype="int32").reshape(1, config.max_context)
  start_pos, n_toks = UOp.variable("start_pos", 0, config.max_context-1).bind(0), UOp.variable("toks", 1, 32).bind(len(prompt))
  return (model.forward(tokens[:, start_pos:start_pos+n_toks].contiguous(), Tensor([0.0]), start_pos),)

@save_uop(patches=patches)
def conv2d(): return (Tensor.empty(1, 3, 5, 7).conv2d(Tensor.empty(6, 1, 3, 3), groups=3),)

@save_uop(patches=patches)
def conv2d_ceildiv_edge_case():
  v = UOp.variable("v", 11, 100).bind(79)
  return (Tensor.randn(1, 8, 100)[:, :, :v].conv2d(weight=Tensor.randn(16, 8, 12), groups=1, stride=6, dilation=1, padding=(3, 3))[:, :, :13],)

def grads(fxn, *shapes) -> tuple[Tensor, ...]:
  ts = tuple(Tensor.empty(*s) for s in shapes)
  return tuple(fxn(*ts).sum().gradient(*ts))

@save_uop(patches=patches)
def broadcast_full_bwd(): return grads(Tensor.pow, (1, 3, 1, 7, 1), (2, 1, 5, 1, 8))

@save_uop(patches=patches)
def broadcast_partial_bwd(): return grads(Tensor.pow, (5, 13, 24, 16, 2), (1, 13, 24, 1, 1))

@save_uop(patches=patches)
def scaled_dot_product_attention_bwd(): return grads(Tensor.scaled_dot_product_attention, (32, 8, 16, 64), (32, 8, 16, 64), (32, 8, 16, 64))

@save_uop(patches=patches)
def scaled_dot_product_attention_mismatch_ls_bwd():
  return grads(Tensor.scaled_dot_product_attention, (32, 8, 4, 64), (32, 8, 16, 64), (32, 8, 16, 64))

@save_uop(patches=patches)
def padded_conv3d_bwd(): return grads(lambda x, w: x.conv2d(w, padding=[1, 1, 1, 1, 1, 1]), (1, 4, 5, 5, 5), (4, 4, 3, 3, 3))

@save_uop(patches=patches)
def dilated_conv_transpose2d_bwd(): return grads(lambda x, w: x.conv_transpose2d(w, dilation=(1, 2)), (2, 4, 9, 9), (4, 4, 3, 3))

@save_uop(patches=patches)
def conv2d_bwd(): return grads(lambda x, w: x.conv2d(w, groups=3), (1, 3, 5, 7), (6, 1, 3, 3))

@save_uop(patches=patches)
def svd_general(): return Tensor.empty(2, 2, 2, 2, 3).svd()

@save_uop(patches=patches)
def svd_identity_4x4(): return Tensor.eye(4).clone().svd()

@save_uop(patches=patches)
def qr_general(): return Tensor.empty(2, 2, 2, 2, 2).qr()

@save_uop(patches=patches)
def sort(): return Tensor.empty(8, 8, 6).sort(-1, descending=True)

def sharded_gemm(a_shape, b_shape, ref:bool) -> tuple[Tensor, ...]:
  from extra.gemm.cdna_asm_gemm import asm_gemm
  devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(2))
  a, b = (Tensor.empty(*s, dtype=dtypes.bfloat16).clone() for s in (a_shape, b_shape))
  a, b = a.shard(devs, axis=0), b.shard(devs, axis=None)
  out = a @ b if ref else asm_gemm(a, b)
  out.sum().backward()
  return (out, a.grad, b.grad)

@save_uop(patches=patches)
def gemm_multi(): return sharded_gemm((2, 64, 32), (32, 32), ref=False)

@save_uop(patches=patches)
def gemm_multi_ref(): return sharded_gemm((2, 64, 32), (32, 32), ref=True)

@save_uop(patches=patches)
def gemm_m_sharded(): return sharded_gemm((2*64, 32), (32, 64), ref=False)

@save_uop(patches=patches)
def gemm_m_sharded_ref(): return sharded_gemm((2*64, 32), (32, 64), ref=True)

@save_uop(patches=patches)
def backprop_conv_wino_step():
  with Context(TRAINING=1, WINO=1):
    devs = (f"{Device.DEFAULT}:1", f"{Device.DEFAULT}:2")
    conv = nn.Conv2d(3, 8, 3)
    for p in nn.state.get_parameters(conv): p.shard_(devs)
    opt = nn.optim.Adam(nn.state.get_parameters(conv))
    out = conv(Tensor.rand(2, 3, 16, 16).shard(devs, axis=0))
    opt.zero_grad()
    out.mean().backward()
    return tuple(opt.schedule_step())

# should be tuples

CORE = [mnist, conv_pool_bwd, tinyconv_bwd, gdb, newton_schulz, last_row_softmax]

KCE = [apply_rope_jit_prune, image_conv_fusion_minimal, preserve_multistage_reduce, multireduce_pad_reduce_unsafe,
       custom_kernel_2d_shrink, custom_kernel_double_permute, custom_kernel_expand, custom_kernel_flip, custom_kernel_offset_shrink,
       custom_kernel_pad, custom_kernel_permute, custom_kernel_reshape, custom_kernel_shrink, multi_tensor_local_shard_amax,
       symbolic_attention, cache_reduce_multiple_children, arange_fuse_grouped_children, partial_fuse1, partial_fuse3, partial_fuse4,
       reduce_expand_child, reduce_ext_reduce_child, reduce_multiple_paths, multireduce_reduce_multiple_paths,
       reduce_multiple_paths_midreduce_fused, reduce_same_size, reduce_shrink_child, adam_2convs_fuse, adam_conv_fuse, adam_step_fusion,
       fold_conv_batchnorm_optim_adam]

SLOW = [newton_schulz, gatedeltanet_reference_prefill, keccak_variable_bs, slice_fancy_indexing_dim_inject_none,
        slice_fancy_indexing_dim_inject_none_bwd, two_overlapping_binops, broadcast_full, broadcast_partial, randperm, logcumsumexp,
        logcumsumexp_bwd, kda_prefill, scaled_dot_product_attention, multinomial_replacement, multinomial_no_replacement,
        backprop_conv_wino_forward, padded_conv3d, cumprod, cumprod_bwd, quant_linear_fallback, dilated_conv_transpose2d,
        kv_cache_prefill, conv2d, keccak_referenced, scaled_dot_product_attention_mismatch_ls, conv2d_ceildiv_edge_case,
        broadcast_full_bwd, broadcast_partial_bwd, scaled_dot_product_attention_bwd, scaled_dot_product_attention_mismatch_ls_bwd,
        padded_conv3d_bwd, dilated_conv_transpose2d_bwd, conv2d_bwd, svd_general, svd_identity_4x4, qr_general, sort, gemm_multi,
        gemm_multi_ref, gemm_m_sharded, gemm_m_sharded_ref, backprop_conv_wino_step]

ALL = list(dict.fromkeys(CORE + KCE + SLOW))

CASES = {fxn.__name__:(fxn,) for fxn in ALL}
CASES.update(CORE=CORE, KCE=KCE, SLOW=SLOW, ALL=ALL)


