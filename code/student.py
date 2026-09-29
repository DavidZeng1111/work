"""Your algorithm goes here. The default is a complete, runnable baseline.

Required work: diagnose a limitation and implement a structural/training/memory
change. Explain it, measure its cost and perform a mechanism ablation. Merely
renaming the baseline or reporting a lucky seed is not an algorithmic contribution.
You can replace this factory/model completely while keeping the two model interfaces.
"""
"""进阶改进版：RMSNorm + SwiGLU(3x) + QKNorm + RoPE + Dropout
严格通过合约测试，仅改模型，接口完全兼容
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


# ==================== 1. RMSNorm ====================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ==================== 2. SwiGLU（3倍隐藏维度，平衡表达与效率） ====================
class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or 3 * width  # 从4x降到3x，减少过拟合，推理更快
        self.gate = nn.Linear(width, hidden_dim)
        self.up = nn.Linear(width, hidden_dim)
        self.down = nn.Linear(hidden_dim, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ==================== 3. 优化版 RoPE（短上下文适配 base=5000） ====================
class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 256, base: float = 5000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int, device=None):
        t = torch.arange(seq_len, device=device or self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos().repeat_interleave(2, dim=-1)
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        self.register_buffer('cos_cached', cos)
        self.register_buffer('sin_cached', sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[2]
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len, x.device)
        
        cos = self.cos_cached[:seq_len].view(1, 1, seq_len, self.head_dim)
        sin = self.sin_cached[:seq_len].view(1, 1, seq_len, self.head_dim)
        
        x1, x2 = x.chunk(2, dim=-1)
        x_rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + x_rotated * sin


# ==================== 4. Transformer Block（新增 QKNorm） ====================
class Block(nn.Module):
    def __init__(self, width=128, heads=4, dropout_rate=0.05):
        super().__init__()
        self.heads = heads
        head_dim = width // heads

        self.norm1 = RMSNorm(width)
        self.norm2 = RMSNorm(width)

        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        
        # 新增：QK 归一化，稳定注意力，提升表达能力
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        
        self.rope = RoPE(head_dim)

        self.dropout_attn = nn.Dropout(dropout_rate)
        self.dropout_mlp = nn.Dropout(dropout_rate)

        self.mlp = SwiGLU(width)

    def forward(self, x):
        batch, length, width = x.shape
        head_dim = width // self.heads

        # 注意力分支
        qkv = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # 核心改进：QK 归一化 + RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rope(q)
        k = self.rope(k)

        # 严格因果注意力
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        x = x + self.dropout_attn(self.proj(attended))

        # MLP 分支
        x = x + self.dropout_mlp(self.mlp(self.norm2(x)))
        return x


# ==================== 5. GPT 主模型 ====================
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        # 将默认 Dropout 降低至 0.02，避免小模型欠拟合
        dropout_rate = config.get('dropout', 0.02)

        self.token = nn.Embedding(config['vocab'], width)
        self.blocks = nn.ModuleList([
            Block(width, config['heads'], dropout_rate)
            for _ in range(config['depth'])
        ])
        self.norm = RMSNorm(width)

        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        
        # 核心改进：深度残差缩放初始化 (GPT-2 / Llama 标准做法)
        # 按层数的平方根缩小残差分支（Proj和Down）的初始权重，极大地稳定深层训练
        depth = config['depth']
        for name, param in self.named_parameters():
            if name.endswith('proj.weight') or name.endswith('down.weight'):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * depth))
                
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)

def build_model(config):
    return GPT(config)
