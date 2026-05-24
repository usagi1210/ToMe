# 兼容性说明：timm 0.4.12 → 0.9.16

官方 ToMe 要求 `timm==0.4.12`，但该版本与 PyTorch 2.7 不兼容。
本仓库升级至 `timm==0.9.16`，对 `tome/patch/timm.py` 做了以下三处修改。

## 变更 1：ToMeAttention — head_dim 获取方式

timm 0.9.x 引入了 `self.head_dim` 显式属性（0.4.12 中通过 `C // self.num_heads` 计算）。

**修改前（tome/patch/timm.py, ToMeAttention.forward）：**

    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        .permute(2, 0, 3, 1, 4)
    )

**修改后：**

    head_dim = getattr(self, 'head_dim', C // self.num_heads)
    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, head_dim)
        .permute(2, 0, 3, 1, 4)
    )

## 变更 2：ToMeAttention — q_norm / k_norm

timm 0.9.x 在 Attention 中新增了可选的 QK 归一化层（标准 ViT/DeiT 默认为 Identity）。

在 `q, k, v = qkv[0], qkv[1], qkv[2]` 之后添加：

    if hasattr(self, 'q_norm'):
        q = self.q_norm(q)
    if hasattr(self, 'k_norm'):
        k = self.k_norm(k)

## 变更 3：ToMeBlock — LayerScale (ls1 / ls2)

timm 0.9.x 在 Block 中引入了 LayerScale（ls1/ls2），标准模型默认为 Identity。

**修改前：**

    x_attn, metric = self.attn(self.norm1(x), attn_size)
    x = x + self._drop_path1(x_attn)
    ...
    x = x + self._drop_path2(self.mlp(self.norm2(x)))

**修改后：**

    x_attn, metric = self.attn(self.norm1(x), attn_size)
    x = x + self._drop_path1(getattr(self, 'ls1', lambda z: z)(x_attn))
    ...
    x = x + self._drop_path2(getattr(self, 'ls2', lambda z: z)(self.mlp(self.norm2(x))))
