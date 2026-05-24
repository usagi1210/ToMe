# ToMe 论文复现 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fork facebookresearch/ToMe，修复 timm 0.9.16 兼容性，编写纯 Python 脚本替代官方 Jupyter Notebooks，在 Ubuntu 服务器（RTX 5090 / CUDA 12.8）上完整复现论文 Table 1 的速度 + 精度结果。

**Architecture:** GitHub Fork（usagi1210/ToMe）作为唯一代码中枢；本地 Windows（`f:\Research\Project\ToMe`）编辑代码并 push；Ubuntu 服务器（`/home/cassi_116/SPI/ToMe`）pull 并运行所有实验。核心代码修改为 `tome/patch/timm.py`（3 处 API 兼容性修复），新增 `scripts/` 目录存放替代 notebooks 的独立 Python 脚本。

**Tech Stack:** Python 3.10, PyTorch 2.7.1+cu128, timm 0.9.16, torchvision 0.22.1, scipy 1.15.3, tqdm 4.67.1, pytest

---

## 文件变更地图

| 操作 | 文件 | 说明 |
|------|------|------|
| 修改 | `tome/patch/timm.py` | 3 处 timm 0.9.x 兼容性修复 |
| 修改 | `setup.py` | timm 版本 0.4.12 → 0.9.16 |
| 新建 | `tests/__init__.py` | pytest 包标记（空文件） |
| 新建 | `tests/test_tome_patch.py` | 兼容性 smoke tests |
| 新建 | `scripts/__init__.py` | 包标记（空文件） |
| 新建 | `scripts/demo.py` | 单次推理验证脚本 |
| 新建 | `scripts/benchmark.py` | throughput + ImageNet 精度 |
| 新建 | `scripts/visualize.py` | token 合并可视化 PNG |
| 新建 | `.gitignore` | 排除 Datasets/, results/*.csv 等 |
| 新建 | `results/.gitkeep` | 占位（保留空目录） |
| 新建 | `environment_5090.yml` | 环境文档（非 conda create 用） |
| 新建 | `docs/compatibility-notes.md` | 兼容性变更说明 |

---

## Task 0: 在 GitHub 上 Fork 仓库（浏览器手动操作）

**Files:** 无代码

- [ ] **Step 1: Fork 官方仓库**

  浏览器打开 https://github.com/facebookresearch/ToMe，点击右上角 **Fork** 按钮，Owner 选 `usagi1210`，Repository name 保持 `ToMe`，点击 **Create fork**。

  完成后访问 https://github.com/usagi1210/ToMe 确认 Fork 成功，页面顶部显示 `forked from facebookresearch/ToMe`。

---

## Task 1: 本地仓库初始化

**Files:**
- Create: `.gitignore`
- Create: `results/.gitkeep`

- [ ] **Step 1: 备份现有 docs/ 目录**

  在 PowerShell 中执行（注意在 `f:\Research\Project\` 层级，不是 ToMe 目录内）：

  ```powershell
  Copy-Item -Recurse "f:\Research\Project\ToMe\docs" "f:\Research\Project\ToMe_docs_backup"
  Remove-Item -Recurse -Force "f:\Research\Project\ToMe"
  ```

- [ ] **Step 2: Clone Fork 到本地**

  ```powershell
  git clone https://github.com/usagi1210/ToMe "f:\Research\Project\ToMe"
  cd "f:\Research\Project\ToMe"
  ```

  预期输出包含：`Cloning into 'f:\Research\Project\ToMe'...` 及 `done.`

- [ ] **Step 3: 恢复 docs/ 目录**

  ```powershell
  New-Item -ItemType Directory -Force "f:\Research\Project\ToMe\docs\superpowers"
  Copy-Item -Recurse "f:\Research\Project\ToMe_docs_backup\superpowers\specs" "f:\Research\Project\ToMe\docs\superpowers\specs"
  Copy-Item -Recurse "f:\Research\Project\ToMe_docs_backup\superpowers\plans" "f:\Research\Project\ToMe\docs\superpowers\plans"
  Remove-Item -Recurse -Force "f:\Research\Project\ToMe_docs_backup"
  ```

- [ ] **Step 4: 新建 .gitignore**

  新建 `f:\Research\Project\ToMe\.gitignore`，内容：

  ```gitignore
  # 数据集 - 体积过大，不上传
  Datasets/

  # 实验结果 - 本地生成，不同机器结果不同
  results/*.csv
  results/*.png

  # Python 编译产物
  __pycache__/
  *.py[cod]
  *.egg-info/
  build/
  dist/
  .eggs/

  # Jupyter
  .ipynb_checkpoints/

  # OS
  .DS_Store
  Thumbs.db
  ```

- [ ] **Step 5: 新建 results/.gitkeep（保留空目录）**

  新建 `f:\Research\Project\ToMe\results\.gitkeep`，内容为空。

- [ ] **Step 6: 提交并推送**

  ```bash
  git add .gitignore results/.gitkeep docs/
  git commit -m "chore: add gitignore, results dir placeholder, design spec and plan docs"
  git push origin main
  ```

  预期：`main -> main` 推送成功。

---

## Task 2: 项目脚手架文件

**Files:**
- Modify: `setup.py`
- Create: `environment_5090.yml`
- Create: `docs/compatibility-notes.md`

- [ ] **Step 1: 修改 setup.py 中的 timm 版本**

  打开 `setup.py`，找到第 18 行：
  ```python
  "timm==0.4.12",
  ```
  改为：
  ```python
  "timm==0.9.16",
  ```

- [ ] **Step 2: 新建 environment_5090.yml**

  新建 `f:\Research\Project\ToMe\environment_5090.yml`，内容：

  ```yaml
  # 记录 RTX 5090 (CUDA 12.8) 上实际可运行的包版本
  # 本文件仅作文档用途，不用于 conda create --file
  # 实际运行环境：cassi_116 conda env（已预装 torch/torchvision/scipy 等）

  name: cassi_116-tome-doc
  dependencies:
    - python=3.10.18
    - pip:
      - torch==2.7.1+cu128
      - torchvision==0.22.1+cu128
      - timm==0.9.16        # 官方要求 0.4.12，已修复兼容性（见 docs/compatibility-notes.md）
      - scipy==1.15.3
      - pillow==11.0.0
      - matplotlib==3.10.3
      - numpy==2.1.2
      - tqdm==4.67.1
      - pytest              # 用于运行 tests/

  # 服务器额外安装步骤（cassi_116 env 已有 torch 等，只需追加）：
  #   pip install timm==0.9.16
  #   pip install pytest
  #   cd /home/cassi_116/SPI/ToMe && pip install -e .
  ```

- [ ] **Step 3: 新建 docs/compatibility-notes.md**

  新建 `f:\Research\Project\ToMe\docs\compatibility-notes.md`，内容：

  ```markdown
  # 兼容性说明：timm 0.4.12 → 0.9.16

  官方 ToMe 要求 `timm==0.4.12`，但该版本与 PyTorch 2.7 不兼容。
  本仓库升级至 `timm==0.9.16`，对 `tome/patch/timm.py` 做了以下三处修改。

  ## 变更 1：ToMeAttention — head_dim 获取方式

  timm 0.9.x 引入了 `self.head_dim` 显式属性（0.4.12 中通过 `C // self.num_heads` 计算）。

  **修改前（tome/patch/timm.py, ToMeAttention.forward）：**
  ```python
  qkv = (
      self.qkv(x)
      .reshape(B, N, 3, self.num_heads, C // self.num_heads)
      .permute(2, 0, 3, 1, 4)
  )
  ```

  **修改后：**
  ```python
  head_dim = getattr(self, 'head_dim', C // self.num_heads)
  qkv = (
      self.qkv(x)
      .reshape(B, N, 3, self.num_heads, head_dim)
      .permute(2, 0, 3, 1, 4)
  )
  ```

  ## 变更 2：ToMeAttention — q_norm / k_norm

  timm 0.9.x 在 `Attention.__init__` 中新增了可选的 QK 归一化层
  （标准 ViT/DeiT 默认为 `nn.Identity()`，但需要显式调用）。

  **在 `q, k, v = qkv[0], qkv[1], qkv[2]` 之后添加：**
  ```python
  # timm 0.9.x compat: apply optional QK normalization
  if hasattr(self, 'q_norm'):
      q = self.q_norm(q)
  if hasattr(self, 'k_norm'):
      k = self.k_norm(k)
  ```

  ## 变更 3：ToMeBlock — LayerScale (ls1 / ls2)

  timm 0.9.x 在 `Block` 中引入了 `ls1`/`ls2` LayerScale 模块
  （标准 ViT/DeiT 默认为 `nn.Identity()`，但需要显式调用）。

  **修改前（ToMeBlock.forward）：**
  ```python
  x_attn, metric = self.attn(self.norm1(x), attn_size)
  x = x + self._drop_path1(x_attn)
  ...
  x = x + self._drop_path2(self.mlp(self.norm2(x)))
  ```

  **修改后：**
  ```python
  x_attn, metric = self.attn(self.norm1(x), attn_size)
  # timm 0.9.x compat: apply LayerScale if present
  x = x + self._drop_path1(getattr(self, 'ls1', lambda z: z)(x_attn))
  ...
  x = x + self._drop_path2(getattr(self, 'ls2', lambda z: z)(self.mlp(self.norm2(x))))
  ```
  ```

- [ ] **Step 4: 提交并推送**

  ```bash
  git add setup.py environment_5090.yml docs/compatibility-notes.md
  git commit -m "chore: setup.py timm version bump, add environment doc and compatibility notes"
  git push origin main
  ```

---

## Task 3: 修复 timm 0.9.16 兼容性

**Files:**
- Modify: `tome/patch/timm.py`（3 处）
- Create: `tests/__init__.py`
- Create: `tests/test_tome_patch.py`

- [ ] **Step 1: 新建 tests/__init__.py（空文件）**

  新建 `f:\Research\Project\ToMe\tests\__init__.py`，内容为空。

- [ ] **Step 2: 新建 tests/test_tome_patch.py（先写测试）**

  新建 `f:\Research\Project\ToMe\tests\test_tome_patch.py`，内容：

  ```python
  """
  Smoke tests: 验证 ToMe patch 在 timm 0.9.16 下正常工作。
  服务器运行：pytest tests/test_tome_patch.py -v
  """
  import torch
  import timm
  import tome


  def test_patch_applies_without_error():
      """ToMe patch 应能无报错地应用到 timm 模型。"""
      model = timm.create_model('deit_small_patch16_224', pretrained=False)
      tome.patch.timm(model, prop_attn=False)
      model.r = 8
      assert hasattr(model, '_tome_info'), "_tome_info 应在 patch 后存在"
      assert model._tome_info['class_token'] is True


  def test_forward_r0_correct_shape():
      """r=0（不合并）时，输出 shape 应为 (batch, 1000)，无 NaN。"""
      model = timm.create_model('deit_small_patch16_224', pretrained=False)
      tome.patch.timm(model, prop_attn=False)
      model.r = 0
      model.eval()

      x = torch.randn(2, 3, 224, 224)
      with torch.no_grad():
          out = model(x)

      assert out.shape == (2, 1000), f"期望 (2, 1000)，实际 {out.shape}"
      assert not torch.isnan(out).any(), "输出包含 NaN"
      assert not torch.isinf(out).any(), "输出包含 Inf"


  def test_forward_r16_correct_shape():
      """r=16 时，输出 shape 应为 (batch, 1000)，无 NaN。"""
      model = timm.create_model('deit_small_patch16_224', pretrained=False)
      tome.patch.timm(model, prop_attn=False)
      model.r = 16
      model.eval()

      x = torch.randn(2, 3, 224, 224)
      with torch.no_grad():
          out = model(x)

      assert out.shape == (2, 1000), f"期望 (2, 1000)，实际 {out.shape}"
      assert not torch.isnan(out).any(), "输出包含 NaN"
      assert not torch.isinf(out).any(), "输出包含 Inf"


  def test_prop_attn_no_error():
      """prop_attn=True（离线评估模式）也应正常工作。"""
      model = timm.create_model('deit_small_patch16_224', pretrained=False)
      tome.patch.timm(model, prop_attn=True)
      model.r = 8
      model.eval()

      x = torch.randn(1, 3, 224, 224)
      with torch.no_grad():
          out = model(x)

      assert out.shape == (1, 1000)
      assert not torch.isnan(out).any()


  def test_source_tracking():
      """trace_source=True 时，_tome_info['source'] 应被填充。"""
      model = timm.create_model('deit_tiny_patch16_224', pretrained=False)
      tome.patch.timm(model, trace_source=True, prop_attn=False)
      model.r = 8
      model.eval()

      x = torch.randn(1, 3, 224, 224)
      with torch.no_grad():
          _ = model(x)

      source = model._tome_info.get('source')
      assert source is not None, "source tracking 应返回非 None"
      assert source.dim() == 3, f"source 应为 3D tensor，实际 {source.dim()}D"
  ```

- [ ] **Step 3: 修改 tome/patch/timm.py — 变更 1（head_dim）**

  打开 `f:\Research\Project\ToMe\tome\patch\timm.py`，找到 `ToMeAttention.forward()` 中的以下代码：

  ```python
          B, N, C = x.shape
          qkv = (
              self.qkv(x)
              .reshape(B, N, 3, self.num_heads, C // self.num_heads)
              .permute(2, 0, 3, 1, 4)
          )
          q, k, v = (
              qkv[0],
              qkv[1],
              qkv[2],
          )  # make torchscript happy (cannot use tensor as tuple)
  ```

  替换为：

  ```python
          B, N, C = x.shape
          # timm 0.9.x compat: use self.head_dim if available, else compute
          head_dim = getattr(self, 'head_dim', C // self.num_heads)
          qkv = (
              self.qkv(x)
              .reshape(B, N, 3, self.num_heads, head_dim)
              .permute(2, 0, 3, 1, 4)
          )
          q, k, v = (
              qkv[0],
              qkv[1],
              qkv[2],
          )  # make torchscript happy (cannot use tensor as tuple)
          # timm 0.9.x compat: apply optional QK normalization layers
          if hasattr(self, 'q_norm'):
              q = self.q_norm(q)
          if hasattr(self, 'k_norm'):
              k = self.k_norm(k)
  ```

- [ ] **Step 4: 修改 tome/patch/timm.py — 变更 2（LayerScale ls1）**

  在同一文件，找到 `ToMeBlock.forward()` 中的：

  ```python
          x_attn, metric = self.attn(self.norm1(x), attn_size)
          x = x + self._drop_path1(x_attn)
  ```

  替换为：

  ```python
          x_attn, metric = self.attn(self.norm1(x), attn_size)
          # timm 0.9.x compat: apply LayerScale (ls1) if present, else identity
          x = x + self._drop_path1(getattr(self, 'ls1', lambda z: z)(x_attn))
  ```

- [ ] **Step 5: 修改 tome/patch/timm.py — 变更 3（LayerScale ls2）**

  在同一文件，找到 `ToMeBlock.forward()` 末尾的：

  ```python
          x = x + self._drop_path2(self.mlp(self.norm2(x)))
          return x
  ```

  替换为：

  ```python
          # timm 0.9.x compat: apply LayerScale (ls2) if present, else identity
          x = x + self._drop_path2(getattr(self, 'ls2', lambda z: z)(self.mlp(self.norm2(x))))
          return x
  ```

- [ ] **Step 6: 提交并推送**

  ```bash
  git add tome/patch/timm.py tests/
  git commit -m "fix: timm 0.9.16 compat — head_dim, q/k_norm, LayerScale in patch/timm.py"
  git push origin main
  ```

---

## Task 4: 编写 scripts/demo.py

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/demo.py`

- [ ] **Step 1: 新建 scripts/__init__.py（空文件）**

  新建 `f:\Research\Project\ToMe\scripts\__init__.py`，内容为空。

- [ ] **Step 2: 新建 scripts/demo.py**

  新建 `f:\Research\Project\ToMe\scripts\demo.py`，内容：

  ```python
  #!/usr/bin/env python3
  """
  ToMe Demo: 加载 timm ViT 模型，应用 ToMe patch，进行单次推理。
  验证 ToMe 安装与兼容性修复是否正确。

  用法：
      python scripts/demo.py                                  # 默认 deit_small, r=16
      python scripts/demo.py --model deit_tiny_patch16_224 --r 8
      python scripts/demo.py --r 0                           # baseline（不合并）
  """
  import argparse

  import torch
  import timm

  import tome


  def main():
      parser = argparse.ArgumentParser(description='ToMe Demo')
      parser.add_argument('--model', default='deit_small_patch16_224',
                          help='timm model name (default: deit_small_patch16_224)')
      parser.add_argument('--r', type=int, default=16,
                          help='每层合并的 token 数；0 = baseline，不合并 (default: 16)')
      parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
      args = parser.parse_args()

      print(f"Device : {args.device}")
      if torch.cuda.is_available():
          print(f"GPU    : {torch.cuda.get_device_name(0)}")

      print(f"\nLoading model : {args.model}")
      model = timm.create_model(args.model, pretrained=False)
      model = model.to(args.device)
      model.eval()

      n_params = sum(p.numel() for p in model.parameters()) / 1e6
      print(f"Parameters    : {n_params:.1f}M")

      if args.r > 0:
          tome.patch.timm(model, prop_attn=False)
          model.r = args.r
          print(f"ToMe applied  : r={args.r} tokens merged per layer")
      else:
          print("Mode          : baseline (no ToMe)")

      x = torch.randn(1, 3, 224, 224, device=args.device)
      with torch.no_grad():
          out = model(x)

      print(f"\nInput shape   : {tuple(x.shape)}")
      print(f"Output shape  : {tuple(out.shape)}")
      print(f"Top-5 indices : {out.topk(5).indices[0].tolist()}")
      print("\nDemo completed successfully ✓")


  if __name__ == '__main__':
      main()
  ```

- [ ] **Step 3: 提交并推送**

  ```bash
  git add scripts/
  git commit -m "feat: add scripts/demo.py"
  git push origin main
  ```

---

## Task 5: 编写 scripts/benchmark.py

**Files:**
- Create: `scripts/benchmark.py`

- [ ] **Step 1: 新建 scripts/benchmark.py**

  新建 `f:\Research\Project\ToMe\scripts\benchmark.py`，内容：

  ```python
  #!/usr/bin/env python3
  """
  ToMe Benchmark: 测量 throughput (img/s) 和 ImageNet Top-1 精度，复现论文 Table 1。

  用法（仅速度，无需 ImageNet）：
      python scripts/benchmark.py --model deit_small_patch16_224 --r 0 8 16 24 32

  用法（速度 + 精度，需要 ImageNet 验证集）：
      python scripts/benchmark.py \\
          --model deit_small_patch16_224 \\
          --imagenet-val ./Datasets/imagenet/val \\
          --r 0 8 16 24 32 \\
          --batch-size 256 \\
          --output results/benchmark_deit_small.csv

  复现论文 Table 1（多个模型，在服务器上依次运行）：
      for model in deit_tiny_patch16_224 deit_small_patch16_224 deit_base_patch16_224; do
          python scripts/benchmark.py --model $model \\
              --imagenet-val ./Datasets/imagenet/val \\
              --r 0 8 16 24 32 --batch-size 256 \\
              --output results/benchmark_${model}.csv
      done
  """
  import argparse
  import csv
  from pathlib import Path

  import torch
  import torchvision
  import torchvision.transforms as T
  import timm

  import tome
  from tome.utils import benchmark as tome_benchmark


  def get_imagenet_val_loader(val_dir: str, batch_size: int, num_workers: int = 8):
      """标准 ImageNet 验证集加载器（Resize 256 → CenterCrop 224 → 归一化）。"""
      transform = T.Compose([
          T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
          T.CenterCrop(224),
          T.ToTensor(),
          T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ])
      dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform)
      loader = torch.utils.data.DataLoader(
          dataset,
          batch_size=batch_size,
          shuffle=False,
          num_workers=num_workers,
          pin_memory=True,
      )
      return loader


  @torch.no_grad()
  def measure_accuracy(model: torch.nn.Module, loader, device: str) -> float:
      """在 ImageNet 验证集（50000 张）上评估 Top-1 精度。"""
      model.eval()
      correct = 0
      total = 0
      for images, labels in loader:
          images = images.to(device, non_blocking=True)
          labels = labels.to(device, non_blocking=True)
          outputs = model(images)
          _, predicted = outputs.max(1)
          total += labels.size(0)
          correct += predicted.eq(labels).sum().item()
          if total % 10000 == 0:
              print(f"    [{total:>5}/50000] current top-1: {100.0 * correct / total:.2f}%")
      return 100.0 * correct / total


  def run_single(model_name: str, r: int, imagenet_val: str,
                 batch_size: int, device: str) -> dict:
      """针对单个 (model, r) 组合测量 throughput 和 accuracy。"""
      # 每次加载新模型，避免状态污染
      model = timm.create_model(model_name, pretrained=True)
      model = model.to(device)
      model.eval()

      if r > 0:
          tome.patch.timm(model, prop_attn=True)
          model.r = r

      # Throughput（直接使用 tome.utils.benchmark）
      print(f"  Measuring throughput...")
      throughput = tome_benchmark(
          model,
          device=device,
          input_size=(3, 224, 224),
          batch_size=batch_size,
          runs=40,
          throw_out=0.25,
          verbose=False,
      )
      print(f"  Throughput  : {throughput:.1f} img/s")

      # Accuracy（需要 ImageNet 数据集）
      top1 = None
      if imagenet_val:
          print(f"  Measuring accuracy...")
          loader = get_imagenet_val_loader(imagenet_val, batch_size=batch_size)
          top1 = measure_accuracy(model, loader, device)
          print(f"  Top-1 Acc   : {top1:.2f}%")
      else:
          print(f"  Accuracy    : skipped (no --imagenet-val)")

      return {
          'model': model_name,
          'r': r,
          'throughput_img_s': round(throughput, 1),
          'top1_acc': round(top1, 2) if top1 is not None else 'N/A',
      }


  def print_table(rows: list):
      """终端打印整洁的结果表格。"""
      print("\n" + "=" * 68)
      print(f"  {'model':<33} {'r':>4}  {'img/s':>10}  {'top-1 (%)':>10}")
      print("  " + "-" * 64)
      for row in rows:
          print(f"  {row['model']:<33} {row['r']:>4}  "
                f"{str(row['throughput_img_s']):>10}  {str(row['top1_acc']):>10}")
      print("=" * 68)


  def main():
      parser = argparse.ArgumentParser(description='ToMe Benchmark')
      parser.add_argument('--model', default='deit_small_patch16_224',
                          help='timm model name')
      parser.add_argument('--imagenet-val', default=None,
                          help='ImageNet val 目录路径（不提供则跳过精度，仅测速度）')
      parser.add_argument('--r', type=int, nargs='+', default=[0, 8, 16, 24, 32],
                          help='要测试的 r 值列表，0 = baseline（默认: 0 8 16 24 32）')
      parser.add_argument('--batch-size', type=int, default=256,
                          help='batch size（RTX 5090 建议 256，DeiT-B 可降为 128）')
      parser.add_argument('--output', default=None,
                          help='结果 CSV 路径，如 results/benchmark_deit_small.csv')
      parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
      args = parser.parse_args()

      print(f"Device  : {args.device}")
      if torch.cuda.is_available():
          print(f"GPU     : {torch.cuda.get_device_name(0)}")
      print(f"Model   : {args.model}")
      print(f"r values: {args.r}")
      if args.imagenet_val:
          print(f"ImageNet: {args.imagenet_val}")
      else:
          print("ImageNet: skipped — provide --imagenet-val for accuracy")

      rows = []
      for r in args.r:
          print(f"\n{'─' * 50}")
          print(f"r = {r} {'(baseline)' if r == 0 else ''}")
          row = run_single(
              model_name=args.model,
              r=r,
              imagenet_val=args.imagenet_val,
              batch_size=args.batch_size,
              device=args.device,
          )
          rows.append(row)

      print_table(rows)

      if args.output:
          Path(args.output).parent.mkdir(parents=True, exist_ok=True)
          with open(args.output, 'w', newline='') as f:
              writer = csv.DictWriter(f, fieldnames=rows[0].keys())
              writer.writeheader()
              writer.writerows(rows)
          print(f"\nResults saved → {args.output}")


  if __name__ == '__main__':
      main()
  ```

- [ ] **Step 2: 提交并推送**

  ```bash
  git add scripts/benchmark.py
  git commit -m "feat: add scripts/benchmark.py (throughput + ImageNet Top-1)"
  git push origin main
  ```

---

## Task 6: 编写 scripts/visualize.py

**Files:**
- Create: `scripts/visualize.py`

- [ ] **Step 1: 新建 scripts/visualize.py**

  新建 `f:\Research\Project\ToMe\scripts\visualize.py`，内容：

  ```python
  #!/usr/bin/env python3
  """
  ToMe Visualization: 可视化哪些 token 被合并，输出对比 PNG。
  使用 ToMe 内置的 make_visualization 函数（tome/vis.py）。

  用法：
      python scripts/visualize.py --image path/to/image.jpg
      python scripts/visualize.py --image path/to/image.jpg --r 8 --output results/vis_r8.png
      python scripts/visualize.py --image path/to/image.jpg --r 0  # r=0 显示无合并结果
  """
  import argparse
  from pathlib import Path

  import torch
  import timm
  from PIL import Image
  import matplotlib.pyplot as plt
  from torchvision import transforms

  import tome
  from tome.vis import make_visualization


  def load_and_preprocess(image_path: str, size: int = 224):
      """加载图片，返回预处理后的 tensor 和用于显示的 PIL Image。"""
      transform = transforms.Compose([
          transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
          transforms.CenterCrop(size),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ])
      img_pil = Image.open(image_path).convert('RGB')
      img_display = img_pil.resize((size, size), Image.BICUBIC)
      x = transform(img_pil).unsqueeze(0)
      return x, img_display


  def main():
      parser = argparse.ArgumentParser(description='ToMe Visualization')
      parser.add_argument('--image', required=True,
                          help='输入图片路径（JPG/PNG）')
      parser.add_argument('--model', default='deit_small_patch16_224',
                          help='timm model name (default: deit_small_patch16_224)')
      parser.add_argument('--r', type=int, default=16,
                          help='每层合并的 token 数 (default: 16)')
      parser.add_argument('--output', default='results/visualization.png',
                          help='输出 PNG 路径 (default: results/visualization.png)')
      parser.add_argument('--patch-size', type=int, default=16,
                          help='模型 patch size，deit/vit 系列为 16 (default: 16)')
      parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
      args = parser.parse_args()

      print(f"Loading model : {args.model}")
      model = timm.create_model(args.model, pretrained=True)
      model = model.to(args.device)
      model.eval()

      # patch with source tracking enabled
      tome.patch.timm(model, trace_source=True, prop_attn=False)
      model.r = args.r
      print(f"ToMe applied  : r={args.r}, trace_source=True")

      print(f"Loading image : {args.image}")
      x, img_display = load_and_preprocess(args.image)
      x = x.to(args.device)

      with torch.no_grad():
          _ = model(x)

      source = model._tome_info.get('source')
      if source is None:
          print("ERROR: source is None. "
                "Make sure trace_source=True and r > 0.")
          return

      # 使用 tome.vis.make_visualization 生成可视化图
      vis_img = make_visualization(
          img_display,
          source,
          patch_size=args.patch_size,
          class_token=(model._tome_info.get('class_token', True)),
      )

      # 保存并列对比图
      fig, axes = plt.subplots(1, 2, figsize=(10, 5))
      axes[0].imshow(img_display)
      axes[0].set_title('Original Image', fontsize=13)
      axes[0].axis('off')

      axes[1].imshow(vis_img)
      axes[1].set_title(f'Token Merging  r={args.r}', fontsize=13)
      axes[1].axis('off')

      plt.tight_layout()
      Path(args.output).parent.mkdir(parents=True, exist_ok=True)
      plt.savefig(args.output, dpi=150, bbox_inches='tight')
      plt.close()

      print(f"Visualization saved → {args.output}")


  if __name__ == '__main__':
      main()
  ```

- [ ] **Step 2: 提交并推送**

  ```bash
  git add scripts/visualize.py
  git commit -m "feat: add scripts/visualize.py (token merging visualization)"
  git push origin main
  ```

---

## Task 7: 服务器部署与验证

> 以下所有命令在 Ubuntu 服务器上通过 SSH 执行，conda env 为 `cassi_116`。

**Files:** 无新代码，全为服务器执行步骤

- [ ] **Step 1: 克隆 Fork 到服务器**

  ```bash
  conda activate cassi_116
  mkdir -p /home/cassi_116/SPI/ToMe
  cd /home/cassi_116/SPI/ToMe
  git clone https://github.com/usagi1210/ToMe .
  ls
  ```

  预期：`ls` 输出包含 `tome/  scripts/  tests/  setup.py  .gitignore`

- [ ] **Step 2: 安装缺失依赖**

  ```bash
  conda activate cassi_116
  cd /home/cassi_116/SPI/ToMe
  pip install timm==0.9.16
  pip install pytest
  pip install -e .
  ```

  预期最后输出：`Successfully installed tome-0.1`

- [ ] **Step 3: 运行兼容性测试**

  ```bash
  cd /home/cassi_116/SPI/ToMe
  pytest tests/test_tome_patch.py -v
  ```

  预期输出：
  ```
  tests/test_tome_patch.py::test_patch_applies_without_error PASSED
  tests/test_tome_patch.py::test_forward_r0_correct_shape     PASSED
  tests/test_tome_patch.py::test_forward_r16_correct_shape    PASSED
  tests/test_tome_patch.py::test_prop_attn_no_error           PASSED
  tests/test_tome_patch.py::test_source_tracking              PASSED
  5 passed in XX.XXs
  ```

  若有 FAILED：查看错误信息，在本地修复 `tome/patch/timm.py`，`git commit && git push`，
  然后服务器执行 `git pull && pytest tests/ -v` 重试。

- [ ] **Step 4: 运行 demo 验证基本功能**

  ```bash
  python scripts/demo.py --model deit_small_patch16_224 --r 16
  ```

  预期输出：
  ```
  Device : cuda
  GPU    : NVIDIA GeForce RTX 5090

  Loading model : deit_small_patch16_224
  Parameters    : 22.1M
  ToMe applied  : r=16 tokens merged per layer

  Input shape   : (1, 3, 224, 224)
  Output shape  : (1, 1000)
  Top-5 indices : [...]

  Demo completed successfully ✓
  ```

- [ ] **Step 5: 运行速度 benchmark（无需 ImageNet）**

  ```bash
  python scripts/benchmark.py \
      --model deit_small_patch16_224 \
      --r 0 8 16 24 32 \
      --batch-size 256 \
      --output results/throughput_deit_small.csv
  ```

  预期趋势：r 越大，throughput 越高（相对 r=0 有约 1.5× ～ 2.5× 提升）。

- [ ] **Step 6: 准备 ImageNet 验证集（待数据集就绪后执行）**

  将 ImageNet 验证集放置于：`/home/cassi_116/SPI/ToMe/Datasets/imagenet/val/`

  目录结构验证：
  ```bash
  ls /home/cassi_116/SPI/ToMe/Datasets/imagenet/val/ | head -5
  # 预期：n01440764  n01443537  n01484850 ...（1000 个 synset 文件夹）

  ls /home/cassi_116/SPI/ToMe/Datasets/imagenet/val/n01440764/ | wc -l
  # 预期：50（每个类别 50 张图片）
  ```

- [ ] **Step 7: 运行完整 benchmark（速度 + 精度，复现论文 Table 1）**

  ```bash
  # DeiT-Ti
  python scripts/benchmark.py \
      --model deit_tiny_patch16_224 \
      --imagenet-val ./Datasets/imagenet/val \
      --r 0 8 16 24 32 \
      --batch-size 256 \
      --output results/benchmark_deit_tiny.csv

  # DeiT-S
  python scripts/benchmark.py \
      --model deit_small_patch16_224 \
      --imagenet-val ./Datasets/imagenet/val \
      --r 0 8 16 24 32 \
      --batch-size 256 \
      --output results/benchmark_deit_small.csv

  # DeiT-B（显存较大，batch 降为 128）
  python scripts/benchmark.py \
      --model deit_base_patch16_224 \
      --imagenet-val ./Datasets/imagenet/val \
      --r 0 8 16 24 32 \
      --batch-size 128 \
      --output results/benchmark_deit_base.csv
  ```

  **验收标准（趋势对比论文 Table 1，绝对速度因 GPU 不同会有差异）：**

  | 模型 | r=0 Top-1 | r=16 Top-1 | r=16 速度提升 |
  |------|----------|-----------|------------|
  | deit_tiny | ~72.2% | ~71.9% | ~1.5× |
  | deit_small | ~79.8% | ~79.5% | ~1.7× |
  | deit_base | ~81.8% | ~81.4% | ~1.7× |

- [ ] **Step 8: 运行可视化（准备一张测试图片）**

  ```bash
  # 下载一张测试图片（或用任意 JPG）
  wget -O /tmp/test_img.jpg https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg

  python scripts/visualize.py \
      --image /tmp/test_img.jpg \
      --model deit_small_patch16_224 \
      --r 16 \
      --output results/visualization_r16.png
  ```

  预期：`results/visualization_r16.png` 文件生成，显示原图与 token 合并分组对比图。

---

## 日常工作流（供参考）

本地修改代码后同步到服务器：

```bash
# 本地（Windows PowerShell）
git add .
git commit -m "描述你的修改"
git push origin main

# 服务器（SSH）
cd /home/cassi_116/SPI/ToMe
git pull origin main
# 然后重新运行相关脚本
```
