# ToMe 论文复现设计文档

**日期**：2026-05-24  
**论文**：Token Merging: Your ViT But Faster（ICLR 2023）  
**官方仓库**：[facebookresearch/ToMe](https://github.com/facebookresearch/ToMe)（已归档，只读）  
**目标**：运行官方 benchmark，验证论文结果可复现

---

## 1. 项目目标

运行 ToMe 官方 examples 和 benchmark notebooks，验证论文 Table 1 中的速度与精度结果在当前硬件环境下可复现。不涉及重新训练模型，也不涉及在自定义数据集上的扩展实验。

---

## 2. 整体架构

```text
本地 Windows (f:\Research\Project\ToMe)
  ├── git clone 自己的 Fork
  ├── 代码编辑（VSCode）
  └── git add / commit / push
          │
          ▼
GitHub (your-username/ToMe)   ← Fork 自 facebookresearch/ToMe
          │
          ▼ git pull
Ubuntu 服务器 (/home/cassi_116/SPI/ToMe)
  ├── conda env: cassi_116（已有完整 CUDA 12.8 环境）
  ├── 额外安装：timm==0.9.16, jupyter
  ├── pip install -e .（安装 ToMe 包）
  └── 运行 benchmark notebooks
```

**职责划分：**
- **本地 Windows**：代码编辑、git 管理，不运行实验
- **GitHub Fork**：唯一的代码同步中枢
- **Ubuntu 服务器**：所有计算，RTX 5090 + CUDA 12.8

---

## 3. 仓库结构

Fork 后的仓库保持官方目录结构，新增少量文件：

```text
ToMe/
├── tome/                          # 核心库
│   ├── __init__.py
│   ├── merge.py                   # 核心 Bipartite Soft Matching 算法
│   ├── patch/
│   │   ├── timm.py                # timm ViT patch（需兼容性修复）
│   │   ├── mae.py                 # MAE patch
│   │   └── ...
│   └── utils.py
├── examples/                      # 官方示例 notebooks
│   ├── 0_example_timm.ipynb
│   ├── 1_benchmark_timm.ipynb     # 主要复现目标
│   ├── 2_visualization.ipynb
│   └── ...
├── docs/
│   └── superpowers/specs/         # 本设计文档所在位置
├── scripts/
│   └── run_benchmark.sh           # 新增：一键运行 benchmark 脚本
├── environment_5090.yml           # 新增：记录 RTX 5090 实际可用的包版本（文档用途，非 conda create 用）
├── setup.py
└── README.md
```

---

## 4. 依赖与兼容性

### 4.1 服务器现有环境（conda env: cassi_116）

| 包 | 版本 | ToMe 需求 | 状态 |
|----|------|---------|------|
| python | 3.10.18 | ≥3.8 | ✅ |
| torch | 2.7.1+cu128 | ≥1.12.1 | ✅ |
| torchvision | 0.22.1+cu128 | 匹配 torch | ✅ |
| scipy | 1.15.3 | 需要 | ✅ |
| pillow | 11.0.0 | 需要 | ✅ |
| matplotlib | 3.10.3 | 需要 | ✅ |
| timm | 未安装 | ==0.4.12 | ⚠️ |
| jupyter | 未安装 | 需要 | ⚠️ |

### 4.2 timm 版本策略

官方要求 `timm==0.4.12`，但该版本与 PyTorch 2.7 不兼容。  
**采用 `timm==0.9.16`**，该版本：
- 与 PyTorch 2.7 完全兼容
- API 与 0.4.12 差异较小，主要集中在 `tome/patch/timm.py`

**需要确认的兼容性变更点（在 `tome/patch/timm.py` 中）：**
- `timm.models.vision_transformer.Attention` 类的 `forward` 方法签名
- `timm.models.vision_transformer.Block` 类的属性名
- `timm.models.vision_transformer.VisionTransformer.forward` 的参数

如果 0.9.16 存在问题，备选方案是 `timm==1.0.11`（需更多改动）。

### 4.3 安装命令

```bash
# 在服务器的 cassi_116 环境中执行
pip install timm==0.9.16
# 注意：不再需要 jupyter

# 克隆并安装 ToMe
cd /home/cassi_116/SPI/ToMe
git clone https://github.com/usagi1210/ToMe .
pip install -e .
```

---

## 5. Git 工作流

### 5.1 初始化

```bash
# 1. 在 GitHub 上 Fork facebookresearch/ToMe

# 2. 本地 Windows 克隆
cd f:\Research\Project\ToMe
git clone https://github.com/usagi1210/ToMe .

# 3. 服务器克隆
cd /home/cassi_116/SPI/ToMe
git clone https://github.com/usagi1210/ToMe .
```

### 5.2 日常工作流

```bash
# 本地（Windows）修改代码后：
git add .
git commit -m "描述修改内容"
git push origin main

# 服务器拉取：
git pull origin main
```

### 5.3 分支策略

- `main`：可运行的稳定代码，直接用于服务器
- （可选）`dev`：实验性修改，测试通过后 merge 到 main

---

## 6. Benchmark 运行方案

### 6.1 运行方式

**不使用 Jupyter Notebook**，改为纯 Python 脚本，在服务器上直接执行。

### 6.2 脚本列表（新建于 `scripts/` 目录）

| 脚本 | 对应原 notebook | 功能 | 优先级 |
|------|---------------|------|--------|
| `scripts/demo.py` | `0_example_timm.ipynb` | 基础用法：加载模型、应用 ToMe、单次推理验证 | 高（先跑） |
| `scripts/benchmark.py` | `1_benchmark_timm.ipynb` | 速度（throughput img/s）+ 精度（ImageNet Top-1） | 高 |
| `scripts/visualize.py` | `2_visualization.ipynb` | 可视化 token 合并效果，输出 PNG | 中 |

### 6.3 benchmark.py 设计

```bash
# 运行示例
python scripts/benchmark.py \
    --model deit_small_patch16_224 \
    --imagenet-val /path/to/imagenet/val \
    --r 0 8 16 24 32 \
    --batch-size 256 \
    --output results/benchmark_deit_small.csv
```

**命令行参数：**
- `--model`：timm 模型名称（deit_tiny/small/base, vit_base/large）
- `--imagenet-val`：ImageNet 验证集路径（服务器上已有则直接指定）
- `--r`：要测试的 token 合并数量列表（0 = baseline，无 ToMe）
- `--batch-size`：批大小，RTX 5090 可用 256 或更大
- `--output`：结果保存为 CSV 文件

**输出格式（CSV + 终端打印）：**

```
model                    r    throughput(img/s)    top1_acc(%)
deit_small_patch16_224   0    1024.3               79.8
deit_small_patch16_224   8    1456.7               79.5
deit_small_patch16_224   16   1821.2               79.1
deit_small_patch16_224   24   2103.4               78.4
```

### 6.4 ImageNet 验证集

benchmark 需要 ImageNet 验证集（约 6.3GB，50000 张图片）。

**数据集路径约定（本地与服务器保持一致的相对路径）：**

```text
ToMe/
└── Datasets/
    └── imagenet/
        └── val/
            ├── n01440764/   ← synset 文件夹
            │   ├── ILSVRC2012_val_00000293.JPEG
            │   └── ...
            └── ...
```

- 本地 Windows：`F:\Research\Project\ToMe\Datasets\imagenet\val`
- Ubuntu 服务器：`/home/cassi_116/SPI/ToMe/Datasets\imagenet\val`

脚本中使用相对路径 `./Datasets/imagenet/val`，本地与服务器无需修改命令。  
数据集路径待后续确认，`Datasets/` 目录加入 `.gitignore`（不上传 GitHub）。

---

## 7. 兼容性修复策略

如果 `timm==0.9.16` 导致运行错误，按以下顺序排查：

1. **`AttributeError`**：timm 内部类属性名变更 → 在 `tome/patch/timm.py` 中查找并更新属性引用
2. **`ImportError`**：模块路径变更（如 `timm.models.vision_transformer` → `timm.models._vision_transformer`）→ 更新 import 路径
3. **模型精度异常**：检查 patch 是否正确应用 → 对比有无 ToMe 的模型输出

所有修复记录在 `docs/compatibility-notes.md` 中，方便后续参考。

---

## 8. 验收标准

- [ ] `pip install -e .` 在服务器上成功执行
- [ ] `python scripts/demo.py` 无错误运行，输出单张图片的推理结果
- [ ] `python scripts/benchmark.py --r 0 8 16 24 32 ...` 无错误运行，输出速度/精度 CSV
- [ ] benchmark 结果趋势与论文 Table 1 一致（r 越大速度越快，精度轻微下降）
- [ ] `python scripts/visualize.py` 输出 token 合并可视化 PNG 图片
