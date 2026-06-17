#!/usr/bin/env python3
"""
ToMe Visualization: 多图/多模型对比，展示 token 合并边界随 r 增大的变化。

模式 1 — 多图单模型（默认）:
    python scripts/visualize.py \
        --imagenet-val ./Dataset/imagenet/val \
        --num-images 5 --model deit_small_patch16_224 \
        --r 8 16 --device cuda:6 \
        --output results/vis_multi_image.png

模式 2 — 单图多模型:
    python scripts/visualize.py \
        --imagenet-val ./Dataset/imagenet/val \
        --models deit_tiny_patch16_224 deit_small_patch16_224 \
                 deit_base_patch16_224 vit_base_patch16_224 \
        --r 8 16 --device cuda:6 \
        --output results/vis_multi_model.png
"""
import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import timm
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms

import tome
from tome.vis import make_visualization


# ---------------------------------------------------------------------------
# 图像加载
# ---------------------------------------------------------------------------

def load_image(path: str, size: int = 224):
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_pil = Image.open(path).convert('RGB')
    img_display = img_pil.resize((size, size), Image.BICUBIC)
    return transform(img_pil).unsqueeze(0), img_display


def pick_images(val_dir: str, n: int, seed: int = 42) -> list[str]:
    """从不同 synset 各选一张图，保证内容多样性。"""
    random.seed(seed)
    synsets = [d for d in Path(val_dir).iterdir() if d.is_dir()]
    if not synsets:
        raise FileNotFoundError(f"No synset directories in {val_dir}")
    chosen = random.sample(synsets, min(n, len(synsets)))
    paths = []
    for s in chosen:
        imgs = sorted(s.glob('*.JPEG')) + sorted(s.glob('*.jpg')) + sorted(s.glob('*.png'))
        if imgs:
            paths.append(str(random.choice(imgs)))
    return paths[:n]


# ---------------------------------------------------------------------------
# 可视化核心
# ---------------------------------------------------------------------------

def safe_visualize(img_display, source, patch_size: int, class_token: bool):
    """调用官方 make_visualization，处理除零 NaN，降级回原图色。"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        vis = make_visualization(img_display, source,
                                 patch_size=patch_size, class_token=class_token)
    vis_arr = np.array(vis, dtype=np.float32)
    orig_arr = np.array(img_display, dtype=np.float32)
    nan_mask = np.isnan(vis_arr)
    if nan_mask.any():
        vis_arr[nan_mask] = orig_arr[nan_mask]
    return Image.fromarray(np.clip(vis_arr, 0, 255).astype(np.uint8))


def run_model(model_name: str, r_values: list[int],
              images_data: list, device: str, patch_size: int = 16) -> dict:
    """
    对一个模型跑所有 r 值和所有图片。
    返回 {r: [vis_img_0, vis_img_1, ...]}
    """
    results = {r: [] for r in r_values}

    for r in r_values:
        print(f"  r={r} ...", end=' ', flush=True)
        model = timm.create_model(model_name, pretrained=True)
        model = model.to(device).eval()
        tome.patch.timm(model, trace_source=True, prop_attn=False)
        model.r = r

        for x, img_display in images_data:
            with torch.no_grad():
                model(x.to(device))
            source = model._tome_info.get('source')
            class_token = model._tome_info.get('class_token', True)
            if source is not None:
                vis = safe_visualize(img_display, source, patch_size, class_token)
                remaining = source[0].shape[0]
                total = (224 // patch_size) ** 2 + 1
            else:
                vis = img_display
                remaining = total = (224 // patch_size) ** 2 + 1
            results[r].append((vis, remaining, total))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        counts = [f"{res[1]}/{res[2]}" for res in results[r]]
        print(f"tokens remaining: {', '.join(counts)}")

    # 只返回可视化图片
    return {r: [item[0] for item in results[r]] for r in r_values}


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def build_grid(orig_images, vis_by_r, r_values: list[int],
               row_labels: list[str], title: str, output: str):
    """
    生成对比大图。
    行 = 图片（或模型），列 = Original | r=r1 | r=r2 | ...
    """
    n_rows = len(orig_images)
    n_cols = 1 + len(r_values)
    fig_w = 3.8 * n_cols
    fig_h = 3.8 * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             gridspec_kw={'hspace': 0.05, 'wspace': 0.05})

    # 保证 axes 总是 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    col_titles = ['Original'] + [f'r = {r}' for r in r_values]
    for j, ctitle in enumerate(col_titles):
        axes[0, j].set_title(ctitle, fontsize=13, fontweight='bold', pad=6)

    for i, (orig, label) in enumerate(zip(orig_images, row_labels)):
        axes[i, 0].imshow(orig)
        axes[i, 0].set_ylabel(label, fontsize=10, rotation=0,
                              labelpad=72, va='center')
        for j in range(n_cols):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            for spine in axes[i, j].spines.values():
                spine.set_visible(False)
        for j, r in enumerate(r_values):
            axes[i, j + 1].imshow(vis_by_r[r][i])

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved → {output}")


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='ToMe Visualization')

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--imagenet-val', metavar='DIR',
                     help='ImageNet val 目录（synset 子目录结构）')
    src.add_argument('--images', nargs='+', metavar='IMG',
                     help='直接指定图片路径列表')

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--model', default=None,
                      help='单模型名称（多图模式）')
    mode.add_argument('--models', nargs='+', default=None,
                      help='多模型名称列表（单图多模型模式）')

    parser.add_argument('--num-images', type=int, default=5,
                        help='多图模式下随机选取的图片数（默认 5）')
    parser.add_argument('--r', type=int, nargs='+', default=[8, 16],
                        help='r 值列表（默认: 8 16）')
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--output', default='results/vis_compare.png')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    r_values = sorted(set(args.r))
    multi_model_mode = bool(args.models)
    model_list = args.models if multi_model_mode else [args.model or 'deit_small_patch16_224']

    # ---- 选图 ----
    if multi_model_mode:
        n_needed = 1
    else:
        n_needed = args.num_images

    if args.imagenet_val:
        image_paths = pick_images(args.imagenet_val, n_needed, args.seed)
    else:
        image_paths = args.images[:n_needed]

    print(f"Images ({len(image_paths)}):")
    images_data = []
    for p in image_paths:
        print(f"  {p}")
        x, img_display = load_image(p)
        images_data.append((x, img_display))

    # ---- 推理 ----
    if multi_model_mode:
        # 单图多模型：对每个模型跑一遍
        vis_by_r = {r: [] for r in r_values}
        for m in model_list:
            print(f"\n[{m}]")
            result = run_model(m, r_values, images_data, args.device, args.patch_size)
            for r in r_values:
                vis_by_r[r].append(result[r][0])

        short_names = []
        for m in model_list:
            name = m.replace('_patch16_224', '')
            name = name.replace('deit_', 'DeiT-').replace('vit_', 'ViT-')
            short_names.append(name)

        orig_images = [images_data[0][1]] * len(model_list)
        title = 'ToMe Token Merging — Multi-Model Comparison'
        row_labels = short_names

    else:
        # 多图单模型
        model_name = model_list[0]
        print(f"\n[{model_name}]")
        vis_by_r = run_model(model_name, r_values, images_data, args.device, args.patch_size)

        orig_images = [d[1] for d in images_data]
        row_labels = [f'Image {i+1}' for i in range(len(images_data))]
        short = model_name.replace('_patch16_224', '').replace('deit_', 'DeiT-').replace('vit_', 'ViT-')
        title = f'ToMe Token Merging — {short}'

    # ---- 绘图 ----
    build_grid(orig_images, vis_by_r, r_values, row_labels, title, args.output)


if __name__ == '__main__':
    main()
