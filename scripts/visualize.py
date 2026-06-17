#!/usr/bin/env python3
"""
ToMe Visualization: 多 r 值对比图，展示 token 合并边界随 r 增大的变化。

用法（从 ImageNet 验证集随机取图）：
    python scripts/visualize.py \
        --imagenet-val ./Dataset/imagenet/val \
        --r 8 16 \
        --device cuda:4 \
        --output results/vis_compare.png

用法（指定图片）：
    python scripts/visualize.py \
        --image path/to/image.jpg \
        --r 8 16 \
        --output results/vis_compare.png
"""
import argparse
import random
from pathlib import Path

import torch
import timm
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms

import tome
from tome.vis import make_visualization


def load_image(image_path: str, size: int = 224):
    """加载图片，返回预处理 tensor 和用于显示的 PIL Image。"""
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


def pick_random_image(val_dir: str) -> str:
    """从 ImageNet val 目录（synset 子目录结构）随机选一张图片。"""
    val_path = Path(val_dir)
    synsets = [d for d in val_path.iterdir() if d.is_dir()]
    if not synsets:
        raise FileNotFoundError(f"No synset subdirectories found in {val_dir}")
    synset = random.choice(synsets)
    images = list(synset.glob('*.JPEG')) + list(synset.glob('*.jpg')) + list(synset.glob('*.png'))
    if not images:
        raise FileNotFoundError(f"No images found in {synset}")
    return str(random.choice(images))


def run_tome(model_name: str, r: int, x: torch.Tensor, device: str):
    """对单个 r 值运行推理，返回 source 矩阵。"""
    model = timm.create_model(model_name, pretrained=True)
    model = model.to(device).eval()
    tome.patch.timm(model, trace_source=True, prop_attn=False)
    model.r = r
    x = x.to(device)
    with torch.no_grad():
        model(x)
    source = model._tome_info.get('source')
    class_token = model._tome_info.get('class_token', True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return source, class_token


def main():
    parser = argparse.ArgumentParser(description='ToMe Multi-r Visualization')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', help='指定输入图片路径（JPG/PNG）')
    group.add_argument('--imagenet-val', help='ImageNet val 目录，随机取一张图')
    parser.add_argument('--model', default='deit_small_patch16_224')
    parser.add_argument('--r', type=int, nargs='+', default=[8, 16],
                        help='要可视化的 r 值列表（默认: 8 16）')
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--output', default='results/vis_compare.png')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=None, help='随机种子（固定选图）')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # 选图
    if args.imagenet_val:
        image_path = pick_random_image(args.imagenet_val)
        print(f"Selected image: {image_path}")
    else:
        image_path = args.image
        print(f"Using image   : {image_path}")

    x, img_display = load_image(image_path)

    # 生成各 r 值的可视化
    r_values = sorted(set(args.r))
    vis_images = {}
    for r in r_values:
        print(f"Running r={r} ...")
        source, class_token = run_tome(args.model, r, x, args.device)
        if source is None:
            print(f"  WARNING: source is None for r={r}, skipping")
            continue
        vis = make_visualization(img_display, source,
                                 patch_size=args.patch_size,
                                 class_token=class_token)
        vis_images[r] = vis
        remaining = len([v for v in source[0].sum(0) if v > 0])
        total = (224 // args.patch_size) ** 2
        print(f"  Tokens remaining: {remaining}/{total} "
              f"({100*remaining/total:.1f}%)")

    if not vis_images:
        print("No visualizations generated.")
        return

    # 绘图：原图 + 各 r 值
    n_panels = 1 + len(vis_images)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    axes[0].imshow(img_display)
    axes[0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    for i, (r, vis) in enumerate(sorted(vis_images.items())):
        total = (224 // args.patch_size) ** 2
        axes[i + 1].imshow(vis)
        axes[i + 1].set_title(f'ToMe  r={r}', fontsize=14, fontweight='bold')
        axes[i + 1].axis('off')

    fig.suptitle(f'Token Merging Visualization — {args.model}',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved → {args.output}")


if __name__ == '__main__':
    main()
