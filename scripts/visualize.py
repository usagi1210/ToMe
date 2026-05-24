#!/usr/bin/env python3
"""
ToMe Visualization: 可视化哪些 token 被合并，输出对比 PNG。
使用 ToMe 内置的 make_visualization 函数（tome/vis.py）。

用法：
    python scripts/visualize.py --image path/to/image.jpg
    python scripts/visualize.py --image path/to/image.jpg --r 8 --output results/vis_r8.png
    python scripts/visualize.py --image path/to/image.jpg --r 0  # r=0 时 source=None，会提示错误
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
