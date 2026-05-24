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
