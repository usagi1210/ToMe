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
