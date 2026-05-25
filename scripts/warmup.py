#!/usr/bin/env python3
"""
预热模型缓存，加速后续推理启动。
在 tmux 窗口中运行，Ctrl+C 停止。

用法：
    python scripts/warmup.py --gpu 4
    python scripts/warmup.py --gpu 4 --mem-gb 28 --compute-size 8192
"""
import argparse
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=4, help="GPU id (default: 4)")
parser.add_argument("--mem-gb", type=float, default=28.0, help="显存占用 GB (default: 28.0)")
parser.add_argument("--compute-size", type=int, default=8192, help="矩阵乘法维度，控制 GPU 利用率 (default: 8192)")
args = parser.parse_args()

device = f"cuda:{args.gpu}"
print(f"Warming up on GPU {args.gpu} ({torch.cuda.get_device_name(args.gpu)})")

# 占显存
n_elements = int(args.mem_gb * 1024**3 / 4)
cache = torch.zeros(n_elements, dtype=torch.float32, device=device)
print(f"Memory allocated: {args.mem_gb:.1f} GB")

# 持续计算，保持 GPU-Util 非零
n = args.compute_size
a = torch.randn(n, n, device=device, dtype=torch.float16)
b = torch.randn(n, n, device=device, dtype=torch.float16)
print(f"Running continuous compute (matrix {n}x{n})... Ctrl+C to stop.")

step = 0
try:
    while True:
        _ = torch.mm(a, b)
        torch.cuda.synchronize(device)
        step += 1
        if step % 100000 == 0:
            print(f"  step {step}")
except KeyboardInterrupt:
    del cache, a, b
    print("Warmup cache released.")
