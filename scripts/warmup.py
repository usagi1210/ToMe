#!/usr/bin/env python3
"""
预热模型缓存，加速后续推理启动。
在服务器后台运行，Ctrl+C 停止。

用法：
    # 前台运行
    python scripts/warmup.py --gpu 4

    # 后台运行（推荐，关闭终端不会停止）
    nohup python scripts/warmup.py --gpu 4 > /tmp/warmup.log 2>&1 &
    echo "PID: $!"

    # 停止
    kill <PID>
"""
import argparse
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=4, help="GPU id (default: 4)")
parser.add_argument("--mem-gb", type=float, default=4.0, help="预加载缓存大小 GB (default: 4.0)")
args = parser.parse_args()

device = f"cuda:{args.gpu}"
print(f"Warming up on GPU {args.gpu} ({torch.cuda.get_device_name(args.gpu)})")

n_elements = int(args.mem_gb * 1024**3 / 4)
cache = torch.zeros(n_elements, dtype=torch.float32, device=device)
print(f"Cache ready: {args.mem_gb:.1f} GB on {device}. Waiting for main job...")

try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    del cache
    print("Warmup cache released.")
