#!/usr/bin/env python3
"""
整理 ImageNet 验证集：将 flat 目录中的图片按 synset 子文件夹分类。

使用方法：
    python scripts/organize_imagenet_val.py
    python scripts/organize_imagenet_val.py --val-dir ./Dataset/imagenet/val

说明：
    原始解压后的图片全部放在 val/ 下（flat 结构），
    torchvision.datasets.ImageFolder 需要 val/n01440764/*.JPEG 这样的结构。
    本脚本将图片移动到对应的 synset 子文件夹中。
    脚本可重复运行：已整理过的图片不会重复移动。

标签来源：
    ILSVRC2012 验证集官方标签（50000 行，每行一个 synset ID）
    主 URL：TensorFlow Models 仓库（GitHub raw）
    备用 URL：本地缓存或手动放置
"""
import os
import shutil
import urllib.request
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────────
VAL_DIR = Path(__file__).parent.parent / "Dataset" / "imagenet" / "val"
LABELS_FILE = VAL_DIR.parent / "imagenet_2012_validation_synset_labels.txt"

# 标签文件下载地址（50000 行，每行对应一张图片的 synset ID，按文件名排序）
LABELS_URLS = [
    # TensorFlow Models（主）
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/slim/datasets/imagenet_2012_validation_synset_labels.txt",
    # CLASSY Vision 备用
    "https://raw.githubusercontent.com/facebookresearch/ClassyVision/main/"
    "classy_vision/dataset/classy_imagenet_dataset.py",
]


def download_labels(save_path: Path) -> bool:
    """尝试从多个 URL 下载标签文件，成功返回 True。"""
    for url in LABELS_URLS:
        try:
            print(f"  Downloading labels from:\n    {url}")
            urllib.request.urlretrieve(url, save_path)
            # 验证：应该有 50000 行
            with open(save_path) as f:
                lines = [l.strip() for l in f if l.strip().startswith('n')]
            if len(lines) >= 50000:
                print(f"  OK: {len(lines)} labels downloaded.")
                return True
            else:
                print(f"  Warning: only {len(lines)} lines, retrying next URL...")
                save_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"  Failed: {e}")
    return False


def load_labels(labels_file: Path):
    """从文件加载标签列表（50000 个 synset ID，按图片编号顺序）。"""
    with open(labels_file) as f:
        labels = [line.strip() for line in f if line.strip()]
    # 只保留 synset ID（形如 n01440764）
    labels = [l for l in labels if l.startswith('n') and len(l) == 9]
    if len(labels) != 50000:
        raise ValueError(
            f"标签文件应有 50000 行，实际读取到 {len(labels)} 行。\n"
            f"请检查文件：{labels_file}"
        )
    return labels


def organize(val_dir: Path, labels: list):
    """将 flat 目录中的图片移动到 synset 子文件夹。"""
    # 找到所有还在根目录的图片（已整理的在子文件夹里，不会被匹配）
    flat_images = sorted(val_dir.glob("ILSVRC2012_val_*.JPEG"))
    total = len(flat_images)

    if total == 0:
        print("没有找到需要整理的图片（可能已全部整理完成）。")
        return

    print(f"找到 {total} 张图片待整理，开始移动...")
    moved = 0
    skipped = 0

    for img_path in flat_images:
        # 从文件名提取图片编号（1-based）
        # ILSVRC2012_val_00000079.JPEG → 79
        try:
            img_num = int(img_path.stem.split("_")[-1])
        except ValueError:
            print(f"  [跳过] 无法解析文件名：{img_path.name}")
            skipped += 1
            continue

        if img_num < 1 or img_num > 50000:
            print(f"  [跳过] 图片编号超出范围：{img_num}")
            skipped += 1
            continue

        synset = labels[img_num - 1]  # 转为 0-based 索引
        target_dir = val_dir / synset
        target_dir.mkdir(exist_ok=True)
        target_path = target_dir / img_path.name

        if target_path.exists():
            skipped += 1
            continue

        shutil.move(str(img_path), str(target_path))
        moved += 1

        if moved % 2000 == 0:
            print(f"  进度：{moved}/{total} 已移动...")

    print(f"\n完成！")
    print(f"  移动：{moved} 张")
    print(f"  跳过：{skipped} 张")
    synset_dirs = [d for d in val_dir.iterdir() if d.is_dir()]
    print(f"  synset 文件夹数：{len(synset_dirs)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="整理 ImageNet 验证集目录结构")
    parser.add_argument(
        "--val-dir",
        default=str(VAL_DIR),
        help=f"val 目录路径（默认：{VAL_DIR}）"
    )
    parser.add_argument(
        "--labels",
        default=str(LABELS_FILE),
        help="标签文件路径（默认自动下载）"
    )
    args = parser.parse_args()

    val_dir = Path(args.val_dir)
    labels_file = Path(args.labels)

    print(f"Val 目录：{val_dir}")

    if not val_dir.exists():
        print(f"错误：目录不存在：{val_dir}")
        return

    # 下载标签文件
    if not labels_file.exists():
        print("标签文件不存在，尝试下载...")
        if not download_labels(labels_file):
            print(
                "\n自动下载失败。请手动下载标签文件：\n"
                "  URL: https://raw.githubusercontent.com/tensorflow/models/master/"
                "research/slim/datasets/imagenet_2012_validation_synset_labels.txt\n"
                f"  保存到：{labels_file}"
            )
            return
    else:
        print(f"使用已有标签文件：{labels_file}")

    labels = load_labels(labels_file)
    organize(val_dir, labels)


if __name__ == "__main__":
    main()
