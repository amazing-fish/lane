"""
sample_keyframes.py - 从解码帧序列中按规则采样关键帧候选

采样策略: 中间区域高密度、首尾低密度
用法:
    python sample_keyframes.py --config config.yaml
    python sample_keyframes.py --frames_dir ./data/frames --output_dir ./data/keyframes
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np
import yaml


def density_sampling(total_frames, num_keyframes, center_ratio=0.6,
                     center_density=3, min_interval=5):
    """
    密度采样：中间区域密、首尾疏。

    将帧序列分为三段：头部、中间、尾部。
    中间区域分配 center_density 倍于边缘的采样密度。
    """
    if total_frames <= num_keyframes:
        return list(range(total_frames))

    center_start = int(total_frames * (1 - center_ratio) / 2)
    center_end = int(total_frames * (1 + center_ratio) / 2)
    edge_len = center_start + (total_frames - center_end)
    center_len = center_end - center_start

    # 按密度比分配采样数
    total_weight = center_len * center_density + edge_len
    center_count = max(1, int(num_keyframes * center_len * center_density / total_weight))
    edge_count = max(1, num_keyframes - center_count)
    head_count = max(1, edge_count // 2)
    tail_count = max(1, edge_count - head_count)

    # 各区域均匀采样
    head_indices = np.linspace(0, center_start - 1, head_count, dtype=int).tolist() \
        if center_start > 0 else []
    center_indices = np.linspace(center_start, center_end - 1, center_count, dtype=int).tolist()
    tail_indices = np.linspace(center_end, total_frames - 1, tail_count, dtype=int).tolist() \
        if center_end < total_frames else []

    indices = sorted(set(head_indices + center_indices + tail_indices))

    # 去除间隔过小的帧
    if min_interval > 1 and len(indices) > 1:
        filtered = [indices[0]]
        for idx in indices[1:]:
            if idx - filtered[-1] >= min_interval:
                filtered.append(idx)
        indices = filtered

    return indices


def sample_clip(clip_dir, output_dir, cfg):
    """对单个 clip 目录进行关键帧采样."""
    index_path = os.path.join(clip_dir, "timestamp_index.csv")
    if not os.path.exists(index_path):
        print(f"  [SKIP] 无时间戳索引: {clip_dir}")
        return []

    # 读取帧索引
    with open(index_path, "r") as f:
        reader = csv.DictReader(f)
        frames = list(reader)

    total = len(frames)
    if total == 0:
        return []

    scfg = cfg["sampling"]
    num_kf = scfg["total_keyframes_per_clip"]
    center_ratio = scfg["center_ratio"]
    center_density = scfg["center_density"]
    min_interval = scfg["min_interval_frames"]

    selected_indices = density_sampling(total, num_kf, center_ratio,
                                        center_density, min_interval)

    clip_name = Path(clip_dir).name
    clip_kf_dir = os.path.join(output_dir, clip_name)
    os.makedirs(clip_kf_dir, exist_ok=True)

    keyframe_rows = []
    for idx in selected_indices:
        frame = frames[idx]
        src = os.path.join(clip_dir, frame["filename"])
        if not os.path.exists(src):
            continue
        dst = os.path.join(clip_kf_dir, frame["filename"])
        shutil.copy2(src, dst)
        keyframe_rows.append({
            "clip": clip_name,
            "frame_idx": frame["frame_idx"],
            "timestamp": frame["timestamp"],
            "filename": frame["filename"],
            "keyframe_path": os.path.relpath(dst, output_dir),
        })

    # 保存该 clip 的关键帧索引
    if keyframe_rows:
        kf_index = os.path.join(clip_kf_dir, "keyframe_index.csv")
        with open(kf_index, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keyframe_rows[0].keys())
            writer.writeheader()
            writer.writerows(keyframe_rows)

    return keyframe_rows


def main():
    parser = argparse.ArgumentParser(description="关键帧采样：中间密、首尾疏")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--frames_dir", default=None, help="解码帧目录（覆盖 config）")
    parser.add_argument("--output_dir", default=None, help="关键帧输出目录（覆盖 config）")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    frames_dir = args.frames_dir or cfg["data"]["output_dir"]
    output_dir = args.output_dir or cfg["data"]["keyframe_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 遍历每个 clip 目录
    clip_dirs = sorted([
        d for d in Path(frames_dir).iterdir()
        if d.is_dir() and (d / "timestamp_index.csv").exists()
    ])

    if not clip_dirs:
        print(f"[ERROR] 在 {frames_dir} 中未找到已解码的 clip 目录")
        return

    print(f"找到 {len(clip_dirs)} 个 clip")
    all_keyframes = []

    for clip_dir in clip_dirs:
        print(f"采样: {clip_dir.name}")
        rows = sample_clip(str(clip_dir), output_dir, cfg)
        all_keyframes.extend(rows)
        print(f"  选取 {len(rows)} 个关键帧")

    # 保存全局关键帧索引
    if all_keyframes:
        global_index = os.path.join(output_dir, "all_keyframes.csv")
        with open(global_index, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keyframes[0].keys())
            writer.writeheader()
            writer.writerows(all_keyframes)
        print(f"\n总计 {len(all_keyframes)} 个关键帧 -> {global_index}")


if __name__ == "__main__":
    main()
