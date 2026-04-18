"""
prepare_data.py - 整理训练数据：合并 clip 级标签与关键帧标签，生成 train/val manifest

用法:
    python prepare_data.py --config config.yaml
"""

import argparse
import csv
import os
import random
from pathlib import Path

import yaml


DIRECTION_MAP = {"yes": 1, "no": 0, "unknown": -1}
LANE_MAP = {"1": 0, "2": 1, "2+": 2, "unknown": -1}
LEGACY_LANE_TO_NEW = {"3": "2+", "4": "2+", "5": "2+", "6+": "2+"}


def normalize_lane_label(raw):
    lane = str(raw or "unknown").strip().lower()
    lane = LEGACY_LANE_TO_NEW.get(lane, lane)
    return lane if lane in LANE_MAP else "unknown"


def load_clip_labels(clip_label_file):
    """加载 clip 级标签 (由用户提供)."""
    labels = {}
    if not os.path.exists(clip_label_file):
        print(f"[WARN] clip 标签文件不存在: {clip_label_file}")
        return labels
    with open(clip_label_file, "r", newline="") as f:
        for row in csv.DictReader(f):
            clip = row["clip"]
            labels[clip] = {
                "is_bidirectional": DIRECTION_MAP.get(row.get("is_bidirectional", "unknown"), -1),
                "lane_count": LANE_MAP.get(normalize_lane_label(row.get("lane_count", "unknown")), -1),
            }
    return labels


def load_keyframe_labels(kf_label_file):
    """加载关键帧级标签 (标注工具产出)."""
    labels = {}
    if not os.path.exists(kf_label_file):
        print(f"[WARN] 关键帧标签文件不存在: {kf_label_file}")
        return labels
    with open(kf_label_file, "r", newline="") as f:
        for row in csv.DictReader(f):
            path = row["image_path"]
            clip = path.split("/")[0] if "/" in path else path.split("\\")[0]
            if clip not in labels:
                labels[clip] = []
            labels[clip].append({
                "image_path": path,
                "is_bidirectional": DIRECTION_MAP.get(row.get("is_bidirectional", "unknown"), -1),
                "lane_count": LANE_MAP.get(normalize_lane_label(row.get("lane_count", "unknown")), -1),
            })
    return labels


def build_manifest(frames_dir, clip_labels, kf_labels):
    """构建训练 manifest，每行一个 clip."""
    manifest = []
    clips_dir = Path(frames_dir)

    for clip_dir in sorted(clips_dir.iterdir()):
        if not clip_dir.is_dir():
            continue
        clip_name = clip_dir.name
        index_file = clip_dir / "timestamp_index.csv"
        if not index_file.exists():
            continue

        # 统计帧数
        with open(index_file, "r") as f:
            frame_count = sum(1 for _ in csv.DictReader(f))

        # clip 级标签
        cl = clip_labels.get(clip_name, {"is_bidirectional": -1, "lane_count": -1})

        # 关键帧标签（取众数作为补充验证）
        kf = kf_labels.get(clip_name, [])
        has_keyframe_labels = len(kf) > 0

        manifest.append({
            "clip": clip_name,
            "clip_dir": str(clip_dir),
            "frame_count": frame_count,
            "is_bidirectional": cl["is_bidirectional"],
            "lane_count": cl["lane_count"],
            "keyframe_label_count": len(kf),
        })

    return manifest


def split_train_val(manifest, val_ratio, seed=42):
    """按比例划分 train/val，保证每个类别都有样本."""
    random.seed(seed)
    # 过滤掉无标签样本
    valid = [m for m in manifest if m["is_bidirectional"] >= 0 or m["lane_count"] >= 0]
    invalid = [m for m in manifest if m["is_bidirectional"] < 0 and m["lane_count"] < 0]

    if invalid:
        print(f"[INFO] {len(invalid)} 个 clip 无有效标签，跳过")

    random.shuffle(valid)
    val_size = max(1, int(len(valid) * val_ratio))
    val_set = valid[:val_size]
    train_set = valid[val_size:]

    return train_set, val_set


def save_manifest(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="生成训练/验证 manifest")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    clip_labels = load_clip_labels(cfg["data"]["clip_label_file"])
    kf_labels = load_keyframe_labels(cfg["data"]["keyframe_label_file"])

    manifest = build_manifest(cfg["data"]["output_dir"], clip_labels, kf_labels)
    print(f"共 {len(manifest)} 个 clip")

    train_set, val_set = split_train_val(manifest, cfg["data"]["val_ratio"],
                                          cfg["training"].get("seed", 42))
    print(f"训练集: {len(train_set)}, 验证集: {len(val_set)}")

    save_manifest(train_set, cfg["data"]["train_manifest"])
    save_manifest(val_set, cfg["data"]["val_manifest"])
    print("manifest 已保存")


if __name__ == "__main__":
    main()
