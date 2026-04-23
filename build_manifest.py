"""
build_manifest.py - 从 segment_labels.csv 生成 train/val manifest（segment 级）

用法:
  python build_manifest.py --config config.yaml
"""

import argparse
import csv
import os
import random
from collections import defaultdict

import yaml

DIRECTION_MAP = {"yes": 1, "no": 0, "unknown": -1}
LANE_MAP = {"1": 0, "2": 1, "2+": 2, "unknown": -1}
LEGACY_LANE_TO_NEW = {"3": "2+", "4": "2+", "5": "2+", "6+": "2+"}


def safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_segment_row(row):
    segment_type = str(row.get("segment_type", "unknown") or "unknown").strip().lower()
    quality = str(row.get("quality", "unknown") or "unknown").strip().lower()
    direction = str(row.get("is_bidirectional", "unknown") or "unknown").strip().lower()
    lane = str(row.get("lane_count", "unknown") or "unknown").strip().lower()
    lane = LEGACY_LANE_TO_NEW.get(lane, lane)

    if segment_type != "slope":
        direction = "unknown"
        lane = "unknown"

    return {
        "segment_id": str(row.get("segment_id", "") or "").strip(),
        "clip_id": str(row.get("clip_id", "") or "").strip(),
        "clip_dir": str(row.get("clip_dir", "") or "").strip(),
        "start_frame": safe_int(row.get("start_frame"), 0),
        "end_frame": safe_int(row.get("end_frame"), -1),
        "segment_type": segment_type,
        "is_bidirectional": direction if direction in DIRECTION_MAP else "unknown",
        "lane_count": lane if lane in LANE_MAP else "unknown",
        "quality": quality,
    }


def load_segment_labels(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            rows.append(normalize_segment_row(raw))
    return rows


def filter_segments(rows, include_review_segments=False):
    filtered = []
    dropped = defaultdict(int)

    for row in rows:
        if row["segment_type"] != "slope":
            dropped["non_slope_type"] += 1
            continue
        if row["quality"] == "bad":
            dropped["quality_bad"] += 1
            continue
        if (not include_review_segments) and row["quality"] in {"review", "need_review"}:
            dropped["quality_review"] += 1
            continue
        if row["end_frame"] < row["start_frame"]:
            dropped["invalid_range"] += 1
            continue
        if not row["clip_dir"] or not row["clip_id"]:
            dropped["missing_clip"] += 1
            continue
        filtered.append(row)

    return filtered, dropped


def split_by_clip_id(rows, val_ratio, seed):
    clip_groups = defaultdict(list)
    for row in rows:
        clip_groups[row["clip_id"]].append(row)

    clips = list(clip_groups.keys())
    if len(clips) <= 1:
        return rows[:], []
    random.Random(seed).shuffle(clips)

    val_clip_count = max(1, int(len(clips) * val_ratio)) if clips else 0
    if val_clip_count >= len(clips):
        val_clip_count = len(clips) - 1
    val_clips = set(clips[:val_clip_count])

    train_rows, val_rows = [], []
    for clip_id in clips:
        if clip_id in val_clips:
            val_rows.extend(clip_groups[clip_id])
        else:
            train_rows.extend(clip_groups[clip_id])

    return train_rows, val_rows


def to_manifest_rows(rows):
    out = []
    for row in rows:
        frame_count = row["end_frame"] - row["start_frame"] + 1
        out.append({
            "sample_id": row["segment_id"],
            "clip": row["clip_id"],
            "clip_dir": row["clip_dir"],
            "start_frame": row["start_frame"],
            "end_frame": row["end_frame"],
            "frame_count": frame_count,
            "is_bidirectional": DIRECTION_MAP[row["is_bidirectional"]],
            "lane_count": LANE_MAP[row["lane_count"]],
        })
    return out


def save_manifest(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_id", "clip", "clip_dir", "start_frame", "end_frame", "frame_count", "is_bidirectional", "lane_count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="从 segment_labels.csv 生成 train/val manifest")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg["data"]
    path = dcfg["segment_label_file"]
    rows = load_segment_labels(path)
    print(f"[INFO] 读取 segment 标签: {len(rows)}")

    filtered, dropped = filter_segments(rows, dcfg.get("include_review_segments", False))
    print(f"[INFO] 过滤后 segment: {len(filtered)}")
    if dropped:
        print("[INFO] 丢弃统计:", dict(dropped))

    if dcfg.get("split_by_clip", True):
        train_rows, val_rows = split_by_clip_id(filtered, dcfg.get("val_ratio", 0.2), cfg["training"].get("seed", 42))
    else:
        all_rows = filtered[:]
        random.Random(cfg["training"].get("seed", 42)).shuffle(all_rows)
        if len(all_rows) <= 1:
            train_rows, val_rows = all_rows, []
        else:
            val_size = max(1, int(len(all_rows) * dcfg.get("val_ratio", 0.2))) if all_rows else 0
            if val_size >= len(all_rows):
                val_size = len(all_rows) - 1
            val_rows = all_rows[:val_size]
            train_rows = all_rows[val_size:]

    train_manifest = to_manifest_rows(train_rows)
    val_manifest = to_manifest_rows(val_rows)

    save_manifest(dcfg["train_manifest"], train_manifest)
    save_manifest(dcfg["val_manifest"], val_manifest)

    print(f"[INFO] 训练集: {len(train_manifest)}, 验证集: {len(val_manifest)}")
    print(f"[INFO] train_manifest -> {dcfg['train_manifest']}")
    print(f"[INFO] val_manifest -> {dcfg['val_manifest']}")


if __name__ == "__main__":
    main()
