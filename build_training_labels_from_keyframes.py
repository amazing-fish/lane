"""
build_training_labels_from_keyframes.py

从 clip_labels.csv + keyframe_labels.csv 自动推断坡道 segment，并生成：
- auto_segments.csv（审计/复核）
- train_manifest.csv / val_manifest.csv（训练直接消费）
"""

import argparse
import csv
import os
import random
from collections import defaultdict
from pathlib import Path

import yaml

DIRECTION_MAP = {"yes": 1, "no": 0, "unknown": -1}
LANE_MAP = {"1": 0, "2": 1, "2+": 2, "unknown": -1}
VALID_SCOPE = {"slope", "non_slope", "unknown", "transition"}
VALID_QUALITY = {"ok", "review", "bad", "unknown"}
LEGACY_LANE_TO_NEW = {"3": "2+", "4": "2+", "5": "2+", "6+": "2+"}


def safe_int(v, default=-1):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def normalize_lane(v):
    lane = str(v or "unknown").strip().lower()
    lane = LEGACY_LANE_TO_NEW.get(lane, lane)
    return lane if lane in LANE_MAP else "unknown"


def load_clip_labels(path):
    labels = {}
    if not os.path.exists(path):
        return labels
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clip_id = str(row.get("clip_id", "")).strip()
            if not clip_id:
                continue
            quality = str(row.get("quality", "ok") or "ok").strip().lower()
            labels[clip_id] = {
                "is_bidirectional": str(row.get("is_bidirectional", "unknown") or "unknown").strip().lower(),
                "lane_count": normalize_lane(row.get("lane_count", "unknown")),
                "quality": quality if quality in VALID_QUALITY else "unknown",
                "notes": str(row.get("notes", "") or "").strip(),
            }
    return labels


def load_keyframe_labels(path):
    grouped = defaultdict(list)
    if not os.path.exists(path):
        return grouped
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clip_id = str(row.get("clip_id", "")).strip()
            if not clip_id:
                continue
            frame_idx = safe_int(row.get("frame_idx"), -1)
            if frame_idx < 0:
                continue
            scope = str(row.get("frame_scope", "unknown") or "unknown").strip().lower()
            grouped[clip_id].append({
                "image_path": row.get("image_path", ""),
                "frame_idx": frame_idx,
                "frame_scope": scope if scope in VALID_SCOPE else "unknown",
            })
    scope_priority = {"slope": 3, "non_slope": 2, "transition": 1, "unknown": 0}
    for clip_id in grouped:
        grouped[clip_id].sort(key=lambda x: x["frame_idx"])
        deduped = {}
        for item in grouped[clip_id]:
            idx = item["frame_idx"]
            if idx not in deduped:
                deduped[idx] = item
                continue
            prev = deduped[idx]
            if scope_priority[item["frame_scope"]] >= scope_priority[prev["frame_scope"]]:
                deduped[idx] = item
        grouped[clip_id] = [deduped[k] for k in sorted(deduped.keys())]
    return grouped


def clip_max_frame(frames_root, clip_id, keyframes=None):
    clip_dir = Path(frames_root) / clip_id
    idx_csv = clip_dir / "timestamp_index.csv"
    if idx_csv.exists():
        max_idx = -1
        with idx_csv.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                max_idx = max(max_idx, safe_int(row.get("frame_idx"), -1))
        if max_idx >= 0:
            return max_idx

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted([p for p in clip_dir.iterdir() if p.suffix.lower() in exts]) if clip_dir.exists() else []
    if files:
        return len(files) - 1

    if keyframes:
        return max((k.get("frame_idx", -1) for k in keyframes), default=-1)
    return -1


def infer_segments_for_clip(clip_id, keyframes, clip_label, max_frame):
    anchors = [k for k in keyframes if k["frame_scope"] in {"slope", "non_slope"}]
    if not anchors or max_frame < 0:
        return []

    unknown_frames = {k["frame_idx"] for k in keyframes if k["frame_scope"] == "unknown"}
    transition_frames = {k["frame_idx"] for k in keyframes if k["frame_scope"] == "transition"}

    segments = []
    for i, anchor in enumerate(anchors):
        prev_anchor = anchors[i - 1] if i > 0 else None
        next_anchor = anchors[i + 1] if i < len(anchors) - 1 else None

        start = 0 if prev_anchor is None else (prev_anchor["frame_idx"] + anchor["frame_idx"]) // 2 + 1
        end = max_frame if next_anchor is None else (anchor["frame_idx"] + next_anchor["frame_idx"]) // 2

        if end < start:
            continue
        if anchor["frame_scope"] != "slope":
            continue

        quality = "ok"
        reasons = []
        if prev_anchor is None or next_anchor is None:
            quality = "review"
            reasons.append("edge_anchor")

        inspect_start = prev_anchor["frame_idx"] + 1 if prev_anchor else start
        inspect_end = next_anchor["frame_idx"] - 1 if next_anchor else end
        if inspect_start <= inspect_end:
            has_unknown = any(inspect_start <= x <= inspect_end for x in unknown_frames)
            has_transition = any(inspect_start <= x <= inspect_end for x in transition_frames)
            if has_unknown:
                quality = "review"
                reasons.append("unknown_between_anchors")
            if has_transition:
                quality = "review"
                reasons.append("transition_between_anchors")

        if clip_label["quality"] in {"review", "bad"}:
            quality = clip_label["quality"]
            reasons.append(f"clip_quality_{clip_label['quality']}")
        if clip_label.get("notes"):
            reasons.append(f"clip_notes:{clip_label['notes']}")

        segments.append({
            "clip_id": clip_id,
            "clip_dir": str(Path("./data/frames") / clip_id),
            "start_frame": start,
            "end_frame": end,
            "segment_type": "slope",
            "is_bidirectional": clip_label["is_bidirectional"],
            "lane_count": clip_label["lane_count"],
            "quality": quality,
            "notes": ";".join(reasons),
        })

    if not segments:
        return []

    merged = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg["start_frame"] <= last["end_frame"] + 1 and seg["is_bidirectional"] == last["is_bidirectional"] and seg["lane_count"] == last["lane_count"]:
            last["end_frame"] = max(last["end_frame"], seg["end_frame"])
            if last["quality"] != "bad":
                if "review" in {last["quality"], seg["quality"]}:
                    last["quality"] = "review"
            if seg["notes"]:
                last["notes"] = ";".join([x for x in [last["notes"], seg["notes"]] if x])
        else:
            merged.append(seg)
    return merged


def split_by_clip_id(rows, val_ratio, seed):
    clip_groups = defaultdict(list)
    for row in rows:
        clip_groups[row["clip_id"]].append(row)

    clips = list(clip_groups.keys())
    random.Random(seed).shuffle(clips)
    val_clip_count = max(1, int(len(clips) * val_ratio)) if clips else 0
    val_clips = set(clips[:val_clip_count])

    train_rows, val_rows = [], []
    for clip_id in clips:
        if clip_id in val_clips:
            val_rows.extend(clip_groups[clip_id])
        else:
            train_rows.extend(clip_groups[clip_id])
    return train_rows, val_rows


def to_manifest(rows):
    out = []
    for r in rows:
        out.append({
            "sample_id": r["segment_id"],
            "clip": r["clip_id"],
            "clip_dir": r["clip_dir"],
            "start_frame": r["start_frame"],
            "end_frame": r["end_frame"],
            "frame_count": r["end_frame"] - r["start_frame"] + 1,
            "is_bidirectional": DIRECTION_MAP.get(r["is_bidirectional"], -1),
            "lane_count": LANE_MAP.get(r["lane_count"], -1),
        })
    return out


def save_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="从关键帧标签自动构建训练标签")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg["data"]
    clip_labels = load_clip_labels(dcfg["clip_label_file"])
    keyframes = load_keyframe_labels(dcfg["keyframe_label_file"])
    if not clip_labels:
        print(f"[WARN] clip labels not found or empty: {dcfg['clip_label_file']}")
    if not keyframes:
        print(f"[WARN] keyframe labels not found or empty: {dcfg['keyframe_label_file']}")

    segments = []
    all_clips = sorted(set(keyframes.keys()) | set(clip_labels.keys()))
    for clip_id in all_clips:
        clip_label = clip_labels.get(clip_id, {
            "is_bidirectional": "unknown",
            "lane_count": "unknown",
            "quality": "review",
            "notes": "missing_clip_label",
        })
        clip_keyframes = keyframes.get(clip_id, [])
        if not clip_keyframes:
            continue
        max_frame = clip_max_frame(dcfg["output_dir"], clip_id, clip_keyframes)
        inferred = infer_segments_for_clip(clip_id, clip_keyframes, clip_label, max_frame)
        for i, seg in enumerate(inferred, start=1):
            seg["segment_id"] = f"{clip_id}_auto_{i:03d}"
            seg["clip_dir"] = str(Path(dcfg["output_dir"]) / clip_id)
        segments.extend(inferred)

    auto_segment_file = dcfg.get("auto_segment_file", "./data/auto_segments.csv")
    save_csv(auto_segment_file, segments, [
        "segment_id", "clip_id", "clip_dir", "start_frame", "end_frame",
        "segment_type", "is_bidirectional", "lane_count", "quality", "notes"
    ])

    include_review = dcfg.get("include_review_segments", False)
    filtered = [
        s for s in segments
        if s["segment_type"] == "slope" and s["quality"] != "bad" and (include_review or s["quality"] != "review")
    ]

    if dcfg.get("split_by_clip", True):
        train_rows, val_rows = split_by_clip_id(filtered, dcfg.get("val_ratio", 0.2), cfg["training"].get("seed", 42))
    else:
        all_rows = filtered[:]
        random.Random(cfg["training"].get("seed", 42)).shuffle(all_rows)
        val_size = max(1, int(len(all_rows) * dcfg.get("val_ratio", 0.2))) if all_rows else 0
        val_rows = all_rows[:val_size]
        train_rows = all_rows[val_size:]

    save_csv(dcfg["train_manifest"], to_manifest(train_rows), [
        "sample_id", "clip", "clip_dir", "start_frame", "end_frame", "frame_count", "is_bidirectional", "lane_count"
    ])
    save_csv(dcfg["val_manifest"], to_manifest(val_rows), [
        "sample_id", "clip", "clip_dir", "start_frame", "end_frame", "frame_count", "is_bidirectional", "lane_count"
    ])

    print(f"[INFO] auto_segments -> {auto_segment_file}: {len(segments)}")
    print(f"[INFO] train_manifest -> {dcfg['train_manifest']}: {len(train_rows)}")
    print(f"[INFO] val_manifest -> {dcfg['val_manifest']}: {len(val_rows)}")


if __name__ == "__main__":
    main()
