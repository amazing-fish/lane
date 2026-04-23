"""annotate_web.py - 轻量本地 Web 关键帧标注工具（clip+keyframe 双真值输出）"""

import argparse
import csv
import os
import re
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

VALID_SCOPE = {"slope", "transition", "non_slope", "unknown"}
VALID_DIR = {"yes", "no", "unknown"}
VALID_LANE = {"1", "2", "2+", "unknown"}
VALID_QUALITY = {"ok", "review", "bad", "unknown"}
LEGACY_LANE_TO_NEW = {"3": "2+", "4": "2+", "5": "2+", "6+": "2+"}


def collect_images(keyframe_dir):
    return sorted([
        str(p.relative_to(keyframe_dir))
        for p in Path(keyframe_dir).rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp") and p.name != "timestamp_index.csv"
    ])


def extract_meta(rel_path):
    norm = rel_path.replace("\\", "/")
    parts = norm.split("/")
    clip_id = parts[0] if parts else "unknown"
    stem = Path(rel_path).stem
    found = re.findall(r"\d+", stem)
    frame_idx = int(found[-1]) if found else -1
    return clip_id, frame_idx


def normalize_scope(v):
    scope = str(v or "unknown").strip().lower()
    return scope if scope in VALID_SCOPE else "unknown"


def normalize_clip_label(v):
    if not isinstance(v, dict):
        v = {}
    direction = str(v.get("is_bidirectional", "unknown") or "unknown").strip().lower()
    lane = str(v.get("lane_count", "unknown") or "unknown").strip().lower()
    lane = LEGACY_LANE_TO_NEW.get(lane, lane)
    quality = str(v.get("quality", "ok") or "ok").strip().lower()
    notes = str(v.get("notes", "") or "").strip()
    if direction not in VALID_DIR:
        direction = "unknown"
    if lane not in VALID_LANE:
        lane = "unknown"
    if quality not in VALID_QUALITY:
        quality = "unknown"
    return {
        "is_bidirectional": direction,
        "lane_count": lane,
        "quality": quality,
        "notes": notes,
    }


def load_keyframe_labels(label_file, images):
    labels = {}
    if not os.path.exists(label_file):
        return labels
    image_set = set(images)
    with open(label_file, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = row.get("image_path", "")
            if rel in image_set:
                labels[rel] = normalize_scope(row.get("frame_scope", "unknown"))
    return labels


def load_clip_labels(clip_label_file, valid_clips=None):
    labels = {}
    if not os.path.exists(clip_label_file):
        return labels
    valid = set(valid_clips or [])
    with open(clip_label_file, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clip_id = str(row.get("clip_id", "")).strip()
            if clip_id and (not valid or clip_id in valid):
                labels[clip_id] = normalize_clip_label(row)
    return labels


def save_keyframe_labels(path, images, keyframe_labels):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "clip_id", "frame_idx", "frame_scope"])
        writer.writeheader()
        for rel in images:
            clip_id, frame_idx = extract_meta(rel)
            writer.writerow({
                "image_path": rel,
                "clip_id": clip_id,
                "frame_idx": frame_idx,
                "frame_scope": normalize_scope(keyframe_labels.get(rel, "unknown")),
            })


def save_clip_labels(path, clip_labels):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "is_bidirectional", "lane_count", "quality", "notes"])
        writer.writeheader()
        for clip_id in sorted(clip_labels.keys()):
            writer.writerow({"clip_id": clip_id, **normalize_clip_label(clip_labels[clip_id])})


def labeled_count(keyframe_labels):
    return sum(1 for v in keyframe_labels.values() if normalize_scope(v) != "unknown")


def create_app(keyframe_dir, keyframe_label_file, clip_label_file):
    app = Flask(__name__, template_folder="templates")

    keyframe_root = Path(keyframe_dir).resolve()
    images = collect_images(keyframe_root)
    if not images:
        raise RuntimeError(f"在 {keyframe_root} 中未找到图片")
    valid_clips = {extract_meta(rel)[0] for rel in images}

    keyframe_labels = load_keyframe_labels(keyframe_label_file, images)
    clip_labels = load_clip_labels(clip_label_file, valid_clips)
    image_set = set(images)

    @app.get("/")
    def index():
        return render_template("annotate_web.html")

    @app.get("/api/state")
    def api_state():
        return jsonify({
            "images": images,
            "keyframe_labels": keyframe_labels,
            "clip_labels": clip_labels,
            "idx": 0,
            "total": len(images),
            "labeled": labeled_count(keyframe_labels),
        })

    @app.post("/api/save")
    def api_save():
        payload = request.get_json(silent=True) or {}
        incoming_kf = payload.get("keyframe_labels", {})
        incoming_clip = payload.get("clip_labels", {})

        if isinstance(incoming_kf, dict):
            for rel, scope in incoming_kf.items():
                if rel in image_set:
                    keyframe_labels[rel] = normalize_scope(scope)

        if isinstance(incoming_clip, dict):
            for clip_id, val in incoming_clip.items():
                normalized_clip_id = str(clip_id).strip()
                if normalized_clip_id in valid_clips:
                    clip_labels[normalized_clip_id] = normalize_clip_label(val)

        save_keyframe_labels(keyframe_label_file, images, keyframe_labels)
        save_clip_labels(clip_label_file, clip_labels)
        return jsonify({
            "ok": True,
            "labeled": labeled_count(keyframe_labels),
            "message": f"已保存 keyframe={len(images)} -> {keyframe_label_file}; clip={len(clip_labels)} -> {clip_label_file}",
        })

    @app.get("/image/<path:rel_path>")
    def image(rel_path):
        target = (keyframe_root / rel_path).resolve()
        try:
            target.relative_to(keyframe_root)
        except ValueError:
            return jsonify({"ok": False, "message": "非法路径"}), 400
        if not target.exists():
            return jsonify({"ok": False, "message": "图片不存在"}), 404
        return send_from_directory(keyframe_root, rel_path)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="轻量本地 Web 关键帧标注工具")
    parser.add_argument("--keyframe_dir", default="./data/keyframes", help="关键帧目录")
    parser.add_argument("--keyframe_label_file", default="./data/keyframe_labels.csv", help="关键帧标注输出")
    parser.add_argument("--clip_label_file", default="./data/clip_labels.csv", help="clip级标注输出")
    parser.add_argument("--host", default="127.0.0.1", help="服务地址")
    parser.add_argument("--port", type=int, default=8765, help="服务端口")
    args = parser.parse_args()

    app = create_app(args.keyframe_dir, args.keyframe_label_file, args.clip_label_file)
    print(f"[INFO] 打开浏览器: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
