"""
annotate_web.py - 轻量本地 Web 关键帧标注工具

输出 schema:
  image_path,clip_id,frame_idx,frame_scope,is_bidirectional,lane_count
"""

import argparse
import csv
import os
import re
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

VALID_SCOPE = {"slope", "transition", "non_slope", "unknown"}
VALID_DIR = {"yes", "no", "unknown"}
VALID_LANE = {"1", "2", "3", "4", "5", "6+", "unknown"}


def collect_images(keyframe_dir):
    return sorted([
        str(p.relative_to(keyframe_dir))
        for p in Path(keyframe_dir).rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        and p.name != "timestamp_index.csv"
    ])


def extract_meta(rel_path):
    norm = rel_path.replace("\\", "/")
    parts = norm.split("/")
    clip_id = parts[0] if parts else "unknown"
    stem = Path(rel_path).stem
    found = re.findall(r"\d+", stem)
    frame_idx = int(found[-1]) if found else -1
    return clip_id, frame_idx


def normalize_label(val):
    if not isinstance(val, dict):
        val = {}
    frame_scope = val.get("frame_scope", "unknown")
    direction = val.get("is_bidirectional", "unknown")
    lane = val.get("lane_count", "unknown")

    if frame_scope not in VALID_SCOPE:
        frame_scope = "unknown"
    if direction not in VALID_DIR:
        direction = "unknown"
    if lane not in VALID_LANE:
        lane = "unknown"

    if frame_scope != "slope":
        direction = "unknown"
        lane = "unknown"

    return {
        "frame_scope": frame_scope,
        "is_bidirectional": direction,
        "lane_count": lane,
    }


def load_labels(label_file, images):
    labels = {}
    if not os.path.exists(label_file):
        return labels
    image_set = set(images)
    with open(label_file, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = row.get("image_path", "")
            if rel in image_set:
                labels[rel] = normalize_label({
                    "frame_scope": row.get("frame_scope", "unknown"),
                    "is_bidirectional": row.get("is_bidirectional", "unknown"),
                    "lane_count": row.get("lane_count", "unknown"),
                })
    return labels


def save_labels(label_file, images, labels):
    os.makedirs(os.path.dirname(label_file) or ".", exist_ok=True)
    rows = []
    for rel in images:
        clip_id, frame_idx = extract_meta(rel)
        lbl = normalize_label(labels.get(rel, {}))
        rows.append({
            "image_path": rel,
            "clip_id": clip_id,
            "frame_idx": frame_idx,
            "frame_scope": lbl["frame_scope"],
            "is_bidirectional": lbl["is_bidirectional"],
            "lane_count": lbl["lane_count"],
        })
    with open(label_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_path", "clip_id", "frame_idx", "frame_scope", "is_bidirectional", "lane_count"
        ])
        writer.writeheader()
        writer.writerows(rows)


def labeled_count(labels):
    count = 0
    for value in labels.values():
        if value.get("frame_scope", "unknown") != "unknown":
            count += 1
    return count


def create_app(keyframe_dir, label_file):
    app = Flask(__name__, template_folder="templates")

    keyframe_root = Path(keyframe_dir).resolve()
    images = collect_images(keyframe_root)
    if not images:
        raise RuntimeError(f"在 {keyframe_root} 中未找到图片")

    labels = load_labels(label_file, images)
    image_set = set(images)

    @app.get("/")
    def index():
        return render_template("annotate_web.html")

    @app.get("/api/state")
    def api_state():
        return jsonify({
            "images": images,
            "labels": labels,
            "idx": 0,
            "total": len(images),
            "labeled": labeled_count(labels),
        })

    @app.post("/api/save")
    def api_save():
        payload = request.get_json(silent=True) or {}
        incoming = payload.get("labels", {})
        if isinstance(incoming, dict):
            for rel, val in incoming.items():
                if rel in image_set:
                    labels[rel] = normalize_label(val)
        save_labels(label_file, images, labels)
        return jsonify({
            "ok": True,
            "labeled": labeled_count(labels),
            "message": f"已保存 {len(images)} 条标注 -> {label_file}",
        })

    @app.get("/api/meta/<path:rel_path>")
    def api_meta(rel_path):
        if rel_path not in image_set:
            return jsonify({"ok": False, "message": "图片不存在"}), 404
        clip_id, frame_idx = extract_meta(rel_path)
        return jsonify({"clip_id": clip_id, "frame_idx": frame_idx})

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


def main():
    parser = argparse.ArgumentParser(description="轻量本地 Web 关键帧标注工具")
    parser.add_argument("--keyframe_dir", default="./data/keyframes", help="关键帧目录")
    parser.add_argument("--label_file", default="./data/keyframe_labels.csv", help="标注输出文件")
    parser.add_argument("--host", default="127.0.0.1", help="服务地址")
    parser.add_argument("--port", type=int, default=8765, help="服务端口")
    args = parser.parse_args()

    app = create_app(args.keyframe_dir, args.label_file)
    print(f"[INFO] 打开浏览器: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
