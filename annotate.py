"""
annotate.py - 关键帧标注工具 (Tkinter)

支持:
  - 图片浏览 (左右键翻页)
  - 标注: frame_scope / is_bidirectional / lane_count
  - 保存/加载 csv（兼容旧 schema）
  - 快捷键: ←→翻页, b/n=双向yes/no, u=unknown, 1/2/3=车道数(3=2+), Ctrl+S=保存

输出 schema:
  image_path,clip_id,frame_idx,frame_scope,is_bidirectional,lane_count
"""

import argparse
import csv
import os
import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk

VALID_SCOPE = {"slope", "transition", "non_slope", "unknown"}
VALID_DIR = {"yes", "no", "unknown"}
VALID_LANE = {"1", "2", "2+", "unknown"}


class AnnotationTool:
    def __init__(self, root, keyframe_dir, label_file):
        self.root = root
        self.root.title("Lane MVP - 关键帧标注")
        self.keyframe_dir = keyframe_dir
        self.label_file = label_file

        self.images = sorted([
            str(p) for p in Path(keyframe_dir).rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
            and p.name != "timestamp_index.csv"
        ])
        if not self.images:
            messagebox.showerror("错误", f"在 {keyframe_dir} 中未找到图片")
            root.destroy()
            return

        self.idx = 0
        self.labels = {}
        self._load_existing_labels()

        self.canvas = tk.Canvas(root, width=800, height=500, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        ctrl = tk.Frame(root)
        ctrl.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(ctrl, text="帧范围: ").grid(row=0, column=0, sticky="w")
        self.scope_var = tk.StringVar(value="unknown")
        scope_options = [
            ("坡道(slope)", "slope"),
            ("过渡(transition)", "transition"),
            ("非坡道(non_slope)", "non_slope"),
            ("未知(unknown)", "unknown"),
        ]
        for i, (text, val) in enumerate(scope_options):
            tk.Radiobutton(ctrl, text=text, variable=self.scope_var, value=val,
                           command=self._on_label_change).grid(row=0, column=i + 1, padx=5)

        tk.Label(ctrl, text="是否双向: ").grid(row=1, column=0, sticky="w")
        self.dir_var = tk.StringVar(value="unknown")
        for i, (text, val) in enumerate([("是(B)", "yes"), ("否(N)", "no"), ("未知(U)", "unknown")]):
            tk.Radiobutton(ctrl, text=text, variable=self.dir_var, value=val,
                           command=self._on_label_change).grid(row=1, column=i + 1, padx=5)

        tk.Label(ctrl, text="总车道数: ").grid(row=2, column=0, sticky="w")
        self.lane_var = tk.StringVar(value="unknown")
        lane_options = [("1", "1"), ("2", "2"), ("2+", "2+"), ("未知", "unknown")]
        for i, (text, val) in enumerate(lane_options):
            tk.Radiobutton(ctrl, text=text, variable=self.lane_var, value=val,
                           command=self._on_label_change).grid(row=2, column=i + 1, padx=3)

        nav = tk.Frame(root)
        nav.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(nav, text="← 上一张", command=self._prev).pack(side=tk.LEFT)
        tk.Button(nav, text="下一张 →", command=self._next).pack(side=tk.LEFT, padx=10)
        tk.Button(nav, text="保存 (Ctrl+S)", command=self._save).pack(side=tk.RIGHT)

        self.status_var = tk.StringVar()
        tk.Label(root, textvariable=self.status_var, anchor="w", relief=tk.SUNKEN).pack(
            fill=tk.X, side=tk.BOTTOM)

        root.bind("<Left>", lambda e: self._prev())
        root.bind("<Right>", lambda e: self._next())
        root.bind("b", lambda e: self._set_dir("yes"))
        root.bind("n", lambda e: self._set_dir("no"))
        root.bind("u", lambda e: self._set_dir("unknown"))
        for k in "123":
            root.bind(k, lambda e, v=k: self._set_lane(v))
        root.bind("<Control-s>", lambda e: self._save())
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._show_current()

    def _extract_meta(self, rel_path):
        norm = rel_path.replace("\\", "/")
        parts = norm.split("/")
        clip_id = parts[0] if parts else "unknown"
        stem = Path(rel_path).stem
        found = re.findall(r"\d+", stem)
        frame_idx = int(found[-1]) if found else -1
        return clip_id, frame_idx

    def _normalize_label(self, label):
        scope = label.get("frame_scope", "unknown")
        direction = label.get("is_bidirectional", "unknown")
        lane = label.get("lane_count", "unknown")

        if scope not in VALID_SCOPE:
            scope = "unknown"
        if direction not in VALID_DIR:
            direction = "unknown"
        if lane not in VALID_LANE:
            lane = "unknown"

        if scope != "slope":
            direction = "unknown"
            lane = "unknown"

        return {
            "frame_scope": scope,
            "is_bidirectional": direction,
            "lane_count": lane,
        }

    def _load_existing_labels(self):
        if not os.path.exists(self.label_file):
            return
        with open(self.label_file, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rel = row.get("image_path", "")
                if not rel:
                    continue
                self.labels[rel] = self._normalize_label({
                    "frame_scope": row.get("frame_scope", "unknown"),
                    "is_bidirectional": row.get("is_bidirectional", "unknown"),
                    "lane_count": row.get("lane_count", "unknown"),
                })
        print(f"加载已有标注 {len(self.labels)} 条")

    def _rel_path(self, abs_path):
        return os.path.relpath(abs_path, self.keyframe_dir)

    def _show_current(self):
        path = self.images[self.idx]
        img = Image.open(path)
        cw, ch = self.canvas.winfo_width() or 800, self.canvas.winfo_height() or 500
        img.thumbnail((cw, ch), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor=tk.CENTER)

        rel = self._rel_path(path)
        clip_id, frame_idx = self._extract_meta(rel)
        current = self.labels.get(rel, self._normalize_label({}))
        self.scope_var.set(current["frame_scope"])
        self.dir_var.set(current["is_bidirectional"])
        self.lane_var.set(current["lane_count"])

        labeled = sum(1 for p in self.images if self._rel_path(p) in self.labels)
        self.status_var.set(
            f"[{self.idx + 1}/{len(self.images)}] clip={clip_id} frame={frame_idx} path={rel} | 已标注: {labeled}/{len(self.images)}")

    def _on_label_change(self):
        rel = self._rel_path(self.images[self.idx])
        self.labels[rel] = self._normalize_label({
            "frame_scope": self.scope_var.get(),
            "is_bidirectional": self.dir_var.get(),
            "lane_count": self.lane_var.get(),
        })
        self._show_current()

    def _set_dir(self, val):
        self.dir_var.set(val)
        self._on_label_change()

    def _set_lane(self, val):
        self.lane_var.set("2+" if val == "3" else val)
        self._on_label_change()

    def _prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._show_current()

    def _next(self):
        if self.idx < len(self.images) - 1:
            self.idx += 1
            self._show_current()

    def _save(self):
        os.makedirs(os.path.dirname(self.label_file) or ".", exist_ok=True)
        rows = []
        for img_path in self.images:
            rel = self._rel_path(img_path)
            clip_id, frame_idx = self._extract_meta(rel)
            lbl = self.labels.get(rel, self._normalize_label({}))
            rows.append({
                "image_path": rel,
                "clip_id": clip_id,
                "frame_idx": frame_idx,
                "frame_scope": lbl["frame_scope"],
                "is_bidirectional": lbl["is_bidirectional"],
                "lane_count": lbl["lane_count"],
            })
        with open(self.label_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "image_path", "clip_id", "frame_idx", "frame_scope", "is_bidirectional", "lane_count"
            ])
            writer.writeheader()
            writer.writerows(rows)
        self.status_var.set(f"已保存 {len(rows)} 条标注 -> {self.label_file}")

    def _on_close(self):
        if self.labels and messagebox.askyesno("保存", "退出前是否保存标注？"):
            self._save()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="关键帧标注工具")
    parser.add_argument("--keyframe_dir", default="./data/keyframes", help="关键帧目录")
    parser.add_argument("--label_file", default="./data/keyframe_labels.csv", help="标注输出文件")
    args = parser.parse_args()

    root = tk.Tk()
    root.geometry("980x760")
    AnnotationTool(root, args.keyframe_dir, args.label_file)
    root.mainloop()


if __name__ == "__main__":
    main()
