"""
annotate.py - 关键帧标注工具 (Tkinter)

新流程输出两份真值：
  1) clip_labels.csv: clip 级属性（is_bidirectional/lane_count/quality/notes）
  2) keyframe_labels.csv: 关键帧范围（frame_scope）
"""

import argparse
import csv
import os
import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk

VALID_SCOPE = {"slope", "non_slope", "unknown", "transition"}
VALID_DIR = {"yes", "no", "unknown"}
VALID_LANE = {"1", "2", "2+", "unknown"}
VALID_QUALITY = {"ok", "review", "bad", "unknown"}
LEGACY_LANE_TO_NEW = {"3": "2+", "4": "2+", "5": "2+", "6+": "2+"}


class AnnotationTool:
    def __init__(self, root, keyframe_dir, keyframe_label_file, clip_label_file):
        self.root = root
        self.root.title("Lane MVP - 关键帧标注")
        self.keyframe_dir = keyframe_dir
        self.keyframe_label_file = keyframe_label_file
        self.clip_label_file = clip_label_file

        self.images = sorted([
            str(p) for p in Path(keyframe_dir).rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp") and p.name != "timestamp_index.csv"
        ])
        if not self.images:
            messagebox.showerror("错误", f"在 {keyframe_dir} 中未找到图片")
            root.destroy()
            return

        self.idx = 0
        self.keyframe_labels = {}
        self.clip_labels = {}
        self._load_existing_labels()

        self.canvas = tk.Canvas(root, width=860, height=520, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        ctrl = tk.Frame(root)
        ctrl.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(ctrl, text="frame_scope:").grid(row=0, column=0, sticky="w")
        self.scope_var = tk.StringVar(value="unknown")
        for i, (text, val) in enumerate([
            ("slope", "slope"),
            ("non_slope", "non_slope"),
            ("transition", "transition"),
            ("unknown", "unknown"),
        ]):
            tk.Radiobutton(ctrl, text=text, variable=self.scope_var, value=val,
                           command=self._on_scope_change).grid(row=0, column=i + 1, padx=5)

        tk.Label(ctrl, text="clip: is_bidirectional").grid(row=1, column=0, sticky="w")
        self.dir_var = tk.StringVar(value="unknown")
        for i, (text, val) in enumerate([("yes(B)", "yes"), ("no(N)", "no"), ("unknown(U)", "unknown")]):
            tk.Radiobutton(ctrl, text=text, variable=self.dir_var, value=val,
                           command=self._on_clip_change).grid(row=1, column=i + 1, padx=5)

        tk.Label(ctrl, text="clip: lane_count").grid(row=2, column=0, sticky="w")
        self.lane_var = tk.StringVar(value="unknown")
        for i, (text, val) in enumerate([("1", "1"), ("2", "2"), ("2+", "2+"), ("unknown", "unknown")]):
            tk.Radiobutton(ctrl, text=text, variable=self.lane_var, value=val,
                           command=self._on_clip_change).grid(row=2, column=i + 1, padx=5)

        tk.Label(ctrl, text="clip: quality").grid(row=3, column=0, sticky="w")
        self.quality_var = tk.StringVar(value="ok")
        for i, (text, val) in enumerate([("ok", "ok"), ("review", "review"), ("bad", "bad"), ("unknown", "unknown")]):
            tk.Radiobutton(ctrl, text=text, variable=self.quality_var, value=val,
                           command=self._on_clip_change).grid(row=3, column=i + 1, padx=5)

        tk.Label(ctrl, text="clip: notes").grid(row=4, column=0, sticky="w")
        self.notes_var = tk.StringVar(value="")
        self.note_entry = tk.Entry(ctrl, textvariable=self.notes_var, width=70)
        self.note_entry.grid(row=4, column=1, columnspan=5, sticky="we", padx=5)
        self.note_entry.bind("<KeyRelease>", lambda _e: self._on_clip_change())

        nav = tk.Frame(root)
        nav.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(nav, text="← 上一张", command=self._prev).pack(side=tk.LEFT)
        tk.Button(nav, text="下一张 →", command=self._next).pack(side=tk.LEFT, padx=10)
        tk.Button(nav, text="保存 (Ctrl+S)", command=self._save).pack(side=tk.RIGHT)

        self.status_var = tk.StringVar()
        tk.Label(root, textvariable=self.status_var, anchor="w", relief=tk.SUNKEN).pack(fill=tk.X, side=tk.BOTTOM)

        root.bind("<Left>", lambda _e: self._prev())
        root.bind("<Right>", lambda _e: self._next())
        root.bind("b", lambda e: self._on_hotkey_dir(e, "yes"))
        root.bind("n", lambda e: self._on_hotkey_dir(e, "no"))
        root.bind("u", lambda e: self._on_hotkey_dir(e, "unknown"))
        root.bind("1", lambda e: self._on_hotkey_lane(e, "1"))
        root.bind("2", lambda e: self._on_hotkey_lane(e, "2"))
        root.bind("3", lambda e: self._on_hotkey_lane(e, "2+"))
        root.bind("<Control-s>", lambda _e: self._save())
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

    def _rel_path(self, abs_path):
        return os.path.relpath(abs_path, self.keyframe_dir)

    def _normalize_scope(self, value):
        value = str(value or "unknown").strip().lower()
        return value if value in VALID_SCOPE else "unknown"

    def _normalize_clip_attrs(self, direction="unknown", lane="unknown", quality="ok", notes=""):
        direction = str(direction or "unknown").strip().lower()
        lane = LEGACY_LANE_TO_NEW.get(str(lane or "unknown").strip().lower(), str(lane or "unknown").strip().lower())
        quality = str(quality or "unknown").strip().lower()
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
            "notes": str(notes or "").strip(),
        }

    def _load_existing_labels(self):
        if os.path.exists(self.keyframe_label_file):
            with open(self.keyframe_label_file, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rel = row.get("image_path", "")
                    if not rel:
                        continue
                    self.keyframe_labels[rel] = self._normalize_scope(row.get("frame_scope", "unknown"))
                    clip_id = row.get("clip_id") or self._extract_meta(rel)[0]
                    if clip_id not in self.clip_labels:
                        self.clip_labels[clip_id] = self._normalize_clip_attrs(
                            row.get("is_bidirectional", "unknown"),
                            row.get("lane_count", "unknown"),
                            "ok",
                            "",
                        )

        if os.path.exists(self.clip_label_file):
            with open(self.clip_label_file, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    clip_id = str(row.get("clip_id", "")).strip()
                    if clip_id:
                        self.clip_labels[clip_id] = self._normalize_clip_attrs(
                            row.get("is_bidirectional", "unknown"),
                            row.get("lane_count", "unknown"),
                            row.get("quality", "ok"),
                            row.get("notes", ""),
                        )

    def _current_clip_id(self):
        rel = self._rel_path(self.images[self.idx])
        clip_id, _ = self._extract_meta(rel)
        return clip_id

    def _current_clip_label(self):
        clip_id = self._current_clip_id()
        if clip_id not in self.clip_labels:
            self.clip_labels[clip_id] = self._normalize_clip_attrs()
        return self.clip_labels[clip_id]

    def _show_current(self):
        path = self.images[self.idx]
        img = Image.open(path)
        cw, ch = self.canvas.winfo_width() or 860, self.canvas.winfo_height() or 520
        img.thumbnail((cw, ch), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor=tk.CENTER)

        rel = self._rel_path(path)
        clip_id, frame_idx = self._extract_meta(rel)
        scope = self.keyframe_labels.get(rel, "unknown")
        self.scope_var.set(scope)

        clip_lbl = self._current_clip_label()
        self.dir_var.set(clip_lbl["is_bidirectional"])
        self.lane_var.set(clip_lbl["lane_count"])
        self.quality_var.set(clip_lbl["quality"])
        self.notes_var.set(clip_lbl["notes"])

        labeled = sum(1 for p in self.images if self.keyframe_labels.get(self._rel_path(p), "unknown") != "unknown")
        self.status_var.set(
            f"[{self.idx + 1}/{len(self.images)}] clip={clip_id} frame={frame_idx} path={rel} | 已标frame_scope:{labeled}/{len(self.images)} | clip标签:{len(self.clip_labels)}"
        )

    def _on_scope_change(self):
        rel = self._rel_path(self.images[self.idx])
        self.keyframe_labels[rel] = self._normalize_scope(self.scope_var.get())
        self._show_current()

    def _on_clip_change(self):
        clip_id = self._current_clip_id()
        self.clip_labels[clip_id] = self._normalize_clip_attrs(
            self.dir_var.get(), self.lane_var.get(), self.quality_var.get(), self.notes_var.get()
        )

    def _set_dir(self, value):
        self.dir_var.set(value)
        self._on_clip_change()
        self._show_current()

    def _set_lane(self, value):
        self.lane_var.set(value)
        self._on_clip_change()
        self._show_current()

    def _is_text_input_focus(self, event):
        widget = getattr(event, "widget", None)
        if widget is None:
            return False
        return isinstance(widget, tk.Entry) or widget == self.note_entry

    def _on_hotkey_dir(self, event, value):
        if self._is_text_input_focus(event):
            return
        self._set_dir(value)

    def _on_hotkey_lane(self, event, value):
        if self._is_text_input_focus(event):
            return
        self._set_lane(value)

    def _prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._show_current()

    def _next(self):
        if self.idx < len(self.images) - 1:
            self.idx += 1
            self._show_current()

    def _save(self):
        os.makedirs(os.path.dirname(self.keyframe_label_file) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.clip_label_file) or ".", exist_ok=True)

        kf_rows = []
        for img_path in self.images:
            rel = self._rel_path(img_path)
            clip_id, frame_idx = self._extract_meta(rel)
            kf_rows.append({
                "image_path": rel,
                "clip_id": clip_id,
                "frame_idx": frame_idx,
                "frame_scope": self.keyframe_labels.get(rel, "unknown"),
            })

        with open(self.keyframe_label_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "clip_id", "frame_idx", "frame_scope"])
            writer.writeheader()
            writer.writerows(kf_rows)

        clip_rows = []
        for clip_id in sorted(self.clip_labels.keys()):
            label = self.clip_labels[clip_id]
            clip_rows.append({
                "clip_id": clip_id,
                "is_bidirectional": label["is_bidirectional"],
                "lane_count": label["lane_count"],
                "quality": label["quality"],
                "notes": label["notes"],
            })

        with open(self.clip_label_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["clip_id", "is_bidirectional", "lane_count", "quality", "notes"])
            writer.writeheader()
            writer.writerows(clip_rows)

        self.status_var.set(
            f"已保存 keyframe={len(kf_rows)} -> {self.keyframe_label_file} | clip={len(clip_rows)} -> {self.clip_label_file}"
        )

    def _on_close(self):
        if (self.keyframe_labels or self.clip_labels) and messagebox.askyesno("保存", "退出前是否保存标注？"):
            self._save()
        self.root.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="关键帧标注工具")
    parser.add_argument("--keyframe_dir", default="./data/keyframes", help="关键帧目录")
    parser.add_argument("--keyframe_label_file", default="./data/keyframe_labels.csv", help="关键帧标注输出")
    parser.add_argument("--clip_label_file", default="./data/clip_labels.csv", help="clip级标签输出")
    args = parser.parse_args()

    root = tk.Tk()
    root.geometry("1020x840")
    AnnotationTool(root, args.keyframe_dir, args.keyframe_label_file, args.clip_label_file)
    root.mainloop()
