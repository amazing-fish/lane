"""
annotate.py - 极简关键帧标注工具 (Tkinter)

支持:
  - 图片浏览 (左右键翻页)
  - 标注: 是否双向 / 总车道数 / unknown
  - 保存/加载 csv
  - 快捷键: ←→翻页, b/n=双向yes/no, 1-6=车道数, u=unknown, Ctrl+S=保存

用法:
    python annotate.py --keyframe_dir ./data/keyframes
    python annotate.py --keyframe_dir ./data/keyframes --label_file ./data/keyframe_labels.csv
"""

import argparse
import csv
import os
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk


class AnnotationTool:
    def __init__(self, root, keyframe_dir, label_file):
        self.root = root
        self.root.title("Lane MVP - 关键帧标注")
        self.keyframe_dir = keyframe_dir
        self.label_file = label_file

        # 收集所有图片
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
        self.labels = {}  # path -> {is_bidirectional, lane_count}
        self._load_existing_labels()

        # --- UI 布局 ---
        # 图片区域
        self.canvas = tk.Canvas(root, width=800, height=500, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 控制面板
        ctrl = tk.Frame(root)
        ctrl.pack(fill=tk.X, padx=10, pady=5)

        # 是否双向
        tk.Label(ctrl, text="是否双向:").grid(row=0, column=0, sticky="w")
        self.dir_var = tk.StringVar(value="unknown")
        for i, (text, val) in enumerate([("是(B)", "yes"), ("否(N)", "no"), ("未知(U)", "unknown")]):
            tk.Radiobutton(ctrl, text=text, variable=self.dir_var, value=val,
                           command=self._on_label_change).grid(row=0, column=i + 1, padx=5)

        # 总车道数
        tk.Label(ctrl, text="总车道数:").grid(row=1, column=0, sticky="w")
        self.lane_var = tk.StringVar(value="unknown")
        lane_options = [("1", "1"), ("2", "2"), ("3", "3"), ("4", "4"), ("5", "5"), ("6+", "6+"), ("未知", "unknown")]
        for i, (text, val) in enumerate(lane_options):
            tk.Radiobutton(ctrl, text=text, variable=self.lane_var, value=val,
                           command=self._on_label_change).grid(row=1, column=i + 1, padx=3)

        # 导航与保存
        nav = tk.Frame(root)
        nav.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(nav, text="← 上一张", command=self._prev).pack(side=tk.LEFT)
        tk.Button(nav, text="下一张 →", command=self._next).pack(side=tk.LEFT, padx=10)
        tk.Button(nav, text="保存 (Ctrl+S)", command=self._save).pack(side=tk.RIGHT)

        # 状态栏
        self.status_var = tk.StringVar()
        tk.Label(root, textvariable=self.status_var, anchor="w", relief=tk.SUNKEN).pack(
            fill=tk.X, side=tk.BOTTOM)

        # 快捷键
        root.bind("<Left>", lambda e: self._prev())
        root.bind("<Right>", lambda e: self._next())
        root.bind("b", lambda e: self._set_dir("yes"))
        root.bind("n", lambda e: self._set_dir("no"))
        root.bind("u", lambda e: self._set_dir("unknown"))
        for k in "123456":
            root.bind(k, lambda e, v=k: self._set_lane(v))
        root.bind("<Control-s>", lambda e: self._save())
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._show_current()

    def _load_existing_labels(self):
        if not os.path.exists(self.label_file):
            return
        with open(self.label_file, "r", newline="") as f:
            for row in csv.DictReader(f):
                self.labels[row["image_path"]] = {
                    "is_bidirectional": row.get("is_bidirectional", "unknown"),
                    "lane_count": row.get("lane_count", "unknown"),
                }
        print(f"加载已有标注 {len(self.labels)} 条")

    def _rel_path(self, abs_path):
        return os.path.relpath(abs_path, self.keyframe_dir)

    def _show_current(self):
        path = self.images[self.idx]
        img = Image.open(path)
        # 缩放适配 canvas
        cw, ch = self.canvas.winfo_width() or 800, self.canvas.winfo_height() or 500
        img.thumbnail((cw, ch), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor=tk.CENTER)

        # 恢复已有标注
        rel = self._rel_path(path)
        if rel in self.labels:
            self.dir_var.set(self.labels[rel]["is_bidirectional"])
            self.lane_var.set(self.labels[rel]["lane_count"])
        else:
            self.dir_var.set("unknown")
            self.lane_var.set("unknown")

        labeled = sum(1 for p in self.images if self._rel_path(p) in self.labels)
        self.status_var.set(
            f"[{self.idx + 1}/{len(self.images)}] {rel}  |  已标注: {labeled}/{len(self.images)}")

    def _on_label_change(self):
        rel = self._rel_path(self.images[self.idx])
        self.labels[rel] = {
            "is_bidirectional": self.dir_var.get(),
            "lane_count": self.lane_var.get(),
        }
        self._show_current()

    def _set_dir(self, val):
        self.dir_var.set(val)
        self._on_label_change()

    def _set_lane(self, val):
        self.lane_var.set(val if val != "6" else "6+")
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
            lbl = self.labels.get(rel, {"is_bidirectional": "", "lane_count": ""})
            rows.append({"image_path": rel, **lbl})
        with open(self.label_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "is_bidirectional", "lane_count"])
            writer.writeheader()
            writer.writerows(rows)
        self.status_var.set(f"已保存 {len(self.labels)} 条标注 -> {self.label_file}")

    def _on_close(self):
        if self.labels:
            if messagebox.askyesno("保存", "退出前是否保存标注？"):
                self._save()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="极简关键帧标注工具")
    parser.add_argument("--keyframe_dir", default="./data/keyframes", help="关键帧目录")
    parser.add_argument("--label_file", default="./data/keyframe_labels.csv", help="标注输出文件")
    args = parser.parse_args()

    root = tk.Tk()
    root.geometry("900x700")
    AnnotationTool(root, args.keyframe_dir, args.label_file)
    root.mainloop()


if __name__ == "__main__":
    main()
