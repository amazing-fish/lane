"""
dataset.py - PyTorch Dataset: 从 segment manifest 加载 snippet bags

每个样本 = 一个 segment 的所有 snippet，每个 snippet = snippet_length 帧
"""

import csv
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_transforms(image_size, is_train=True):
    """构建图像变换."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class LaneSegmentDataset(Dataset):
    """Segment 级 Dataset。"""

    def __init__(self, manifest_path, cfg, is_train=True):
        self.cfg = cfg
        self.snippet_length = cfg["model"]["snippet_length"]
        self.snippet_stride = cfg["model"]["snippet_stride"]
        self.is_train = is_train
        self.transform = build_transforms(cfg["model"]["image_size"], is_train)

        self.samples = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.samples.append({
                    "sample_id": row.get("sample_id", row.get("clip", "unknown")),
                    "clip": row["clip"],
                    "clip_dir": row["clip_dir"],
                    "start_frame": int(row.get("start_frame", 0)),
                    "end_frame": int(row.get("end_frame", int(row.get("frame_count", 0)) - 1)),
                    "frame_count": int(row.get("frame_count", 0)),
                    "is_bidirectional": int(row["is_bidirectional"]),
                    "lane_count": int(row["lane_count"]),
                })

    def __len__(self):
        return len(self.samples)

    def _load_frame_list(self, clip_dir):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        if not os.path.isdir(clip_dir):
            print(f"[WARN] clip_dir 不存在或不可访问: {clip_dir}")
            return []
        return sorted([
            os.path.join(clip_dir, f) for f in os.listdir(clip_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])

    def _slice_segment_frames(self, frame_paths, start_frame, end_frame, sample_id):
        if not frame_paths:
            return []

        max_idx = len(frame_paths) - 1
        clipped_start = max(0, start_frame)
        clipped_end = min(max_idx, end_frame)

        if clipped_start != start_frame or clipped_end != end_frame:
            print(
                f"[WARN] segment {sample_id} 区间越界，已裁剪: "
                f"[{start_frame}, {end_frame}] -> [{clipped_start}, {clipped_end}]"
            )

        if clipped_start > clipped_end:
            print(f"[WARN] segment {sample_id} 裁剪后无有效帧")
            return []

        return frame_paths[clipped_start:clipped_end + 1]

    def _create_snippets(self, frame_paths):
        n = len(frame_paths)
        if n < self.snippet_length:
            return [0]
        indices = list(range(0, n - self.snippet_length + 1, self.snippet_stride))
        return indices if indices else [0]

    def _load_snippet(self, frame_paths, start_idx):
        frames = []
        for i in range(self.snippet_length):
            idx = min(start_idx + i, len(frame_paths) - 1)
            img = Image.open(frame_paths[idx]).convert("RGB")
            frames.append(self.transform(img))
        return torch.stack(frames, dim=0)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_paths = self._load_frame_list(sample["clip_dir"])
        frame_paths = self._slice_segment_frames(
            frame_paths,
            sample["start_frame"],
            sample["end_frame"],
            sample["sample_id"],
        )

        if not frame_paths:
            t = self.snippet_length
            s = self.cfg["model"]["image_size"]
            dummy = torch.zeros(1, t, 3, s, s)
            return {
                "snippets": dummy,
                "direction_label": sample["is_bidirectional"],
                "lane_count_label": sample["lane_count"],
                "clip_id": sample["clip"],
                "sample_id": sample["sample_id"],
                "segment_start": sample["start_frame"],
                "segment_end": sample["end_frame"],
            }

        snippet_starts = self._create_snippets(frame_paths)
        max_snippets = 8
        if self.is_train and len(snippet_starts) > max_snippets:
            snippet_starts = sorted(random.sample(snippet_starts, max_snippets))

        snippets = torch.stack([self._load_snippet(frame_paths, start) for start in snippet_starts], dim=0)
        return {
            "snippets": snippets,
            "direction_label": sample["is_bidirectional"],
            "lane_count_label": sample["lane_count"],
            "clip_id": sample["clip"],
            "sample_id": sample["sample_id"],
            "segment_start": sample["start_frame"],
            "segment_end": sample["end_frame"],
        }


# 向后兼容旧命名
LaneClipDataset = LaneSegmentDataset


def collate_fn(batch):
    max_n = max(item["snippets"].shape[0] for item in batch)
    t, c, h, w = batch[0]["snippets"].shape[1:]

    padded_snippets = []
    masks = []
    for item in batch:
        n = item["snippets"].shape[0]
        if n < max_n:
            pad = torch.zeros(max_n - n, t, c, h, w)
            padded = torch.cat([item["snippets"], pad], dim=0)
        else:
            padded = item["snippets"]
        padded_snippets.append(padded)
        mask = torch.zeros(max_n, dtype=torch.bool)
        mask[:n] = True
        masks.append(mask)

    return {
        "snippets": torch.stack(padded_snippets),
        "masks": torch.stack(masks),
        "direction_label": torch.tensor([b["direction_label"] for b in batch]),
        "lane_count_label": torch.tensor([b["lane_count_label"] for b in batch]),
        "clip_id": [b["clip_id"] for b in batch],
        "sample_id": [b["sample_id"] for b in batch],
        "segment_start": torch.tensor([b["segment_start"] for b in batch]),
        "segment_end": torch.tensor([b["segment_end"] for b in batch]),
    }
