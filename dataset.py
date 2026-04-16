"""
dataset.py - PyTorch Dataset: 从 manifest 加载 clip snippet bags

每个样本 = 一个 clip 的所有 snippet，每个 snippet = snippet_length 帧
"""

import csv
import os
import random

import numpy as np
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


class LaneClipDataset(Dataset):
    """
    Clip 级 Dataset。

    每个样本返回:
        snippets: (N, T, C, H, W) - N 个 snippet
        direction_label: int (0=单向, 1=双向, -1=unknown)
        lane_count_label: int (0~5 对应 1~6+, -1=unknown)
        clip_id: str
    """

    def __init__(self, manifest_path, cfg, is_train=True):
        self.cfg = cfg
        self.snippet_length = cfg["model"]["snippet_length"]
        self.snippet_stride = cfg["model"]["snippet_stride"]
        self.is_train = is_train
        self.transform = build_transforms(cfg["model"]["image_size"], is_train)

        # 读取 manifest
        self.samples = []
        with open(manifest_path, "r") as f:
            for row in csv.DictReader(f):
                self.samples.append({
                    "clip": row["clip"],
                    "clip_dir": row["clip_dir"],
                    "frame_count": int(row["frame_count"]),
                    "is_bidirectional": int(row["is_bidirectional"]),
                    "lane_count": int(row["lane_count"]),
                })

    def __len__(self):
        return len(self.samples)

    def _load_frame_list(self, clip_dir):
        """加载 clip 目录下的帧文件列表（按文件名排序）."""
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        frames = sorted([
            os.path.join(clip_dir, f) for f in os.listdir(clip_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])
        return frames

    def _create_snippets(self, frame_paths):
        """将帧序列切分为 snippet，返回 snippet 起始索引列表."""
        n = len(frame_paths)
        if n < self.snippet_length:
            return [0]  # 帧不够时只取一个 snippet（会 pad）
        indices = list(range(0, n - self.snippet_length + 1, self.snippet_stride))
        if not indices:
            indices = [0]
        return indices

    def _load_snippet(self, frame_paths, start_idx):
        """加载一个 snippet 的帧并应用变换."""
        frames = []
        for i in range(self.snippet_length):
            idx = min(start_idx + i, len(frame_paths) - 1)  # pad by repeating last
            img = Image.open(frame_paths[idx]).convert("RGB")
            img = self.transform(img)
            frames.append(img)
        return torch.stack(frames, dim=0)  # (T, C, H, W)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_paths = self._load_frame_list(sample["clip_dir"])

        if not frame_paths:
            # 空 clip 兜底：返回零张量
            T = self.snippet_length
            dummy = torch.zeros(1, T, 3, self.cfg["model"]["image_size"],
                                self.cfg["model"]["image_size"])
            return {
                "snippets": dummy,
                "direction_label": sample["is_bidirectional"],
                "lane_count_label": sample["lane_count"],
                "clip_id": sample["clip"],
            }

        snippet_starts = self._create_snippets(frame_paths)

        # 训练时随机采样部分 snippet 控制显存
        max_snippets = 8
        if self.is_train and len(snippet_starts) > max_snippets:
            snippet_starts = sorted(random.sample(snippet_starts, max_snippets))

        snippets = []
        for start in snippet_starts:
            snippet = self._load_snippet(frame_paths, start)
            snippets.append(snippet)

        snippets = torch.stack(snippets, dim=0)  # (N, T, C, H, W)

        return {
            "snippets": snippets,
            "direction_label": sample["is_bidirectional"],
            "lane_count_label": sample["lane_count"],
            "clip_id": sample["clip"],
        }


def collate_fn(batch):
    """
    自定义 collate: 不同 clip 的 snippet 数量可能不同，pad 到同一长度。
    """
    max_n = max(item["snippets"].shape[0] for item in batch)
    T, C, H, W = batch[0]["snippets"].shape[1:]

    padded_snippets = []
    masks = []
    for item in batch:
        n = item["snippets"].shape[0]
        if n < max_n:
            pad = torch.zeros(max_n - n, T, C, H, W)
            padded = torch.cat([item["snippets"], pad], dim=0)
        else:
            padded = item["snippets"]
        padded_snippets.append(padded)
        mask = torch.zeros(max_n, dtype=torch.bool)
        mask[:n] = True
        masks.append(mask)

    return {
        "snippets": torch.stack(padded_snippets),       # (B, N, T, C, H, W)
        "masks": torch.stack(masks),                      # (B, N)
        "direction_label": torch.tensor([b["direction_label"] for b in batch]),
        "lane_count_label": torch.tensor([b["lane_count_label"] for b in batch]),
        "clip_id": [b["clip_id"] for b in batch],
    }
