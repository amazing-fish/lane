"""
infer.py - 推理脚本: 对视频片段推理，输出预测类别、置信度、证据片段

用法:
    python infer.py --config config.yaml
    python infer.py --checkpoint checkpoints/best.pth --clip_dir ./data/frames/bag001
"""

import argparse
import csv
import os

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from model import build_model

DIRECTION_NAMES = ["unidirectional", "bidirectional"]
LANE_NAMES = ["1", "2", "3", "4", "5", "6+"]


def load_model(cfg, checkpoint_path, device):
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


def build_infer_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_clip_frames(clip_dir, transform):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    frame_files = sorted([
        f for f in os.listdir(clip_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])
    frames = [transform(Image.open(os.path.join(clip_dir, f)).convert("RGB"))
              for f in frame_files]
    return frames, frame_files


def create_snippets(frames, snippet_length, snippet_stride):
    n = len(frames)
    if n == 0:
        return None, []
    starts = list(range(0, max(1, n - snippet_length + 1), snippet_stride))
    if not starts:
        starts = [0]
    snippets = []
    for s in starts:
        snip = [frames[min(s + i, n - 1)] for i in range(snippet_length)]
        snippets.append(torch.stack(snip))
    return torch.stack(snippets), starts


@torch.no_grad()
def infer_clip(model, clip_dir, cfg, device):
    mcfg = cfg["model"]
    transform = build_infer_transform(mcfg["image_size"])
    frames, frame_files = load_clip_frames(clip_dir, transform)
    if not frames:
        return None

    snippets, starts = create_snippets(frames, mcfg["snippet_length"], mcfg["snippet_stride"])
    if snippets is None:
        return None

    snippets = snippets.unsqueeze(0).to(device)
    # 推理路径显式传入 masks，保持与训练/验证一致的模型调用方式
    masks = torch.ones((1, snippets.shape[1]), dtype=torch.bool, device=device)
    out = model(snippets, masks)

    dir_probs = torch.softmax(out["direction_logits"], dim=1)[0].cpu().numpy()
    dir_pred = int(dir_probs.argmax())
    lane_probs = torch.softmax(out["lane_count_logits"], dim=1)[0].cpu().numpy()
    lane_pred = int(lane_probs.argmax())

    attn = out["attention_weights"][0].cpu().numpy()
    top_k = min(cfg["inference"].get("top_k_evidence", 5), len(starts))
    top_indices = np.argsort(attn)[::-1][:top_k]

    evidence = []
    for rank, si in enumerate(top_indices):
        sf = starts[si]
        ef = min(sf + mcfg["snippet_length"] - 1, len(frame_files) - 1)
        evidence.append({
            "rank": rank + 1, "snippet_idx": int(si),
            "attention": float(attn[si]),
            "start_frame": frame_files[sf], "end_frame": frame_files[ef],
        })

    return {
        "direction": DIRECTION_NAMES[dir_pred],
        "direction_confidence": float(dir_probs[dir_pred]),
        "lane_count": LANE_NAMES[lane_pred],
        "lane_count_confidence": float(lane_probs[lane_pred]),
        "evidence": evidence,
        "num_frames": len(frames),
        "num_snippets": len(starts),
    }


def main():
    parser = argparse.ArgumentParser(description="Lane MVP inference")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--clip_dir", default=None, help="Single clip dir")
    parser.add_argument("--manifest", default=None, help="Manifest CSV for batch")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or cfg["inference"]["checkpoint"]
    model = load_model(cfg, ckpt_path, device)

    output_dir = cfg["inference"].get("output_dir", "./results")
    os.makedirs(output_dir, exist_ok=True)

    # 收集 clips
    clips = []
    if args.clip_dir:
        clips = [("custom", args.clip_dir)]
    else:
        manifest = args.manifest or cfg["data"].get("val_manifest")
        if manifest and os.path.exists(manifest):
            with open(manifest, "r") as f:
                for row in csv.DictReader(f):
                    clips.append((row["clip"], row["clip_dir"]))

    if not clips:
        print("[ERROR] No clips. Use --clip_dir or --manifest.")
        return

    print(f"Inferring {len(clips)} clips...")
    results = []

    for clip_name, clip_dir in clips:
        result = infer_clip(model, clip_dir, cfg, device)
        if result is None:
            print(f"  [SKIP] {clip_name}: no frames")
            continue
        results.append({"clip": clip_name, **result})
        ev_str = " | ".join(
            f"#{e['rank']}({e['attention']:.3f}): {e['start_frame']}-{e['end_frame']}"
            for e in result["evidence"])
        print(f"  {clip_name}: dir={result['direction']}({result['direction_confidence']:.2f}) "
              f"lanes={result['lane_count']}({result['lane_count_confidence']:.2f}) "
              f"evidence=[{ev_str}]")

    # 保存结果
    output_file = cfg["inference"].get("output_file", os.path.join(output_dir, "predictions.csv"))
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clip", "direction", "direction_confidence",
            "lane_count", "lane_count_confidence",
            "num_frames", "num_snippets", "top_evidence"])
        writer.writeheader()
        for r in results:
            top_ev = r["evidence"][0]["start_frame"] if r["evidence"] else ""
            writer.writerow({
                "clip": r["clip"], "direction": r["direction"],
                "direction_confidence": f"{r['direction_confidence']:.4f}",
                "lane_count": r["lane_count"],
                "lane_count_confidence": f"{r['lane_count_confidence']:.4f}",
                "num_frames": r["num_frames"], "num_snippets": r["num_snippets"],
                "top_evidence": top_ev})

    print(f"\nResults -> {output_file}")


if __name__ == "__main__":
    main()
