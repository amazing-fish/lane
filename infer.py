"""
infer.py - segment 级推理脚本

用法:
  python infer.py --config config.yaml --clip_dir ./data/frames/bag001 --start_frame 100 --end_frame 220
  python infer.py --config config.yaml --manifest ./data/val_manifest.csv
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
LANE_NAMES = ["1", "2", "2+"]


def safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def load_segment_frames(clip_dir, start_frame, end_frame, transform):
    if not os.path.isdir(clip_dir):
        return [], [], (start_frame, end_frame)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    frame_files = sorted([
        f for f in os.listdir(clip_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])
    if not frame_files:
        return [], [], (start_frame, end_frame)

    max_idx = len(frame_files) - 1
    clipped_start = max(0, start_frame)
    clipped_end = min(max_idx, end_frame)

    if clipped_start > clipped_end:
        return [], [], (clipped_start, clipped_end)

    selected_files = frame_files[clipped_start:clipped_end + 1]
    frames = [transform(Image.open(os.path.join(clip_dir, f)).convert("RGB")) for f in selected_files]
    return frames, selected_files, (clipped_start, clipped_end)


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
def infer_segment(model, clip_dir, start_frame, end_frame, cfg, device):
    mcfg = cfg["model"]
    transform = build_infer_transform(mcfg["image_size"])
    frames, frame_files, (actual_start, actual_end) = load_segment_frames(clip_dir, start_frame, end_frame, transform)
    if not frames:
        return None

    snippets, starts = create_snippets(frames, mcfg["snippet_length"], mcfg["snippet_stride"])
    if snippets is None:
        return None

    snippets = snippets.unsqueeze(0).to(device)
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
            "rank": rank + 1,
            "snippet_idx": int(si),
            "attention": float(attn[si]),
            "start_frame": actual_start + sf,
            "end_frame": actual_start + ef,
            "start_frame_name": frame_files[sf],
            "end_frame_name": frame_files[ef],
        })

    return {
        "direction": DIRECTION_NAMES[dir_pred],
        "direction_confidence": float(dir_probs[dir_pred]),
        "lane_count": LANE_NAMES[lane_pred],
        "lane_count_confidence": float(lane_probs[lane_pred]),
        "evidence": evidence,
        "num_frames": len(frames),
        "num_snippets": len(starts),
        "start_frame": actual_start,
        "end_frame": actual_end,
    }


def manifest_row_to_segment(row):
    start_frame = safe_int(row.get("start_frame"), 0)
    if "end_frame" in row and str(row.get("end_frame", "")).strip() != "":
        end_frame = safe_int(row.get("end_frame"), start_frame)
    else:
        frame_count = safe_int(row.get("frame_count"), 0)
        end_frame = start_frame + max(0, frame_count - 1)

    return {
        "sample_id": row.get("sample_id", row.get("clip", "unknown")),
        "clip": row.get("clip", "unknown"),
        "clip_dir": row.get("clip_dir", ""),
        "start_frame": start_frame,
        "end_frame": end_frame,
    }


def main():
    parser = argparse.ArgumentParser(description="Lane MVP segment inference")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--clip_dir", default=None, help="Single segment clip dir")
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--manifest", default=None, help="Manifest CSV for batch")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or cfg["inference"]["checkpoint"]
    model = load_model(cfg, ckpt_path, device)

    output_dir = cfg["inference"].get("output_dir", "./results")
    os.makedirs(output_dir, exist_ok=True)

    segments = []
    if args.clip_dir:
        if args.start_frame is None or args.end_frame is None:
            raise ValueError("使用 --clip_dir 时必须同时传 --start_frame 与 --end_frame")
        if args.end_frame < args.start_frame:
            raise ValueError("--end_frame 不能小于 --start_frame")
        segments = [{
            "sample_id": "custom",
            "clip": "custom",
            "clip_dir": args.clip_dir,
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
        }]
    else:
        manifest = args.manifest or cfg["data"].get("val_manifest")
        if manifest and os.path.exists(manifest):
            with open(manifest, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    seg = manifest_row_to_segment(row)
                    if not seg["clip_dir"]:
                        print(f"[WARN] skip row without clip_dir: {seg['sample_id']}")
                        continue
                    segments.append(seg)

    if not segments:
        print("[ERROR] No segments. Use --clip_dir + --start_frame + --end_frame or --manifest.")
        return

    print(f"Inferring {len(segments)} segments...")
    results = []

    for seg in segments:
        result = infer_segment(model, seg["clip_dir"], seg["start_frame"], seg["end_frame"], cfg, device)
        if result is None:
            print(f"  [SKIP] {seg['sample_id']}: no frames in range [{seg['start_frame']}, {seg['end_frame']}]")
            continue
        out_row = {**seg, **result}
        results.append(out_row)
        ev_str = " | ".join(
            f"#{e['rank']}({e['attention']:.3f}): {e['start_frame']}-{e['end_frame']}"
            for e in result["evidence"]
        )
        print(
            f"  {seg['sample_id']}: range=[{result['start_frame']},{result['end_frame']}] "
            f"dir={result['direction']}({result['direction_confidence']:.2f}) "
            f"lanes={result['lane_count']}({result['lane_count_confidence']:.2f}) "
            f"evidence=[{ev_str}]"
        )

    output_file = cfg["inference"].get("output_file", os.path.join(output_dir, "predictions.csv"))
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id", "clip", "clip_dir", "start_frame", "end_frame",
            "direction", "direction_confidence", "lane_count", "lane_count_confidence",
            "num_frames", "num_snippets", "top_evidence"
        ])
        writer.writeheader()
        for r in results:
            top_ev = ""
            if r["evidence"]:
                ev0 = r["evidence"][0]
                top_ev = f"{ev0['start_frame']}-{ev0['end_frame']}"
            writer.writerow({
                "sample_id": r["sample_id"],
                "clip": r["clip"],
                "clip_dir": r["clip_dir"],
                "start_frame": r["start_frame"],
                "end_frame": r["end_frame"],
                "direction": r["direction"],
                "direction_confidence": f"{r['direction_confidence']:.4f}",
                "lane_count": r["lane_count"],
                "lane_count_confidence": f"{r['lane_count_confidence']:.4f}",
                "num_frames": r["num_frames"],
                "num_snippets": r["num_snippets"],
                "top_evidence": top_ev,
            })

    print(f"\nResults -> {output_file}")


if __name__ == "__main__":
    main()
