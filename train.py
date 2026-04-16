"""
train.py - 训练脚本: 配置读取 / 训练 / 验证 / 保存权重 / 输出指标

用法:
    python train.py --config config.yaml
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
import yaml

from dataset import LaneClipDataset, collate_fn
from model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, cfg, steps_per_epoch):
    tcfg = cfg["training"]
    warmup_steps = tcfg["warmup_epochs"] * steps_per_epoch
    total_steps = tcfg["epochs"] * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_loss_and_preds(out, dir_lbl, lane_lbl, criterion_dir, criterion_lane, tcfg):
    """计算损失和预测，忽略 unknown 标签 (-1)."""
    loss = torch.tensor(0.0, device=dir_lbl.device)
    dir_preds_list, dir_labels_list = [], []
    lane_preds_list, lane_labels_list = [], []

    dir_mask = dir_lbl >= 0
    if dir_mask.any():
        loss_dir = criterion_dir(out["direction_logits"][dir_mask], dir_lbl[dir_mask])
        loss = loss + tcfg["direction_loss_weight"] * loss_dir
        dir_preds_list = out["direction_logits"][dir_mask].argmax(1).cpu().tolist()
        dir_labels_list = dir_lbl[dir_mask].cpu().tolist()

    lane_mask = lane_lbl >= 0
    if lane_mask.any():
        loss_lane = criterion_lane(out["lane_count_logits"][lane_mask], lane_lbl[lane_mask])
        loss = loss + tcfg["lane_count_loss_weight"] * loss_lane
        lane_preds_list = out["lane_count_logits"][lane_mask].argmax(1).cpu().tolist()
        lane_labels_list = lane_lbl[lane_mask].cpu().tolist()

    has_valid = dir_mask.any() or lane_mask.any()
    return loss, has_valid, dir_preds_list, dir_labels_list, lane_preds_list, lane_labels_list


def train_one_epoch(model, loader, optimizer, scheduler, criterion_dir, criterion_lane, cfg, device):
    model.train()
    tcfg = cfg["training"]
    total_loss = 0
    all_dir_p, all_dir_l, all_lane_p, all_lane_l = [], [], [], []

    for batch in loader:
        snippets = batch["snippets"].to(device)
        masks = batch["masks"].to(device)
        dir_lbl = batch["direction_label"].to(device)
        lane_lbl = batch["lane_count_label"].to(device)
        out = model(snippets, masks)

        loss, has_valid, dp, dl, lp, ll = compute_loss_and_preds(
            out, dir_lbl, lane_lbl, criterion_dir, criterion_lane, tcfg)
        all_dir_p.extend(dp); all_dir_l.extend(dl)
        all_lane_p.extend(lp); all_lane_l.extend(ll)

        if has_valid:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

    metrics = {"loss": total_loss / max(len(loader), 1)}
    if all_dir_l:
        metrics["dir_f1"] = f1_score(all_dir_l, all_dir_p, average="binary", zero_division=0)
    if all_lane_l:
        metrics["lane_f1"] = f1_score(all_lane_l, all_lane_p, average="macro", zero_division=0)
    return metrics


@torch.no_grad()
def validate(model, loader, criterion_dir, criterion_lane, cfg, device):
    model.eval()
    tcfg = cfg["training"]
    total_loss = 0
    all_dir_p, all_dir_l, all_lane_p, all_lane_l = [], [], [], []

    for batch in loader:
        snippets = batch["snippets"].to(device)
        masks = batch["masks"].to(device)
        dir_lbl = batch["direction_label"].to(device)
        lane_lbl = batch["lane_count_label"].to(device)
        out = model(snippets, masks)

        loss, _, dp, dl, lp, ll = compute_loss_and_preds(
            out, dir_lbl, lane_lbl, criterion_dir, criterion_lane, tcfg)
        all_dir_p.extend(dp); all_dir_l.extend(dl)
        all_lane_p.extend(lp); all_lane_l.extend(ll)
        total_loss += loss.item()

    metrics = {"val_loss": total_loss / max(len(loader), 1)}
    if all_dir_l:
        metrics["val_dir_f1"] = f1_score(all_dir_l, all_dir_p, average="binary", zero_division=0)
        metrics["dir_report"] = classification_report(
            all_dir_l, all_dir_p, target_names=["unidirectional", "bidirectional"], zero_division=0)
    if all_lane_l:
        metrics["val_lane_f1"] = f1_score(all_lane_l, all_lane_p, average="macro", zero_division=0)
        metrics["lane_report"] = classification_report(all_lane_l, all_lane_p, zero_division=0)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Lane MVP training")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tcfg = cfg["training"]
    set_seed(tcfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 数据
    train_ds = LaneClipDataset(cfg["data"]["train_manifest"], cfg, is_train=True)
    val_ds = LaneClipDataset(cfg["data"]["val_manifest"], cfg, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                              num_workers=tcfg["num_workers"], collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False,
                            num_workers=tcfg["num_workers"], collate_fn=collate_fn, pin_memory=True)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # 模型
    model = build_model(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    criterion_dir = nn.CrossEntropyLoss()
    criterion_lane = nn.CrossEntropyLoss()

    os.makedirs(tcfg["save_dir"], exist_ok=True)
    os.makedirs(tcfg["log_dir"], exist_ok=True)
    best_score = -1
    history = []

    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, scheduler,
                                  criterion_dir, criterion_lane, cfg, device)
        val_m = validate(model, val_loader, criterion_dir, criterion_lane, cfg, device)
        elapsed = time.time() - t0

        score = val_m.get("val_dir_f1", 0) * 0.5 + val_m.get("val_lane_f1", 0) * 0.5
        log = {"epoch": epoch, "time": f"{elapsed:.1f}s",
               **{k: round(v, 4) for k, v in train_m.items() if isinstance(v, float)},
               **{k: round(v, 4) for k, v in val_m.items() if isinstance(v, float)},
               "score": round(score, 4)}
        history.append(log)
        print(f"Epoch {epoch}/{tcfg['epochs']} | " +
              " | ".join(f"{k}={v}" for k, v in log.items() if k != "epoch"))

        if score > best_score:
            best_score = score
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "score": score, "cfg": cfg},
                        os.path.join(tcfg["save_dir"], "best.pth"))
            print(f"  -> Best model saved (score={score:.4f})")

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                    os.path.join(tcfg["save_dir"], "last.pth"))

    with open(os.path.join(tcfg["log_dir"], "train_history.json"), "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Best score={best_score:.4f}")
    if "dir_report" in val_m:
        print("\n=== Direction Report ===\n" + val_m["dir_report"])
    if "lane_report" in val_m:
        print("=== Lane Count Report ===\n" + val_m["lane_report"])


if __name__ == "__main__":
    main()
