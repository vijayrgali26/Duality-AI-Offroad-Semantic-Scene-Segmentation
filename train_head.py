"""
Train the segmentation head on PCA-reduced 4-layer DINOv2 features.

Pipeline context:
  * Backbone DINOv2-vits14 is frozen; last-4-layer features (1536-d) were cached
    once (run_iou_boost.py extraction), then PCA-reduced to 384-d (reduce_features.py),
    retaining ~97.5% variance. The reduced cache (~1.35GB) fits in RAM, so epochs
    run in ~30s on CPU instead of ~20 min from disk thrashing.
  * Loss = weighted CrossEntropy + soft Dice, computed at the 19x34 token grid.
  * IoU is EVALUATED at full 266x476 resolution (bilinear upsample) so reported
    numbers are real; the best model is selected on validation mean IoU.

Outputs (iou_boost_output/):
  best_model.pth        - trained head weights
  final_results.json    - baseline vs best IoU and per-class IoU
  training_history.json - per-eval epoch metrics
"""

import os
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image

try:
    torch.set_num_threads(os.cpu_count() or 4)
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).parent
RED = PROJECT / "iou_boost_output" / "cache_red"
OUT = PROJECT / "iou_boost_output"

IMAGE_H, IMAGE_W, PATCH = 266, 476, 14
TOKEN_H, TOKEN_W = IMAGE_H // PATCH, IMAGE_W // PATCH   # 19 x 34
FULL_DIM = 384             # dims available in reduced cache (PCA, variance-ordered)
USE_DIM = 192              # use only top-192 PCs -> half RAM, ~2x faster epochs
EMBED_DIM = USE_DIM
NUM_CLASSES = 11
NUM_EPOCHS = 70
BATCH = 32
LR = 2e-3
WD = 1e-2
LABEL_SMOOTH = 0.05
HIDDEN = 192
DICE_W = 1.0
EVAL_EVERY = 3
BASELINE = 0.2794
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {
    0: "Background", 1: "Trees", 2: "Lush Bushes", 3: "Dry Grass", 4: "Dry Bushes",
    5: "Ground Clutter", 6: "Flowers", 7: "Logs", 8: "Rocks", 9: "Landscape", 10: "Sky",
}


class EfficientSegHead(nn.Module):
    def __init__(self, in_channels, out_channels, token_h, token_w, hidden=256, n_blocks=3):
        super().__init__()
        self.H, self.W = token_h, token_w
        self.proj = nn.Sequential(nn.Conv2d(in_channels, hidden, 1), nn.GELU())
        blocks = []
        for _ in range(n_blocks):
            blocks.append(nn.Sequential(
                nn.Conv2d(hidden, hidden, 7, padding=3, groups=hidden),
                nn.GELU(),
                nn.Conv2d(hidden, hidden * 2, 1),
                nn.GELU(),
                nn.Conv2d(hidden * 2, hidden, 1),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.GroupNorm(1, hidden)
        self.drop = nn.Dropout2d(0.1)
        self.classifier = nn.Conv2d(hidden, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.proj(x)
        for blk in self.blocks:
            x = x + blk(x)
        x = self.norm(x)
        x = self.drop(x)
        return self.classifier(x)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DiceLoss(nn.Module):
    def __init__(self, num_classes, weight=None, eps=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.register_buffer("w", weight if weight is not None else torch.ones(num_classes))

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        tgt = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * tgt).sum(dims)
        denom = probs.sum(dims) + tgt.sum(dims)
        dice = (2 * inter + self.eps) / (denom + self.eps)
        present = tgt.sum(dims) > 0
        w = self.w.to(dice.device) * present.float()
        return 1.0 - (dice * w).sum() / w.sum() if w.sum() > 0 else 1.0 - dice.mean()


def downsample_masks(masks):
    n = masks.shape[0]
    out = np.zeros((n, TOKEN_H, TOKEN_W), dtype=np.uint8)
    for i in range(n):
        out[i] = np.array(Image.fromarray(masks[i]).resize((TOKEN_W, TOKEN_H), Image.NEAREST), dtype=np.uint8)
    return out


def median_freq_weights(token_masks):
    counts = np.bincount(token_masks.ravel(), minlength=NUM_CLASSES).astype(np.float64)[:NUM_CLASSES]
    present = counts > 0
    freq = counts / counts.sum()
    med = np.median(freq[present])
    w = np.ones(NUM_CLASSES)
    w[present] = med / freq[present]
    w = np.clip(w, 0.3, 6.0)
    logger.info("Class weights: " + ", ".join(f"{CLASS_NAMES[i]}={w[i]:.2f}" for i in range(NUM_CLASSES)))
    return torch.tensor(w, dtype=torch.float32)


def evaluate(head, feats, masks_full, batch=8):
    head.eval()
    inter = np.zeros(NUM_CLASSES)
    union = np.zeros(NUM_CLASSES)
    n = feats.shape[0]
    with torch.no_grad():
        for s in range(0, n, batch):
            f = torch.from_numpy(np.asarray(feats[s:s+batch], dtype=np.float32)).to(DEVICE)
            m = np.asarray(masks_full[s:s+batch], dtype=np.int64)
            logits = head(f)
            logits = F.interpolate(logits, size=(IMAGE_H, IMAGE_W), mode="bilinear", align_corners=False)
            preds = logits.argmax(1).cpu().numpy()
            for c in range(NUM_CLASSES):
                pm, tm = preds == c, m == c
                inter[c] += np.logical_and(pm, tm).sum()
                union[c] += np.logical_or(pm, tm).sum()
    iou = {c: (float(inter[c]/union[c]) if union[c] > 0 else 0.0) for c in range(NUM_CLASSES)}
    mean_all = float(np.mean(list(iou.values())))
    present = [c for c in range(NUM_CLASSES) if union[c] > 0]
    mean_present = float(np.mean([iou[c] for c in present])) if present else 0.0
    return iou, mean_all, mean_present


def main():
    logger.info("=" * 80)
    logger.info("TRAIN HEAD on PCA-reduced 4-layer features")
    logger.info(f"Device {DEVICE} | token {TOKEN_H}x{TOKEN_W} | dim {EMBED_DIM} | hidden {HIDDEN}")
    logger.info("=" * 80)

    # Load reduced features into RAM, keeping only the top USE_DIM PCs
    # (PCA dims are variance-ordered, so the first USE_DIM are the strongest).
    logger.info(f"Loading reduced features into RAM (top {USE_DIM} of {FULL_DIM} PCs)...")
    tr_mm = np.load(RED / "train_features.npy", mmap_mode="r")    # (n,646,384) fp16
    va_mm = np.load(RED / "val_features.npy", mmap_mode="r")
    train_feats = np.ascontiguousarray(tr_mm[:, :, :USE_DIM])     # ~680MB fp16
    val_feats = np.ascontiguousarray(va_mm[:, :, :USE_DIM])
    del tr_mm, va_mm
    train_meta = json.loads((RED / "train_meta.json").read_text())
    val_meta = json.loads((RED / "val_meta.json").read_text())
    train_masks_full = np.load(train_meta["mask_src"], mmap_mode="r")
    val_masks_full = np.load(val_meta["mask_src"], mmap_mode="r")
    logger.info(f"Train {train_feats.shape} | Val {val_feats.shape}")

    train_token_masks = torch.from_numpy(downsample_masks(train_masks_full).astype(np.int64))
    class_weights = median_freq_weights(train_token_masks.numpy()).to(DEVICE)

    head = EfficientSegHead(EMBED_DIM, NUM_CLASSES, TOKEN_H, TOKEN_W, hidden=HIDDEN, n_blocks=3).to(DEVICE)
    logger.info(f"Head params: {head.num_params():,}")

    ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
    dice = DiceLoss(NUM_CLASSES, weight=class_weights)
    opt = optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)

    n_train = train_feats.shape[0]
    best = 0.0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        head.train()
        perm = np.random.permutation(n_train)
        total, nb = 0.0, 0
        for s in range(0, n_train, BATCH):
            idx = perm[s:s+BATCH]
            f = torch.from_numpy(train_feats[idx].astype(np.float32)).to(DEVICE)
            m = train_token_masks[idx].to(DEVICE)
            logits = head(f)
            loss = ce(logits, m) + DICE_W * dice(logits, m)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1
        sched.step()
        avg = total / max(nb, 1)

        if epoch % EVAL_EVERY == 0 or epoch in (1, NUM_EPOCHS):
            iou, mean_all, mean_present = evaluate(head, val_feats, val_masks_full)
            tag = ""
            if mean_all > best:
                best = mean_all
                torch.save(head.state_dict(), OUT / "best_model.pth")
                tag = "  <-- best (saved)"
            history.append({"epoch": epoch, "loss": avg, "mean_iou": mean_all,
                            "mean_iou_present": mean_present,
                            "per_class_iou": {int(k): v for k, v in iou.items()}})
            logger.info(f"Ep {epoch:3d}/{NUM_EPOCHS} | loss {avg:.4f} | mIoU(11) {mean_all:.4f} | mIoU(present) {mean_present:.4f} | best {best:.4f}{tag}")
            logger.info("  " + ", ".join(f"{CLASS_NAMES[c]}={iou[c]:.2f}" for c in range(NUM_CLASSES)))
        else:
            logger.info(f"Ep {epoch:3d}/{NUM_EPOCHS} | loss {avg:.4f}")

    (OUT / "training_history.json").write_text(json.dumps(history, indent=2))
    head.load_state_dict(torch.load(OUT / "best_model.pth", map_location=DEVICE))
    iou, mean_all, mean_present = evaluate(head, val_feats, val_masks_full)
    result = {
        "baseline_mean_iou": BASELINE,
        "best_mean_iou": round(mean_all, 4),
        "best_mean_iou_present_classes": round(mean_present, 4),
        "improvement": round(mean_all - BASELINE, 4),
        "feature": "DINOv2 vits14 last-4 layers -> PCA 384 (97.5% var)",
        "per_class_iou": {CLASS_NAMES[c]: round(iou[c], 4) for c in range(NUM_CLASSES)},
    }
    (OUT / "final_results.json").write_text(json.dumps(result, indent=2))

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Baseline mIoU          : {BASELINE:.4f}")
    logger.info(f"Best mIoU (11 classes) : {mean_all:.4f}  ({mean_all - BASELINE:+.4f})")
    logger.info(f"Best mIoU (present)    : {mean_present:.4f}")
    for c in range(NUM_CLASSES):
        flag = "OK " if iou[c] >= 0.4 else "low"
        logger.info(f"  [{flag}] {CLASS_NAMES[c]:15s}: {iou[c]:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
