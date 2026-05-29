"""
Clean, memory-light evaluation of the best trained head on the validation set.

Loads the PCA-reduced val features (top-192 PCs) and the saved best_model.pth,
computes full-resolution per-class IoU, and writes final_results.json.
"""
import json, logging
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).parent
RED = PROJECT / "iou_boost_output" / "cache_red"
OUT = PROJECT / "iou_boost_output"
IMAGE_H, IMAGE_W, PATCH = 266, 476, 14
TOKEN_H, TOKEN_W = IMAGE_H // PATCH, IMAGE_W // PATCH
USE_DIM, HIDDEN, NUM_CLASSES, BASELINE = 192, 192, 11, 0.2794
DEVICE = torch.device("cpu")
CLASS_NAMES = {0:"Background",1:"Trees",2:"Lush Bushes",3:"Dry Grass",4:"Dry Bushes",
               5:"Ground Clutter",6:"Flowers",7:"Logs",8:"Rocks",9:"Landscape",10:"Sky"}


class EfficientSegHead(nn.Module):
    def __init__(self, in_channels, out_channels, token_h, token_w, hidden=192, n_blocks=3):
        super().__init__()
        self.H, self.W = token_h, token_w
        self.proj = nn.Sequential(nn.Conv2d(in_channels, hidden, 1), nn.GELU())
        blocks = []
        for _ in range(n_blocks):
            blocks.append(nn.Sequential(
                nn.Conv2d(hidden, hidden, 7, padding=3, groups=hidden), nn.GELU(),
                nn.Conv2d(hidden, hidden*2, 1), nn.GELU(), nn.Conv2d(hidden*2, hidden, 1)))
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
        x = self.norm(x); x = self.drop(x)
        return self.classifier(x)


def main():
    va_mm = np.load(RED / "val_features.npy", mmap_mode="r")
    val_feats = np.ascontiguousarray(va_mm[:, :, :USE_DIM]); del va_mm
    val_meta = json.loads((RED / "val_meta.json").read_text())
    val_masks = np.load(val_meta["mask_src"], mmap_mode="r")
    logger.info(f"Val features {val_feats.shape}")

    head = EfficientSegHead(USE_DIM, NUM_CLASSES, TOKEN_H, TOKEN_W, hidden=HIDDEN, n_blocks=3).to(DEVICE)
    head.load_state_dict(torch.load(OUT / "best_model.pth", map_location=DEVICE))
    head.eval()

    inter = np.zeros(NUM_CLASSES); union = np.zeros(NUM_CLASSES)
    n = val_feats.shape[0]
    with torch.no_grad():
        for s in range(0, n, 8):
            f = torch.from_numpy(np.asarray(val_feats[s:s+8], dtype=np.float32))
            m = np.asarray(val_masks[s:s+8], dtype=np.int64)
            logits = F.interpolate(head(f), size=(IMAGE_H, IMAGE_W), mode="bilinear", align_corners=False)
            preds = logits.argmax(1).numpy()
            for c in range(NUM_CLASSES):
                pm, tm = preds == c, m == c
                inter[c] += np.logical_and(pm, tm).sum()
                union[c] += np.logical_or(pm, tm).sum()
    iou = {c: (float(inter[c]/union[c]) if union[c] > 0 else 0.0) for c in range(NUM_CLASSES)}
    mean_all = float(np.mean(list(iou.values())))
    present = [c for c in range(NUM_CLASSES) if union[c] > 0]
    mean_present = float(np.mean([iou[c] for c in present])) if present else 0.0

    result = {
        "baseline_mean_iou": BASELINE,
        "best_mean_iou_11_classes": round(mean_all, 4),
        "best_mean_iou_present_classes": round(mean_present, 4),
        "improvement_11_classes": round(mean_all - BASELINE, 4),
        "feature": "DINOv2 vits14 last-4 layers -> PCA top-192 (of 384, ~97% var)",
        "per_class_iou": {CLASS_NAMES[c]: round(iou[c], 4) for c in range(NUM_CLASSES)},
    }
    (OUT / "final_results.json").write_text(json.dumps(result, indent=2))

    logger.info("=" * 70)
    logger.info("FINAL EVALUATION (validation set)")
    logger.info("=" * 70)
    logger.info(f"Baseline mIoU           : {BASELINE:.4f}")
    logger.info(f"Best mIoU (11 classes)  : {mean_all:.4f}  ({mean_all - BASELINE:+.4f})")
    logger.info(f"Best mIoU (present 10)  : {mean_present:.4f}")
    for c in range(NUM_CLASSES):
        flag = "OK " if iou[c] >= 0.4 else "low"
        logger.info(f"  [{flag}] {CLASS_NAMES[c]:15s}: {iou[c]:.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
