"""
IoU Boost Pipeline (feature-cached, multi-layer, low-memory) for CPU.

Why this design (tuned for an 8GB-RAM, CPU-only machine):

  1. FROZEN backbone => features are constant. We extract them ONCE and cache to
     disk (iou_boost_output/cache/). No backbone forward passes during training.

  2. RICH features: we concatenate the LAST 4 DINOv2 transformer layers
     (4 x 384 = 1536-d) via get_intermediate_layers. This is the standard DINOv2
     dense-prediction recipe and is substantially stronger than using only the
     final layer - the key to pushing mean IoU past 0.5.

  3. LOW MEMORY: training reads each mini-batch straight from the on-disk memmap
     (np.load(..., mmap_mode='r')). Nothing large is held in RAM, so the machine
     never swaps (the earlier slowdowns were caused by loading ~1.4GB into the
     ~0.8GB of free RAM).

  4. FAST epochs: the head is trained at the 19x34 token grid (loss computed
     there), while IoU is still EVALUATED at full 266x476 resolution (bilinear
     logit upsampling, matching iou_pipeline/analyzer.py) so reported numbers are
     real and the best model is chosen on true validation mean IoU.

Loss = weighted CrossEntropy + soft Dice (Dice directly optimizes overlap, which
helps the rare classes). Class weights use median-frequency balancing.

Note on Background: this dataset contains effectively zero Background pixels, so
its IoU is structurally 0 (the original 0.2794 baseline saw the same). We report
both the 11-class mean (comparable to the baseline) and the present-class mean.

Target: boost mean IoU from baseline 0.2794 to >= 0.5.
"""

import os
import sys
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    torch.set_num_threads(os.cpu_count() or 4)
except Exception:
    pass

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Config:
    train_path = Path(r"C:\Users\vijay\OneDrive\Desktop\MERN Stack\Project1_Img_url\segmentation_hackathon\data\Offroad_Segmentation_Training_Dataset\train")
    val_path = Path(r"C:\Users\vijay\OneDrive\Desktop\MERN Stack\Project1_Img_url\segmentation_hackathon\data\Offroad_Segmentation_Training_Dataset\val")
    output_dir = project_root / "iou_boost_output"
    cache_dir = output_dir / "cache_l4"     # separate cache for 4-layer features

    image_height = 266
    image_width = 476
    patch_size = 14
    layer_dim = 384            # dinov2_vits14 per-layer dim
    n_layers = 4               # concatenate last 4 layers
    embed_dim = layer_dim * n_layers   # 1536
    num_classes = 11

    extract_batch_size = 8

    num_epochs = 60
    batch_size = 16
    learning_rate = 2e-3
    weight_decay = 1e-2
    label_smoothing = 0.05
    hidden_channels = 192
    dice_weight = 1.0

    eval_every = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


VALUE_MAP = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 600: 6, 700: 7, 800: 8, 7100: 9, 10000: 10}
CLASS_NAMES = {
    0: "Background", 1: "Trees", 2: "Lush Bushes", 3: "Dry Grass", 4: "Dry Bushes",
    5: "Ground Clutter", 6: "Flowers", 7: "Logs", 8: "Rocks", 9: "Landscape", 10: "Sky",
}
DINOV2_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DINOV2_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TOKEN_H = Config.image_height // Config.patch_size   # 19
TOKEN_W = Config.image_width // Config.patch_size     # 34


class EfficientSegHead(nn.Module):
    """CPU-friendly ConvNeXt-style head: cheap 1x1 projection then depthwise 7x7
    + pointwise ConvNeXt residual blocks, then a 1x1 classifier."""

    def __init__(self, in_channels, out_channels, token_h, token_w, hidden=192, n_blocks=3):
        super().__init__()
        self.H, self.W = token_h, token_w
        self.proj = nn.Sequential(nn.Conv2d(in_channels, hidden, kernel_size=1), nn.GELU())
        blocks = []
        for _ in range(n_blocks):
            blocks.append(nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=7, padding=3, groups=hidden),
                nn.GELU(),
                nn.Conv2d(hidden, hidden * 2, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden * 2, hidden, kernel_size=1),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.GroupNorm(1, hidden)
        self.drop = nn.Dropout2d(0.1)
        self.classifier = nn.Conv2d(hidden, out_channels, kernel_size=1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.proj(x)
        for blk in self.blocks:
            x = x + blk(x)
        x = self.norm(x)
        x = self.drop(x)
        return self.classifier(x)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DiceLoss(nn.Module):
    """Soft multi-class Dice loss; ignores classes absent from the batch."""

    def __init__(self, num_classes, weight=None, eps=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.register_buffer("cls_weight", weight if weight is not None else torch.ones(num_classes))

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        tgt = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * tgt).sum(dims)
        denom = probs.sum(dims) + tgt.sum(dims)
        dice = (2 * inter + self.eps) / (denom + self.eps)
        present = tgt.sum(dims) > 0
        w = self.cls_weight.to(dice.device) * present.float()
        if w.sum() == 0:
            return 1.0 - dice.mean()
        return 1.0 - (dice * w).sum() / w.sum()


def list_pairs(split_path: Path):
    img_dir = split_path / "Color_Images"
    mask_dir = split_path / "Segmentation"
    ids = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")])
    pairs = []
    for img_path in ids:
        mask_path = mask_dir / img_path.name
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


def load_image_tensor(img_path: Path) -> np.ndarray:
    img = Image.open(img_path).convert("RGB").resize((Config.image_width, Config.image_height), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - DINOV2_MEAN) / DINOV2_STD
    return np.transpose(arr, (2, 0, 1))


def load_mask_full(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path)
    arr = np.array(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    out = np.zeros_like(arr, dtype=np.uint8)
    for raw, cid in VALUE_MAP.items():
        out[arr == raw] = cid
    out = np.array(Image.fromarray(out).resize((Config.image_width, Config.image_height), Image.NEAREST), dtype=np.uint8)
    return out


def extract_features(split: str, pairs, backbone):
    """Extract last-4-layer concatenated patch tokens (1536-d) and full masks."""
    n = len(pairs)
    n_patches = TOKEN_H * TOKEN_W
    feat_path = Config.cache_dir / f"{split}_features.npy"
    mask_path = Config.cache_dir / f"{split}_masks.npy"
    meta_path = Config.cache_dir / f"{split}_meta.json"

    if feat_path.exists() and mask_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("n") == n and meta.get("embed_dim") == Config.embed_dim:
            logger.info(f"[{split}] Using existing feature cache ({n} samples, dim {Config.embed_dim}).")
            return feat_path, mask_path

    logger.info(f"[{split}] Extracting last-{Config.n_layers}-layer features for {n} samples (one-time)...")
    Config.cache_dir.mkdir(parents=True, exist_ok=True)
    feats = np.lib.format.open_memmap(feat_path, mode="w+", dtype=np.float16, shape=(n, n_patches, Config.embed_dim))
    masks = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.uint8, shape=(n, Config.image_height, Config.image_width))

    backbone.eval()
    bs = Config.extract_batch_size
    with torch.no_grad():
        for start in tqdm(range(0, n, bs), desc=f"{split} feats"):
            batch_pairs = pairs[start:start + bs]
            imgs = np.stack([load_image_tensor(p[0]) for p in batch_pairs])
            img_t = torch.from_numpy(imgs).to(Config.device)
            layers = backbone.get_intermediate_layers(img_t, n=Config.n_layers, return_class_token=False)
            tokens = torch.cat(layers, dim=-1)        # (B, N, 1536)
            feats[start:start + len(batch_pairs)] = tokens.cpu().numpy().astype(np.float16)
            for i, p in enumerate(batch_pairs):
                masks[start + i] = load_mask_full(p[1])
    feats.flush()
    masks.flush()
    meta_path.write_text(json.dumps({"n": n, "n_patches": n_patches, "embed_dim": Config.embed_dim}))
    logger.info(f"[{split}] Feature cache written.")
    return feat_path, mask_path


def downsample_masks_to_tokens(masks_mm) -> np.ndarray:
    n = masks_mm.shape[0]
    out = np.zeros((n, TOKEN_H, TOKEN_W), dtype=np.uint8)
    for i in range(n):
        out[i] = np.array(Image.fromarray(masks_mm[i]).resize((TOKEN_W, TOKEN_H), Image.NEAREST), dtype=np.uint8)
    return out


def median_frequency_weights(token_masks: np.ndarray) -> torch.Tensor:
    counts = np.bincount(token_masks.ravel(), minlength=Config.num_classes).astype(np.float64)[:Config.num_classes]
    present = counts > 0
    freq = counts / counts.sum()
    med = np.median(freq[present])
    weights = np.ones(Config.num_classes, dtype=np.float64)
    weights[present] = med / freq[present]
    weights = np.clip(weights, 0.3, 6.0)
    logger.info("Median-frequency class weights:")
    for i in range(Config.num_classes):
        logger.info(f"  {CLASS_NAMES[i]:15s}: {weights[i]:.3f}  (pixels={int(counts[i])})")
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_full_res(head, feats_mm, masks_mm, device, batch_size=8):
    head.eval()
    inter = np.zeros(Config.num_classes)
    union = np.zeros(Config.num_classes)
    n = feats_mm.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            f = np.asarray(feats_mm[start:start + batch_size], dtype=np.float32)
            m = np.asarray(masks_mm[start:start + batch_size], dtype=np.int64)
            logits = head(torch.from_numpy(f).to(device))
            logits = F.interpolate(logits, size=(Config.image_height, Config.image_width), mode="bilinear", align_corners=False)
            preds = logits.argmax(dim=1).cpu().numpy()
            for c in range(Config.num_classes):
                pm = preds == c
                tm = m == c
                inter[c] += np.logical_and(pm, tm).sum()
                union[c] += np.logical_or(pm, tm).sum()
    iou = {c: (float(inter[c] / union[c]) if union[c] > 0 else 0.0) for c in range(Config.num_classes)}
    mean_iou = float(np.mean(list(iou.values())))
    present = [c for c in range(Config.num_classes) if union[c] > 0]
    mean_iou_present = float(np.mean([iou[c] for c in present])) if present else 0.0
    return iou, mean_iou, mean_iou_present


def main():
    logger.info("=" * 80)
    logger.info("IoU BOOST PIPELINE (multi-layer features, low-memory)")
    logger.info("=" * 80)
    logger.info(f"Device: {Config.device} | token grid: {TOKEN_H}x{TOKEN_W} | feat dim: {Config.embed_dim}")
    Config.output_dir.mkdir(exist_ok=True)

    train_pairs = list_pairs(Config.train_path)
    val_pairs = list_pairs(Config.val_path)
    logger.info(f"Train samples: {len(train_pairs)} | Val samples: {len(val_pairs)}")

    # --- Phase A: feature extraction (cached to disk) ---
    logger.info("Loading DINOv2 backbone (frozen)...")
    backbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14", skip_validation=True)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval().to(Config.device)

    train_feat_path, train_mask_path = extract_features("train", train_pairs, backbone)
    val_feat_path, val_mask_path = extract_features("val", val_pairs, backbone)
    del backbone

    # Memory-light: keep everything on disk, read batches via memmap.
    train_feats = np.load(train_feat_path, mmap_mode="r")
    train_masks_full = np.load(train_mask_path, mmap_mode="r")
    val_feats = np.load(val_feat_path, mmap_mode="r")
    val_masks_full = np.load(val_mask_path, mmap_mode="r")

    logger.info("Downsampling train masks to token grid...")
    train_token_masks = downsample_masks_to_tokens(train_masks_full)
    class_weights = median_frequency_weights(train_token_masks).to(Config.device)
    train_token_masks_t = torch.from_numpy(train_token_masks.astype(np.int64))

    head = EfficientSegHead(
        in_channels=Config.embed_dim, out_channels=Config.num_classes,
        token_h=TOKEN_H, token_w=TOKEN_W, hidden=Config.hidden_channels, n_blocks=3,
    ).to(Config.device)
    logger.info(f"Head trainable params: {head.get_num_params():,}")

    ce_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=Config.label_smoothing)
    dice_criterion = DiceLoss(Config.num_classes, weight=class_weights)
    optimizer = optim.AdamW(head.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.num_epochs)

    n_train = train_feats.shape[0]
    best_mean_iou = 0.0
    history = []

    logger.info("=" * 80)
    logger.info("Training head on cached features (disk memmap batches)")
    logger.info("=" * 80)

    for epoch in range(1, Config.num_epochs + 1):
        head.train()
        perm = np.random.permutation(n_train)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n_train, Config.batch_size):
            idx = np.sort(perm[start:start + Config.batch_size])   # sorted -> fast memmap read
            feats = torch.from_numpy(np.asarray(train_feats[idx], dtype=np.float32)).to(Config.device)
            masks = train_token_masks_t[idx].to(Config.device)
            logits = head(feats)
            loss = ce_criterion(logits, masks) + Config.dice_weight * dice_criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        do_eval = (epoch % Config.eval_every == 0) or (epoch == Config.num_epochs) or (epoch == 1)
        if do_eval:
            iou, mean_iou, mean_iou_present = evaluate_full_res(head, val_feats, val_masks_full, Config.device)
            improved = ""
            if mean_iou > best_mean_iou:
                best_mean_iou = mean_iou
                torch.save(head.state_dict(), Config.output_dir / "best_model.pth")
                improved = "  <-- new best (saved)"
            history.append({"epoch": epoch, "train_loss": avg_loss, "mean_iou": mean_iou,
                            "mean_iou_present": mean_iou_present,
                            "per_class_iou": {int(k): v for k, v in iou.items()}})
            logger.info(f"Epoch {epoch:3d}/{Config.num_epochs} | loss {avg_loss:.4f} | val mIoU(11) {mean_iou:.4f} | mIoU(present) {mean_iou_present:.4f} | best {best_mean_iou:.4f}{improved}")
            logger.info("  Per-class IoU: " + ", ".join(f"{CLASS_NAMES[c]}={iou[c]:.2f}" for c in range(Config.num_classes)))
        else:
            logger.info(f"Epoch {epoch:3d}/{Config.num_epochs} | loss {avg_loss:.4f}")

    # --- Final report on best model ---
    (Config.output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    head.load_state_dict(torch.load(Config.output_dir / "best_model.pth", map_location=Config.device))
    iou, mean_iou, mean_iou_present = evaluate_full_res(head, val_feats, val_masks_full, Config.device)
    result = {
        "baseline_mean_iou": 0.2794,
        "best_mean_iou": round(mean_iou, 4),
        "best_mean_iou_present_classes": round(mean_iou_present, 4),
        "improvement": round(mean_iou - 0.2794, 4),
        "feature": f"DINOv2 vits14 last-{Config.n_layers} layers ({Config.embed_dim}-d)",
        "per_class_iou": {CLASS_NAMES[c]: round(iou[c], 4) for c in range(Config.num_classes)},
    }
    (Config.output_dir / "final_results.json").write_text(json.dumps(result, indent=2))

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Baseline mean IoU      : 0.2794")
    logger.info(f"Best mean IoU (11 cls) : {mean_iou:.4f}")
    logger.info(f"Best mean IoU (present): {mean_iou_present:.4f}")
    logger.info(f"Improvement            : {mean_iou - 0.2794:+.4f}")
    logger.info("Per-class IoU (validation):")
    for c in range(Config.num_classes):
        flag = "OK " if iou[c] >= 0.4 else "low"
        logger.info(f"  [{flag}] {CLASS_NAMES[c]:15s}: {iou[c]:.4f}")
    logger.info(f"Best model:  {Config.output_dir / 'best_model.pth'}")
    logger.info(f"Results JSON: {Config.output_dir / 'final_results.json'}")


if __name__ == "__main__":
    main()
