"""
Duality AI Offroad Segmentation - Visualization Script
Produces high-contrast colored overlays for any RGB + predicted mask pair.

Usage:
    python visualize.py --rgb_dir Offroad_Segmentation_testImages --pred_dir runs/test_out/predictions --out_dir runs/test_out/viz_hc
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CLASS_NAMES = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"
]

# High-contrast distinct colors
HC_COLORS = np.array([
    [0,   255,   0],   # Trees        - lime
    [0,   128,   0],   # Lush Bushes  - dark green
    [255, 255,   0],   # Dry Grass    - yellow
    [139,  69,  19],   # Dry Bushes   - saddle brown
    [255, 140,   0],   # Ground Clutter - dark orange
    [255,   0, 255],   # Flowers      - magenta
    [101,  67,  33],   # Logs         - brown
    [128, 128, 128],   # Rocks        - gray
    [255, 228, 181],   # Landscape    - moccasin
    [0,   191, 255],   # Sky          - deep sky blue
], dtype=np.uint8)


def find_images_in_dir(path: Path):
    return sorted(list(path.glob("*.png")) + list(path.glob("*.jpg")) + list(path.glob("*.jpeg")))


def resolve_rgb_dir(rgb_dir):
    root = Path(rgb_dir)
    if len(find_images_in_dir(root)) > 0:
        return root
    if (root / "Color_Images").exists():
        return root / "Color_Images"
    return root


def colorize(pred_arr):
    h, w = pred_arr.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in enumerate(HC_COLORS):
        color[pred_arr == i] = c
    return color


def main(args):
    rgb_dir  = resolve_rgb_dir(args.rgb_dir)
    pred_dir = Path(args.pred_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = find_images_in_dir(rgb_dir)
    if len(images) == 0:
        raise FileNotFoundError(f"No RGB images found in {rgb_dir}")
    patches = [mpatches.Patch(color=c/255, label=n)
               for c, n in zip(HC_COLORS, CLASS_NAMES)]

    for img_path in images:
        pred_path = pred_dir / img_path.name
        if not pred_path.exists():
            pred_path = pred_dir / (img_path.stem + ".png")
        if not pred_path.exists():
            print(f"  Skip: no prediction for {img_path.name}")
            continue

        rgb  = np.array(Image.open(img_path).convert("RGB"))
        pred = np.array(Image.open(pred_path))
        if pred.ndim == 3:
            pred = pred[:, :, 0]

        color = colorize(pred)
        # Resize color to match rgb if needed
        if color.shape[:2] != rgb.shape[:2]:
            color = np.array(Image.fromarray(color).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST))

        overlay = (rgb * 0.45 + color * 0.55).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(rgb);     axes[0].set_title("RGB");             axes[0].axis("off")
        axes[1].imshow(color);   axes[1].set_title("HC Segmentation"); axes[1].axis("off")
        axes[2].imshow(overlay); axes[2].set_title("Overlay");         axes[2].axis("off")
        fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8)
        plt.suptitle(img_path.name, fontsize=11)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.savefig(out_dir / (img_path.stem + "_hc.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {img_path.stem}_hc.png")

    print(f"\nDone! Saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_dir",  required=True, help="Folder of RGB images")
    parser.add_argument("--pred_dir", required=True, help="Folder of prediction masks")
    parser.add_argument("--out_dir",  default="runs/viz_hc", help="Output folder")
    args = parser.parse_args()
    main(args)
