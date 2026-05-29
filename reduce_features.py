"""
One-time PCA reduction of cached 4-layer DINOv2 features: 1536 -> 384.

The 4-layer concatenated features (1536-d, ~5.4GB) don't fit in this machine's
scarce RAM, causing ~20 min/epoch from random disk reads. We reduce to 384-d with
PCA so the compact cache (~1.35GB) fits in RAM and epochs run in ~30s, while
retaining the most informative variance from the richer multi-layer features
(PCA keeps far more signal than a random projection).

Memory-light approach:
  * Fit PCA from a streaming covariance accumulated over a subsample of patches
    (covariance is only 1536x1536 = ~9MB; we never hold all patches in RAM).
  * Eigen-decompose the covariance, keep the top-384 eigenvectors.
  * Stream all train/val features through the projection, writing fp16 output.
"""

import json
import numpy as np
from pathlib import Path

SRC = Path("iou_boost_output/cache_l4")
DST = Path("iou_boost_output/cache_red")
IN_DIM = 1536
OUT_DIM = 384
CHUNK = 64                 # images per streaming chunk
FIT_IMAGES = 700           # images sampled to fit PCA (700*646 ~ 452k patch vectors)


def fit_pca(train_feats):
    n = train_feats.shape[0]
    sel = np.linspace(0, n - 1, min(FIT_IMAGES, n)).astype(int)
    # Streaming first/second moments
    count = 0
    mean = np.zeros(IN_DIM, dtype=np.float64)
    cov = np.zeros((IN_DIM, IN_DIM), dtype=np.float64)
    print(f"Fitting PCA from {len(sel)} images...")
    for k, i in enumerate(sel):
        X = np.asarray(train_feats[i], dtype=np.float32)      # (N, 1536)
        count += X.shape[0]
        mean += X.sum(axis=0)
        cov += X.T @ X
        if k % 100 == 0:
            print(f"  fit {k}/{len(sel)}")
    mean /= count
    cov = cov / count - np.outer(mean, mean)
    # Top eigenvectors (cov is symmetric PSD)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    comps = eigvecs[:, order[:OUT_DIM]].astype(np.float32)     # (1536, 384)
    var_kept = eigvals[order[:OUT_DIM]].sum() / eigvals.sum()
    print(f"PCA fitted. Variance retained by top {OUT_DIM}: {var_kept:.3f}")
    return mean.astype(np.float32), comps


def main():
    DST.mkdir(parents=True, exist_ok=True)
    train_feats = np.load(SRC / "train_features.npy", mmap_mode="r")
    mean, comps = fit_pca(train_feats)
    np.save(DST / "pca_mean.npy", mean)
    np.save(DST / "pca_comps.npy", comps)

    for split in ("train", "val"):
        feats = np.load(SRC / f"{split}_features.npy", mmap_mode="r")
        n, n_patches, dim = feats.shape
        assert dim == IN_DIM
        out_path = DST / f"{split}_features.npy"
        out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16,
                                        shape=(n, n_patches, OUT_DIM))
        print(f"[{split}] projecting {n} samples {IN_DIM}->{OUT_DIM} ...")
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            block = np.asarray(feats[start:end], dtype=np.float32)     # (b, N, 1536)
            b = block.shape[0]
            flat = block.reshape(-1, IN_DIM) - mean
            red = (flat @ comps).reshape(b, n_patches, OUT_DIM)
            out[start:end] = red.astype(np.float16)
            if (start // CHUNK) % 10 == 0:
                print(f"  {end}/{n}")
        out.flush()
        meta = {"n": n, "n_patches": n_patches, "embed_dim": OUT_DIM,
                "mask_src": str((SRC / f"{split}_masks.npy").resolve())}
        (DST / f"{split}_meta.json").write_text(json.dumps(meta))
        print(f"[{split}] done -> {out_path}")

    print("Reduction complete.")


if __name__ == "__main__":
    main()
