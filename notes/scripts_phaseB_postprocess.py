"""Phase B postprocess: compare patch_interval 16 vs Phase A interval 32 on pack_B 500fr."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio

BASE = Path(r"F:\bPACNewData2026\PreProcessing Optimization\Level3b copy")
TAG = BASE / "support_runs" / "phaseB_interval16_packB_500fr"
PHASE_A = BASE / "support_runs" / "packB_vs_raw_500fr"


def load(path: Path) -> np.ndarray:
    arr = skio.imread(str(path)).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected TYX, got {arr.shape} for {path}")
    return arr


def mean_img(stack: np.ndarray) -> np.ndarray:
    return stack.mean(axis=0)


def pnorm(img: np.ndarray, lo=1.0, hi=99.0) -> np.ndarray:
    a, b = np.percentile(img, [lo, hi])
    if b <= a:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - a) / (b - a), 0, 1).astype(np.float32)


def fourier_y_ridge_frac(img: np.ndarray, ky_thresh: float = 0.05) -> float:
    x = img.astype(np.float64) - img.mean()
    F = np.fft.fftshift(np.fft.fft2(x))
    P = np.abs(F) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(P.shape[0]))
    tot = P.sum()
    if tot <= 0:
        return float("nan")
    return float(P[np.abs(ky) > ky_thresh, :].sum() / tot)


def block_seam_score(img: np.ndarray, grid: int = 32) -> dict:
    h, w = img.shape
    scores_v, scores_h = [], []
    for x in range(grid, w, grid):
        left, right = img[:, x - 1], img[:, x]
        denom = 0.5 * (np.abs(left).mean() + np.abs(right).mean()) + 1e-6
        scores_v.append(float(np.abs(left - right).mean() / denom))
    for y in range(grid, h, grid):
        up, down = img[y - 1, :], img[y, :]
        denom = 0.5 * (np.abs(up).mean() + np.abs(down).mean()) + 1e-6
        scores_h.append(float(np.abs(up - down).mean() / denom))
    return {
        "grid": grid,
        "seam_mean": float(np.mean(scores_v + scores_h)) if (scores_v or scores_h) else float("nan"),
        "seam_v_mean": float(np.mean(scores_v)) if scores_v else float("nan"),
        "seam_h_mean": float(np.mean(scores_h)) if scores_h else float("nan"),
    }


def center_crop(stack: np.ndarray, n_out: int) -> np.ndarray:
    if stack.shape[0] == n_out:
        return stack
    off = (stack.shape[0] - n_out) // 2
    return stack[off : off + n_out]


def save_row(title: str, panels: list[tuple[str, np.ndarray]], out: Path):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, (lab, img) in zip(axes, panels):
        ax.imshow(pnorm(img), cmap="gray", vmin=0, vmax=1)
        ax.set_title(lab, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def process(ch: str) -> dict:
    pack = load(BASE / "defringe_runs" / "v21_sweep_500fr" / "accepted" / "pack_B" / f"{ch}_raw_500fr_v21.tif")
    s32 = load(PHASE_A / ch / f"{ch}_packB_500fr_support.tif")
    s16 = load(TAG / ch / f"{ch}_packB_500fr_support_i16.tif")
    n = min(pack.shape[0], s32.shape[0], s16.shape[0])
    pack_c, s32_c, s16_c = center_crop(pack, n), center_crop(s32, n), center_crop(s16, n)
    means = {
        "packB": mean_img(pack_c),
        "support_i32": mean_img(s32_c),
        "support_i16": mean_img(s16_c),
    }
    fi = n // 2
    # also seam on single frame (more sensitive to boxes)
    frame_metrics = {
        "frame_idx": fi,
        "seam_mean_i32": block_seam_score(s32_c[fi], grid=32)["seam_mean"],
        "seam_mean_i16": block_seam_score(s16_c[fi], grid=32)["seam_mean"],
        "seam16grid_i32": block_seam_score(s32_c[fi], grid=16)["seam_mean"],
        "seam16grid_i16": block_seam_score(s16_c[fi], grid=16)["seam_mean"],
    }
    save_row(
        f"{ch} temporal mean — pack_B | SUPPORT i32 | SUPPORT i16",
        [
            ("pack_B mean", means["packB"]),
            ("SUPPORT i32 (Phase A)", means["support_i32"]),
            ("SUPPORT i16 (Phase B)", means["support_i16"]),
        ],
        TAG / "montages" / f"{ch}_mean_i32_vs_i16.png",
    )
    save_row(
        f"{ch} frame {fi} — pack_B | SUPPORT i32 | SUPPORT i16",
        [
            ("pack_B", pack_c[fi]),
            ("SUPPORT i32", s32_c[fi]),
            ("SUPPORT i16", s16_c[fi]),
        ],
        TAG / "montages" / f"{ch}_frame{fi:04d}_i32_vs_i16.png",
    )
    # difference |i16-i32| on mean for stitch change visibility
    diff = np.abs(means["support_i16"] - means["support_i32"])
    save_row(
        f"{ch} |SUPPORT i16 − i32| on mean",
        [("abs diff", diff)],
        TAG / "montages" / f"{ch}_absdiff_i16_minus_i32_mean.png",
    )
    return {
        "channel": ch,
        "n_frames_compared": int(n),
        "shapes": {"packB": list(pack.shape), "support_i32": list(s32.shape), "support_i16": list(s16.shape)},
        "ridge_frac_ky_gt_0.05": {k: fourier_y_ridge_frac(v) for k, v in means.items()},
        "block_seam_mean_img": {k: block_seam_score(v, grid=32) for k, v in means.items()},
        "block_seam_single_frame": frame_metrics,
        "mean_abs_diff_i16_vs_i32": float(diff.mean()),
        "mean_abs_diff_i16_vs_i32_over_i32_std": float(diff.mean() / (means["support_i32"].std() + 1e-6)),
    }


def main():
    metrics = {ch: process(ch) for ch in ("ChanA", "ChanB")}
    out = TAG / "metrics" / "phaseB_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {out}")
    for ch, m in metrics.items():
        print(ch, "ridge", m["ridge_frac_ky_gt_0.05"])
        print(ch, "seam_frame", m["block_seam_single_frame"])
        print(ch, "mean_abs_diff_norm", m["mean_abs_diff_i16_vs_i32_over_i32_std"])


if __name__ == "__main__":
    main()
