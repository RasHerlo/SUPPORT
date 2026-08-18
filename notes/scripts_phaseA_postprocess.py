"""Phase A postprocess: montages + simple fringe/block metrics for packB_vs_raw_500fr."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio

TAG = Path(r"F:\bPACNewData2026\PreProcessing Optimization\Level3b copy\support_runs\packB_vs_raw_500fr")
BASE = Path(r"F:\bPACNewData2026\PreProcessing Optimization\Level3b copy")


def load(path: Path) -> np.ndarray:
    arr = skio.imread(str(path)).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected TYX stack, got {arr.shape} for {path}")
    return arr


def mean_img(stack: np.ndarray) -> np.ndarray:
    return stack.mean(axis=0)


def pnorm(img: np.ndarray, lo=1.0, hi=99.0) -> np.ndarray:
    a, b = np.percentile(img, [lo, hi])
    if b <= a:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - a) / (b - a), 0, 1).astype(np.float32)


def fourier_y_ridge_frac(img: np.ndarray, ky_thresh: float = 0.05) -> float:
    """Fraction of 2D-FFT power at |ky| > ky_thresh (suite2p-style horizontal ridge band)."""
    x = img.astype(np.float64)
    x = x - x.mean()
    F = np.fft.fftshift(np.fft.fft2(x))
    P = np.abs(F) ** 2
    h, w = P.shape
    ky = np.fft.fftshift(np.fft.fftfreq(h))
    row_mask = np.abs(ky) > ky_thresh
    tot = P.sum()
    if tot <= 0:
        return float("nan")
    return float(P[row_mask, :].sum() / tot)


def block_seam_score(img: np.ndarray, grid: int = 32) -> dict:
    """Discontinuity across nominal SUPPORT patch boundaries (xy patch 64, interval 32)."""
    h, w = img.shape
    scores_v, scores_h = [], []
    # vertical seams (constant x = k*grid)
    for x in range(grid, w, grid):
        left = img[:, x - 1]
        right = img[:, x]
        denom = 0.5 * (np.abs(left).mean() + np.abs(right).mean()) + 1e-6
        scores_v.append(float(np.abs(left - right).mean() / denom))
    # horizontal seams (constant y = k*grid)
    for y in range(grid, h, grid):
        up = img[y - 1, :]
        down = img[y, :]
        denom = 0.5 * (np.abs(up).mean() + np.abs(down).mean()) + 1e-6
        scores_h.append(float(np.abs(up - down).mean() / denom))
    return {
        "grid": grid,
        "seam_v_mean": float(np.mean(scores_v)) if scores_v else float("nan"),
        "seam_h_mean": float(np.mean(scores_h)) if scores_h else float("nan"),
        "seam_mean": float(np.mean(scores_v + scores_h)) if (scores_v or scores_h) else float("nan"),
    }


def save_montage(channel: str, panels: list[tuple[str, np.ndarray]], frame_idx: int, out_path: Path):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(pnorm(img), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{channel}  frame {frame_idx}  (display %-ile stretch)", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_mean_row(channel: str, panels: list[tuple[str, np.ndarray]], out_path: Path):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(pnorm(img), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{channel}  temporal mean", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def process_channel(ch: str, paths: dict) -> dict:
    raw = load(paths["raw"])
    pack = load(paths["packB"])
    s_raw = load(paths["support_raw"])
    s_pack = load(paths["support_packB"])

    # Align lengths: SUPPORT trims temporal edges; compare on overlapping central region.
    # Prefer matching SUPPORT length by trimming inputs to SUPPORT frame count centered.
    n = min(raw.shape[0], pack.shape[0], s_raw.shape[0], s_pack.shape[0])
    # If SUPPORT is shorter, take the middle n frames of inputs
    def center_crop(stack, n_out):
        if stack.shape[0] == n_out:
            return stack
        off = (stack.shape[0] - n_out) // 2
        return stack[off : off + n_out]

    raw_c = center_crop(raw, n)
    pack_c = center_crop(pack, n)
    # SUPPORT outputs may already be trimmed; if longer, center too
    s_raw_c = center_crop(s_raw, n)
    s_pack_c = center_crop(s_pack, n)

    means = {
        "raw": mean_img(raw_c),
        "packB": mean_img(pack_c),
        "support_raw": mean_img(s_raw_c),
        "support_packB": mean_img(s_pack_c),
    }

    # Pick a mid frame for montage
    fi = n // 2
    save_montage(
        ch,
        [
            ("raw", raw_c[fi]),
            ("pack_B", pack_c[fi]),
            ("SUPPORT(raw)", s_raw_c[fi]),
            ("SUPPORT(pack_B)", s_pack_c[fi]),
        ],
        fi,
        TAG / "montages" / f"{ch}_frame{fi:04d}.png",
    )
    save_mean_row(
        ch,
        [
            ("raw mean", means["raw"]),
            ("pack_B mean", means["packB"]),
            ("SUPPORT(raw) mean", means["support_raw"]),
            ("SUPPORT(pack_B) mean", means["support_packB"]),
        ],
        TAG / "montages" / f"{ch}_mean.png",
    )

    metrics = {
        "channel": ch,
        "n_frames_compared": int(n),
        "shapes": {
            "raw": list(raw.shape),
            "packB": list(pack.shape),
            "support_raw": list(s_raw.shape),
            "support_packB": list(s_pack.shape),
        },
        "ridge_frac_ky_gt_0.05": {
            k: fourier_y_ridge_frac(v) for k, v in means.items()
        },
        "block_seam": {k: block_seam_score(v, grid=32) for k, v in means.items()},
    }
    # Amplification ratios (output / input) on matching pairs
    metrics["ridge_ratio_support_over_input"] = {
        "support_raw_vs_raw": metrics["ridge_frac_ky_gt_0.05"]["support_raw"]
        / max(metrics["ridge_frac_ky_gt_0.05"]["raw"], 1e-12),
        "support_packB_vs_packB": metrics["ridge_frac_ky_gt_0.05"]["support_packB"]
        / max(metrics["ridge_frac_ky_gt_0.05"]["packB"], 1e-12),
    }
    return metrics


def main():
    jobs = {
        "ChanA": {
            "raw": BASE / "inputs" / "slices_500fr" / "raw" / "ChanA_raw_500fr.tif",
            "packB": BASE / "defringe_runs" / "v21_sweep_500fr" / "accepted" / "pack_B" / "ChanA_raw_500fr_v21.tif",
            "support_raw": TAG / "ChanA" / "ChanA_raw_500fr_support.tif",
            "support_packB": TAG / "ChanA" / "ChanA_packB_500fr_support.tif",
        },
        "ChanB": {
            "raw": BASE / "inputs" / "slices_500fr" / "raw" / "ChanB_raw_500fr.tif",
            "packB": BASE / "defringe_runs" / "v21_sweep_500fr" / "accepted" / "pack_B" / "ChanB_raw_500fr_v21.tif",
            "support_raw": TAG / "ChanB" / "ChanB_raw_500fr_support.tif",
            "support_packB": TAG / "ChanB" / "ChanB_packB_500fr_support.tif",
        },
    }
    for ch, paths in jobs.items():
        for k, p in paths.items():
            if not p.is_file():
                raise FileNotFoundError(f"Missing {ch} {k}: {p}")

    all_metrics = {ch: process_channel(ch, paths) for ch, paths in jobs.items()}
    out_json = TAG / "metrics" / "phaseA_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Wrote {out_json}")
    for ch, m in all_metrics.items():
        print(ch, "ridge", m["ridge_frac_ky_gt_0.05"])
        print(ch, "amplification", m["ridge_ratio_support_over_input"])
        print(ch, "seam_mean", {k: v["seam_mean"] for k, v in m["block_seam"].items()})


if __name__ == "__main__":
    main()
