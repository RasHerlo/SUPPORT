"""Full-stack v21 + model_10 baseline: montages + ridge/seam metrics."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio

BASE = Path(r"F:\bPACNewData2026\PreProcessing Optimization\Level3b copy")
TAG = BASE / "support_runs" / "fullstack_v21_model10"


def load(path: Path) -> np.ndarray:
    arr = skio.imread(str(path)).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected TYX, got {arr.shape} for {path}")
    return arr


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


def block_seam_score(img: np.ndarray, grid: int = 32) -> float:
    h, w = img.shape
    scores = []
    for x in range(grid, w, grid):
        left, right = img[:, x - 1], img[:, x]
        denom = 0.5 * (np.abs(left).mean() + np.abs(right).mean()) + 1e-6
        scores.append(float(np.abs(left - right).mean() / denom))
    for y in range(grid, h, grid):
        up, down = img[y - 1, :], img[y, :]
        denom = 0.5 * (np.abs(up).mean() + np.abs(down).mean()) + 1e-6
        scores.append(float(np.abs(up - down).mean() / denom))
    return float(np.mean(scores)) if scores else float("nan")


def center_crop(stack: np.ndarray, n_out: int) -> np.ndarray:
    if stack.shape[0] == n_out:
        return stack
    off = (stack.shape[0] - n_out) // 2
    return stack[off : off + n_out]


def save_row(title: str, panels: list[tuple[str, np.ndarray]], out: Path):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.5))
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
    inp = load(BASE / "inputs" / "defringed_v21" / ch / f"{ch}_stk_defringed_v21.tif")
    out = load(TAG / ch / f"{ch}_stk_defringed_v21_support.tif")
    n = min(inp.shape[0], out.shape[0])
    inp_c, out_c = center_crop(inp, n), center_crop(out, n)
    mean_in, mean_out = inp_c.mean(0), out_c.mean(0)
    fi = n // 2
    # also a late frame and an early frame
    frames = {"early": n // 8, "mid": fi, "late": (7 * n) // 8}
    save_row(
        f"{ch} temporal mean — defringed_v21 | SUPPORT model_10",
        [("defringed_v21 mean", mean_in), ("SUPPORT mean", mean_out)],
        TAG / "montages" / f"{ch}_mean.png",
    )
    for name, idx in frames.items():
        save_row(
            f"{ch} frame {idx} ({name}) — defringed_v21 | SUPPORT",
            [(f"v21 f{idx}", inp_c[idx]), (f"SUPPORT f{idx}", out_c[idx])],
            TAG / "montages" / f"{ch}_frame{idx:04d}_{name}.png",
        )
    ridge_in = fourier_y_ridge_frac(mean_in)
    ridge_out = fourier_y_ridge_frac(mean_out)
    return {
        "channel": ch,
        "shapes": {"input": list(inp.shape), "support": list(out.shape), "compared_n": int(n)},
        "ridge_frac_mean": {"defringed_v21": ridge_in, "support": ridge_out},
        "ridge_amp_support_over_v21": ridge_out / max(ridge_in, 1e-12),
        "seam_mean_img": {
            "defringed_v21": block_seam_score(mean_in),
            "support": block_seam_score(mean_out),
        },
        "seam_mid_frame": {
            "defringed_v21": block_seam_score(inp_c[fi]),
            "support": block_seam_score(out_c[fi]),
        },
    }


def main():
    metrics = {ch: process(ch) for ch in ("ChanA", "ChanB")}
    out = TAG / "metrics" / "fullstack_v21_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {out}")
    for ch, m in metrics.items():
        print(ch, m["shapes"], "amp", m["ridge_amp_support_over_v21"], "ridge", m["ridge_frac_mean"])


if __name__ == "__main__":
    main()
