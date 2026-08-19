"""FFT scores that separate cell sharpness from PMT-family exacerbation.

Kept in sync with suite2p ``lab/pipeline/mc_fft_metrics.py`` (same mask geometry
from defringe ``signature.json``). SUPPORT adds mean + per-frame evaluation for
defringed vs denoised stacks.

Pass (promote) on means: cell power up/flat, fringe power not up
(``cell_up_fringe_ok``). Box/tile grids are a separate fail — not in this mask.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

# Tight scoring mask (matches derippling / suite2p).
Y_PAD = 2
DC_EXCLUDE_R = 8
CELL_R_LO = 8
CELL_R_HI = 48

SANDBOX = (
    Path(r"F:\bPACNewData2026") / "PreProcessing Optimization" / "Level3b copy"
)
DEFAULT_SIGNATURE_ROOT = SANDBOX / "defringe_runs" / "v21_full_seeded500"


def load_signature(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_default_signature(letter, start: Path | None = None):
    """ChanA/B signature.json next to a sandbox run, else the v2.1 pack."""
    letter = str(letter).upper().replace("CHAN", "")
    if letter not in ("A", "B"):
        raise ValueError(f"channel letter must be A or B, got {letter!r}")
    name = f"Chan{letter}"
    rel = Path("defringe_runs") / "v21_full_seeded500" / name / "diagnostics" / "signature.json"
    if start is not None:
        for p in [Path(start), *Path(start).resolve().parents]:
            cand = p / rel
            if cand.exists():
                return cand
    packed = DEFAULT_SIGNATURE_ROOT / name / "diagnostics" / "signature.json"
    return packed if packed.exists() else None


def infer_channel_letter(*paths) -> str | None:
    """Return 'A' or 'B' from path names containing ChanA/ChanB."""
    for path in paths:
        if path is None:
            continue
        m = re.search(r"chan\s*([ab])", str(path), flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def _fx_select(width, fam, cx):
    fx = np.arange(width) - cx
    x_sel = np.zeros(width, dtype=bool)
    ranges = fam.get("fx_ranges_weight_gt_0.20") or fam.get("fx_ranges")
    if ranges:
        for lo, hi in ranges:
            x_sel |= (fx >= lo) & (fx <= hi)
        return x_sel
    xw = fam.get("x_weight")
    if xw is not None:
        xw = np.asarray(xw, dtype=float)
        if xw.size == width:
            return xw > 0.20
    return x_sel


def _family_qs(fam, tracking_blocks, family_idx):
    qs = {int(round(float(fam["q"])))}
    hi = fam.get("hi")
    if hi is not None:
        qs.add(int(round(float(hi))))
    for blk in tracking_blocks or []:
        if blk.get("q") is None:
            continue
        if blk.get("family", 0) not in (family_idx, None):
            continue
        qs.add(int(round(float(blk["q"]))))
    return sorted(qs)


def fringe_mask_from_signature(shape, sig, y_pad=Y_PAD, dc_r=DC_EXCLUDE_R):
    """Boolean FFT-centered mask covering the PMT family (and tracked q)."""
    h, w = int(shape[0]), int(shape[1])
    cy, cx = h // 2, w // 2
    mask = np.zeros((h, w), dtype=bool)
    blocks = sig.get("tracking_blocks") or []
    for i, fam in enumerate(sig.get("families") or []):
        x_sel = _fx_select(w, fam, cx)
        if not x_sel.any():
            continue
        for q in _family_qs(fam, blocks, i):
            for sgn in (-1, +1):
                yc = cy + sgn * q
                for yp in range(yc - y_pad, yc + y_pad + 1):
                    if 0 <= yp < h:
                        mask[yp, x_sel] = True
    yy, xx = np.ogrid[:h, :w]
    mask[(yy - cy) ** 2 + (xx - cx) ** 2 < dc_r**2] = False
    return mask


def cell_mask_from_fringe(shape, fringe, r_lo=CELL_R_LO, r_hi=CELL_R_HI):
    """Isotropic mid-band minus the fringe family (and DC)."""
    h, w = int(shape[0]), int(shape[1])
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    cell = (r2 >= r_lo**2) & (r2 <= r_hi**2) & (~np.asarray(fringe, dtype=bool))
    cell[cy, cx] = False
    return cell


def _power_spectrum(img):
    img = np.squeeze(np.asarray(img, dtype=np.float64))
    if img.ndim != 2:
        raise ValueError(f"expected 2D image, got shape {img.shape}")
    img = img - img.mean()
    spec = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(spec) ** 2
    cy, cx = power.shape[0] // 2, power.shape[1] // 2
    power[cy, cx] = 0.0
    return power


def vertical_ridge_energy(img, ky_cut=0.05):
    """Legacy |ky| half-plane fraction (too wide; kept for comparison only)."""
    power = _power_spectrum(img)
    ly = power.shape[0]
    cy = ly // 2
    ky = (np.arange(ly) - cy) / float(ly)
    ridge = power[np.abs(ky) > ky_cut, :].sum()
    tot = power.sum()
    return float(ridge / (tot + 1e-12))


def band_scores(img, fringe_mask, cell_mask):
    power = _power_spectrum(img)
    tot = float(power.sum())
    fringe_p = float(power[fringe_mask].sum())
    cell_p = float(power[cell_mask].sum())
    return {
        "fringe_power": fringe_p,
        "cell_power": cell_p,
        "total_power": tot,
        "fringe_frac": fringe_p / (tot + 1e-12),
        "cell_frac": cell_p / (tot + 1e-12),
    }


def _ratio(a, b):
    if b is None or b == 0 or not np.isfinite(b):
        return float("nan")
    return float(a / b)


def score_pair(pre, post, signature, signature_path=None):
    """Compare pre vs post means (defringed vs SUPPORT, or unreg vs reg)."""
    sig = signature if isinstance(signature, dict) else load_signature(signature)
    shape = np.squeeze(np.asarray(post)).shape[-2:]
    fringe = fringe_mask_from_signature(shape, sig)
    cell = cell_mask_from_fringe(shape, fringe)
    pre_s = band_scores(pre, fringe, cell)
    post_s = band_scores(post, fringe, cell)
    out = {
        "signature_path": None if signature_path is None else str(signature_path),
        "fringe_n_bins": int(fringe.sum()),
        "cell_n_bins": int(cell.sum()),
        "pre": pre_s,
        "post": post_s,
        # suite2p aliases
        "unreg": pre_s,
        "reg": post_s,
        "fringe_power_ratio": _ratio(post_s["fringe_power"], pre_s["fringe_power"]),
        "cell_power_ratio": _ratio(post_s["cell_power"], pre_s["cell_power"]),
        "fringe_frac_ratio": _ratio(post_s["fringe_frac"], pre_s["fringe_frac"]),
        "cell_frac_ratio": _ratio(post_s["cell_frac"], pre_s["cell_frac"]),
        "ridge_pre": vertical_ridge_energy(pre),
        "ridge_post": vertical_ridge_energy(post),
        "ridge_unreg": vertical_ridge_energy(pre),
        "ridge_reg": vertical_ridge_energy(post),
    }
    cell_up = out["cell_power_ratio"] > 1.02
    fringe_ok = out["fringe_power_ratio"] <= 1.05
    if cell_up and fringe_ok:
        out["verdict"] = "cell_up_fringe_ok"
    elif cell_up and not fringe_ok:
        out["verdict"] = "both_up"
    elif (not cell_up) and fringe_ok:
        out["verdict"] = "no_sharpen_fringe_ok"
    else:
        out["verdict"] = "fringe_up_cell_flat"
    return out


def center_align_stacks(pre, post):
    """Crop to common temporal length (center) for unequal SUPPORT cuts."""
    pre = np.asarray(pre)
    post = np.asarray(post)
    if pre.ndim != 3 or post.ndim != 3:
        raise ValueError(f"expected TYX stacks, got {pre.shape} vs {post.shape}")
    n = min(pre.shape[0], post.shape[0])
    if pre.shape[0] != n:
        off = (pre.shape[0] - n) // 2
        pre = pre[off : off + n]
    if post.shape[0] != n:
        off = (post.shape[0] - n) // 2
        post = post[off : off + n]
    return pre.astype(np.float32, copy=False), post.astype(np.float32, copy=False), int(n)


def score_frames(pre_stack, post_stack, signature, frame_stride=1, max_frames=None):
    """Per-frame fringe/cell power ratios (diagnostic; fringe more trusted)."""
    sig = signature if isinstance(signature, dict) else load_signature(signature)
    pre, post, n = center_align_stacks(pre_stack, post_stack)
    shape = pre.shape[1:]
    fringe = fringe_mask_from_signature(shape, sig)
    cell = cell_mask_from_fringe(shape, fringe)
    idxs = list(range(0, n, max(1, int(frame_stride))))
    if max_frames is not None:
        idxs = idxs[: int(max_frames)]

    fringe_ratios = []
    cell_ratios = []
    for i in idxs:
        pre_s = band_scores(pre[i], fringe, cell)
        post_s = band_scores(post[i], fringe, cell)
        fringe_ratios.append(_ratio(post_s["fringe_power"], pre_s["fringe_power"]))
        cell_ratios.append(_ratio(post_s["cell_power"], pre_s["cell_power"]))

    fringe_ratios = np.asarray(fringe_ratios, dtype=np.float64)
    cell_ratios = np.asarray(cell_ratios, dtype=np.float64)
    return {
        "n_frames_scored": int(len(idxs)),
        "frame_stride": int(frame_stride),
        "fringe_power_ratio_median": float(np.nanmedian(fringe_ratios)),
        "fringe_power_ratio_p90": float(np.nanpercentile(fringe_ratios, 90)),
        "fringe_power_ratio_mean": float(np.nanmean(fringe_ratios)),
        "frac_frames_fringe_ratio_gt_1": float(np.nanmean(fringe_ratios > 1.0)),
        "frac_frames_fringe_ratio_gt_1_05": float(np.nanmean(fringe_ratios > 1.05)),
        "cell_power_ratio_median": float(np.nanmedian(cell_ratios)),
        "cell_power_ratio_p90": float(np.nanpercentile(cell_ratios, 90)),
        "note": (
            "Per-frame cell ratios are noisy (shot noise vs denoise). "
            "Prefer mean verdict for cell sharpening; use frame fringe tails for exacerbation."
        ),
    }


def evaluate_denoise_pair(
    pre_stack,
    post_stack,
    signature,
    signature_path=None,
    frame_stride=1,
    max_frames=None,
    skip_frames=False,
):
    """Full SUPPORT QC: mean promote score + optional per-frame fringe tails."""
    sig = signature if isinstance(signature, dict) else load_signature(signature)
    pre, post, n = center_align_stacks(pre_stack, post_stack)
    mean_pair = score_pair(pre.mean(0), post.mean(0), sig, signature_path=signature_path)
    out = {
        "n_frames_compared": n,
        "pre_shape": list(np.asarray(pre_stack).shape),
        "post_shape": list(np.asarray(post_stack).shape),
        "mean": mean_pair,
        "verdict": mean_pair["verdict"],
        "promote_ok": mean_pair["verdict"] == "cell_up_fringe_ok",
        "boxes_note": "Tile/box grids are not scored by the PMT family mask — inspect visually.",
    }
    if not skip_frames:
        out["frames"] = score_frames(
            pre, post, sig, frame_stride=frame_stride, max_frames=max_frames
        )
    return out


def resolve_signature(channel=None, signature=None, search_from=None):
    """Return (sig_dict, path)."""
    if signature is not None:
        path = Path(signature)
        return load_signature(path), path
    letter = channel
    if letter is None and search_from is not None:
        letter = infer_channel_letter(search_from)
    if letter is None:
        raise ValueError(
            "Need --channel A|B or a path containing ChanA/ChanB, or --signature"
        )
    path = find_default_signature(letter, start=search_from)
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            f"No signature.json for Chan{letter}. Pass --signature explicitly."
        )
    return load_signature(path), Path(path)


def write_metrics_json(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path
