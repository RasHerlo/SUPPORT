# SUPPORT optimization status (for overview repo)

**Collected by:** https://github.com/RasHerlo/figure_for_cAMP_Neu_paper  
**Repo:** https://github.com/RasHerlo/SUPPORT  
**Sandbox:** `F:\bPACNewData2026\PreProcessing Optimization\Level3b copy`  
**Manifest:** [`optimization_manifest.json`](optimization_manifest.json)  
**Last updated:** 2026-09-03  

**Session:** awaiting user QC — Haj Grant v4 bakeoff scored; old models win FFT gate (`SESSION.md`)

---

## Failure modes

1. Blocks / patches after denoise  
2. Fringe exacerbation after denoise  

---

## Current approach

| Field | Value |
|---|---|
| **ID** | `support_v22_thorlabs_bakeoff` |
| **ChanA model (new)** | `F:\SUPPORT trained\ChanA_trained\20260821_003756\model_10.pth` |
| **ChanB model (new)** | `F:\SUPPORT trained\ChanB_trained\20260821_112325\model_10.pth` |
| **Baseline models** | `…\20250130_131021` / `…\20250131_090838` |
| **Inference (code default)** | `patch=[61,64,64]`, `interval=[1,32,32]`, `bs_size=3`, mirror, stitch=hard |
| **Inference (stitch winner)** | `patch=[61,128,128]`, `interval=[1,64,64]` hard — not yet the code default |
| **Bakeoff tag** | `support_runs/bakeoff_v22_Level3b/` |
| **Status** | New models preferred on holdout; Haj Grant ChanB boxes are 64-px infer geometry. Prefer hard **128/64**. |

### Headline (full stack)

| Channel | Model | signature verdict | cell_r | fringe_r | Visual |
|---|---|---|---:|---:|---|
| ChanA | `...\20250130_131021\model_10.pth` | `no_sharpen_fringe_ok` | 0.48 | 0.49 | Strong tile grid on mean |
| ChanB | `...\20250131_090838\model_10.pth` | `no_sharpen_fringe_ok` | 0.28 | 0.24 | Tile grid + edge banding |

Legacy `|ky|>0.05` ChanA “1.59× amp” is **superseded** — signature family power **falls**.
Metrics: `support_runs/fullstack_v21_model10/metrics/Chan*_signature_fft_metrics.json`

Default pipelines now write these scores automatically after denoise
(`src/utils/fft_metrics.py`, `--no_score` to skip).


---

## Attempts log

| When | Attempt | Result | Artifacts |
|---|---|---|---|
| 2026-08-18 | Phase A 500fr pack_B vs raw | ChanA fringe amp; ChanB pack_B helps | `packB_vs_raw_500fr/` |
| 2026-08-18 | Phase B interval 16 | Demoted — no box gain | `phaseB_interval16_packB_500fr/` |
| 2026-08-19 | **Full-stack model_10 on defringed_v21** | ChanA amp **1.59×**; boxes clear on A/B means | `fullstack_v21_model10/` |
| 2026-08-20–21 | THORLABS ChanA/B retrain on defringed_v22 | New `model_10` pair ready | `thorlabs_retrain_v22/`; `F:\SUPPORT trained\…\20260821_*` |
| 2026-08-25 | Holdout bakeoff old vs new on Level3b v22 | New markedly better visually; residual ChanB boxes | `bakeoff_v22_Level3b/` (+ PDF/means) |
| 2026-08-25 | Haj Grant ChanB smoke (new model) | Wrote `SUPPORT_v22_ChanB`; FFT `both_up` | `F:\…\Haj Grant Example\DATA\SUPPORT_v22_ChanB\` |
| 2026-08-27 | Haj Grant stitch-grid forensics (no new denoise) | Inputs no boxes; old SUPPORT weak 32-px seam; new SUPPORT same geometry, stronger | `F:\…\Haj Grant Example\analysis\patch_grid_overview.pdf` |
| 2026-08-28 | Haj Grant 2×2 raw/v22 × old/new model | **MODEL-driven.** New model ~6× seam on both inputs; v22 ≈ raw | `analysis/patch_grid_2x2.pdf`; `DATA/SUPPORT_2x2_*` |
| 2026-08-28 | Level3b excess@16 + one-tile + Haj Grant stitch arms | Holdout ChanA boxes gone after retrain; one-tile already boxed at 64 px; **128/64 hard best** (peak +0.56 vs +1.54); uniform blend **worse** (peak +2.73 @ phase 0); i16 still demoted | `support_runs/stitch_troubleshoot/`; `analysis/haj_grant_stitch_trials.pdf`; `DATA/SUPPORT_stitch_ChanB/` |
| 2026-09-03 | Haj Grant v4 old vs new at hard 128/64 | Old hits `cell_up_fringe_ok`; new ChanA fringe **1.40×** (`both_up`); new ChanB seam **+0.527 @32**. Not promote | `support_runs/haj_v4_bakeoff/`; `DATA/SUPPORT_v4_bakeoff/` |
| 2026-09-03 | Same bakeoff on registered stacks | MC does not rescue 2026 models. Reg new ChanA fringe **1.46×**; reg old ChanB still passes. | `DATA/SUPPORT_v4_bakeoff/reg/` |

---

## Retrain: necessary vs recommendable

**Verdict: retrain done on THORLABS v22.** Promote still gated.
Haj Grant v4 bakeoff (2026-09-03, hard 128/64): **old 2025 models** hit
`cell_up_fringe_ok`; **new 2026** fail it (ChanA fringe 1.40×). New ChanB still
has a phase-32 seam (+0.527, same family as v22 128/64). Do not retrain on v4.
Uniform blend still demoted. Interval 16 still demoted. 64/32 code default not switched.

---

## Handoffs

- suite2p: diagnostic paths in `HANDOFF_SUITE2P.md` — **no full promote yet**  
- Overview: this file + manifest + `SESSION.md`  

## Next (awaiting user)

1. Visual QC of `haj_v4_bakeoff_report.pdf` (now includes registered arms).
2. Do **not** default blend. Do **not** retrain (including not on v4). Do **not** promote.
3. Do **not** switch the 64/32 code default yet.
