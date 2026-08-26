# SUPPORT optimization status (for overview repo)

**Collected by:** https://github.com/RasHerlo/figure_for_cAMP_Neu_paper  
**Repo:** https://github.com/RasHerlo/SUPPORT  
**Sandbox:** `F:\bPACNewData2026\PreProcessing Optimization\Level3b copy`  
**Manifest:** [`optimization_manifest.json`](optimization_manifest.json)  
**Last updated:** 2026-08-26  

**Session:** parked — THORLABS v22 retrain + Level3b bakeoff done; residual boxes next (`SESSION.md`)

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
| **Inference** | `patch=[61,64,64]`, `interval=[1,32,32]`, `bs_size=3`, mirror |
| **Bakeoff tag** | `support_runs/bakeoff_v22_Level3b/` |
| **Status** | New models preferred visually; **boxes remain** (esp. ChanB) |

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

---

## Retrain: necessary vs recommendable

**Verdict: retrain done on THORLABS v22.** Promote still gated by residual boxes
(stitch), not by lack of a defringed-trained model. Old ECF1 `SUPPORT_ChanB`
used same batch defaults (user-confirmed); less-boxed look there ≠ special settings.

---

## Handoffs

- suite2p: diagnostic paths in `HANDOFF_SUITE2P.md` — **no full promote yet**  
- Overview: this file + manifest + `SESSION.md`  

## Next (user choice)

1. Box stitch bakeoff on new model (overlap and/or blending in `validate`)  
2. If boxes OK for suite2p → `batch_denoise_v22` + `SUPPORT_v22_ChanA/B`  
3. Park further SUPPORT; MC on defringed alone  
