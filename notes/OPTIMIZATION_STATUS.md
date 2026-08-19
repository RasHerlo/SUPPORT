# SUPPORT optimization status (for overview repo)

**Collected by:** https://github.com/RasHerlo/figure_for_cAMP_Neu_paper  
**Repo:** https://github.com/RasHerlo/SUPPORT  
**Sandbox:** `F:\bPACNewData2026\PreProcessing Optimization\Level3b copy`  
**Manifest:** [`optimization_manifest.json`](optimization_manifest.json)  
**Last updated:** 2026-08-19  

**Session:** full-stack baseline complete — awaiting retrain / promote decision (`SESSION.md`)

---

## Failure modes

1. Blocks / patches after denoise  
2. Fringe exacerbation after denoise  

---

## Current approach

| Field | Value |
|---|---|
| **ID** | `support_model10_on_defringed_v21` |
| **ChanA model** | `F:\SUPPORT trained\ChanA_trained\20250130_131021\model_10.pth` |
| **ChanB model** | `F:\SUPPORT trained\ChanB_trained\20250131_090838\model_10.pth` |
| **Inference** | `patch_interval=[1,32,32]` (i16 demoted), `bs_size=3` |
| **Full-stack tag** | `support_runs/fullstack_v21_model10/` |
| **Status** | Baseline **done**; **not** suite2p-promote ready |

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

---

## Retrain: necessary vs recommendable

**Verdict: recommendable; promote blocked mainly by boxes + no `cell_up_fringe_ok`.**

Signature QC (2026-08-19) shows fringe **family power down** after SUPPORT on both
channels (not exacerbated by the mask metric). Retrain still useful to match the
defringed distribution and chase cell-band gain without tiles. Skipping SUPPORT
and MC on `defringed_v21` alone remains valid.

---

## Handoffs

- suite2p: diagnostic full-stack paths in `HANDOFF_SUITE2P.md` — **no promote**  
- Optional parallel path: MC directly on `inputs/defringed_v21/` without SUPPORT  
- Overview: this file + manifest  

## Next (user choice)

1. Start retrain on `defringed_v21` (ChanA required; ChanB recommended)  
2. Park SUPPORT; let suite2p MC on `defringed_v21` only  
3. Promote diagnostic stacks only for inspection (not paper traces)  
