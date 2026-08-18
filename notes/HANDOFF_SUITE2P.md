# Handoff: suite2p agent

You are working in https://github.com/RasHerlo/suite2p  
(local: `C:\Users\rasmu\Projects\Repos\suite2p`).

SUPPORT denoising lives in https://github.com/RasHerlo/SUPPORT.  
Shared sandbox (single data tree):

`F:\bPACNewData2026\PreProcessing Optimization\Level3b copy`

---

## What SUPPORT will deliver (when ready)

| Item | Location / rule |
|---|---|
| Trial denoised stacks | `support_runs/<tag>/` only |
| Status for overview | SUPPORT repo `notes/OPTIMIZATION_STATUS.md` + `optimization_manifest.json` |
| Mirror for data-tree readers | `support_runs/_handoff/` |
| Promoted input (optional) | Only by explicit promote — **never** overwrite `inputs/support` or `inputs/support_on_defringed` without user OK |

**Frame-length warning:** SUPPORT default path drops temporal edges (`model` receptive field / optional cut). Typical full-stack: raw/defringe **5400** → SUPPORT **5340**. **Never share MC shifts** across unequal lengths.

---

## Do not disturb (still in force)

- Do not rename/move `mc_runs/` mid-bakeoff.
- Do not overwrite `inputs/raw`, `inputs/defringed`, `inputs/support`.
- Do not treat `inputs/defringed` (v2) or `inputs/support_on_defringed` (SUPPORT-on-v2) as the v2.1 winner.
- **Use** `inputs/defringed_v21/` for current full-stack defringe (promoted 2026-08-18).
- 500fr pack_B trials remain under `defringe_runs/v21_sweep_500fr/accepted/pack_B/`.

---

## Current SUPPORT state (2026-08-18)

- Phase A complete: `support_runs/packB_vs_raw_500fr/` — ChanA fringe amplified after pack_B+SUPPORT; ChanB benefits from defringe-first.
- Phase B complete: `support_runs/phaseB_interval16_packB_500fr/` — **`patch_interval=16` demoted** (no box/fringe gain vs 32). Keep `[1,32,32]`.
- Full-stack `inputs/defringed_v21/` is ready; SUPPORT baseline on those stacks and/or retrain are **pending user choice**.
- Retrain: **highly recommendable**, not yet strictly proven necessary (see `notes/OPTIMIZATION_STATUS.md`).
- Baseline weights still ChanA/B `model_10.pth`. **Not promote-ready** for production MC/ROI.

Continue MC on **raw** and/or **`inputs/defringed_v21/`** per your notes. Do **not** treat Phase A/B 500fr SUPPORT outputs as production delivered stacks.

---

## Delivered stacks

_Diagnostic only — not for suite2p production MC._

| tag | channel | path | n_frames | notes |
|---|---|---|---|---|
| packB_vs_raw_500fr | ChanA | `support_runs/packB_vs_raw_500fr/ChanA/*_support.tif` | 440 | fringe amp ~1.11× on pack_B; do not promote |
| packB_vs_raw_500fr | ChanB | `support_runs/packB_vs_raw_500fr/ChanB/*_support.tif` | 440 | pack_B helps vs raw; diagnostic |
| phaseB_interval16_packB_500fr | ChanA/B | `support_runs/phaseB_interval16_packB_500fr/.../*_i16.tif` | 440 | interval-16 demoted; diagnostic |

---

## What suite2p should do when a tag is delivered

1. Read SUPPORT `notes/OPTIMIZATION_STATUS.md` attempts log for that tag.
2. Run the **same MC eval** as raw (`fringe_robust_register` + `compare_AB`): ridge vs `stk_avg`, share-A test.
3. Write under `mc_runs/<related_tag>/` — not into original session `DATA/`.
4. Pass bar (from your notes): registered ridge ≤ `stk_avg`; share-A should **lower** ChanB ridge if shifts followed tissue.
5. Only then segmentation / traces for overview.

Reasonable **now** without SUPPORT: MC bakeoff on `inputs/defringed_v21/` (defringe-first, no denoise).

---

## Pipeline order (agreed direction)

```text
assemble → defringe v2.1 → SUPPORT (optional but intended) → register → segment → traces
```

Legacy `SUPPORT → register → defringe` is demoted (feeds phasecorr the worst texture).

---

## Return path for overview

Overview repo: https://github.com/RasHerlo/figure_for_cAMP_Neu_paper  

suite2p keeps reflecting in `lab/notes/`; SUPPORT keeps `notes/OPTIMIZATION_STATUS.md`.  
Overview **collects** — agents do not push catalog edits from SUPPORT/suite2p unless the user opens that repo.
