# SUPPORT optimization status (for overview repo)

**Collected by (do not edit from here):** https://github.com/RasHerlo/figure_for_cAMP_Neu_paper  
**Stage:** SUPPORT denoising  
**Repo:** https://github.com/RasHerlo/SUPPORT  
**Sandbox:** `F:\bPACNewData2026\PreProcessing Optimization\Level3b copy`  
**Manifest:** [`optimization_manifest.json`](optimization_manifest.json)  
**Last updated:** 2026-08-18  

**Session:** Phase B complete — awaiting user on retrain / full-stack SUPPORT baseline (`SESSION.md`)

---

## Failure modes

1. Blocks / patches after denoise  
2. Fringe exacerbation after denoise  

---

## Current inference approach

| Field | Value |
|---|---|
| **ID** | `support_model10_inference_baseline` |
| **Checkpoint** | ChanA/B `model_10.pth` |
| **patch_interval** | **Keep [1,32,32]** — Phase B i16 demoted |
| **Full-stack v2.1 input** | `inputs/defringed_v21/ChanA\|B/*_stk_defringed_v21.tif` (**ready**) |

### Phase B result (boxes)

`patch_interval` 16 vs 32 on pack_B 500fr: **no meaningful visual or metric gain** on boxes or fringe. Demote i16. Details: `support_runs/phaseB_interval16_packB_500fr/STATUS.md`.

### Phase A result (fringe) — unchanged

ChanA SUPPORT amplifies fringe ~1.1× even after pack_B; ChanB benefits from pack_B before SUPPORT.

---

## Attempts log

| When | Attempt | Result | Artifacts |
|---|---|---|---|
| 2026-08-18 | Phase A pack_B vs raw 500fr | ChanA fringe amp; ChanB pack_B helps | `support_runs/packB_vs_raw_500fr/` |
| 2026-08-18 | Phase B interval 16 | **No box gain** — demote | `support_runs/phaseB_interval16_packB_500fr/` |
| — | Full-stack SUPPORT on `defringed_v21` | Not run yet | inputs ready |
| — | Retrain on `defringed_v21` | Not started | gate open |

---

## Retrain: necessary vs recommendable (current call)

**Verdict: highly recommendable; not yet strictly proven necessary.**

| Evidence | Weight |
|---|---|
| Models trained on fringed data; ChanA fringe amp after defringe+SUPPORT | Strong → retrain likely needed for ChanA |
| Inference tiling knob failed to fix boxes | Medium → boxes won’t be solved by interval alone; retrain may or may not help boxes |
| Full-stack `defringed_v21` now available | Gate open → can train properly |
| Have not yet run default SUPPORT on full `defringed_v21` | One baseline inference would confirm fringe amp on the *production* input before multi-hour retrain |

**Practical recommendation:**  
1. Optional cheap confirm: SUPPORT `model_10` + i32 on full `defringed_v21` (or 500fr slice from it) — expect ChanA fringe amp to persist.  
2. Then **retrain** ChanA (and likely ChanB for consistency) on `defringed_v21`.  
3. Re-run Phase A–style compare: new `model_10` vs old on held-out frames / metrics.  
Until that beats old weights on fringe amp + visuals, keep calling it **recommendable**; after a failed full-v21 baseline (fringe still amplified) and especially after a successful new-vs-old bakeoff, call it **necessary for ChanA production**.

---

## Handoffs

- suite2p: still **no promote** (`HANDOFF_SUITE2P.md`)  
- Defringe: full-stack v2.1 received under `inputs/defringed_v21/`  
- Overview: this file + manifest  

## Next (user choice)

1. Full-stack SUPPORT baseline on `defringed_v21` (confirm)  
2. Start retrain on `defringed_v21`  
3. Park SUPPORT and let suite2p MC on defringed_v21 without SUPPORT first  
