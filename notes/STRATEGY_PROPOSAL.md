# SUPPORT strategy proposal

**Status:** `agreed`  
**Agreed phases:** A  
**Agreed date:** 2026-08-18

**Goals:** reduce (1) block/patch appearance and (2) fringe exacerbation after
SUPPORT, without harming biology or breaking suite2p handoff.

**Non-goals for Phase A:** full retrain; editing suite2p; changing defringe code.

---

## Decision tree (where to spend effort)

```text
Measure SUPPORT(raw_500) vs SUPPORT(pack_B_500)
        │
        ├─ Fringe worse after SUPPORT even when pack_B residual is low
        │     → inference / model prior issue → Phase B knobs, then retrain plan
        │
        ├─ Fringe worse mainly when pack_B residual is still high (ChanA ~9%)
        │     → escalate to defringe first; SUPPORT compare stays diagnostic
        │
        └─ Blocks present on both raw and pack_B (or only on one)
              → stitch / patch_interval / normalization → Phase B
```

Retrain is **not** the first lever. Prefer: measure → cheap inference knobs →
defringe residual if needed → retrain on promoted full-stack `defringed_v21`.

---

## Phase A — Baseline inference compare (first real work)

**Tag:** `support_runs/packB_vs_raw_500fr/`

| Input | Model |
|---|---|
| `inputs/slices_500fr/raw/ChanA_raw_500fr.tif` | ChanA `model_10` |
| `inputs/slices_500fr/raw/ChanB_raw_500fr.tif` | ChanB `model_10` |
| `defringe_runs/v21_sweep_500fr/accepted/pack_B/ChanA_raw_500fr_v21.tif` | ChanA `model_10` |
| `…/ChanB_raw_500fr_v21.tif` | ChanB `model_10` |

**Deliverables in the tag folder:**

1. Denoised TIFFs (name clearly: `*_support_raw.tif` / `*_support_packB.tif`)
2. Same-frame montages: raw | pack_B | SUPPORT(raw) | SUPPORT(pack_B)
3. Mean / max projections + optional coolwarm PNG (as in batch script)
4. Short `STATUS.md`: qualitative fringe + block notes
5. Simple quantitative checks (implement only what’s cheap and documented):
   - **Fringe:** Fourier-y ridge power on mean (same spirit as suite2p MC ridge) on input vs SUPPORT output
   - **Blocks:** e.g. patch-grid seam energy or local mean discontinuity at 32/64 grid — only if we can define it clearly in STATUS

**Pass / interpret:**

- If SUPPORT(pack_B) fringe metric ≪ SUPPORT(raw) and blocks acceptable → promote path toward suite2p on defringed+SUPPORT once full stack exists.
- If SUPPORT(pack_B) still amplifies fringe a lot → Phase B and/or defringe escalate.
- If blocks dominate regardless of fringe → Phase B stitch knobs first.

---

## Phase B — Inference knobs (only after Phase A)

Try **one change per tagged run** under `support_runs/`:

| Lever | Hypothesis | Constraint |
|---|---|---|
| Smaller `patch_interval` xy (e.g. 16) | More overlap → weaker seams | Cost ↑ |
| Larger spatial patch (if VRAM allows) | Fewer tiles | Must fit GPU |
| `include_first_last=mirror` | Edge frames | Temporal only |
| Do **not** change `bs_size` | Blind-spot geometry | Must match training (=3) |

Document each run in `OPTIMIZATION_STATUS.md` attempts log.

---

## Phase C — Escalate / retrain gates

| Gate | Action |
|---|---|
| pack_B ChanA residual still high **and** SUPPORT amplifies it | Ask defringe agent for stronger residual pass / full-stack promote before more SUPPORT |
| Blocks persist after overlap increase | Inspect dataset mean/std stitch; consider code fix in `DatasetSUPPORT_test_stitch` / validate |
| Defringe full-stack promoted to `inputs/defringed_v21/` | Schedule **retrain** on defringed distribution; keep `model_10` as baseline until new epoch-10 beats it on Phase A metrics |

---

## Phase D — Handoff to suite2p

When a candidate is good enough for MC bakeoff:

1. Write/update [`HANDOFF_SUITE2P.md`](HANDOFF_SUITE2P.md) with exact paths + frame counts.
2. Optionally hardlink/promote into a named `inputs/` slot **only if user agrees** (do not overwrite existing `inputs/support*`).
3. suite2p runs MC under `mc_runs/<tag>/` — SUPPORT agent does not touch that.

---

## Explicitly deferred (beyond Phase A)

- [x] Phase A inference unlocked
- [ ] No Phase B knobs until Phase A STATUS
- [ ] No retrain
- [ ] No edits to defringe or suite2p code from this agent
- [ ] No writes under `mc_runs/` or overwrite of protected `inputs/`
