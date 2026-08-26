# THORLABS training manifest (draft)

**What this is:** a written split of which defringed stacks train ChanA/ChanB models,
and which we keep aside to test whether the new model beats old `model_10`.
That list (paths + roles) is the “concrete train/holdout manifest.”

**Scope:** `THORLABS_30_016` only (USER-PC deferred).  
**Inputs:** `*_stk_defringed_v22.tif` under `F:\bPACNewData2026\AC_cAMP_Neu_Ca_C1_C2`  
**Machine-readable twin:** [`thorlabs_train_manifest.json`](thorlabs_train_manifest.json)

---

## Recommended split (sufficient for training)

### Hold out (do not train on)

Same FOV as the existing SUPPORT baseline so we can compare apples-to-apples:

- `260511\C1_RLV_LW_maybe\LED_x15_Level3b\DATA\ChanA\ChanA_stk_defringed_v22.tif`
- `…\ChanB\ChanB_stk_defringed_v22.tif`

### Core train (use these)

Long physiology / LED stacks (~12.4 GB per channel). Enough temporal diversity for SUPPORT.

**ChanA**

1. `260511\C1_RLV_LW_maybe\LED_x15_Level1\DATA\ChanA\ChanA_stk_defringed_v22.tif` (2.83 GB)
2. `260511\C1_RLV_LW_maybe\LED_x15_Level3\DATA\ChanA\ChanA_stk_defringed_v22.tif` (2.83 GB)
3. `260511\C1_RLV_LW_maybe\LED_x15_Level5_001\DATA\ChanA\ChanA_stk_defringed_v22.tif` (2.83 GB)
4. `260511\C1_RLV_LW_maybe\LED_x15_Level5b\DATA\ChanA\ChanA_stk_defringed_v22.tif` (2.83 GB)
5. `260516\C1 LV\BestAttempt980\DATA\ChanA\ChanA_stk_defringed_v22.tif` (1.12 GB)

**ChanB** — same five trials, `ChanB_stk_defringed_v22.tif`

### Optional extras (only if we want more FOV/animal diversity)

Medium C2 / wavelength runs ≥ ~0.7 GB (different `families_q` on ChanA, q≈19):

- `260516\C1 LRV\920nm test` (+ `_001`)
- `260516\Test\LED again_002`
- `260517\C2 RV\980nm test`
- `260521\RW\1040nm test1/2`, `980nm test2`, `980nm test_001`

Skip tiny probes (&lt; ~0.3 GB) and the junk `260510\...\Trial` ChanB.

---

## Models

| Model | Train on | Test on |
|---|---|---|
| THORLABS ChanA | core ChanA (+ optional extras if agreed) | Level3b ChanA defringed_v22 |
| THORLABS ChanB | core ChanB (+ optional extras if agreed) | Level3b ChanB defringed_v22 |

Old checkpoints stay the baseline:  
`F:\SUPPORT trained\ChanA_trained\20250130_131021\model_10.pth` and ChanB twin.

---

## Status

**Locked:** core-only (no optional extras). Training launched under  
`support_runs/thorlabs_retrain_v22/` (see `STATUS.md` / `progress.json`).
