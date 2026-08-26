# SUPPORT session lock

**Last updated:** 2026-08-26  
**Repo:** https://github.com/RasHerlo/SUPPORT  
**Sandbox:** `F:\bPACNewData2026\PreProcessing Optimization\Level3b copy`

```yaml
status: parked
strategy: agreed
active_run_tag: bakeoff_v22_Level3b
next_action: "Resume box optimization (stitch/patch bakeoff) and/or batch_denoise_v22 after user OK"
do_not:
  - touch mc_runs/
  - overwrite inputs/raw|defringed|support
  - train on USER-PC until more QC-ok data
  - train on holdout LED_x15_Level3b
defaults:
  include_first_last: mirror
  signature_fft_score: on
  inference_patch: [61, 64, 64] / [1, 32, 32]   # batch_denoise defaults (unchanged historically)
```

## Results to remember (close-down 2026-08-26)

### 1) THORLABS retrain (done)
Separate ChanA/B models on core defringed_v22 stacks (holdout Level3b excluded).

| Channel | New `model_10.pth` |
|---|---|
| ChanA | `F:\SUPPORT trained\ChanA_trained\20260821_003756\model_10.pth` |
| ChanB | `F:\SUPPORT trained\ChanB_trained\20260821_112325\model_10.pth` |

Manifest: `notes/THORLABS_TRAIN_MANIFEST.md` + `thorlabs_train_manifest.json`  
Run tag: `support_runs/thorlabs_retrain_v22/`

### 2) Holdout bakeoff old vs new on Level3b v22 (done)
Tag: `support_runs/bakeoff_v22_Level3b/`  
PDF: `bakeoff_v22_Level3b_visual_report.pdf` + `means/`

- User visual QC: **both channels markedly better with new models**; residual **boxes** especially ChanB.
- FFT mean: all `no_sharpen_fringe_ok`; none `promote_ok` / `cell_up_fringe_ok`.
- ChanA new: lower fringe ratio than old; ChanB FFT fringe still favors old; boxes need stitch work.

### 3) “Why did old ECF1 SUPPORT_ChanB look less boxed?”
Folder: `E:\…\240916_…\DATA\SUPPORT_ChanB` was **`batch_denoise`**, not GUI  
(sibling `ChanB\SUPPORT\20250201_*` *is* GUI — do not confuse).  
User confirmed **patch args were not overridden** → same defaults `[61,64,64]` / `[1,32,32]` as now.  
So fewer boxes there is **not** a special setting; likely data/display (raw FOV, contrast), not a hidden CLI mode.

### 4) Extra FOV smoke test
`F:\bPACNewData2026\Haj Grant Example\DATA\SUPPORT_v22_ChanB\`  
New ChanB model on `ChanB_stk_defringed_v22.tif` → full-length support + FFT (`both_up`, cell↑ fringe slightly↑).

## Anticipated next steps (tomorrow)

1. **Box residual (priority)** — controlled stitch bakeoff on holdout ChanB (and/or Haj Grant):
   - Re-test denser overlap (`patch_interval` 16/8) on **new** model (old i16 demotion was pre-retrain).
   - Prefer **overlap blending** in `validate` / stitch if overlap alone fails (most likely real fix).
   - Optional: try GUI-matched `[61,128,128]` / `[1,64,64]` as a contrast arm (not required to explain ECF1).
2. **Decide promote path** — if boxes acceptable for suite2p, sketch `batch_denoise_v22` (v22 inputs, new models, `SUPPORT_v22_ChanA/B` folders).
3. Do **not** retrain until stitch levers are checked.

## Resume checklist (new day / new chat)

1. Read this file → if `paused_*` / `parked`, do **not** invent experiments until unlocked.
2. Read `OPTIMIZATION_STATUS.md` + `optimization_manifest.json`.
3. Read incoming: `INCOMING_FROM_DEFRINGE.md` and (if present)  
   `../derippling_PMT_noise/notes/HANDOFF_SUPPORT.md`.
4. If continuing: set `status: active`, pick `active_run_tag`, write under `support_runs/<tag>/`.
5. End of session: update this file + status + manifest; mirror into `support_runs/_handoff/`.

## Pipeline position

```text
raw  →  defringe (v2.1 / v22)  →  SUPPORT (this repo)  →  suite2p MC/seg
                                         │
                                         ▼
                              support_runs/<tag>/
                              + HANDOFF_SUITE2P.md
```

Overview orchestrator: https://github.com/RasHerlo/figure_for_cAMP_Neu_paper  
(collects status; SUPPORT agent does not edit that repo unless asked there).
