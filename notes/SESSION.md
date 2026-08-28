# SUPPORT session lock

**Last updated:** 2026-08-28 (parked overnight)  
**Repo:** https://github.com/RasHerlo/SUPPORT  
**Sandbox:** `F:\bPACNewData2026\PreProcessing Optimization\Level3b copy`

```yaml
status: parked
strategy: agreed
active_run_tag: stitch_troubleshoot
next_action: "After user OK: switch 2026-model infer to hard [61,128,128]/[1,64,64] (optional batch_denoise too). Do not default blend. Do not retrain. Do not promote."
do_not:
  - touch mc_runs/
  - overwrite inputs/raw|defringed|support
  - train on USER-PC until more QC-ok data
  - train on holdout LED_x15_Level3b
  - overwrite Haj Grant SUPPORT_ChanB/ or SUPPORT_v22_ChanB/
defaults:
  include_first_last: mirror
  signature_fft_score: on
  inference_patch: [61, 64, 64] / [1, 32, 32]   # still the code default; 128/64 is the stitch winner, not yet switched
```

## Results to remember

### 1) THORLABS retrain (done, 2026-08-20–21)
Separate ChanA/B models on core defringed_v22 stacks (holdout Level3b excluded).

| Channel | New `model_10.pth` |
|---|---|
| ChanA | `F:\SUPPORT trained\ChanA_trained\20260821_003756\model_10.pth` |
| ChanB | `F:\SUPPORT trained\ChanB_trained\20260821_112325\model_10.pth` |

Baseline (old): `…\20250130_131021` / `…\20250131_090838`  
Manifest: `notes/THORLABS_TRAIN_MANIFEST.md` + `thorlabs_train_manifest.json`  
Run tag: `support_runs/thorlabs_retrain_v22/`

### 2) Holdout bakeoff old vs new on Level3b v22 (done, 2026-08-25)
Tag: `support_runs/bakeoff_v22_Level3b/`  
PDF: `bakeoff_v22_Level3b_visual_report.pdf` + `means/`

- User visual QC: **both channels markedly better with new models**; residual **boxes** especially ChanB.
- FFT mean: all `no_sharpen_fringe_ok`; none `promote_ok` / `cell_up_fringe_ok`.
- ChanA new: lower fringe ratio than old; ChanB FFT fringe still favors old; boxes need stitch work.

### 3) Haj Grant / ECF1 boxes (2026-08-27 — forensics)
Working copy: `F:\bPACNewData2026\Haj Grant Example\DATA\`  
Original session: `E:\Rasmus-Guillermo\ECF1\F1_RV_pdf\New recordings PFC\LED 2s\240916_pl100_pc001_LED_min10_ex02\DATA\`  
PDF: `F:\bPACNewData2026\Haj Grant Example\analysis\patch_grid_overview.pdf`

| Role | Full path |
|---|---|
| raw | `F:\bPACNewData2026\Haj Grant Example\DATA\ChanB\ChanB_stk.tif` |
| defringed_v22 | `F:\bPACNewData2026\Haj Grant Example\DATA\ChanB\ChanB_stk_defringed_v22.tif` |
| old SUPPORT (Apr 2025) | `F:\bPACNewData2026\Haj Grant Example\DATA\SUPPORT_ChanB\denoised_cut.tif` |
| new SUPPORT (Aug 2026) | `F:\bPACNewData2026\Haj Grant Example\DATA\SUPPORT_v22_ChanB\ChanB_stk_defringed_v22_support.tif` |

- Old `SUPPORT_ChanB` = `batch_denoise` on **raw** + 2025 ChanB `model_10` (`20250131_090838`). Frame count 1520 = `denoised_cut` (±30).  
  GUI sibling (do **not** confuse): `E:\…\DATA\ChanB\SUPPORT\20250201_143145\` (`model_info.txt` names the same 2025 checkpoint).
- New `SUPPORT_v22_ChanB` = `single_stack` on **defringed_v22** + 2026 ChanB `model_10` (`20260821_112325`). Full length 1580, mirror pad.
- Patch geometry **identical** on both SUPPORT runs: `[61,64,64]` / `[1,32,32]` / `bs_size=3` → expected seam **phase 16 mod 32**.
- Inputs: no stitch boxes. Own-scale “checkered” PDF page is fold-noise wallpaper (peaks at phase 10, not 16); not visible in ImageJ.
- Old SUPPORT: weak same-family seam (phase 15–17, excess@16 ≈ 0.24). Usually not obvious in ImageJ.
- New SUPPORT: same geometry, locked on phase 16, several times stronger (vert z +2.40, excess@16 ≈ 1.54) — the visible boxes.
- **2×2 done 2026-08-28 (MODEL-driven).** Same `[61,64,64]` / `[1,32,32]`. Vertical excess@16: raw+old 0.24, v22+old 0.23, raw+new 1.55, v22+new 1.54. Input does nothing; new model ~6× the seam on both inputs. PDF: `analysis/patch_grid_2x2.pdf`. Outputs: `DATA/SUPPORT_2x2_raw_new_ChanB/`, `DATA/SUPPORT_2x2_v22_old_ChanB/`.

### 4) Extra FOV smoke test (already on disk)
`F:\bPACNewData2026\Haj Grant Example\DATA\SUPPORT_v22_ChanB\`  
New ChanB model on `ChanB_stk_defringed_v22.tif` → full-length support + FFT (`both_up`, cell↑ fringe slightly↑).

### 5) Stitch troubleshoot (done 2026-08-28; this unlock)

Tag: `support_runs/stitch_troubleshoot/`  
PDF: `F:\bPACNewData2026\Haj Grant Example\analysis\haj_grant_stitch_trials.pdf`

**Level3b bakeoff rescored with excess@16** (`level3b_stitch_grid.pdf`):

| source | vert excess@16 | vert z / phase |
|---|---:|---|
| ChanA v22 | −0.077 | no grid |
| ChanA **old** | **+0.705** | +3.94 @17 |
| ChanA **new** | **~0** | no grid |
| ChanB v22 | −0.034 | no grid |
| ChanB old | +0.626 | +2.25 @16 |
| ChanB new | +0.360 | +5.54 @16 (smoother → z inflates) |

Retrain **killed ChanA boxes** on the holdout and **lowered ChanB absolute seam**. Haj Grant ChanB (ECF1, not in the THORLABS v22 train set) is the FOV where new ≫ old.

**One-tile:** 64×64 2026-model output already has a dark 32×32 box on the keep region (+26 counts edge-ring minus interior). Inferring at 64 px puts the CNN edge-effect on the paste boundary; training is 128 px.

**Haj Grant stitch arms** (v22 + 2026 ChanB, did not overwrite `SUPPORT_v22_ChanB`):

| arm | vert peak excess | note |
|---|---:|---|
| hard 64/32 (baseline) | **+1.537 @16** | visible boxes |
| blend 64/32 | **+2.733 @0** | phase-16 gone; **worse** seam at tile origin |
| hard 128/64 (train geometry) | **+0.556 @32** | **weakest**; ~3× below baseline |
| hard 64/16 | **+0.628 @8** | finer grid, not gone; ~4× tiles |

`--stitch {hard,blend}` is in `src/single_stack.py` only (`batch_denoise` still hard). Uniform blend is demoted. Infer code default is still 64/32 until the user switches it.

## Tomorrow (do not start until unlocked)

1. **Switch infer default** to hard `[61,128,128]/[1,64,64]` in `src/single_stack.py` (and optionally `batch_denoise`) if the user agrees. That matches training (`src/utils/util.py`) and was the stitch winner.
2. Do **not** default `--stitch blend` (phase-0 seam worse than hard 64/32). Leave the flag; it is demoted.
3. Do **not** retrain. Do **not** promote to suite2p.
4. Do **not** rerun the Haj Grant stitch arms — stacks are already on disk under `DATA/SUPPORT_stitch_ChanB/`.

Open first: this file, then `haj_grant_stitch_trials.pdf`.

## Resume checklist (new day / new chat)

1. Read this file → if `paused_*` / `parked`, do **not** invent experiments until unlocked.
2. Read `OPTIMIZATION_STATUS.md` + `optimization_manifest.json`.
3. Read incoming: `INCOMING_FROM_DEFRINGE.md` and (if present)  
   `../derippling_PMT_noise/notes/HANDOFF_SUPPORT.md`.
4. If continuing: set `status: active`, pick `active_run_tag`, write under `support_runs/<tag>/` (Haj Grant stitch lives under the grant DATA tree, not Level3b `mc_runs/`).
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
