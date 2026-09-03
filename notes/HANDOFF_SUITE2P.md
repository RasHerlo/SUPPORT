# Handoff: suite2p agent

You are working in https://github.com/RasHerlo/suite2p  
SUPPORT: https://github.com/RasHerlo/SUPPORT  
Sandbox: `F:\bPACNewData2026\PreProcessing Optimization\Level3b copy`

**Frame length:** `single_stack` / `batch_denoise` now default to `--include_first_last mirror`
(full-length out, e.g. 5400→5400). Older diagnostic runs like `fullstack_v21_model10`
are **5340**. Never share MC shifts across unequal lengths. Opt out: `--include_first_last none`.

**From suite2p (2026-08-19):** score denoised vs defringed means with cell-band
vs PMT-family power — `notes/INCOMING_FROM_SUITE2P.md`. Pass: cell up, fringe
family not up. Tile grids are a separate fail.

---

## Current SUPPORT state (2026-09-03)

Haj Grant **defringed_v4** old vs new at hard 128/64 is scored, including registered
stacks (`support_runs/haj_v4_bakeoff/`). Old 2025 models still win the FFT gate.
New 2026 fail it (ChanA fringe 1.40× unreg / 1.46× after MC). Registration does
not rescue the new models. **Not promote-ready.**

PDFs:
- `F:\bPACNewData2026\PreProcessing Optimization\Level3b copy\support_runs\haj_v4_bakeoff\haj_v4_bakeoff_report.pdf`
- `F:\bPACNewData2026\Haj Grant Example\analysis\haj_v4_bakeoff_report.pdf`
- `F:\bPACNewData2026\Haj Grant Example\analysis\patch_grid_2x2.pdf`
- `F:\bPACNewData2026\Haj Grant Example\analysis\haj_grant_stitch_trials.pdf`

## Earlier SUPPORT state (2026-08-28)

Parked overnight. Retrain on THORLABS defringed_v22 is **done**. Holdout bakeoff prefers
the new models visually; ChanA boxes gone on Level3b; ChanB residual weaker.
Haj Grant ChanB boxes are **64-px infer geometry** on the 2026 model (one-tile
already boxed). Stitch winner: **hard `[61,128,128]/[1,64,64]`** (~3× weaker
seam). Uniform blend is worse. **Not promote-ready** until infer geometry is
decided and Level3b/Haj Grant look OK.

## Earlier SUPPORT state (2026-08-19)

- Full-stack baseline **done:** `support_runs/fullstack_v21_model10/`  
  - **ChanA** `F:\SUPPORT trained\ChanA_trained\20250130_131021\model_10.pth`  
  - **ChanB** `F:\SUPPORT trained\ChanB_trained\20250131_090838\model_10.pth`  
  - on `inputs/defringed_v21/` → 5340-frame outputs  
  - Signature QC: both `no_sharpen_fringe_ok` (fringe_power_ratio ≪ 1; cell also down)  
  - Clear **box/tile** grid on A and B means → **not promote-ready**  
  - Legacy `|ky|>0.05` ChanA 1.59× amp **superseded** by signature scores  
- Phase B `patch_interval=16` remains **demoted**  
- Default denoise path now emits signature FFT metrics automatically  

**Reasonable now without SUPPORT:** MC bakeoff on `inputs/defringed_v21/` alone.

---

## Do not disturb

- Do not overwrite `inputs/raw`, `inputs/defringed`, `inputs/support`  
- Do not treat `inputs/defringed` (v2) as winner — use **`inputs/defringed_v21/`**  
- Do not share MC shifts across 5400 vs 5340 lengths  
- Do not write into original session `DATA/` — only sandbox `mc_runs/`

---

## Delivered stacks (diagnostic)

| tag | path | n_frames | notes |
|---|---|---|---|
| fullstack_v21_model10 | `support_runs/fullstack_v21_model10/ChanA\|B/*_support.tif` | 5340 | model_10 on v21; ChanA fringe amp; boxes; **do not promote** |
| packB_vs_raw_500fr | `support_runs/packB_vs_raw_500fr/...` | 440 | Phase A diagnostic |
| phaseB_interval16_packB_500fr | `support_runs/phaseB_interval16_...` | 440 | demoted interval-16 |

---

## Pipeline order

```text
assemble → defringe v2.2 → SUPPORT (optional; boxes still block promote) → register → segment → traces
```

Retrain is done; v4 bakeoff shows new models **amplify fringe** on Haj Grant ChanA
and still plant a 128/64 seam on ChanB. Suite2p may proceed
**defringe → register** without SUPPORT. Do not promote SUPPORT stacks yet.

Overview: https://github.com/RasHerlo/figure_for_cAMP_Neu_paper collects SUPPORT `notes/OPTIMIZATION_STATUS.md`.
