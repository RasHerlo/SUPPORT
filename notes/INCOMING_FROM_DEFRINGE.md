# Incoming from defringe (read-only)

Source handoff (authoritative in defringe repo):  
`C:\Users\rasmu\Projects\Repos\derippling_PMT_noise\notes\HANDOFF_SUPPORT.md`  
also: `notes/OPTIMIZATION_STATUS.md`, `notes/optimization_manifest.json` there.

## Hard constraints (from sandbox + defringe)

- Do **not** touch `mc_runs/` (suite2p in progress).
- Do **not** rename/overwrite `inputs/raw`, `inputs/defringed`, or `inputs/support`.
- Write new SUPPORT outputs only under `support_runs/<tag>/`.

## Defringe context relevant to SUPPORT

| Item | Value |
|---|---|
| Current winner (500fr) | **`pack_B`** (`gpt_raw_adaptive_v21`) |
| Stacks (500fr) | `defringe_runs/v21_sweep_500fr/accepted/pack_B/ChanA\|B_raw_500fr_v21.tif` |
| **Full-stack v2.1** | **`inputs/defringed_v21/ChanA\|B/*_stk_defringed_v21.tif` (ready 2026-08-18)** |
| Full-stack `inputs/defringed` | Older **v2**; ChanA detection was wrong/weak — **not** final |
| ChanA residual (strong25, 500fr pack_B) | ~9.3% ridge excess after pack_B |
| ChanB residual (strong25, 500fr pack_B) | ~1.8% |

## Known interaction with SUPPORT

Defringe + prior SUPPORT-block compares already noted: **SUPPORT can amplify
residual fringe** (denoiser may treat periodic texture as signal), especially
when models were trained on fringed stacks.

Existing sandbox products (do not overwrite):

| Path | Meaning |
|---|---|
| `inputs/support/` | SUPPORT on **raw** (`denoised_cut.tif`, 5340 fr) |
| `inputs/support_on_defringed/` | SUPPORT after **v2** defringe (not pack_B) |

## Models in use (baseline)

| Channel | Checkpoint |
|---|---|
| ChanA | `F:\SUPPORT trained\ChanA_trained\20250130_131021\model_10.pth` |
| ChanB | `F:\SUPPORT trained\ChanB_trained\20250131_090838\model_10.pth` |

`bs_size=3`, default inference via `python -m src.single_stack`  
(`patch_size=[61,64,64]`, `patch_interval=[1,32,32]`).  
`model_10` = final of 10 training epochs (no separate val-best).
