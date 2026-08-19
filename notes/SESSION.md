# SUPPORT session lock

**Last updated:** 2026-08-18  
**Repo:** https://github.com/RasHerlo/SUPPORT  
**Sandbox:** `F:\bPACNewData2026\PreProcessing Optimization\Level3b copy`

```yaml
status: parked                               # paused | active | parked
strategy: agreed
active_run_tag: fullstack_v21_model10
next_action: "Resume tomorrow — retrain / MC-on-v21 / next SUPPORT bakeoff (user choice)"
do_not:
  - touch mc_runs/
  - overwrite inputs/raw|defringed|support
  - adopt patch_interval 16 as default (demoted)
  - promote fullstack_v21_model10 (boxes; no cell_up_fringe_ok)
defaults:
  include_first_last: mirror
  signature_fft_score: on   # --no_score to skip
```

## Resume checklist (new day / new chat)

1. Read this file → if `paused_*`, do **not** invent experiments.
2. Read `OPTIMIZATION_STATUS.md` + `optimization_manifest.json`.
3. Read incoming: `INCOMING_FROM_DEFRINGE.md` and (if present)  
   `../derippling_PMT_noise/notes/HANDOFF_SUPPORT.md`.
4. If continuing after agreement: set `status: active`, pick `active_run_tag`,
   write under `support_runs/<tag>/` only, append the attempts log.
5. End of session: update this file + status + manifest; mirror into  
   `support_runs/_handoff/`.

## Pipeline position

```text
raw  →  defringe (v2.1 pack_B)  →  SUPPORT (this repo)  →  suite2p MC/seg
                                         │
                                         ▼
                              support_runs/<tag>/
                              + HANDOFF_SUITE2P.md
```

Overview orchestrator: https://github.com/RasHerlo/figure_for_cAMP_Neu_paper  
(collects status; SUPPORT agent does not edit that repo unless asked there).
