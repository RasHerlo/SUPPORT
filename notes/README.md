# SUPPORT optimization notes

Entry point for agents working on **SUPPORT denoising** in the three-repo
preprocessing pipeline.

| File | Who reads it |
|---|---|
| [SESSION.md](SESSION.md) | **Start here** every new day — locked/unlocked state |
| [OPTIMIZATION_STATUS.md](OPTIMIZATION_STATUS.md) | Overview + sibling agents — current approach, pros/cons, log |
| [optimization_manifest.json](optimization_manifest.json) | Machine-readable twin of status (overview catalog) |
| [STRATEGY_PROPOSAL.md](STRATEGY_PROPOSAL.md) | Troubleshooting order (Phase A/B agreed; later phases pending) |
| [HANDOFF_SUITE2P.md](HANDOFF_SUITE2P.md) | suite2p agent — what we deliver and when |
| [INCOMING_FROM_DEFRINGE.md](INCOMING_FROM_DEFRINGE.md) | Defringe pack_B + full-stack `defringed_v21` + constraints |

**Before new experiments:** read `SESSION.md`. Do not invent runs while status is `*_awaiting_*` without user OK.

## Progress snapshot (2026-08-18)

- Phase A done: fringe amp on ChanA; ChanB benefits from pack_B→SUPPORT  
- Phase B done: `patch_interval=16` **demoted**; keep `[1,32,32]`  
- `inputs/defringed_v21/` full stacks ready; retrain **highly recommendable**  
- No suite2p promote yet  

Sandbox (data only):

`F:\bPACNewData2026\PreProcessing Optimization\Level3b copy`

Mirrored handoff copies:

`support_runs/_handoff/`
