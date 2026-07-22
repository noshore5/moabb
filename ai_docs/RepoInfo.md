# Repository Quick Info

MOABB LeftRightImagery (BNCI2014-001) CWT/WCT/XWT GNN + baselines.

- Code: put new work under `coheriqs_contributions/` (preferably inside a module there) unless it clearly belongs
  elsewhere (`Coherent_Multiplex/`, upstream `moabb/`).
- Run scripts / pytest from: **repo root** (imports). Edit code under
  `coheriqs_contributions/`.
- Environment: venvs live at repo root and are (usually) named as `moabb-env-win` (Windows) / `moabb-env` (Unix).
- Main run: `python coheriqs_contributions/run_wct_gnn.py --subjects 1 --pipeline WCT-Evidence-GNN`
or `bash coheriqs_contributions/run_canonical_setup.sh` from repo root.
- Tests: `pytest coheriqs_contributions/tests -q`
- Subject scope: default `--subjects 1`. Before using any other subjects, read
  and update `ai_docs/experimental_policy.md` (scope rules and exploitation
  tracker).
- `run_wct_gnn.py` prints its configured data/result roots and writes each run
  under a unique result ID with HDF5 plus human-readable CSV/Markdown outputs.
- `run_wct_gnn.py` also writes a complete UTF-8 experiment log while keeping
  console output compact by default: every fifth epoch, no selector or
  per-batch diagnostics, hidden progress bars, and final result summaries.

## Paths (open only when needed)

- `orchestration/` - project profile, WCT adapter/preflight, and WCT-only
  context fragment for the reusable local multi-worktree framework. These do
  not activate orchestration by themselves.

- `coheriqs_contributions/run_wct_gnn.py` — CLI, pipeline registry, param grids.
  Start here for runs and hyperparams.
- `coheriqs_contributions/moabb_pipelines/common.py` — shared torch fit/train,
  CWT feature prep helpers. Open for training loop / shared preprocessing.
- `coheriqs_contributions/moabb_pipelines/wct_evidence_gnn_classifier.py` —
  default active model. Open for WCT-Evidence architecture changes.
- `coheriqs_contributions/moabb_pipelines/xwt_phase_gnn_classifier.py` —
  XWT cores **and** `_BaseCWTGNNClassifier` shared by WCT/XWT. Open only when
  changing that base init/feature hook or an XWT model — not for routine
  Evidence-only edits.
- `Coherent_Multiplex/utils/coherence_utils.py` — fcwt CWT / coherence. Open
  only when changing the wavelet transform itself; classifiers call it via
  `common.py`.

## Chores

- Keep `run_canonical_setup.sh` in sync with `run_wct_gnn.py` —
  when the active run profile changes - update the arguments accordingly.
