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

## Paths (open only when needed)

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
