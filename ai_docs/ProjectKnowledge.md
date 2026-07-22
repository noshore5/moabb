# Project Knowledge

Durable conventions and gotchas for `coheriqs_contributions/`. The conditional
local multi-worktree project-control bundle is under `orchestration/`.

## Where work goes

- Prefer `coheriqs_contributions/` for pipelines, models, tests, and experiment
  scripts. Change `Coherent_Multiplex/` only for shared wavelet/coherence code;
  change upstream `moabb/` only when the contribution genuinely belongs there.
- Launch and pytest from the **repo root**. `run_wct_gnn.py` puts the root on
  `sys.path`; the contributions tree is not an installable package.
- `coheriqs_contributions/run_canonical_setup.sh` — maintained script
  wrapper for the canonical experimental setup; update when the active
  experimental CLI profile changes.

## Argument / config threading

Hyperparameters are assembled in `run_wct_gnn.py`: `_make_*` builders construct
sklearn estimators; optional `PIPELINE_PARAM_GRIDS` are applied via `set_params`
or MOABB `param_grid`. GNN classifiers store args in `__init__`, then chain
`_init_cwt_gnn_classifier` (`xwt_phase_gnn_classifier.py`) →
`_init_torch_classifier` (`common.py`). Fit path: MOABB →
`TorchEEGClassifier.fit` → `_prepare_features` (CWT via `Coherent_Multiplex`) →
`_build_model*` → `_train_loop`.

`_make_*` defaults, `PIPELINE_PARAM_GRIDS`, and historical logs can disagree —
treat the post-`set_params` / grid outcome as the true run config. Do not assume
today’s runner defaults match experimental results/logs.

`run_wct_gnn.py` accepts fixed estimator overrides through `--param-names` and
`--param-values`. Names may be passed as separate tokens or as one quoted,
comma-separated string; values may be separate safe literals or one quoted
literal list. Overrides must be supported by every selected pipeline, are
applied before evaluation, and remove the same dimensions from the copied grid.

## Hot-path inheritance

XWT / WCT classifiers → `_BaseCWTGNNClassifier` (`xwt_phase_gnn_classifier.py`)
→ `TorchEEGClassifier` (`common.py`). Some baselines (e.g. EEGNet) extend
`TorchEEGClassifier` directly. CWT comes from
`Coherent_Multiplex/utils/coherence_utils.py`; that sibling tree must be present.

## Experiment logging

`run_wct_gnn.py` configures one process-wide standard-library logger. Its file
handler retains all INFO-level experiment events; its console handler filters
semantic categories independently. Console flags control initial model details,
CWT and MOABB progress, per-batch diagnostics, final results, and epoch/selector
cadence. Direct estimator use outside the runner keeps the legacy `verbose`
behavior. Experiment-log paths are collision-safe and existing files are never
overwritten.

## Training / eval gotchas

- Checkpointing uses validation **loss**; MOABB reports outer **ROC-AUC** — they
  can disagree on small val splits.
- No reusable on-disk CWT/noise-bank cache yet: every `fit` / `predict`
  recomputes CWT (noise banks add another pass). A future cache must be
  keyed from the actual transform inputs and configuration; fold-specific
  normalization makes raw subject/session identity alone insufficient.
  Per-execution MNE scratch state is deliberately not that cache.
- Grouped validation needs groups/metadata in `fit()`; without them the trainer
  falls back to a random stratified split.
- EEG/MOABB artifacts live under `MNE_DATA` / `MOABB_RESULTS`. Manual runs use
  the user MNE configuration; managed runs isolate `MOABB_RESULTS` per execution.
  `run_wct_gnn.py` writes uniquely suffixed HDF5 plus CSV/Markdown companions.
