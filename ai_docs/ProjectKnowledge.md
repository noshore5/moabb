# Project Knowledge

Durable conventions and gotchas for `coheriqs_contributions/`. The conditional
local multi-worktree project-control bundle is under `orchestration/`.

## Where work goes

- Prefer `coheriqs_contributions/` for pipelines, models, tests, and experiment
  scripts. Change `Coherent_Multiplex/` only for shared wavelet/coherence code;
  change upstream `moabb/` only when the contribution genuinely belongs there.
- Launch and pytest from the **repo root**. `run_wct_gnn.py` puts the root on
  `sys.path`; the contributions tree is not an installable package.
- `coheriqs_contributions/run_canonical_setup.sh` and
  `coheriqs_contributions/run_canonical_setup.py` are maintained wrappers for
  the canonical experimental setup. The Python wrapper is a zero-argument IDE
  entry point that reuses the active interpreter and environment. Update both
  wrappers when the active experimental CLI profile changes.

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
cadence. `--console-all` enables every category and cadence; explicit
`--no-console-*` and cadence arguments override that baseline. Direct estimator
use outside the runner keeps the legacy `verbose` behavior. The runner's unified
`--overwrite/--no-overwrite` policy applies to MOABB results, experiment logs,
and CSV/Markdown companions; overwrite is enabled by default.

Selected durable events carry `data=` JSON after their readable message. The
serializer keeps common configuration values JSON-compatible, orders fields
deterministically, and marks unknown values with a type-qualified fallback.
`runtime_context.py` records lightweight source, platform, package, and device
facts once per run; unavailable optional probes are reported in its payload
instead of failing the run. Per-group effective configuration and the optional
`--description` are included in the Markdown summaries.

## Training / eval gotchas

- Checkpointing defaults to validation **loss**, but may use any registered
  final checkpoint score; MOABB reports outer **ROC-AUC**. The within-fit
  selection utility and outer outcome can disagree on small validation splits.
- The named checkpoint policy parameters are `final_checkpoint_score`,
  `checkpoint_scores`, `clean_train_scores`, `candidate_sources`,
  `prediction_metrics`, and `checkpoint_reporting`.
- Built-in checkpoint scores are implicit names. Custom scores use
  `CheckpointScorer(function=..., name=...)`; all selection, candidate, and
  reporting references use registered names. Utilities are maximized, exact
  ties prefer the earlier epoch, and `None` means scorer warm-up/abstention.
  There is no `min_delta`.
- `ScoreRanking` stores lightweight top-K observations for diagnostics.
  `CandidateSource` retains a smaller state-bearing prefix of an explicitly
  configured ranking. `selection_runner_ups` controls a separate lightweight
  final-score presentation and never retains runner-up states or triggers
  evaluation passes. `checkpoint_reporting` is operational and excluded from
  MOABB result identity; scoring, gating, prediction metrics, and candidate
  depths remain semantic. The final score already uses the authoritative
  `selection_runner_ups` output, so its `ScoreRanking` normally keeps
  `show_at_end=False`; setting it to `True` remains valid and deliberately adds
  a second diagnostic rendering.
- Non-clean-gated scores run after every validation record. Clean-gated scores
  run only with a fresh clean pass: on eligible epochs in `interval`, and over
  the fully enriched candidate union after training in `deferred_candidates`.
  `disabled` permits clean-gated diagnostics to remain configured but dormant.
- Online scorer history contains only prior records. Deferred clean-gated
  scorers receive the complete available trajectory except their own current
  record, including later epochs. Use `history_before_epoch` for causal
  behavior, `history_after_epoch` for future records, and
  `history_except_epoch` for the complete non-current view; `recent_history`
  remains a mechanical tail operation on the sequence supplied to it.
- `PredictionMetric(function=..., name=...)` consumes encoded targets and the
  full probability matrix already accumulated during validation/clean loader
  evaluation. It adds fields such as `val_brier_score` and
  `clean_train_brier_score` without another model forward pass.
- Schema-v2 fit summaries and readable checkpoint messages include the selected
  raw metrics and named scores, authoritative final ranking/scope, candidate
  nominations, configured end diagnostics, and inactive-score reasons.
  Runner-up and diagnostic output reuses cached observations. Structured epoch
  data records the authoritative selection-ranking update independently of
  optional observational rankings, and final nomination provenance excludes
  epochs displaced from every candidate-source prefix.
- Validation and clean-training checkpoint metrics use the same non-shuffled
  evaluation path. It temporarily switches the model to evaluation mode, runs
  without gradients, bypasses training-only input augmentation, and restores
  the prior model mode afterward. Selectors default to deterministic `argmax`
  evaluation; explicitly configuring a stochastic selector with
  `eval_mode="same"` intentionally makes evaluation stochastic.
- Checkpoint candidate snapshots remain on the model's device. `disabled` and
  direct interval selection retain only the current final-score winner, plus
  any explicitly configured candidate-source prefixes. Deferred mode retains
  at most `min(eligible_epochs, sum(source.top_k))` distinct epoch states. A
  model-only snapshot is the tensor size of `model.state_dict()`. A saved Adam
  continuation snapshot adds approximately twice the trainable-parameter bytes.
  For the active 22-channel, two-class WCT-Evidence grid this is about 7.0 KiB
  model-only or 20.1 KiB with Adam; even 180 distinct candidates are about
  1.23 MiB or 3.54 MiB respectively. Recalculate before using substantially
  wider models or large `top_k` values.
- Full optimizer snapshots are retained only while selecting a final refit with
  `save_selected_checkpoint=True`, then written to the continuation bundle and
  released. Other fits retain model-only candidates and release them after
  selection.
- Optional input-CWT and noise-bank caches share `wct_cache_root`. Input CWT
  entries are session-level memory-mapped arrays filled incrementally by trial
  identity; noise-bank entries are deterministic whole-bank arrays. Noise-bank
  cache preparation and device materialization are timed separately. Before the
  training loop, CPU fits copy the full bank out of the mmap into contiguous
  RAM, while accelerator fits transfer the full bank to the selected device, so
  cache I/O does not leak into epoch timings.
- Grouped validation needs groups/metadata in `fit()`; without them the trainer
  falls back to a random stratified split. Group-subset selection examines all
  combinations only up to 4,096; larger spaces use a seeded uniform sample of
  at most 4,096 distinct subsets.
- EEG/MOABB artifacts live under `MNE_DATA` / `MOABB_RESULTS`. Manual runs use
  the user MNE configuration; managed runs isolate `MOABB_RESULTS` per execution.
  `run_wct_gnn.py` writes uniquely suffixed HDF5 plus CSV/Markdown companions.
