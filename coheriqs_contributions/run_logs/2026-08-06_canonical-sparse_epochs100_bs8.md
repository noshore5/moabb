# Sparse-Evidence-GNN canonical run — 2026-08-06 21:23

- Run ID: `canonical-sparse`
- Source artifacts: `results_canonical-sparse__cross__inner-none.{hdf5,csv}`,
  `summary_canonical-sparse__cross__inner-none.md` (under
  `/Users/noahshore/mne_data/results/LeftRightImagery/CrossSessionEvaluation/`)
- Experiment log: `/Users/noahshore/mne_data/canonical-sparse/experiment.log`
- Dataset: BNCI2014-001, subjects 1-4, `LeftRightImagery` paradigm,
  `CrossSessionEvaluation`

This is the first canonical run at `epochs=100, batch_size=8` — both changed
from the prior defaults (`epochs=50, batch_size=16`) after today's findings:
epochs=50→100 was a real gain on subject 1 (0.799→0.828, plateaued by 150,
see [[sparse-evidence-gnn-native-resolution-fix]]); batch_size=8 showed the
best combination of speed and seed-to-seed stability in a 4-seed sweep
(subject 1 only) vs. 4/16/32, though with only 4 seeds tested per setting
that finding should be treated as suggestive, not conclusive — batch_size=16
and 4 both reproduced a catastrophic single-seed failure (`0train` collapsing
to ~0.57) that batch_size=8 happened not to hit.

## Results

| subject | session | score | mean (subject) |
| --- | --- | --- | --- |
| 1 | 0train | 0.879244 | **0.838349** |
| 1 | 1test  | 0.797454 | |
| 2 | 0train | 0.493056 | **0.582851** |
| 2 | 1test  | 0.672647 | |
| 3 | 0train | 0.931520 | **0.942419** |
| 3 | 1test  | 0.953318 | |
| 4 | 0train | 0.660108 | **0.624325** |
| 4 | 1test  | 0.588542 | |

**Pipeline mean (subject-balanced): 0.746986**

Compare to the epochs=50/batch_size=16 canonical run earlier the same day:
subj1=0.799 subj2=0.568 subj3=0.946 subj4=0.539, mean=0.713. Every subject
improved except subject 3 (already near ceiling, -0.004, noise-level).
Subject 4 saw the biggest jump (0.539→0.624).

## Epoch timing

Extracted from `experiment.log`, all 8 outer-CV folds (4 subjects x 2
sessions), 100 epochs each, `device=auto` resolved to `cpu`:

| fold (chronological) | epochs | avg epoch_time |
| --- | --- | --- |
| 1 | 100 | 0.406s |
| 2 | 100 | 0.421s |
| 3 | 100 | 0.473s |
| 4 | 100 | 0.515s |
| 5 | 100 | 0.519s |
| 6 | 100 | 0.501s |
| 7 | 100 | 0.516s |
| 8 | 100 | 0.533s |

**Overall average: 0.485s/epoch** across all 800 epoch-records. Note this
climbs steadily fold-over-fold (0.406s -> 0.533s) despite every fold having
the same 144 trials/session -- likely background system load accumulating
over the ~7-minute run rather than a property of the model itself; the
isolated single-fold timing tests earlier today (subject 1 only, nothing else
running) measured ~0.31-0.40s/epoch at batch_size=8/16, consistent with the
low end of this range.

## Full effective parameters

```
batch_size: 8
channel_embed_dim: 8
channel_encoder_dilation: 5
channel_subset: [1, 5, 7, 8, 9, 10, 11, 13, 17]
coherence_threshold: 0.5
coi_enabled: True
cwt_resample_n_time: None
device: 'auto'  (resolved to cpu)
early_stopping_patience: None
epochs: 100
grad_clip_norm: 0.1
hidden_dim: 8
highest: 35.0
last_batch_min_ratio: 0.0
learning_rate: 0.001
lowest: 8.0
nfreqs: 16
noise_apply_prob: 0.0
noise_augmentation_enabled: False
noise_bank_seed: None
noise_bank_size: 128
noise_strength: 0.0
normalize_input: True
optimizer_step_batch_mode: 'credit'
optimizer_step_batch_size: None
optimizer_step_remainder_policy: 'flush'
phase_threshold_deg: 30.0
raw_x_resample_n_time: None
sampling_rate: 250
seed: 42
selector_alpha_val_update_rate: 1.0
smooth_kernel_sigma: (None, None)
smooth_kernel_size: (5, 3)
validation_group_column: None
validation_split: 0.2
verbose: 2
weight_decay: 0.0001
```

See also [[sparse-evidence-gnn-native-resolution-fix]] and
[[run-wct-gnn-concurrent-write-race]] for the investigation and known issues
behind this configuration.
