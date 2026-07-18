# Experimental Subject-Scope Policy

Default subject policy for BNCI2014-001 runs. Research plans that **explicitly**
name a different subject set override this; otherwise follow this file.
Read before choosing `--subjects`; update the tracker after counted runs (below).

## Why

Same subjects used for many pipeline/hyperparam decisions become overfit as
evidence (even when unused for gradients). Cap who is in play; track spend.

- **Compute:** a single subject is cheapest. Subject 1 is reserved for frequent iteration.
- **Hygiene:** more use → less trust as fresh evidence later.

## Allowed scope

Until a wider scope is **approved** (and noted here):

| Use | Subjects | When |
|---|---|---|
| Frequent / ops / components / params | `1` | default; many iterations |
| Batch whole-pipeline check | `1 2 3 4` | after a keeper on 1 (below) |
| Wider / 5–9 | frozen | until allowed |

Do not use 1–4 for cheap frequent tests — wastes compute and burns 2–4's budget.

When 5–9 are allowed, default them to confirmation (spend sparingly on frozen
candidates), not a third development pool — they are the only within-dataset
held-out set.

**Promote to 1–4 only for a "keeper on 1"** — not a tiny score bump. Any of:

- Significant gain: ≳ 0.02 ROC-AUC on subject 1.
- Motivated change: theoretical / methodological reason to check cross-subject
  even if S1 is roughly flat.
- Cheaper / faster compute at similar result (small decrease tolerated) and/or an
  improvement.

Note: subject 1 results are development-only — never cite them as confirmation.

## Exploitation tracker

Two cumulative counters per subject. A run counts when it (1) finishes, (2) is a
unique config, and (3) reports cross-session outer/test scores; failed/aborted
runs don't count. Subject 1 stays `-` (untracked / already overused).

- **Unique:** +1 per subject per unique config counted.
- **Seeds:** += number of seeds used in that run (so a 3-seed run adds 3).
- Re-running the same config with new seeds does not bump Unique, but adds those
  seeds to Seeds.

On allowing a frozen subject: note date/reason, start its counters from the
first counted run that includes it.

**Spend threshold:** TBD — decide later. As counts grow, revisit whether to
freeze 1–4 for adoption decisions and escalate to 5–9. The Seeds counter (total
outer-score looks) is the better pressure signal than Unique.

| Subject | Unique | Seeds | Notes |
|---|---|---|---|
| 1 | - | - | default cheap subject; stop tracking |
| 2 | 1 | 1 | part of 1–4 batch |
| 3 | 1 | 1 | part of 1–4 batch |
| 4 | 1 | 1 | part of 1–4 batch |
| 5 | 0 | 0 | frozen until allowed |
| 6 | 0 | 0 | frozen until allowed |
| 7 | 0 | 0 | frozen until allowed |
| 8 | 0 | 0 | frozen until allowed |
| 9 | 0 | 0 | frozen until allowed |
