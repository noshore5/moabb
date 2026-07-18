# Experimental Subject-Scope Policy

Default subject policy for BNCI2014-001 runs. Research plans that **explicitly**
name a different subject set override this; otherwise follow this file.
Read before choosing `--subjects`; update the tracker after decision-informing runs.

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
| Batch whole-pipeline check | `1 2 3 4` | after a few series on 1 |
| Wider / 5–9 | held back | not without approval |

Do not use 1–4 for cheap frequent tests — wastes compute and burns 2–4's budget.

## Exploitation tracker

Integer = decision-informing uses under this policy. `-` = untracked / already
overused. Bump 2–4 by +1 each 1–4 batch run. Keep 5–9 at 0 until approved.

| Subject | Count | Notes |
|---|---|---|
| 1 | - | default cheap subject; stop tracking |
| 2 | 1 | +1 per 1–4 batch |
| 3 | 1 | +1 per 1–4 batch |
| 4 | 1 | +1 per 1–4 batch |
| 5 | 0 | held back |
| 6 | 0 | held back |
| 7 | 0 | held back |
| 8 | 0 | held back |
| 9 | 0 | held back |

**Update:** after a decision-informing run, increment each used subject except 1
(stays `-`). On approving a held-back subject: note date/reason, start its count.
