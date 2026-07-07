# Categorical Gating, Gumbel Selection, and Selectable Neural Components

## 0. Purpose

This guide describes a reusable design for categorical neural-network components in `nn_components`.

The core idea is to let a layer or residual site choose among multiple candidate operations, such as:

- activation functions
- convolution kernels
- residual update paths
- zero-update / skip choices
- same-config branches with different random initializations
- cheap vs expensive alternatives

The guide is intentionally broader than a single Gumbel-Softmax wrapper. It explains the surrounding choices that make the idea useful, debuggable, and efficient.

The recommended implementation is not a full neural architecture search framework. It is a small set of primitives that can be used inside normal PyTorch models.

---

## 1. Collected idea checklist

This section checks that the main ideas discussed so far are represented in this guide.

### Included user ideas

- Randomly selecting one path during training, then choosing the strongest/best path at inference.
- Using this not only for path dropout, but also for selecting among different candidate blocks.
- Using Gumbel-Softmax as the differentiable version of hard categorical selection.
- Selecting among activation functions.
- Selecting activation functions per layer, per channel, or per hidden unit.
- Selecting among different convolutional choices, such as different kernel sizes.
- Selecting among same-configuration branches with different random seeds as a kind of local lottery-ticket / local seed search.
- Using a zero-update candidate inside residual path selection.
- Using learned selector logits to estimate usefulness.
- Thinking about whether a gate or scale parameter should control both selection and residual amplitude.
- Considering hard vs soft selection modes.
- Considering temperature behavior.
- Considering whether hard Gumbel respects the intended selection probabilities.
- Considering the inaccuracy/bias of straight-through gradients when candidates have different gradient curves.
- Considering whether non-selected options receive gradients.
- Considering efficiency: search-time may compute many branches, but final exported model should compute only one.
- Wanting the result as a reusable `nn_components` primitive, not a one-off model hack.

### Included response ideas

- Keep selection logits separate from residual write scales.
- Select by logits/probabilities or validation ablation, not by raw residual amplitude.
- Use `ZeroUpdate`, not `Identity`, as the residual no-op candidate.
- Normalize branch outputs before gating when branches have different output scales.
- Avoid post-gate normalization if the gate magnitude is meant to matter.
- Use soft warmup before hard Gumbel for parameterized branches.
- Support hard Gumbel, but expose the fact that the backward pass is a biased soft surrogate.
- Include modes for `soft`, `gumbel_soft`, `gumbel_hard`, `argmax`, and `frozen`.
- Distinguish hard-forward behavior from backward-gradient behavior.
- Support optional cost regularization, entropy logging, and exploration floor.
- Provide export/pruning support.
- Fine-tune after pruning/export.
- Treat advanced mechanisms such as REINFORCE, hard-concrete gates, top-k routing, MoE dispatch, and automatic pruning as optional/future extensions, not required v1 behavior.

### Intentionally not solved here

This guide does not design:

- a complete NAS training loop
- a distributed mixture-of-experts system
- automatic architecture search across whole model graphs
- input-dependent routing for every token/pixel/timepoint
- a full pruning/fine-tuning pipeline
- a scheduler framework

The goal is a clean and useful component layer, not a full research platform.

---

## 2. Core intuition

A normal layer has one operation:

```text
y = op(x)
```

A selectable layer has several candidates:

```text
candidate_0(x)
candidate_1(x)
candidate_2(x)
...
candidate_K(x)
```

and a learned categorical selector chooses or mixes them:

```text
y = combine(selector, candidates)
```

The selector has logits:

```text
alpha = learnable scores over candidates
```

These logits define probabilities:

```text
p = softmax(alpha)
```

Depending on mode, the component may:

- mix all candidates softly
- sample a soft stochastic mixture
- sample a hard one-hot candidate using Gumbel-Softmax
- deterministically choose the argmax candidate
- use a manually frozen candidate

The same gating concept can be reused for activation choice, path choice, kernel choice, and residual update choice.

---

## 3. The main abstraction: categorical gate

The lowest-level primitive should be a categorical gate.

It should not know about activations, convolutions, residuals, EEG, or image tensors. It only knows how to produce categorical weights.

Conceptually:

```text
CategoricalGate:
    owns logits alpha
    produces weights over K choices
    reports useful logging info
```

The gate should return two things:

```text
weights
gate_info
```

`weights` are used to combine candidates.

`gate_info` is used for logging, debugging, regularization, and export.

Useful `gate_info` fields:

- current probabilities
- selected index if hard/argmax/frozen
- entropy
- temperature
- mode
- sampled counts if available
- expected candidate cost if costs are provided
- raw logits
- exploration epsilon if used

This separation matters. It lets higher-level components reuse the same gate behavior without duplicating categorical logic.

---

## 4. Selection logits vs residual scale

Use separate parameters for two different jobs:

```text
alpha = selection logits
gamma = optional residual/write scale
```

`alpha` answers:

```text
Which candidate should be selected?
```

`gamma` answers:

```text
How strongly should the selected update write into the residual stream?
```

Do not make one parameter do both jobs in v1.

Bad conceptual design:

```text
selection frequency and residual amplitude both depend on gamma
```

This entangles usefulness with scale. A path may look useful just because it is louder, or it may be selected less because its useful update is naturally small.

Better design:

```text
weights = gate(alpha)
update = selected_or_mixed_candidate(x)
y = shortcut(x) + residual_scale * update
```

Optional per-path `gamma_i` can be added later, but final selection should usually be based on `alpha`, not on raw `abs(gamma_i)`.

### 4.1 Optimizing the selection logits

`alpha` is a learnable parameter, but it must not be trained the same way as branch weights. Three rules matter.

#### Do not let selection minimize training loss alone

If `alpha` and branch weights are updated by the same training-loss gradient in the same step, the selector optimizes training fit. It will then prefer whichever branch drives training loss down fastest, usually the highest-capacity or loudest branch. That is memorization, not generalizable usefulness, and it directly contradicts the goal in section 4 of separating usefulness from scale.

The established fix (DARTS-style bi-level optimization) trains weights on the training split and `alpha` on a held-out validation split, in alternating steps:

```text
step A: update branch weights on a training minibatch
step B: update alpha on a validation minibatch
```

For parameterized branches (kernel selection, path selection, same-config seed selection), prefer validation-split updates for `alpha`. For parameter-free activation selection, joint training on the training loss is usually acceptable and simpler.

#### Give `alpha` its own optimizer parameter group

Selection logits usually need a different (often higher) learning rate than conv/linear weights so they can move within the search budget. Put `alpha` in a dedicated parameter group.

#### Use zero weight decay on `alpha`

Weight decay pulls parameters toward zero. Zero logits mean a uniform softmax, so weight-decayed logits are silently biased toward staying uncommitted or toward an arbitrary attractor. This corrupts the selection signal and is a common silent bug.

```text
alpha parameter group:
    weight_decay = 0
    separate (often larger) learning rate
    trained on validation split for parameterized branches
```

#### Two independent settings

These two behaviors should be toggleable separately, not bundled into one flag:

```text
alpha_update_split: "train" | "val"
    "train": alpha is updated on the same training minibatch as weights
    "val":   alpha is updated on a held-out validation minibatch (bi-level)

alpha_optim: "shared" | "separate"
    "shared":   alpha shares the main optimizer group
    "separate": alpha is placed in its own optimizer group with
                weight_decay = 0 and its own learning rate
```

Recommended defaults:

```text
parameter-free activation selection:
    alpha_update_split = "train"
    alpha_optim        = "separate"   (still keep zero weight decay)

parameterized branch selection (kernel/path/seed):
    alpha_update_split = "val"
    alpha_optim        = "separate"
```

#### Responsibility split

This is a shared contract, not one owner:

```text
component:
    exposes alpha (selection_parameters()) so it can be isolated
    declares the intended alpha_update_split and alpha_optim settings
    does NOT own the data split or the optimizer

trainer:
    reads those settings
    builds the separate optimizer group (zero weight decay) if requested
    feeds validation minibatches for alpha if val-split is requested
```

A `nn.Module` cannot own a data split or an optimizer, so the trainer must enforce these. The component's job is only to expose `alpha` and carry the declared settings so the wiring is unambiguous for whoever builds the training loop.

---

## 5. Soft selection

Soft selection uses deterministic softmax weights.

```text
weights = softmax(alpha / temperature)
output = sum_i weights[i] * candidate_i(x)
```

### Behavior

All candidates contribute to the forward pass.

All candidates receive gradient.

The selector learns smoothly.

### Useful for

- warmup
- activation mixtures
- debugging search spaces
- early training of parameterized candidate branches
- small data regimes where hard stochastic routing is too noisy

### Main risk

The soft mixture may become a useful ensemble-like operation that does not correspond well to any single candidate.

This is the soft-to-hard gap:

```text
soft supernet works
argmax pruned model performs worse
```

Soft selection is stable, but it is not always faithful to the final discrete model.

---

## 6. Gumbel-soft selection

Gumbel-soft selection adds stochastic Gumbel noise, but keeps a soft mixture.

```text
g = Gumbel noise
weights = softmax((alpha + g) / temperature)
output = sum_i weights[i] * candidate_i(x)
```

### Behavior

All candidates still contribute.

All candidates still receive gradient.

The mixture changes stochastically per forward pass.

### Useful for

- exploration without hard one-hot routing
- making the selector less deterministic early
- testing whether stochastic categorical noise helps

### Practical importance

This is useful, but less essential than plain `soft` and `gumbel_hard`.

It is a good mode to support if the implementation is simple, because it uses nearly the same machinery as hard Gumbel.

---

## 7. Hard Gumbel selection

Hard Gumbel uses a one-hot selection in the forward pass and a soft surrogate in the backward pass.

Conceptually:

```text
forward:
    choose one candidate

backward:
    pretend a differentiable soft choice was used
```

This is usually implemented with a straight-through estimator.

### Does hard Gumbel respect the selection distribution?

Yes, if logits are `alpha`, hard Gumbel samples one-hot choices according to the categorical distribution implied by the logits.

Over many forward passes:

```text
if softmax(alpha) = [0.8, 0.2]

then option 0 is selected about 80% of the time
and option 1 is selected about 20% of the time
```

Temperature mainly affects the softness of the backward surrogate. In the usual Gumbel-Softmax hard setup, temperature does not give a simple “80/20 becomes sharper/flatter” interpretation for the hard argmax frequency. It mostly changes the gradient shape.

### Useful for

- architecture/path selection
- training behavior closer to final discrete inference
- forcing candidates to stand alone
- reducing co-adapted soft mixtures late in training

### Main risk

The backward gradient is biased.

It is not the true gradient of a discrete categorical sample. It is the gradient of a continuous relaxation.

This matters when candidates have very different behavior, for example:

- ReLU vs GELU vs tanh
- small conv kernel vs large conv kernel
- zero-update vs expensive residual block
- same-config branches that train at different speeds

---

## 8. Hard-Gumbel backward variants

A useful implementation should distinguish the hard forward pass from how gradients are assigned.

Two variants are worth supporting.

---

### 8.1 `selected_only` backward

The hard one-hot weights are used directly in the candidate combination.

Conceptually:

```text
output = selected_candidate(x)
```

The selector logits still receive a straight-through surrogate gradient, but candidate parameters mostly receive gradient only when selected.

#### Useful for

- late hardening
- making candidates stand alone
- approximating final argmax behavior
- cheap-ish search if implemented with lazy evaluation

#### Risks

- rare candidates undertrain
- early winners get richer
- parameterized branches can collapse before they have a chance to learn
- same-seed/local-ticket experiments become very sensitive to early noise

This should usually not be the first mode used for parameterized path search.

---

### 8.2 `soft_all` backward

The forward pass uses the selected candidate, but the backward pass behaves like the soft mixture.

Conceptually:

```text
hard_output = selected_candidate(x)
soft_output = sum_i soft_weight[i] * candidate_i(x)

output = hard_output with gradients from soft_output
```

This requires computing all candidate outputs.

#### Exact straight-through estimator

Prose is not enough here; this is exactly where implementations introduce subtle bugs. Pin the estimator explicitly.

```python
# w_hard: one-hot weights [K] (selected branch = 1, else 0)
# w_soft: softmax((alpha + g) / tau) [K], differentiable in alpha
# both index the same K candidates in the same order
w = w_hard + (w_soft - w_soft.detach())
output = sum_i w[i] * candidate_i(x)
```

Why this works:

```text
forward:
    (w_soft - w_soft.detach()) is numerically zero
    so the forward value equals w_hard (true one-hot execution)

backward:
    w_hard is a constant, w_soft.detach() has no gradient
    so gradient flows only through w_soft
    every candidate receives gradient proportional to its soft weight
```

That soft gradient path is what keeps rarely-selected parameterized branches training.

Correctness traps:

- Do not add a second `detach()` around `w_hard` in a way that removes the straight-through path for the logits. The logits get their gradient through `w_soft`; `w_hard` is treated as a constant offset.
- All `candidate_i(x)` must actually be evaluated, so `soft_all` is incompatible with lazy single-branch evaluation.
- For `selected_only`, the forward is the same one-hot combination `sum_i w_hard[i] * candidate_i(x)`, and the straight-through `w_hard + (w_soft - w_soft.detach())` is applied to the gate weights so the logits still get a surrogate gradient, but non-selected branch parameters receive no gradient.

#### Useful for

- parameterized path selection
- kernel-size selection
- same-config different-seed branch selection
- early and mid search phases where all branches still need training

#### Risks

- more expensive
- the gradient is more biased relative to true hard execution
- training-time behavior is still partly supernet-like

Despite the cost, this is often the practical default for trainable candidate branches.

---

## 9. Argmax mode

Argmax mode chooses the highest-logit candidate deterministically.

```text
selected = argmax(alpha)
output = candidate_selected(x)
```

### Useful for

- evaluation after search
- export
- pruning
- final model deployment

Argmax mode should compute only the selected candidate when possible.

This is the efficient final mode.

---

## 10. Frozen mode

Frozen mode uses a manually specified candidate index.

```text
selected = configured_index
```

It ignores learned logits.

### Useful for

- ablations
- reproducing a selected architecture
- forcing a known-good choice
- exported/final models
- comparing candidates fairly

Frozen mode should also compute only the chosen candidate when possible.

---

## 11. Temperature

Temperature controls the sharpness of soft relaxations.

High temperature:

```text
flatter distribution
more exploration
smoother gradients
less commitment
```

Low temperature:

```text
sharper distribution
closer to argmax
more brittle gradients
more commitment
```

### In soft mode

Temperature directly controls mixture sharpness.

```text
high temperature:
    many candidates contribute

low temperature:
    one candidate dominates
```

### In hard Gumbel mode

Forward is one-hot, but temperature controls the soft surrogate used in the backward pass.

```text
high temperature:
    smoother selector gradients

low temperature:
    sharper, noisier, more argmax-like gradients
```

### Recommended lifecycle

A simple practical lifecycle:

```text
phase 1:
    soft, temperature 1.0 or 2.0

phase 2:
    soft or gumbel_soft, temperature decays toward 0.5

phase 3:
    gumbel_hard, temperature 0.5 or 0.25

phase 4:
    argmax or frozen

phase 5:
    fine-tune exported model
```

The component should not own a complex scheduler. The trainer can update temperature.

---

## 12. Activation selection

Activation choice is the simplest and safest use case.

Candidates may include:

- ReLU
- GELU
- SiLU
- ELU
- tanh
- identity

The candidates are parameter-free, so there is no branch-undertraining problem.

### Layer-level activation choice

One activation is selected or mixed for the whole layer.

This is the recommended starting point.

Advantages:

- stable
- interpretable
- cheap
- easy to export

Example use:

```text
layer chooses among ReLU, GELU, SiLU, Identity
```

### Channel-level or hidden-unit-level activation choice

Each channel or hidden feature has its own selector.

This allows different features to prefer different nonlinearities.

Examples:

```text
some channels choose ReLU
some choose SiLU
some choose Identity
```

This can be useful, but it is more complex to log and interpret.

### Per-element activation choice

Each individual timepoint/pixel/token/element has its own activation choice.

This is possible, but should not be v1 default.

Problems:

- expensive
- noisy
- hard to interpret
- resembles dynamic routing rather than simple layer selection
- may overfit small data

Recommended stance:

```text
support layer-level first
optionally support channel-level
avoid per-element in v1
```

---

## 13. Selecting different convolution/path parameters

This is one of the highest-value use cases.

Candidates may encode different inductive biases:

- small kernel
- medium kernel
- large kernel
- dilated kernel
- depthwise convolution
- pointwise/channel mixer
- zero-update

For EEG or 1D signals:

```text
small temporal kernel:
    local transients

medium temporal kernel:
    short motifs

large or dilated kernel:
    slower rhythms or longer dependencies

pointwise/channel mixer:
    cross-channel mixing without temporal context
```

For 2D features such as images or CWT maps:

```text
small kernel:
    local texture

larger kernel:
    broader context

depthwise large kernel:
    cheaper spatial mixing

1x1:
    channel mixing only
```

### Soft path selection

Soft path selection acts like a learnable multi-branch/inception block.

This can perform well but may not prune cleanly to one path.

### Hard path selection

Hard path selection is closer to architecture search.

Use soft warmup first, then harden.

Recommended flow:

```text
soft warmup
gumbel_hard with soft_all backward
optional late selected_only hardening
argmax/frozen export
fine-tune
```

---

## 14. Same-config branches with different seeds

This is an experimental but interesting use case.

Candidates are the same operation/configuration, but initialized differently:

```text
branch 0: ConvBlock config A, seed 0
branch 1: ConvBlock config A, seed 1
branch 2: ConvBlock config A, seed 2
```

This is not selecting a different architectural inductive bias. It is selecting among different local parameter trajectories.

Intuition:

```text
Instead of trying many global seeds and picking one whole model,
each selectable site can pick a local branch that trained well.
```

This resembles:

- local lottery-ticket search
- local ensemble pruning
- supernet branch selection
- overparameterization for optimization

### Why it might help

Different initial branches may learn different useful features. A selector could choose the best branch at each site.

### Why it can mislead

Soft selection may simply build an ensemble. If the soft model performs well but every single candidate performs poorly when forced alone, the selector did not find a better seed; it built a multi-branch model.

Hard selection may collapse early:

```text
branch gets selected slightly more
branch trains more
branch becomes better
branch gets selected even more
```

This rich-get-richer effect can select early luck rather than true potential.

### Recommended handling

Use:

- longer soft warmup
- exploration floor
- no early commitment pressure
- delayed hardening
- final argmax/frozen export
- post-export fine-tuning
- strong comparison against baselines

Baselines should include:

- one normal branch
- a wider branch
- multiple global seeds
- soft ensemble kept at inference
- longer training

This use case should be labeled experimental.

---

## 15. Selectable residual paths

A residual selectable path should select an update, then add it to a shortcut.

Shape-preserving case:

```text
y = x + selected_update(x)
```

Shape-changing case:

```text
y = projection(x) + selected_update(x)
```

### Zero update vs identity

Inside a residual update selector, the no-op candidate should be:

```text
ZeroUpdate
```

not:

```text
Identity
```

Because:

```text
y = x + ZeroUpdate(x) = x
```

but:

```text
y = x + Identity(x) = 2x
```

Identity is not a no-op when it is used as the residual update.

### Useful candidates

- ZeroUpdate
- ConvBlock
- depthwise ConvBlock
- MLP/channel mixer
- different kernel sizes
- same-config branches with different seeds

### Residual scaling

A global residual scale can help stabilize early training:

```text
y = shortcut(x) + residual_scale * selected_update(x)
```

This is separate from selection logits.

### Branch output normalization

For parameterized path selection, branch outputs should be comparable.

Otherwise the selector may prefer whichever branch has larger output scale.

Possible branch output normalization:

- none
- RMS-like no-affine normalization
- LayerNorm no-affine
- GroupNorm no-affine

No-affine normalization is useful because affine parameters can reintroduce scale differences.

For activation choice, do not normalize by default. The activation’s output distribution is part of what is being tested.

---

## 16. Cost regularization

Candidate costs allow the selector to prefer cheaper choices unless expensive ones help.

Example costs:

```text
ZeroUpdate: 0
Conv k=3: 1
Conv k=7: 2
Conv k=15: 4
```

A cost loss can be computed from soft probabilities:

```text
cost_loss = cost_weight * sum_i probability[i] * cost[i]
```

Use probabilities instead of sampled one-hot choices for smoother training.

Useful for:

- kernel-size search
- depthwise vs full convolution
- zero-update vs expensive update
- MLP expansion choices
- final model efficiency

Do not enable cost regularization by default.

---

## 17. Entropy regularization and logging

Entropy measures how committed the selector is.

```text
high entropy:
    uncertain / spread across choices

low entropy:
    committed to one or few choices
```

Entropy can be used in two opposite ways:

```text
encourage exploration:
    maximize entropy early

encourage commitment:
    minimize entropy late
```

The component should always expose entropy for logging.

Optional entropy loss can be supported, but the default should be zero.

Do not apply strong commitment pressure early. It can collapse to random early winners.

---

## 18. Exploration floor

An exploration floor prevents candidates from getting zero probability too early.

Conceptually:

```text
p = (1 - epsilon) * softmax(alpha) + epsilon / K
```

Useful for:

- trainable path candidates
- same-config different-seed branches
- large kernels that learn slowly
- zero-update competition

This is especially useful when hard sampling is used.

The trainer can decay epsilon over time.

Do not force this into every use case. For simple activation selection, it may be unnecessary.

---

## 19. REINFORCE estimator

REINFORCE is an unbiased score-function estimator for discrete choices.

Instead of using a differentiable relaxation, it samples a categorical choice and estimates how the logits should change based on the loss.

Conceptually:

```text
sample choice i ~ Categorical(alpha)
run selected candidate
observe loss
update alpha using log_prob(i) * adjusted_loss
```

### Why it matters

It can train discrete selections without computing all candidate outputs and without pretending the choice was soft.

### Why not v1 default

REINFORCE usually has high variance.

It often needs:

- baselines
- variance reduction
- careful reward/loss handling
- many samples
- more fragile training loops

It is useful when:

- computing all candidates is impossible
- exact discrete sampling is important
- one accepts noisier optimization

Recommended stance:

```text
Mention as future/experimental.
Do not implement in v1 unless there is a strong need.
```

---

## 20. Hard-concrete / L0 gates

Hard-concrete gates are used for sparse on/off decisions.

Instead of choosing exactly one option among K, each candidate gets a stochastic gate:

```text
z_i ≈ 0 or 1
```

The model can learn which paths are active.

### Why L0?

L0 regularization penalizes whether a path is active at all, not how large its weights are.

This matches architecture selection:

```text
Does this branch exist?
```

rather than:

```text
Are this branch’s weights small?
```

### Difference from categorical selection

Categorical selection usually means:

```text
choose one of K
```

Hard-concrete/L0 gates usually mean:

```text
choose any subset of K
```

So it is better for:

- pruning
- optional branches
- sparsifying many candidate paths
- learning whether components should exist

It is less direct for:

- exactly one activation function
- exactly one kernel size
- exactly one branch

Recommended stance:

```text
Future extension.
Useful for pruning and optional-path sparsity.
Not needed for v1 categorical choice.
```

---

## 21. Top-k routing

Top-k routing selects the best K candidates rather than exactly one.

Example:

```text
select top 2 out of 8 experts
```

This is common in mixture-of-experts systems.

### Why use it?

It balances expressiveness and sparsity.

Instead of full soft mixture:

```text
use all candidates
```

and instead of hard one-hot:

```text
use one candidate
```

top-k uses:

```text
use a small subset
```

### Why not v1 default

It adds complexity:

- tie handling
- sparse dispatch
- combining multiple selected candidates
- load balancing
- routing stability
- potentially different behavior between training and inference

Recommended stance:

```text
Not v1.
Consider later if the library grows toward sparse MoE or multi-path routing.
```

---

## 22. Dynamic input-dependent routing

The selector discussed so far is usually static:

```text
alpha is learned parameter
same choice distribution for all inputs
```

Dynamic routing makes selection depend on the input:

```text
alpha = router(x)
```

Then different samples, timepoints, channels, or tokens can choose different candidates.

### Why it is powerful

It allows conditional computation:

```text
easy sample uses cheap path
hard sample uses expensive path
one pattern uses one kernel
another pattern uses another kernel
```

### Why it is risky

It is much more complex:

- router can overfit
- input-dependent choices are harder to debug
- batching/efficiency becomes harder
- logging becomes more complex
- can become a mixture-of-experts problem
- may need load balancing

Recommended stance:

```text
Not v1.
Static learned selectors first.
Input-dependent routers later.
```

---

## 23. Mixture-of-experts dispatch

Mixture-of-experts dispatch is a specialized version of dynamic routing where inputs are routed to expert modules.

It typically involves:

- many experts
- top-k routing
- sparse computation
- load-balancing loss
- dispatch/combination logic
- distributed or memory-sensitive execution

### Why it is related

The same categorical selection ideas appear in MoE routing.

### Why it is not the same as this component

The proposed `nn_components` design is small-scale local selection.

MoE dispatch is a system-level architecture feature.

Recommended stance:

```text
Do not implement MoE dispatch in v1.
Keep the gate API compatible enough that future MoE-like work is not blocked.
```

---

## 24. Stateful norm selection

Selecting among normalization layers seems tempting:

```text
BatchNorm vs GroupNorm vs LayerNorm vs RMSNorm
```

But norms are not always simple stateless candidate functions.

BatchNorm, for example, has running statistics and train/eval behavior.

### Why it is risky

If different norms are candidates, then:

- each norm may maintain different state
- rarely selected BatchNorm paths may have bad running statistics
- train/eval behavior can diverge
- soft mixtures of stateful norms can be hard to interpret
- frozen/exported behavior may not match search behavior

### Safer alternatives

For v1:

- choose norm type by config
- do not make norm type selectable
- if experimenting, prefer stateless/no-running-stat norms
- avoid BatchNorm as a selectable candidate unless carefully controlled

Recommended stance:

```text
Avoid stateful norm selection in v1.
```

---

## 25. Automatic temperature scheduler

Temperature scheduling is important, but the categorical component should not own a complex scheduler in v1.

### Why not put it inside the module?

Training schedules depend on:

- optimizer
- dataset size
- warmup length
- validation behavior
- whether branches are parameterized
- whether final pruning is planned

Putting this logic inside the component makes it harder to reason about.

### Recommended behavior

The component exposes:

```text
temperature
set_temperature(...)
```

The trainer or experiment config updates it.

A simple external schedule is enough:

```text
soft warmup at tau=1.0
decay tau to 0.5
hard Gumbel at tau=0.5
optional decay to 0.25
argmax/frozen export
```

Recommended stance:

```text
Expose temperature control.
Do not implement automatic scheduling in v1.
```

---

## 26. Automatic pruning during training

Automatic pruning means the component removes or disables candidates during training once they look unimportant.

### Why it is appealing

It can reduce compute and force commitment.

### Why it is risky

Early probabilities can be misleading.

A branch may look bad because:

- it was undertrained
- it learns slower
- it has lower output scale
- it is expensive but useful later
- the soft mixture co-adapted
- the current temperature is too high or too low

Pruning too early can permanently remove useful paths.

### Recommended behavior

Do not automatically prune during training in v1.

Instead:

1. train/search
2. inspect probabilities and validation behavior
3. export argmax/frozen model
4. fine-tune
5. compare to original model and baselines

Optional later support:

- manual pruning
- pruning after a specified epoch
- pruning only after validation ablation
- pruning with safety thresholds

Recommended stance:

```text
Provide export/freeze utilities.
Avoid automatic pruning in v1.
```

---

## 27. Efficiency model

Efficiency depends heavily on mode.

### Modes that usually compute all candidates

- soft
- gumbel_soft
- gumbel_hard with `soft_all` backward

These are useful for training/search, but expensive.

### Modes that can compute only one candidate

- argmax
- frozen
- gumbel_hard with `selected_only` backward if selector-gradient requirements are limited

These are useful for final/exported models.

### Practical rule

```text
Search phase:
    compute all candidates if needed for stable learning

Final phase:
    export/freeze and compute only selected candidates
```

Do not expect search-time components to be cheap. The efficiency win comes after pruning/export.

---

## 28. Logging and diagnostics

Categorical components are hard to debug without logs.

Minimum logs:

- probabilities per candidate
- argmax candidate
- entropy
- temperature
- mode
- selected counts for hard modes
- expected cost if costs are configured

For path selection, also consider logging:

- branch output RMS
- branch gradient norm
- branch parameter norm
- forced-candidate validation performance
- post-pruning validation performance

For channel-level activation selection:

- mean probability per activation
- fraction of channels selecting each activation
- entropy distribution across channels

Useful failure signs:

```text
one candidate wins immediately:
    likely early collapse or biased init

expensive branch always wins:
    cost missing or output scale mismatch

zero-update always wins:
    residual branch too weak, too much regularization, or bad init

soft model strong but argmax weak:
    soft-to-hard gap / ensemble behavior

rare branches never improve:
    undertraining, need soft_all backward or exploration floor
```

### Falsification criterion (optional, recommended)

This is an optional evaluation practice, not required v1 module behavior. It is strongly recommended before trusting any selection result.

Logging alone does not tell you whether the selector learned anything useful. The selector will always commit to something, even when candidates are statistically indistinguishable. Define the null explicitly and test it.

A gate has not learned anything useful if the argmax-forced model's validation performance is within noise of both:

```text
- a randomly-frozen-index model
- the single best hand-picked (frozen) candidate
```

Concretely, after search, evaluate each candidate frozen individually. If `argmax` is not reliably better than the best fixed candidate and better than a random fixed candidate, the gate added complexity for no gain. Simplify to a fixed choice or stop.

This is especially important on small datasets, where the difference in validation loss between candidates is often within noise.

### Seed-stability check (small-data / EEG regime) (optional, recommended)

This is an optional evaluation practice, not required v1 module behavior. It is strongly recommended in the small-data / EEG regime.

On small data (for example EEG with hundreds of trials, or MOABB cross-session / cross-subject splits), a single search run cannot distinguish a real inductive-bias preference from fitting noise.

Run the search across several global seeds and/or folds and record the argmax index per selectable site.

```text
if the winning index is stable across seeds/folds:
    the selection reflects a real preference

if the winning index flips across seeds/folds:
    the selector is fitting noise, do not report "the data prefers op X"
```

Useful summaries:

- per-site selection agreement (fraction of seeds/folds choosing the modal index)
- entropy of the argmax index distribution across seeds/folds
- whether within-session winners disagree with cross-subject winners

Only trust and report a selection that is stable across seeds and across the relevant MOABB split.

---

## 29. Recommended v1 components

### 29.1 `CategoricalGate`

Low-level selector.

Responsibilities:

- own logits
- produce weights
- handle mode
- handle temperature
- expose probabilities and diagnostics
- optionally apply exploration floor
- optionally compute entropy and expected cost

Should not know about candidate modules.

---

### 29.2 `CategoricalChoice`

Combines same-shape candidate outputs.

Responsibilities:

- receive candidate callables or already-computed candidate outputs
- query `CategoricalGate`
- combine candidates according to mode
- support dense and lazy evaluation where practical
- expose `gate_info`

Should validate that candidate outputs are compatible.

---

### 29.3 `SelectableActivation`

Convenience wrapper for activation functions.

Responsibilities:

- construct activation candidates
- support layer-level and optionally channel-level gating
- use `CategoricalGate`
- export to selected activation

Recommended default:

```text
candidates: ReLU, GELU, SiLU, Identity
scope: layer
mode: soft
temperature: 1.0
```

---

### 29.4 `SelectPath`

Selects among same-output-shape candidate modules.

Responsibilities:

- hold candidate modules
- optionally include `ZeroUpdate`
- apply optional branch output normalization
- use `CategoricalGate`
- support cost regularization info
- support export/freeze

Use for:

- kernel-size selection
- depthwise vs full conv
- MLP branch vs conv branch
- same-config different-seed branches

---

### 29.5 Residual path selection (composition recipe, not a new component)

There is no dedicated `SelectPathResidual` component in v1. Residual path selection is expressed by composing the existing library `ResidualWrapper` around a `SelectPath`:

```python
ResidualWrapper(branch=SelectPath([ZeroUpdate, ConvBlock, ...]), cfg=...)
```

`ResidualWrapper` already owns the shortcut (identity or projection), residual scale, `rezero`, `layer_scale`, `drop_path`, norm positions, and shape-mismatch checking. `SelectPath` only needs to select among same-output-shape update candidates, so a single shortcut/projection is sufficient.

Conceptual formula (handled by `ResidualWrapper`):

```text
same shape:
    y = x + scale * SelectPath(x)

shape-changing:
    y = projection(x) + scale * SelectPath(x)
```

Rules:

- the no-op candidate is `ZeroUpdate`, not `Identity` (if selected, `y = x`)
- set `cfg.drop_path = 0` during search to avoid double stochasticity (see section 36.6)
- do not reimplement residual logic; reuse `ResidualWrapper`

Export is handled by the tree-walking exporter, which replaces the inner `SelectPath` with its `export()` result and leaves the surrounding `ResidualWrapper` intact (see section 36.5).

---

## 30. Interface sketch

This is not final code. It is an implementation target for an agent.

### `CategoricalGateConfig`

Suggested fields:

```text
num_choices
mode
temperature
scope                 # "layer" | "channel"
num_features          # required when scope="channel" (see section 36.4)
channel_dim           # default 1 (conv); -1 for dense
logits_init
eval_mode
exploration_epsilon
cost_weight
entropy_weight
gradient_mode
frozen_index
alpha_update_split    # "train" | "val"  (declared intent; trainer enforces)
alpha_optim           # "shared" | "separate" (separate => own group, weight_decay=0)
```

Keep defaults simple.

Default mode can be `soft`.

Default logits initialization should be uniform.

`alpha_update_split` and `alpha_optim` are declared training settings, not
things the module executes itself (see section 4.1). The module carries them so
the trainer can build the correct optimizer groups and feed the correct split.

### `CategoricalGate`

Suggested methods:

```text
forward(...)
    returns weights and gate_info

probabilities()
    returns softmax probabilities without sampling

set_temperature(value)

freeze(index=None)
    switches to frozen/argmax behavior

selection_parameters()
    returns the alpha logits so the trainer can place them in a
    separate optimizer group (zero weight decay, own learning rate;
    see section 4.1)

extra_loss(gate_info)
    optional entropy/cost helper if configured
```

### `CategoricalChoice`

Suggested behavior:

```text
given candidates:
    compute or lazily evaluate candidates according to mode
    combine outputs using gate weights
    return output and gate_info
```

Candidate outputs must be shape-compatible.

### `SelectableActivation`

Suggested behavior:

```text
input tensor -> candidate activations -> categorical choice -> output
```

Should support:

```text
layer scope
channel/feature scope if dimension can be inferred/configured safely
```

Avoid per-element scope in v1.

### `SelectPath`

Suggested behavior:

```text
input tensor -> candidate modules -> categorical choice -> output
```

Should support:

```text
candidate costs
optional branch output norm
export selected path
```

### Residual path selection (composition, no new interface)

Not a separate component. Compose the existing `ResidualWrapper` around a `SelectPath`:

```text
ResidualWrapper(branch=SelectPath([ZeroUpdate, ...]), cfg=...)

output = shortcut(x) + residual_scale * SelectPath(x)
```

`ResidualWrapper` provides the shortcut/projection, residual scale, and shape checks. No new class or config is introduced (see section 29.5).

---

## 31. Use case examples

### 31.1 Activation choice

Goal:

```text
Let each layer learn whether it prefers ReLU, GELU, SiLU, or Identity.
```

Recommended:

```text
start with layer-level soft mode
optionally harden later
export argmax activation
```

Good first experiment because activations are cheap and parameter-free.

---

### 31.2 Kernel-size selection

Goal:

```text
Let a convolutional site choose the useful receptive field.
```

Candidates:

```text
ZeroUpdate
small kernel conv
medium kernel conv
large kernel conv
depthwise large kernel
pointwise mixer
```

Recommended:

```text
soft warmup
gumbel_hard with soft_all backward
argmax/frozen export
fine-tune
```

Use candidate costs if large kernels are much more expensive.

---

### 31.3 Same-config seed selection

Goal:

```text
Let a layer choose among several independently initialized copies of the same block.
```

Recommended:

```text
longer soft warmup
exploration floor
delayed hardening
careful baselines
final fine-tune
```

Treat as experimental.

---

### 31.4 Residual update selection

Goal:

```text
Let a residual block decide whether to use no update, a small update, a large update, or a different update type.
```

Use `ZeroUpdate` for no-op.

Use branch normalization if candidate outputs have different scales.

Do not use `Identity` as the residual no-op update.

---

## 32. Efficiency and performance notice

Search-time selectable components may be much slower than normal components.

This is expected.

Training with dense soft or soft-backward modes often requires computing all candidates.

The final model should not pay this cost.

The intended workflow is:

```text
1. train/search with selectable components
2. inspect logs
3. export argmax/frozen choices
4. replace selectable components with ordinary modules
5. fine-tune
6. evaluate final compact model
```

Do not compare search-time compute directly to final model compute without reporting the distinction.

For fair experiments, compare against:

- normal single-path baseline
- wider baseline with similar parameters
- soft ensemble kept at inference
- multiple random seeds
- longer training baseline

---

## 33. Details that can make it buggy or incorrect

### Shape mismatch

Residual addition requires matching shapes.

If a selected path changes channels or spatial/time dimensions, the shortcut must project to the same output shape.

Do not silently broadcast or crop.

### Identity vs zero update

In residual update selection, no-op means `ZeroUpdate`.

Using `Identity` as an update doubles the residual stream.

### Scale mismatch

If candidates have different output scales, the selector may choose the loudest branch.

Use fair initialization and optional branch output normalization.

### Selection logits vs residual scales

Do not use the same parameter for selection frequency and output amplitude.

Keep `alpha` and `gamma` separate.

### Stateful candidates

Be careful with BatchNorm and other stateful layers.

Rarely selected stateful branches may have bad running statistics.

Avoid selectable stateful norms in v1.

### Hard Gumbel misconception

Hard Gumbel forward samples one-hot choices, but backward uses a biased soft surrogate.

Do not assume the gradient is exact.

### Undertrained branches

Parameterized branches that are rarely selected may not learn.

Use soft warmup, `soft_all` backward, or exploration floor.

### Soft-to-hard gap

A soft mixture may work better than any single candidate.

Always evaluate exported argmax/frozen models.

### Automatic pruning risk

Do not prune too early based only on logits.

Validate, export, and fine-tune.

### Logging absence

Without logs, it is difficult to know whether the selector learned anything meaningful.

Logging should be treated as part of the component, not optional decoration.

---

## 34. Recommended v1 implementation plan for an agent

### Step 1: implement `CategoricalGate`

Implement a small gate module with:

- learnable logits
- modes: `soft`, `gumbel_soft`, `gumbel_hard`, `argmax`, `frozen`
- temperature
- optional exploration epsilon
- optional frozen index
- diagnostic output

Do not implement REINFORCE, hard-concrete, top-k, or dynamic routing in v1.

### Step 2: implement `CategoricalChoice`

Implement a wrapper that combines same-shape candidate outputs using a gate.

Support the two hard-Gumbel gradient modes:

```text
selected_only
soft_all
```

Make candidate evaluation efficient where possible:

```text
dense/search modes:
    compute all candidates

argmax/frozen:
    compute only selected candidate
```

### Step 3: implement `SelectableActivation`

Build a convenience component using `CategoricalChoice`.

Start with layer-level selection.

Optionally add channel-level selection if it fits existing tensor conventions cleanly.

Do not add per-element activation selection in v1.

### Step 4: implement `SelectPath`

Build a module for selecting among candidate modules.

Add:

- optional `ZeroUpdate`
- optional candidate costs
- optional branch output norm
- export/freeze support

Require candidate outputs to be compatible.

### Step 5: document residual path selection via composition

Do not build a separate residual component. Residual path selection is the existing `ResidualWrapper` composed around a `SelectPath` (see sections 29.5 and 36.6):

```python
ResidualWrapper(branch=SelectPath([ZeroUpdate, ...]), cfg=...)
```

Rules:

- shape match allows identity shortcut; shape change requires a projection shortcut
  (both provided by `ResidualWrapper`)
- no-op candidate is `ZeroUpdate`
- residual scale is `ResidualWrapper`'s job, separate from selection logits
- set `cfg.drop_path = 0` during search

### Step 6: add logging helpers

Expose easy diagnostics:

- probabilities
- argmax choice
- entropy
- temperature
- selected counts
- expected cost
- mode
- gradient mode

Logging should work for layer-level and channel-level gates.

### Step 7: add export/freeze utilities

Provide a way to convert searched components into ordinary modules.

Expected export behavior:

```text
SelectableActivation -> selected activation
SelectPath -> selected candidate path
ResidualWrapper(SelectPath) -> ResidualWrapper with the inner SelectPath replaced by its selected branch
```

Allow manual frozen index for ablations.

#### Export by extracting the selected submodule

Export must produce an ordinary module that computes only the winning branch, not the supernet run in argmax mode (which keeps all K branches in memory and gives no efficiency win, defeating sections 27 and 32).

Because candidates are stored as real registered `nn.Module`s (see the implementation contract in section 36), export does not need to rebuild a module from config or `load_state_dict`. The winning branch already exists as a trained submodule, so export just extracts it:

```text
SelectPath.export():
    k* = argmax(alpha), or the frozen index
    module = deepcopy(candidates[k*])
    if a branch-output norm wrapped candidate k* during search:
        include that same branch-norm (numeric fidelity)
    drop the other candidates and the gate
    return module
```

For a residual site (`ResidualWrapper(branch=SelectPath(...))`), a tree-walking exporter replaces the inner `SelectPath` with its `export()` result and leaves the surrounding `ResidualWrapper` (shortcut, residual scale, norms) intact. No residual reconstruction is needed.

Weight transplantation is automatic because the returned module is the same trained object (deep-copied), not a re-initialized one.

After export, re-estimate BatchNorm running statistics with a forward pass over training data before fine-tuning (search-time BN stats are stale; see sections 24 and 33).

To support this, each selectable component must keep the branch-output norm paired with its branch (for example `SelectPath` holds `(branch, branch_norm)` pairs) so the extracted module stays numerically faithful.

Add a test that the exported module reproduces the supernet's argmax-mode output on a fixed batch, within BatchNorm-stat tolerance.

Allow manual frozen index for ablations.

### Step 8: add tests

Minimum tests:

- probabilities sum to one
- frozen mode selects configured index
- argmax mode selects max-logit candidate
- eval/export mode is deterministic
- shape mismatch raises a clear error
- `ZeroUpdate` in residual produces true no-op update
- hard modes produce one-hot forward weights
- soft mode gives dense weights
- `soft_all` backward can propagate gradients to non-selected candidates
- `soft_all` forward value equals the one-hot (hard) execution, not the soft mixture
- `selected_only` backward does not unexpectedly train non-selected branches
- argmax/frozen can avoid computing unselected expensive candidates if lazy evaluation is implemented
- exported module reproduces supernet argmax-mode output on a fixed batch, within BatchNorm-stat tolerance

### Step 9: document recommended training lifecycle

Document the default lifecycle:

```text
soft warmup
temperature decay
gumbel_hard hardening
argmax/frozen export
fine-tune
```

Also document that same-config seed selection is experimental and must be compared against simpler baselines.

---

## 35. Final recommended scope

For v1, implement:

```text
CategoricalGate
CategoricalChoice
SelectableActivation
SelectPath
ZeroUpdate
```

Residual path selection is not a separate component; it is `ResidualWrapper(branch=SelectPath(...))` (see section 29.5).

With:

```text
soft
gumbel_soft
gumbel_hard
argmax
frozen
```

And:

```text
selected_only hard backward
soft_all hard backward
temperature control
exploration floor
entropy logging
optional cost loss
branch output normalization
export/freeze support (weight-transplanting)
alpha exposure + declared alpha_update_split / alpha_optim settings
```

Optional but recommended evaluation practices (not required module behavior):

```text
falsification criterion (argmax vs best-fixed and random-fixed candidates)
seed-stability check across seeds / MOABB folds
```

Leave these for later:

```text
REINFORCE estimator
hard-concrete / L0 gates
top-k routing
per-element activation choice
dynamic input-dependent routing
mixture-of-experts dispatch
stateful norm selection
automatic temperature scheduler
automatic pruning during training
```

The guiding principle:

```text
v1 should make local categorical choices easy, inspectable, and exportable.
It should not try to become a full architecture-search framework.
```

---

## 36. Implementation contract

This section pins the concrete decisions an agent must not guess. It resolves the ambiguities left open by the conceptual sections above. Where it conflicts with earlier prose, this section wins.

### 36.1 File layout

Follow the one-concept-per-file convention already used in `nn_components`.

```text
nn_components/selectable.py
    all runtime logic and the GateInfo container:
        ZeroUpdate
        GateInfo
        CategoricalGate
        CategoricalChoice
        SelectableActivation
        SelectPath
    (no SelectPathResidual: residual path selection is
     ResidualWrapper(branch=SelectPath(...)); see section 29.5)

nn_components/configs.py
    all config dataclasses (consistent with existing configs living here):
        CategoricalGateConfig
        any SelectPathConfig

nn_components/__init__.py
    export the new public classes and configs
```

Config dataclasses go in `configs.py`, not in `selectable.py`, because every other config in the package lives there. `GateInfo` stays in `selectable.py` because it is a runtime return value, not a config.

### 36.2 Candidate API

Candidates are passed as callables (`nn.Module`s are callable).

```text
- candidates are registered in an nn.ModuleList so their parameters are
  always tracked and checkpointed, even when not invoked on a given forward
- invocation is lazy per mode:
      soft, gumbel_soft, gumbel_hard(soft_all backward): call all candidates
      argmax, frozen, gumbel_hard(selected_only backward): call only the selected candidate
- registration is never lazy; only invocation is
```

This is what makes the exported/inference path compute a single branch while search-time dense modes compute all branches.

### 36.3 `GateInfo` schema

`CategoricalGate.forward` returns `(weights, GateInfo)`. `GateInfo` is a lightweight dataclass in `selectable.py` (not frozen, because it holds live tensors).

```python
@dataclass
class GateInfo:
    logits: Tensor                 # raw alpha, shape [*scope, K], differentiable
    probs: Tensor                  # softmax(alpha), [*scope, K], differentiable
    weights: Tensor                # combine weights used this forward, [*scope, K]
    entropy: Tensor                # scalar, mean over scope, differentiable
    temperature: float
    mode: str                      # soft | gumbel_soft | gumbel_hard | argmax | frozen
    gradient_mode: str | None      # selected_only | soft_all | None
    selected_index: Tensor | None  # [*scope] for hard/argmax/frozen, else None
    expected_cost: Tensor | None   # sum_i probs_i * cost_i, or None
    exploration_epsilon: float | None
```

Rule: differentiable fields (`logits`, `probs`, `entropy`) stay attached so optional entropy/cost losses can use them. Logging code is responsible for detaching before recording.

### 36.4 Scope and channel axis

```text
scope = "layer"
    logits shape [K]
    one choice for the whole tensor

scope = "channel"
    logits shape [num_features, K]
    requires num_features
    channel_dim default = 1   (Conv1d [B,C,T], Conv2d [B,C,H,W])
    channel_dim = -1          (dense [B, ..., F])

per-element scope
    not in v1
```

For channel scope, the combine step reshapes per-channel weights to broadcast over all non-channel dims (for example `[C, K]` becomes a `[1, C, 1]` weight per candidate on `[B, C, T]`). Channel-scope candidates must preserve the channel dimension.

### 36.5 Export

Export extracts the selected submodule; it does not rebuild from config. See the export block in section 34, Step 7.

```text
SelectPath.export():
    k* = argmax(alpha) or frozen index
    return deepcopy(candidates[k*]) plus its branch-output norm if one was applied
then re-estimate BatchNorm running stats over training data before fine-tuning
```

For a residual site, a tree-walking exporter replaces the inner `SelectPath` with `SelectPath.export()` and leaves the surrounding `ResidualWrapper` (shortcut, residual scale, norms) intact. No residual reconstruction is needed.

Components must keep each branch paired with its branch-output norm (e.g. `SelectPath` holds `(branch, branch_norm)` pairs) so the extracted module stays numerically faithful.

### 36.6 Composition with existing residual/regularization

Residual path selection is composition, not a new component: `ResidualWrapper(branch=SelectPath(...))`. `ResidualWrapper` provides the shortcut/projection, residual scale, `rezero`, `layer_scale`, norm positions, and shape checks; `SelectPath` provides the update-candidate selection. Do not reimplement residual logic.

Broader composition with `ResidualWrapper` and `DropPath` is left to the network composer; the selectable components do not enforce it.

One caveat to document (not enforce): do not stack `drop_path` on a gated residual during search. Combining stochastic path-dropping with stochastic Gumbel selection is double stochasticity that can destabilize the selector. Set `cfg.drop_path = 0` on a `ResidualWrapper` wrapping a `SelectPath` during search.
