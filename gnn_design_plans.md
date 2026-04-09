Below is a structured specification of your **current pipeline** as it exists conceptually now, followed by a set of **incremental level-ups**. Each level is intended to be a viable proof of concept, even when inefficient.

---

# Current pipeline specification

## Goal

Build a dynamic directed graph from multichannel EEG using **cross-wavelet phase relationships**, then run graph message passing over time to predict gesture / motor-imagery class.

---

## Input and notation

Let:

* (C) = number of EEG channels
* (T) = number of time samples in a trial
* (F) = number of wavelet frequencies/scales
* (x_c(t)) = real EEG signal for channel (c)

Input for one trial:

[
X \in \mathbb{R}^{C \times T}
]

Output:

* class label for the trial, or
* frame/window-level label if doing online decoding

---

## Step 1. Preprocessing

For each trial:

1. Select EEG channels.
2. Apply standard EEG preprocessing:

   * notch filter
   * bandpass filter
   * optional rereferencing
   * optional artifact rejection / ICA
3. Normalize per channel if desired.
4. Optionally segment the trial into windows.

Implementation output:

[
\tilde{X} \in \mathbb{R}^{C \times T}
]

This is standard for EEG pipelines before graph construction. Many EEG-GNN pipelines operate on segmented windows rather than entire continuous recordings. For example, EEG graph methods often compute features or connectivity within windows/epochs before graph modeling.

---

## Step 2. Per-channel CWT

For each channel (c), compute the complex continuous wavelet transform:

[
W_c(t,f) \in \mathbb{C}
]

Stack all channels:

[
W \in \mathbb{C}^{C \times T \times F}
]

Interpretation:

* (|W_c(t,f)|): time-frequency magnitude
* (\angle W_c(t,f)): instantaneous wavelet phase

Your current design keeps full time-frequency resolution rather than aggregating early.

---

## Step 3. Pairwise cross-wavelet transform

For each ordered or unordered channel pair ((i,j)), compute cross-wavelet interaction:

[
XWT_{ij}(t,f) = W_i(t,f),\overline{W_j(t,f)}
]

This yields:

[
XWT \in \mathbb{C}^{C \times C \times T \times F}
]

From this, derive:

* magnitude:
  [
  M_{ij}(t,f)=|XWT_{ij}(t,f)|
  ]
* phase difference:
  [
  \phi_{ij}(t,f)=\angle XWT_{ij}(t,f)
  ]

This is the core object in your current design.

---

## Step 4. Dynamic directed graph construction at every ((t,f))

This is the defining property of your present pipeline.

For each time step (t) and frequency (f), construct a graph:

[
G_{t,f} = (V, E_{t,f})
]

where:

* nodes (V) are EEG channels
* edges are determined from the pairwise phase relation (\phi_{ij}(t,f))
* edge weights can depend on (M_{ij}(t,f)), coherence-like quantities, or both

Possible current rule:

* if phase indicates channel (i) leads channel (j), create directed edge (i \to j)
* weight edge by (M_{ij}(t,f)) or by some normalized coherence score

So in practice you are creating:

* **one graph per time-frequency bin**
* total number of graphs per trial:
  [
  T \times F
  ]

This is much finer-grained than most EEG dynamic-GNN pipelines, which usually operate per window or per snapshot rather than per raw ((t,f)) bin. Recent EEG dynamic-GNN work stresses that many models use time-varying graphs, but usually as sequences of snapshots rather than one graph per frequency-time atom.

---

## Step 5. Node state definition

You currently appear to maintain an internal state (s(t)) or (s_{c}(t)), possibly also indexed by frequency.

Two plausible current choices are:

### Option A. State per node

[
s_c(t) \in \mathbb{R}^{d}
]

### Option B. State per node and frequency

[
s_c(t,f) \in \mathbb{R}^{d}
]

Option B is closer to your current diagram and makes the model more expressive, but much more expensive.

---

## Step 6. Message passing

For each graph (G_{t,f}), perform message passing across edges.

General form:

[
m_{i\to j}(t,f)=\text{Msg}\big(s_i(t,f), s_j(t,f), e_{ij}(t,f)\big)
]

where edge attributes may include:

* (M_{ij}(t,f))
* (\phi_{ij}(t,f))
* binary direction mask from phase condition

Aggregate incoming messages:

[
\hat{m}*j(t,f)=\text{Agg}*{i \in \mathcal{N}(j)} m_{i\to j}(t,f)
]

Update state:

[
s_j(t+1,f)=\text{Upd}\big(s_j(t,f), \hat{m}_j(t,f)\big)
]

or, if you update all frequencies together after one time step:

[
s_j(t+1)=\text{Upd}\big(s_j(t), {\hat{m}*j(t,f)}*{f=1}^F\big)
]

This is a recurrent dynamic-GNN flavor. Dynamic EEG-GNN work explicitly discusses recurrent/dynamic graph processing where hidden states evolve with graph snapshots over time.

---

## Step 7. Temporal rollout

Repeat Step 4–6 across time.

Depending on implementation, the recurrence is either:

* over every raw time sample, or
* over decimated / window-center timestamps

At the end, produce either:

* final node states
* pooled graph state
* pooled node-frequency state

---

## Step 8. Readout / classification

Aggregate final representations:

### Node pooling

[
h_{\text{trial}}=\text{Pool}\big({s_c(T_{\text{end}})}_{c=1}^C\big)
]

or, if frequency-aware:

[
h_{\text{trial}}=\text{Pool}\big({s_c(T_{\text{end}},f)}_{c,f}\big)
]

Then classify:

[
\hat{y}=\text{MLP}(h_{\text{trial}})
]

Loss:

* cross-entropy for class prediction

---

## Computational profile of current version

This is the main implementation challenge.

### Storage

You may need to hold:

* (W: C \times T \times F)
* (XWT: C \times C \times T \times F)

### Graph count

[
T \times F
]

### Pairwise channel interactions

[
O(C^2 T F)
]

### Message passing cost

Roughly:
[
O(C^2 T F d)
]
for dense directed graphs

That is why your current pipeline is a valid proof of concept but scales poorly.

---

# Level-up path

Each level below is intended to be independently buildable and testable.

---

## Level 0 — Exact current idea, no compression

### Summary

Implement exactly what you currently designed:

* CWT per channel
* XWT per pair
* phase-conditioned directed graph per ((t,f))
* recurrent message passing over time-frequency graphs
* pooling + classifier

### Why build it

* It is the cleanest proof that your original hypothesis can work
* It gives you the strongest conceptual baseline
* It tells you whether the phase-conditioned graph idea is useful at all

### Expected issues

* very slow
* high memory
* difficult batching
* likely noisy because phase at single-bin resolution is unstable

### Recommended simplifications for feasibility

To make this level actually runnable:

* use few channels first
* use short trials/windows
* use few wavelet scales
* use sparse edge selection rule
* keep state dimension small

### Deliverable

A proof-of-concept model that is scientifically faithful to the original idea, even if impractical.

---

## Level 1 — Temporal downsampling before graph construction

### Summary

Keep the same phase-conditioned graph logic, but do not construct graphs for every raw time sample.

Instead:

1. Compute CWT/XWT at full resolution
2. Select only every (k)-th time point, or
3. Average / summarize within short temporal windows

### Pipeline

* same Steps 1–3
* replace per-sample (t) with window index (w)
* construct graphs (G_{w,f})

### Benefit

Graph count becomes:

[
W \times F \quad \text{instead of} \quad T \times F
]

### Why this is still faithful

You still preserve frequency-specific phase logic.

### Tradeoff

You lose very fast temporal transitions.

### Deliverable

A much more runnable version that still matches your conceptual model closely.

---

## Level 2 — Frequency-band aggregation with circular phase averaging

### Summary

Instead of graph per ((t,f)), build graph per ((t,b)), where (b) is a frequency band.

This is the first major reduction that still preserves phase meaning.

### Key rule

Do **not** average angles directly.

For a band (B), aggregate phase using:

[
z_{ij}(t,B)=\frac{1}{|B|}\sum_{f \in B} e^{i\phi_{ij}(t,f)}
]

Then define:

* aggregated phase:
  [
  \bar{\phi}*{ij}(t,B)=\arg(z*{ij}(t,B))
  ]
* phase consistency:
  [
  R_{ij}(t,B)=|z_{ij}(t,B)|
  ]

### Graph construction

For each (t) and band (B):

* nodes = channels
* direction from (\bar{\phi}_{ij}(t,B))
* weight from (R_{ij}(t,B)), (M_{ij}(t,B)), or both

### Benefit

Graph count becomes:

[
T \times B
]

where (B) may be 4–8 bands instead of tens/hundreds of scales.

Wavelet coherence and other time-frequency connectivity measures are often used at band or summarized levels rather than at every scale for downstream models, because this is much more stable and computationally manageable.

### Why it is viable

* mathematically sensible if done with circular averaging
* still interpretable
* easy to compare against neuroscience bands

### Deliverable

Your first truly practical phase-based dynamic graph baseline.

---

## Level 3 — Rich edge attributes instead of one graph per frequency

### Summary

Stop exploding the number of graphs. Build **one graph per time window**, but make each edge carry a multi-scale descriptor.

### Edge feature example

For each edge ((i,j)) at time (t):

[
e_{ij}(t) = [
R_{ij}(t,B_1), \bar{\phi}*{ij}(t,B_1),
R*{ij}(t,B_2), \bar{\phi}_{ij}(t,B_2),
\dots
]
]

So instead of many graphs:

* one graph per time window
* edges contain bandwise phase/coherence information

### Benefit

Graph count becomes:

[
T \quad \text{or} \quad W
]

### Why this is strong

This keeps the multi-band phase idea, but moves complexity from graph count to edge feature richness.

### Implementation consequence

Use a message function that reads edge attributes explicitly.

### Deliverable

A dynamic graph model that is much closer to standard temporal-GNN tooling.

---

## Level 4 — Window-level graph + recurrent GNN

### Summary

Now combine Level 3 with explicit temporal recurrence.

Pipeline:

1. segment trial into windows
2. compute one graph per window
3. run GNN on each graph
4. feed pooled graph embeddings to GRU/LSTM
5. classify from temporal hidden state

This is structurally close to **graph-then-time** dynamic GNNs. Dynamic EEG-GNN literature explicitly frames graph-then-time as applying GNN per snapshot, then temporal modeling over the resulting embeddings.

### Why useful

* easy to train
* easy to inspect
* much lower memory than per-((t,f)) recurrence

### Deliverable

A strong practical baseline with your phase-based graph logic preserved at window level.

---

## Level 5 — Time-then-graph version

### Summary

Instead of constructing graphs from raw instantaneous phase structure, first compress temporal information within each channel.

Pipeline:

1. window EEG
2. for each channel, temporal encoder:

   * 1D CNN / GRU / TCN
3. use encoded channel features to compute adjacency
4. run GNN
5. classify

This is close to time-then-graph approaches discussed in recent dynamic EEG graph modeling, where temporal dynamics are modeled first and graph learning follows.

### How phase still enters

You can still include phase-derived features in the temporal encoder input:

* per-band phase summaries
* phase consistency
* wavelet magnitude

### Benefit

* fewer graphs
* more stable adjacency
* easier to batch
* more robust than raw single-bin phase

### Deliverable

A modernized version that is still rooted in your wavelet-phase hypothesis.

---

## Level 6 — Learned frequency aggregation

### Summary

Instead of manually aggregating frequencies into delta/theta/alpha/etc., let the model learn how to compress frequency.

Pipeline:

1. compute CWT/XWT
2. for each pair ((i,j)), at each time (t), take frequency vector:
   [
   [M_{ij}(t,f), \phi_{ij}(t,f)]_{f=1}^F
   ]
3. pass through small encoder:

   * 1D conv over frequency
   * MLP
   * attention over frequency
4. produce compact edge embedding
5. build one graph per time window
6. GNN + temporal model

### Why useful

* avoids hand-designed bands
* can discover task-relevant frequency groupings
* still starts from your wavelet/XWT objects

### Tradeoff

Less interpretable.

### Deliverable

A more ML-driven version of your idea.

---

## Level 7 — Data-driven dynamic adjacency instead of explicit phase-threshold graphing

### Summary

At this stage, phase is no longer the direct graph-construction rule. It becomes one of the signals used to learn adjacency.

Pipeline:

1. build node features from temporal encoder / wavelet features
2. optionally add pairwise phase/coherence descriptors
3. compute learned adjacency with attention or similarity network
4. GNN over learned graph
5. temporal modeling

This is closer to models like DGAT / DGDCN / NeuroGNN, which emphasize learning dynamic adjacency from data rather than explicitly hand-coding every edge from one connectivity rule. These models use dynamic adjacency, attention, and temporal encoders to avoid fixed or overly rigid graph construction.

### Why useful

* scalable
* robust
* easier optimization
* lets the model denoise unstable phase signals

### What you keep from the original idea

Phase can still be:

* an edge feature
* an auxiliary supervision signal
* a regularizer for adjacency

### Deliverable

A practical research-grade model inspired by your original mechanism.

---

# Recommended build order

For proof-of-concept work, I would build in this order:

## 1. Level 0

To prove the original idea is coherent.

## 2. Level 2

Band-aggregated circular phase graph.
This is likely your best first practical baseline.

## 3. Level 3

One graph per time window, multi-band edge attributes.

## 4. Level 4

Graph-then-time recurrent baseline.

## 5. Level 5

Time-then-graph baseline.

## 6. Level 6 or 7

Learned aggregation / learned adjacency.

---

# Best “minimum viable research pipeline”

If you want one version that is still faithful to your idea but much easier to evaluate:

## Recommended MVP

* preprocess EEG
* segment into short windows
* compute CWT
* compute XWT per pair
* aggregate phase within frequency bands using circular mean
* compute one directed graph per window and band, or one graph per window with bandwise edge attributes
* run GNN per window
* temporal GRU across windows
* classify

This keeps:

* wavelets
* cross-wavelet phase
* dynamic graphs
* recurrence

while removing the worst computational explosion.

---

# What to hand to an implementation agent

Tell the implementation agent to treat this as a staged project:

## Baseline A

Exact current pipeline, sparse, small-scale, correctness-first.

## Baseline B

Band-aggregated phase graph with circular averaging.

## Baseline C

Window graph with edge feature vectors across bands.

## Baseline D

Graph-then-time recurrent classifier.

## Baseline E

Time-then-graph classifier.

Each baseline should:

* run end-to-end
* log runtime and memory
* save intermediate graph statistics
* support ablation on:

  * channels
  * number of frequencies
  * time stride
  * edge sparsity
  * band aggregation on/off

---