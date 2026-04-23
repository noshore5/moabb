# XWTPhaseGNNCore - Step-by-Step Reader Guide

This document explains the data/signal flow of `XWTPhaseGNNCore` from:

- `D:\Piotr\moabb\my_contributions\moabb_pipelines\xwt_phase_gnn_classifier.py`

It is aligned with:

- `D:\Piotr\moabb\docs\source\images\xwt_phase_gnn_core_flow.svg`
- the equivalent PNG background/diagram

---

## What the module does in one sentence

`XWTPhaseGNNCore` performs recurrent message passing across directed channel pairs, where each message is computed from cross-wavelet features and gated by phase condition, then pooled for trial-level classification.

---

## Inputs, symbols, and shapes

- `B`: batch size
- `C`: number of channels (nodes)
- `T`: number of timesteps
- `F`: number of wavelet frequencies
- `H`: hidden dimension
- `E = C * (C - 1)`: number of ordered directed edges (no self-edges)

Forward inputs:

- `raw_x`: `(B, C, T)`
- `w_real`: `(B, C, T, F)`
- `w_imag`: `(B, C, T, F)`

Forward outputs:

- `logits`: `(B, n_classes)`
- `edge_density`: scalar diagnostic (`gate_sum / gate_count`)

---

## Step-by-step flow

## Step 0: Initialize directed edge index and hidden state

Ordered channel pairs `(src_idx, dst_idx)` are used so `(i -> j)` and `(j -> i)` are distinct edges.

State initialization depends on `state_mode`:

- `per_node`: `state = (B, C, H)`
- `per_node_per_freq`: `state = (B, C, F, H)`

---

## Step 1: Time loop

The model iterates:

- `for t in range(0, T, time_stride)`

Each iteration performs one recurrent update from `state(t)` to `state(t+1)`.

---

## Step 2: Gather source/destination CWT features per edge

At timestep `t`:

- `src_r = w_real[:, src_idx, t, :]`
- `src_i = w_imag[:, src_idx, t, :]`
- `dst_r = w_real[:, dst_idx, t, :]`
- `dst_i = w_imag[:, dst_idx, t, :]`

All are shape `(B, E, F)`.

---

## Step 3: Compute directed cross-wavelet terms

Using complex multiplication with conjugation:

- `xwt_real = src_r * dst_r + src_i * dst_i`
- `xwt_imag = src_i * dst_r - src_r * dst_i`

Shape remains `(B, E, F)`.

---

## Step 4: Derive magnitude/angle and phase gate

- `mag = sqrt(xwt_real^2 + xwt_imag^2 + 1e-12)`
- `ang = atan2(xwt_imag, xwt_real)`
- `delta = atan2(sin(ang), cos(ang))`

Default gating rule (`deadzone_sign`):

- `gate = 1[delta > theta_dead_rad]`

Then numerical cleanup is applied:

- `gate = nan_to_num(gate)`
- also `mag` and `ang` are sanitized with `nan_to_num`

`gate_sum` and `gate_count` are accumulated for the final `edge_density`.

---

## Step 5: Build message payload

Features are concatenated conditionally (based on flags):

- `mag` (if `use_mag`)
- `ang` (if `use_ang`)
- raw source/destination sample at time `t` (if `use_raw`)
- source state (if `use_state_src`)
- destination state (if `use_state_dst`)

Output:

- `payload` of shape `(B, E, F, payload_dim)`

---

## Step 6: Message MLP + phase-conditioned masking

- `msg = message_mlp(payload)` gives `(B, E, F, H)`
- `msg = msg * gate[..., None]`

Only phase-allowed edge-frequency interactions contribute to updates.

---

## Step 7: Aggregate and update recurrent state (two branches)

## Branch A: `state_mode == "per_node"`

1. Sum messages over frequencies:
   - `msg_sum_f = msg.sum(dim=2)` -> `(B, E, H)`
2. Aggregate by destination channel (`index_add_`):
   - `agg` -> `(B, C, H)`
3. GRU update per node:
   - `state = GRUCell(agg_flat, state_flat)` then reshape to `(B, C, H)`

## Branch B: `state_mode == "per_node_per_freq"`

1. Keep frequency dimension (no early sum over `F`)
2. Aggregate by destination and frequency (`index_add_`):
   - `agg` -> `(B, C, F, H)`
3. GRU update per `(channel, frequency)` slot:
   - reshape/update/reshape back to `(B, C, F, H)`

---

## Step 8: Pool and classify

After the time loop:

- `per_node`: `pooled = state.mean(dim=1)` -> `(B, H)`
- `per_node_per_freq`: `pooled = state.mean(dim=(1,2))` -> `(B, H)`

Then:

- `logits = classifier(pooled)` -> `(B, n_classes)`

Final metric:

- `edge_density = gate_sum / gate_count` (or `0.0` if count is zero)

---

## Compact pseudocode view

```text
init state
for t in time:
  gather src/dst CWT at t
  compute xwt_real, xwt_imag
  compute mag, ang, delta
  gate = phase_rule(delta, theta)
  payload = concat(selected features)
  msg = message_mlp(payload)
  msg = msg * gate

  if per_node:
    agg = aggregate_dst(sum_over_freq(msg))
    state = GRUCell(agg, state)
  else:
    agg = aggregate_dst_freq(msg)
    state = GRUCell(agg, state)

pooled = mean(state)
logits = linear(pooled)
edge_density = gate_sum / gate_count
return logits, edge_density
```

---

## Practical interpretation

- The model is a directed, phase-gated recurrent GNN in wavelet space.
- Direction is explicit because edges are ordered.
- Phase gating controls which edge-frequency messages are active.
- GRU dynamics accumulate information over time.
- Readout is trial-level via pooled hidden state.

---

## Quick branch comparison

- `per_node`:
  - lower memory/compute
  - frequency information merged before update
- `per_node_per_freq`:
  - richer frequency-specific state
  - higher compute/memory

