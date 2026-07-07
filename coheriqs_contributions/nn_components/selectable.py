"""Categorical gates and selectable neural-network components."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .activations import build_activation
from .configs import ActConfig, CategoricalGateConfig, SelectPathConfig


@dataclass
class GateInfo:
    """Runtime diagnostics returned by :class:`CategoricalGate`."""

    logits: Tensor
    probs: Tensor
    weights: Tensor
    entropy: Tensor
    temperature: float
    mode: str
    gradient_mode: str | None
    selected_index: Tensor | None
    expected_cost: Tensor | None
    exploration_epsilon: float | None
    soft_weights: Tensor | None = None


class ZeroUpdate(nn.Module):
    """Return a zero update with the same shape as the input."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x)


class CategoricalGate(nn.Module):
    """Produce categorical weights for layer-level or channel-level choices."""

    def __init__(self, cfg: CategoricalGateConfig) -> None:
        super().__init__()
        _validate_gate_config(cfg)

        self.num_choices = int(cfg.num_choices)
        self.scope = cfg.scope
        self.num_features = cfg.num_features
        self.channel_dim = int(cfg.channel_dim)
        self.mode = cfg.mode
        self.eval_mode = cfg.eval_mode
        self.temperature = float(cfg.temperature)
        self.exploration_epsilon = cfg.exploration_epsilon
        self.cost_weight = float(cfg.cost_weight)
        self.entropy_weight = float(cfg.entropy_weight)
        self.gradient_mode = cfg.gradient_mode
        self.frozen_index = cfg.frozen_index
        self.alpha_update_split = cfg.alpha_update_split
        self.alpha_optim = cfg.alpha_optim

        if self.scope == "layer":
            shape = (self.num_choices,)
        else:
            if self.num_features is None:
                raise ValueError("num_features is required when scope='channel'.")
            shape = (self.num_features, self.num_choices)
        self.logits = nn.Parameter(torch.full(shape, float(cfg.logits_init)))

    def forward(
        self,
        *,
        costs: Tensor | Sequence[float] | None = None,
    ) -> tuple[Tensor, GateInfo]:
        mode = self._active_mode()
        probs = self.probabilities()
        weights, selected_index, soft_weights = self._weights_for_mode(mode, probs)
        entropy = -(probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()).sum(
            dim=-1
        )
        expected_cost = self._expected_cost(probs, costs)
        active_gradient_mode = self.gradient_mode if mode == "gumbel_hard" else None
        info = GateInfo(
            logits=self.logits,
            probs=probs,
            weights=weights,
            entropy=entropy.mean(),
            temperature=self.temperature,
            mode=mode,
            gradient_mode=active_gradient_mode,
            selected_index=selected_index,
            expected_cost=expected_cost,
            exploration_epsilon=self.exploration_epsilon,
            soft_weights=soft_weights,
        )
        return weights, info

    def probabilities(self) -> Tensor:
        """Return the current categorical probabilities without sampling."""

        probs = F.softmax(self.logits, dim=-1)
        return self._apply_exploration_floor(probs)

    def set_temperature(self, value: float) -> None:
        value = float(value)
        if value <= 0.0:
            raise ValueError("temperature must be positive.")
        self.temperature = value

    def set_mode(self, mode: str) -> None:
        if mode not in {"soft", "gumbel_soft", "gumbel_hard", "argmax", "frozen"}:
            raise ValueError(f"Unsupported gate mode: {mode!r}.")
        self.mode = mode

    def freeze(self, index: int | None = None) -> None:
        """Switch to frozen mode, defaulting to the current argmax choice."""

        if index is None:
            index = self.export_index()
        self.frozen_index = _validate_index("frozen_index", index, self.num_choices)
        self.mode = "frozen"

    def export_index(self) -> int:
        """Return the single selected index for export.

        Channel-scope gates can only be exported to one ordinary candidate when
        all channels agree on the same selected candidate.
        """

        if self.mode == "frozen":
            if self.frozen_index is None:
                raise ValueError("frozen mode requires frozen_index.")
            return _validate_index("frozen_index", self.frozen_index, self.num_choices)

        selected = torch.argmax(self.logits.detach(), dim=-1)
        unique = torch.unique(selected.reshape(-1))
        if unique.numel() != 1:
            raise ValueError(
                "Cannot export a channel-scope gate with different selected "
                "candidates per channel."
            )
        return int(unique.item())

    def selection_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return selector parameters for a separate optimizer group."""

        return (self.logits,)

    def extra_loss(self, gate_info: GateInfo) -> Tensor:
        """Return optional entropy and cost regularization for a gate forward."""

        loss = gate_info.logits.new_zeros(())
        if self.cost_weight != 0.0:
            if gate_info.expected_cost is None:
                raise ValueError("candidate costs are required for cost regularization.")
            loss = loss + self.cost_weight * gate_info.expected_cost
        if self.entropy_weight != 0.0:
            loss = loss + self.entropy_weight * gate_info.entropy
        return loss

    def _active_mode(self) -> str:
        if self.training or self.eval_mode == "same":
            return self.mode
        return self.eval_mode

    def _weights_for_mode(
        self,
        mode: str,
        probs: Tensor,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if mode == "soft":
            weights = F.softmax(self.logits / self.temperature, dim=-1)
            weights = self._apply_exploration_floor(weights)
            return weights, None, weights

        if mode in {"gumbel_soft", "gumbel_hard"}:
            sampling_logits = probs.clamp_min(torch.finfo(probs.dtype).tiny).log()
            gumbels = _sample_gumbels_like(sampling_logits)
            soft_weights = F.softmax(
                (sampling_logits + gumbels) / self.temperature,
                dim=-1,
            )
            if mode == "gumbel_soft":
                return soft_weights, None, soft_weights
            selected_index = soft_weights.argmax(dim=-1)
            hard_weights = self._one_hot(selected_index)
            weights = hard_weights + (soft_weights - soft_weights.detach())
            return weights, selected_index, soft_weights

        if mode == "argmax":
            selected_index = self.logits.argmax(dim=-1)
            return self._one_hot(selected_index), selected_index, None

        if mode == "frozen":
            if self.frozen_index is None:
                raise ValueError("frozen mode requires frozen_index.")
            index = _validate_index("frozen_index", self.frozen_index, self.num_choices)
            selected_index = torch.full(
                self.logits.shape[:-1],
                index,
                device=self.logits.device,
                dtype=torch.long,
            )
            return self._one_hot(selected_index), selected_index, None

        raise ValueError(f"Unsupported gate mode: {mode!r}.")

    def _one_hot(self, selected_index: Tensor) -> Tensor:
        return F.one_hot(selected_index, self.num_choices).to(
            device=self.logits.device,
            dtype=self.logits.dtype,
        )

    def _apply_exploration_floor(self, probs: Tensor) -> Tensor:
        if self.exploration_epsilon is None:
            return probs
        epsilon = float(self.exploration_epsilon)
        return probs.mul(1.0 - epsilon).add(epsilon / self.num_choices)

    def _expected_cost(
        self,
        probs: Tensor,
        costs: Tensor | Sequence[float] | None,
    ) -> Tensor | None:
        if costs is None:
            return None
        if isinstance(costs, Tensor):
            cost_tensor = costs.to(device=probs.device, dtype=probs.dtype)
        else:
            cost_tensor = torch.as_tensor(costs, device=probs.device, dtype=probs.dtype)
        if cost_tensor.shape != (self.num_choices,):
            raise ValueError(
                "candidate costs must have shape "
                f"({self.num_choices},), got {tuple(cost_tensor.shape)}."
            )
        return (probs * cost_tensor).sum(dim=-1).mean()


class CategoricalChoice(nn.Module):
    """Combine same-shape candidate outputs through a categorical gate."""

    def __init__(
        self,
        candidates: Sequence[nn.Module],
        gate: CategoricalGateConfig | CategoricalGate | None = None,
        *,
        candidate_costs: Tensor | Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if len(candidates) == 0:
            raise ValueError("candidates must not be empty.")
        self.candidates = nn.ModuleList(candidates)
        self.gate = _make_gate(gate, len(self.candidates))
        self.last_gate_info: GateInfo | None = None
        if candidate_costs is None:
            self.register_buffer("_candidate_costs", None, persistent=False)
        else:
            costs = torch.as_tensor(candidate_costs, dtype=torch.float32)
            if costs.shape != (len(self.candidates),):
                raise ValueError(
                    "candidate_costs must have one value per candidate, got "
                    f"{tuple(costs.shape)} for {len(self.candidates)} candidates."
                )
            self.register_buffer("_candidate_costs", costs, persistent=False)

    def forward(self, x: Tensor) -> tuple[Tensor, GateInfo]:
        weights, gate_info = self.gate(costs=self._candidate_costs)
        self.last_gate_info = gate_info

        if self._should_compute_all(gate_info):
            return self._forward_all(x, weights, gate_info), gate_info

        selected = _single_selected_index(gate_info.selected_index)
        if selected is None:
            return self._forward_all(x, weights, gate_info), gate_info
        output = self.candidates[selected](x)
        self._validate_channel_output(output)
        selected_weight = _candidate_weight(
            weights,
            selected,
            output.ndim,
            self.gate.scope,
            self.gate.channel_dim,
            self.gate.num_features,
        )
        return output * selected_weight.to(dtype=output.dtype), gate_info

    def selection_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.gate.selection_parameters()

    def set_temperature(self, value: float) -> None:
        self.gate.set_temperature(value)

    def freeze(self, index: int | None = None) -> None:
        self.gate.freeze(index)

    def extra_loss(self, gate_info: GateInfo | None = None) -> Tensor:
        if gate_info is None:
            if self.last_gate_info is None:
                raise ValueError("extra_loss requires a previous forward or gate_info.")
            gate_info = self.last_gate_info
        return self.gate.extra_loss(gate_info)

    def _should_compute_all(self, gate_info: GateInfo) -> bool:
        if gate_info.mode in {"soft", "gumbel_soft"}:
            return True
        return (
            gate_info.mode == "gumbel_hard"
            and gate_info.gradient_mode == "soft_all"
        )

    def _forward_all(
        self,
        x: Tensor,
        weights: Tensor,
        gate_info: GateInfo,
    ) -> Tensor:
        outputs = [candidate(x) for candidate in self.candidates]
        _validate_same_shape(outputs)
        self._validate_channel_output(outputs[0])
        if (
            gate_info.mode == "gumbel_hard"
            and gate_info.gradient_mode == "soft_all"
        ):
            return self._soft_all_hard_output(outputs, gate_info)
        return self._combine_outputs(outputs, weights)

    def _soft_all_hard_output(
        self,
        outputs: Sequence[Tensor],
        gate_info: GateInfo,
    ) -> Tensor:
        if gate_info.selected_index is None or gate_info.soft_weights is None:
            raise RuntimeError("soft_all hard Gumbel requires selected and soft weights.")
        hard_weights = F.one_hot(
            gate_info.selected_index,
            self.gate.num_choices,
        ).to(device=gate_info.weights.device, dtype=gate_info.weights.dtype)
        hard_output = self._combine_outputs(outputs, hard_weights)
        soft_output = self._combine_outputs(outputs, gate_info.soft_weights)
        return hard_output.detach() + soft_output - soft_output.detach()

    def _combine_outputs(self, outputs: Sequence[Tensor], weights: Tensor) -> Tensor:
        out = torch.zeros_like(outputs[0])
        for index, output in enumerate(outputs):
            weight = _candidate_weight(
                weights,
                index,
                output.ndim,
                self.gate.scope,
                self.gate.channel_dim,
                self.gate.num_features,
            )
            out = out + output * weight.to(dtype=output.dtype)
        return out

    def _validate_channel_output(self, output: Tensor) -> None:
        if self.gate.scope != "channel":
            return
        if self.gate.num_features is None:
            raise RuntimeError("channel-scope gate is missing num_features.")
        channel_dim = _normalize_dim(self.gate.channel_dim, output.ndim)
        actual = int(output.shape[channel_dim])
        if actual != self.gate.num_features:
            raise ValueError(
                "channel-scope candidates must preserve the configured channel "
                f"dimension; expected {self.gate.num_features}, got {actual}."
            )


class SelectableActivation(nn.Module):
    """Choose or mix activation modules with a categorical gate."""

    def __init__(
        self,
        activations: Sequence[str | ActConfig | nn.Module] | None = None,
        gate: CategoricalGateConfig | CategoricalGate | None = None,
    ) -> None:
        super().__init__()
        if activations is None:
            activations = ("relu", "gelu", "silu", "identity")
        candidates = [_make_activation(candidate) for candidate in activations]
        self.choice = CategoricalChoice(candidates, gate)

    def forward(
        self,
        x: Tensor,
        *,
        return_gate_info: bool = False,
    ) -> Tensor | tuple[Tensor, GateInfo]:
        output, gate_info = self.choice(x)
        if return_gate_info:
            return output, gate_info
        return output

    def export(self) -> nn.Module:
        index = self.choice.gate.export_index()
        return deepcopy(self.choice.candidates[index])

    def selection_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.choice.selection_parameters()

    def set_temperature(self, value: float) -> None:
        self.choice.set_temperature(value)

    def freeze(self, index: int | None = None) -> None:
        self.choice.freeze(index)

    def extra_loss(self, gate_info: GateInfo | None = None) -> Tensor:
        return self.choice.extra_loss(gate_info)

    @property
    def gate(self) -> CategoricalGate:
        return self.choice.gate

    @property
    def last_gate_info(self) -> GateInfo | None:
        return self.choice.last_gate_info


class SelectPath(nn.Module):
    """Choose among same-output-shape path candidates."""

    def __init__(
        self,
        candidates: Sequence[nn.Module],
        gate: CategoricalGateConfig | CategoricalGate | None = None,
        *,
        cfg: SelectPathConfig | None = None,
        candidate_costs: Tensor | Sequence[float] | None = None,
        branch_norms: Sequence[nn.Module | None] | None = None,
        include_zero_update: bool | None = None,
    ) -> None:
        super().__init__()
        path_candidates = list(candidates)
        if len(path_candidates) == 0:
            raise ValueError("candidates must not be empty.")

        include_zero = (
            cfg.include_zero_update
            if include_zero_update is None and cfg is not None
            else bool(include_zero_update)
        )
        costs = candidate_costs if candidate_costs is not None else (
            cfg.candidate_costs if cfg is not None else None
        )

        norms = _normalize_branch_norms(
            branch_norms,
            len(path_candidates),
            include_zero=include_zero,
        )
        if include_zero:
            path_candidates = [ZeroUpdate(), *path_candidates]
            costs = _prepend_zero_cost(costs, expected_without_zero=len(candidates))

        modules = [
            _PathCandidate(candidate, norm)
            for candidate, norm in zip(path_candidates, norms, strict=True)
        ]
        path_gate = gate
        if path_gate is None:
            path_gate = CategoricalGateConfig(
                num_choices=len(modules),
                alpha_update_split="val",
            )
        self.choice = CategoricalChoice(modules, path_gate, candidate_costs=costs)

    def forward(
        self,
        x: Tensor,
        *,
        return_gate_info: bool = False,
    ) -> Tensor | tuple[Tensor, GateInfo]:
        output, gate_info = self.choice(x)
        if return_gate_info:
            return output, gate_info
        return output

    def export(self) -> nn.Module:
        index = self.choice.gate.export_index()
        candidate = self.choice.candidates[index]
        if not isinstance(candidate, _PathCandidate):
            return deepcopy(candidate)
        return candidate.export()

    def selection_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.choice.selection_parameters()

    def set_temperature(self, value: float) -> None:
        self.choice.set_temperature(value)

    def freeze(self, index: int | None = None) -> None:
        self.choice.freeze(index)

    def extra_loss(self, gate_info: GateInfo | None = None) -> Tensor:
        return self.choice.extra_loss(gate_info)

    @property
    def gate(self) -> CategoricalGate:
        return self.choice.gate

    @property
    def last_gate_info(self) -> GateInfo | None:
        return self.choice.last_gate_info


def export_selectable_modules(module: nn.Module) -> nn.Module:
    """Deep-copy a module tree and replace selectable components with exports."""

    if isinstance(module, (SelectPath, SelectableActivation)):
        exported_root = module.export()
        _replace_selectable_children(exported_root)
        return exported_root
    exported = deepcopy(module)
    _replace_selectable_children(exported)
    return exported


class _PathCandidate(nn.Module):
    def __init__(self, branch: nn.Module, norm: nn.Module | None = None) -> None:
        super().__init__()
        self.branch = branch
        self.norm = norm

    def forward(self, x: Tensor) -> Tensor:
        output = self.branch(x)
        if self.norm is not None:
            output = self.norm(output)
        return output

    def export(self) -> nn.Module:
        branch = deepcopy(self.branch)
        if self.norm is None:
            return branch
        return nn.Sequential(branch, deepcopy(self.norm))


def _replace_selectable_children(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, (SelectPath, SelectableActivation)):
            replacement = child.export()
            _replace_selectable_children(replacement)
            setattr(module, name, replacement)
        else:
            _replace_selectable_children(child)


def _make_gate(
    gate: CategoricalGateConfig | CategoricalGate | None,
    num_choices: int,
) -> CategoricalGate:
    if gate is None:
        return CategoricalGate(CategoricalGateConfig(num_choices=num_choices))
    if isinstance(gate, CategoricalGate):
        if gate.num_choices != num_choices:
            raise ValueError(
                f"gate has {gate.num_choices} choices, but {num_choices} "
                "candidates were provided."
            )
        return gate
    if gate.num_choices != num_choices:
        raise ValueError(
            f"gate config has {gate.num_choices} choices, but {num_choices} "
            "candidates were provided."
        )
    return CategoricalGate(gate)


def _make_activation(candidate: str | ActConfig | nn.Module) -> nn.Module:
    if isinstance(candidate, nn.Module):
        return candidate
    if isinstance(candidate, ActConfig):
        return build_activation(candidate)
    return build_activation(ActConfig(kind=candidate))


def _candidate_weight(
    weights: Tensor,
    index: int,
    output_ndim: int,
    scope: str,
    channel_dim: int,
    num_features: int | None,
) -> Tensor:
    if scope == "layer":
        return weights[index]
    if num_features is None:
        raise RuntimeError("channel-scope gate is missing num_features.")
    normalized_dim = _normalize_dim(channel_dim, output_ndim)
    shape = [1] * output_ndim
    shape[normalized_dim] = num_features
    return weights[..., index].reshape(shape)


def _validate_same_shape(outputs: Sequence[Tensor]) -> None:
    if not outputs:
        raise ValueError("at least one candidate output is required.")
    expected = tuple(outputs[0].shape)
    for index, output in enumerate(outputs[1:], start=1):
        if tuple(output.shape) != expected:
            raise ValueError(
                "candidate outputs must have matching shapes; candidate 0 has "
                f"{expected}, candidate {index} has {tuple(output.shape)}."
            )


def _single_selected_index(selected_index: Tensor | None) -> int | None:
    if selected_index is None:
        return None
    unique = torch.unique(selected_index.detach().reshape(-1))
    if unique.numel() != 1:
        return None
    return int(unique.item())


def _sample_gumbels_like(logits: Tensor) -> Tensor:
    eps = torch.finfo(logits.dtype).eps
    uniform = torch.rand_like(logits).clamp(min=eps, max=1.0 - eps)
    return -torch.log(-torch.log(uniform))


def _normalize_dim(dim: int, ndim: int) -> int:
    if ndim <= 0:
        raise ValueError("output tensor must have at least one dimension.")
    if dim < -ndim or dim >= ndim:
        raise ValueError(f"channel_dim {dim} is out of range for rank {ndim}.")
    return dim % ndim


def _normalize_branch_norms(
    branch_norms: Sequence[nn.Module | None] | None,
    num_candidates_without_zero: int,
    *,
    include_zero: bool,
) -> list[nn.Module | None]:
    final_count = num_candidates_without_zero + int(include_zero)
    if branch_norms is None:
        return [None] * final_count
    norms = list(branch_norms)
    if include_zero and len(norms) == num_candidates_without_zero:
        norms = [None, *norms]
    if len(norms) != final_count:
        raise ValueError(
            f"branch_norms must have {final_count} entries, got {len(norms)}."
        )
    return norms


def _prepend_zero_cost(
    costs: Tensor | Sequence[float] | None,
    *,
    expected_without_zero: int,
) -> Tensor | Sequence[float] | None:
    if costs is None:
        return None
    if isinstance(costs, Tensor):
        if costs.shape == (expected_without_zero,):
            return torch.cat([costs.new_zeros(1), costs])
        return costs
    cost_list = list(costs)
    if len(cost_list) == expected_without_zero:
        return (0.0, *cost_list)
    return costs


def _validate_gate_config(cfg: CategoricalGateConfig) -> None:
    if int(cfg.num_choices) <= 0:
        raise ValueError("num_choices must be positive.")
    if float(cfg.temperature) <= 0.0:
        raise ValueError("temperature must be positive.")
    if cfg.scope not in {"layer", "channel"}:
        raise ValueError(f"Unsupported gate scope: {cfg.scope!r}.")
    if cfg.scope == "channel":
        if cfg.num_features is None or int(cfg.num_features) <= 0:
            raise ValueError("num_features must be positive for channel scope.")
    if cfg.mode not in {"soft", "gumbel_soft", "gumbel_hard", "argmax", "frozen"}:
        raise ValueError(f"Unsupported gate mode: {cfg.mode!r}.")
    if cfg.eval_mode not in {"same", "argmax", "frozen"}:
        raise ValueError(f"Unsupported eval_mode: {cfg.eval_mode!r}.")
    if cfg.gradient_mode not in {"selected_only", "soft_all"}:
        raise ValueError(f"Unsupported gradient_mode: {cfg.gradient_mode!r}.")
    if cfg.exploration_epsilon is not None:
        epsilon = float(cfg.exploration_epsilon)
        if epsilon < 0.0 or epsilon >= 1.0:
            raise ValueError("exploration_epsilon must be in [0.0, 1.0).")
    if cfg.frozen_index is not None:
        _validate_index("frozen_index", cfg.frozen_index, int(cfg.num_choices))
    if cfg.alpha_update_split not in {"train", "val"}:
        raise ValueError("alpha_update_split must be 'train' or 'val'.")
    if cfg.alpha_optim not in {"shared", "separate"}:
        raise ValueError("alpha_optim must be 'shared' or 'separate'.")


def _validate_index(name: str, index: int, num_choices: int) -> int:
    index = int(index)
    if index < 0 or index >= num_choices:
        raise ValueError(f"{name} must be in [0, {num_choices}).")
    return index
