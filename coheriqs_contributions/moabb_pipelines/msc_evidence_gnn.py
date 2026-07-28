"""Static MSC evidence GNN classifier."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn

from coheriqs_contributions.moabb_pipelines.common import emit_initial_detail

try:
	from coheriqs_contributions.nn_components import (
		ActConfig,
		CategoricalGateConfig,
		Conv2dConfig,
		DenseMLPConfig,
		InitConfig,
		NormConfig,
		RegConfig,
		SelectPath,
		build_conv2d_block,
		build_dense_mlp,
		scoped_torch_init_seed,
	)
except ModuleNotFoundError:
	from nn_components import (
		ActConfig,
		CategoricalGateConfig,
		Conv2dConfig,
		DenseMLPConfig,
		InitConfig,
		NormConfig,
		RegConfig,
		SelectPath,
		build_conv2d_block,
		build_dense_mlp,
		scoped_torch_init_seed,
	)

try:
	from coheriqs_contributions.moabb_pipelines.xwt_phase_gnn_classifier import (
		_BaseCWTGNNClassifier,
		_ordered_pair_indices,
	)
except ModuleNotFoundError:
	from moabb_pipelines.xwt_phase_gnn_classifier import (
		_BaseCWTGNNClassifier,
		_ordered_pair_indices,
	)


WCT_EVIDENCE_COMPONENT_PROFILES = (
	"legacy",
)
MESSAGE_MLP_SELECTOR_MODES = (
	"shared_train",
	"separate_train",
	"separate_val",
)


class _TimeMeanPool2d(nn.Module):
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return x.mean(dim=3, keepdim=True)


def _dtype_nbytes(dtype: torch.dtype) -> int:
	return torch.empty((), dtype=dtype).element_size()


def _shape_numel(shape: tuple[int, ...]) -> int:
	numel = 1
	for dim in shape:
		numel *= max(int(dim), 0)
	return numel


def _format_bytes(n_bytes: int) -> str:
	value = float(n_bytes)
	for unit in ["B", "KiB", "MiB", "GiB"]:
		if value < 1024.0 or unit == "GiB":
			return f"{value:.2f} {unit}"
		value /= 1024.0
	return f"{value:.2f} GiB"


def _format_number_list(values: Sequence[float]) -> str:
	return "[" + ", ".join(f"{float(value):.3f}" for value in values) + "]"


def _selector_mode_from_gate(gate) -> str:
	if gate.alpha_optim == "shared" and gate.alpha_update_split == "train":
		return "shared_train"
	if gate.alpha_optim == "separate" and gate.alpha_update_split == "train":
		return "separate_train"
	if gate.alpha_optim == "separate" and gate.alpha_update_split == "val":
		return "separate_val"
	return f"{gate.alpha_optim}_{gate.alpha_update_split}"


def _build_feature_conv(
	*,
	kernel_size: int,
	intermediate_channels: int,
	out_channels: int,
	pool_size: int,
	intermediate_channels_reduced: int | None = None,
) -> nn.Module:
	conv_blocks = []
	conv1 = build_conv2d_block(
		Conv2dConfig(
			in_channels=1,
			out_channels=intermediate_channels,
			kernel_size=(1, kernel_size),
			padding=0,
			regularization=RegConfig(0.5, 0.0),
			norm=NormConfig("batch"),
			activation=ActConfig(kind="gelu"),
		),
	)
	max_pool1 = nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_size))

	conv_blocks.append(conv1)
	conv_blocks.append(max_pool1)

	conv2_in_channels = intermediate_channels
	if intermediate_channels_reduced is not None:
		conv2_in_channels = intermediate_channels_reduced
		conv1_reduced = build_conv2d_block(
			Conv2dConfig(
				in_channels=intermediate_channels,
				out_channels=intermediate_channels_reduced,
				kernel_size=(1, 1),
				padding=0,
				regularization=RegConfig(),
				norm=NormConfig("batch"),
				activation=ActConfig(kind="gelu"),
			),
		)
		conv_blocks.append(conv1_reduced)
	conv2 = build_conv2d_block(
		Conv2dConfig(
			in_channels=conv2_in_channels,
			out_channels=out_channels,
			kernel_size=(1, kernel_size),
			padding=0,
			regularization=RegConfig(0.0, 0.5),
			norm=NormConfig("batch"),
			activation=ActConfig(kind="gelu"),
		),
	)
	max_pool2 = nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_size))
	conv_blocks.append(conv2)
	conv_blocks.append(max_pool2)
	conv_blocks.append(_TimeMeanPool2d())

	return nn.Sequential(*conv_blocks)


def _build_message_mlp(
	*,
	message_layer_norm: bool,
	in_features: int,
	hidden_features: int,
	out_features: int,
	init_seed: int | None,
	select_message_mlp: list[dict] | None,
	select_message_mlp_gate: dict | None,
	message_mlp_selector_mode: str,
) -> nn.Module:
	if select_message_mlp is not None:
		return _build_selectable_message_mlp(
			message_layer_norm=message_layer_norm,
			in_features=in_features,
			hidden_features=hidden_features,
			out_features=out_features,
			init_seed=init_seed,
			select_message_mlp=select_message_mlp,
			select_message_mlp_gate=select_message_mlp_gate,
			message_mlp_selector_mode=message_mlp_selector_mode,
		)
	return _build_single_message_mlp(
		message_layer_norm=message_layer_norm,
		in_features=in_features,
		hidden_features=hidden_features,
		out_features=out_features,
		depth=2,
		activation="gelu",
		dropout=0.0,
		init_mode="torch_default",
		init_seed=init_seed,
	)


_MESSAGE_MLP_CANDIDATE_KEYS = {
	"activation",
	"depth",
	"dropout",
	"hidden_features",
	"init_mode",
	"init_seed",
	"message_dim",
	"message_layer_norm",
}
_MESSAGE_MLP_SHAPE_KEYS = {"hidden_dim", "in_features", "out_features"}
_MESSAGE_MLP_GATE_KEYS = {
	"entropy_weight",
	"eval_mode",
	"exploration_epsilon",
	"frozen_index",
	"gradient_mode",
	"logits_init",
	"mode",
	"temperature",
}


def _build_selectable_message_mlp(
	*,
	message_layer_norm: bool,
	in_features: int,
	hidden_features: int,
	out_features: int,
	init_seed: int | None,
	select_message_mlp: Sequence[dict],
	select_message_mlp_gate: dict | None,
	message_mlp_selector_mode: str,
) -> nn.Module:
	if isinstance(select_message_mlp, (str, bytes)) or not isinstance(
		select_message_mlp,
		Sequence,
	):
		raise ValueError("select_message_mlp must be a non-empty list of dicts.")
	if len(select_message_mlp) == 0:
		raise ValueError("select_message_mlp must contain at least one candidate.")

	with scoped_torch_init_seed(init_seed):
		candidates = [
			_build_message_mlp_candidate(
				candidate,
				candidate_index=index,
				message_layer_norm=message_layer_norm,
				in_features=in_features,
				hidden_features=hidden_features,
				out_features=out_features,
			)
			for index, candidate in enumerate(select_message_mlp)
		]

	gate = _build_message_mlp_gate(
		num_choices=len(candidates),
		select_message_mlp_gate=select_message_mlp_gate,
		message_mlp_selector_mode=message_mlp_selector_mode,
	)
	return SelectPath(candidates, gate)


def _build_message_mlp_candidate(
	candidate: dict,
	*,
	candidate_index: int,
	message_layer_norm: bool,
	in_features: int,
	hidden_features: int,
	out_features: int,
) -> nn.Module:
	if not isinstance(candidate, dict):
		raise ValueError(
			"select_message_mlp candidates must be dicts; "
			f"candidate {candidate_index} has type {type(candidate).__name__}."
		)

	shape_keys = sorted(set(candidate).intersection(_MESSAGE_MLP_SHAPE_KEYS))
	if shape_keys:
		raise ValueError(
			"select_message_mlp candidates must not override shape-derived keys: "
			f"{shape_keys}."
		)
	unknown = sorted(set(candidate).difference(_MESSAGE_MLP_CANDIDATE_KEYS))
	if unknown:
		raise ValueError(f"Unsupported select_message_mlp candidate keys: {unknown}.")
	if "message_dim" in candidate and "hidden_features" in candidate:
		raise ValueError(
			"select_message_mlp candidate cannot set both 'message_dim' and "
			"'hidden_features'."
		)

	candidate_hidden = candidate.get(
		"hidden_features",
		candidate.get("message_dim", hidden_features),
	)
	return _build_single_message_mlp(
		message_layer_norm=bool(
			candidate.get("message_layer_norm", message_layer_norm)
		),
		in_features=in_features,
		hidden_features=int(candidate_hidden),
		out_features=out_features,
		depth=int(candidate.get("depth", 2)),
		activation=candidate.get("activation", "gelu"),
		dropout=float(candidate.get("dropout", 0.0)),
		init_mode=str(candidate.get("init_mode", "torch_default")),
		init_seed=candidate.get("init_seed"),
	)


def _build_message_mlp_gate(
	*,
	num_choices: int,
	select_message_mlp_gate: dict | None,
	message_mlp_selector_mode: str,
) -> CategoricalGateConfig:
	if message_mlp_selector_mode not in MESSAGE_MLP_SELECTOR_MODES:
		raise ValueError(
			"message_mlp_selector_mode must be one of "
			f"{MESSAGE_MLP_SELECTOR_MODES}."
		)
	gate_overrides = {}
	if select_message_mlp_gate is not None:
		if not isinstance(select_message_mlp_gate, dict):
			raise ValueError("select_message_mlp_gate must be a dict or None.")
		unknown = sorted(set(select_message_mlp_gate).difference(_MESSAGE_MLP_GATE_KEYS))
		if unknown:
			raise ValueError(f"Unsupported select_message_mlp_gate keys: {unknown}.")
		gate_overrides = dict(select_message_mlp_gate)

	alpha_optim = (
		"shared" if message_mlp_selector_mode == "shared_train" else "separate"
	)
	alpha_update_split = (
		"val" if message_mlp_selector_mode == "separate_val" else "train"
	)
	return CategoricalGateConfig(
		num_choices=num_choices,
		alpha_optim=alpha_optim,
		alpha_update_split=alpha_update_split,
		**gate_overrides,
	)


def _build_single_message_mlp(
	*,
	message_layer_norm: bool,
	in_features: int,
	hidden_features: int,
	out_features: int,
	depth: int,
	activation,
	dropout: float,
	init_mode: str,
	init_seed: int | None,
) -> nn.Module:
	activation_cfg = (
		activation if isinstance(activation, ActConfig) else ActConfig(kind=activation)
	)
	return build_dense_mlp(
		DenseMLPConfig(
			depth=depth,
			in_features=in_features,
			hidden_features=hidden_features,
			out_features=out_features,
			activation=activation_cfg,
			norm=NormConfig("layer") if message_layer_norm else NormConfig(kind=None),
			regularization=RegConfig(dropout, dropout),
			init=InitConfig(mode=init_mode),
		),
		init_seed=init_seed,
	)


def _build_readout(
	*,
	in_features: int,
	n_classes: int,
	init_seed: int | None,
) -> nn.Module:
	return build_dense_mlp(
		DenseMLPConfig(
			in_features=in_features,
			hidden_features=n_classes,
			out_features=n_classes,
			depth=1,
			activation=ActConfig(kind="identity"),
			norm=NormConfig(kind=None),
			regularization=RegConfig(),
			init=InitConfig(mode="torch_default"),
		),
		init_seed=init_seed,
	)


class MSCEvidenceGNNCore(nn.Module):
	"""Torch core for static MSC message evidence accumulation."""

	def __init__(
		self,
		n_channels: int,
		nfreqs: int,
		n_classes: int,
		hidden_dim: int = 8,
		message_dim: int = 8,
		coherence_threshold: float = 0.7,
		phase_threshold_deg: float = 30.0,
		use_mag: bool = True,
		use_ang: bool = False,
		use_raw: bool = True,
		use_freq: bool = True,
		readout_mode: str = "mean",
		evidence_norm: str = "all_slots",
		component_profile: str = "legacy",
		message_layer_norm: bool = False,
		model_init_seed: int | None = None,
		message_init_seed: int | None = None,
		readout_init_seed: int | None = None,
		select_message_mlp: list[dict] | None = None,
		select_message_mlp_gate: dict | None = None,
		message_mlp_selector_mode: str = "separate_train",
		feature_conv_kernel_size: int = 5,
		feature_conv_pool_size: int = 4,
		feature_conv_intermediate_channels: int | None = None,
		feature_conv_intermediate_channels_reduced: int | None = None,
		feature_conv_feature_dim: int = 4,
		**kwargs,
	) -> None:
		super().__init__()
		if readout_mode not in {"mean", "flatten"}:
			raise ValueError("readout_mode must be one of {'mean', 'flatten'}")
		if evidence_norm not in {"all_slots", "active_slots", "none"}:
			raise ValueError(
				"evidence_norm must be one of {'all_slots', 'active_slots', 'none'}"
			)
		if message_mlp_selector_mode not in MESSAGE_MLP_SELECTOR_MODES:
			raise ValueError(
				"message_mlp_selector_mode must be one of "
				f"{MESSAGE_MLP_SELECTOR_MODES}."
			)
		if component_profile not in WCT_EVIDENCE_COMPONENT_PROFILES:
			raise ValueError(
				f"Unsupported component_profile={component_profile!r}. "
				f"Expected one of {WCT_EVIDENCE_COMPONENT_PROFILES}."
			)

		self.n_channels = n_channels
		self.nfreqs = nfreqs
		self.hidden_dim = hidden_dim
		self.message_dim = message_dim
		self.coherence_threshold = float(coherence_threshold)
		self.phase_threshold_rad = math.radians(phase_threshold_deg)
		self.use_mag = use_mag
		self.use_ang = use_ang
		self.use_raw = use_raw
		self.use_freq = use_freq
		self.readout_mode = readout_mode
		self.evidence_norm = evidence_norm
		self.component_profile = component_profile

		src_idx, dst_idx = _ordered_pair_indices(n_channels)
		self.register_buffer("src_idx", src_idx, persistent=False)
		self.register_buffer("dst_idx", dst_idx, persistent=False)
		self.register_buffer(
			"edge_pair_idx",
			torch.cat([src_idx, dst_idx]),
			persistent=False,
		)

		self.feature_conv_kernel_size = feature_conv_kernel_size
		self.feature_conv_pool_size = feature_conv_pool_size
		if feature_conv_intermediate_channels is None:
			feature_conv_intermediate_channels = nfreqs
		self.feature_conv_intermediate_channels = feature_conv_intermediate_channels
		self.feature_conv_out_channels = feature_conv_feature_dim * nfreqs
		self.feature_conv_intermediate_channels_reduced = (
			feature_conv_intermediate_channels_reduced
		)
		self.feature_conv_feature_dim = feature_conv_feature_dim

		payload_dim = self.feature_conv_feature_dim * 2
		if self.use_freq:
			payload_dim += 1
		if self.use_mag:
			payload_dim += 1
		if self.use_ang:
			payload_dim += 1
		if self.use_raw:
			payload_dim += 2
		if payload_dim == 0:
			raise ValueError("At least one payload component must be enabled.")
		self.payload_dim = payload_dim
		self._summary_context: dict[str, object] | None = None

		with scoped_torch_init_seed(model_init_seed):
			self.feature_conv = _build_feature_conv(
				kernel_size=self.feature_conv_kernel_size,
				intermediate_channels=self.feature_conv_intermediate_channels,
				out_channels=self.feature_conv_out_channels,
				pool_size=self.feature_conv_pool_size,
				intermediate_channels_reduced=self.feature_conv_intermediate_channels_reduced,
			)
			self.message_mlp = _build_message_mlp(
				message_layer_norm=message_layer_norm,
				in_features=payload_dim,
				hidden_features=message_dim,
				out_features=hidden_dim,
				init_seed=message_init_seed,
				select_message_mlp=select_message_mlp,
				select_message_mlp_gate=select_message_mlp_gate,
				message_mlp_selector_mode=message_mlp_selector_mode,
			)
			readout_dim = (
				hidden_dim * n_channels if readout_mode == "flatten" else hidden_dim
			)
			self.classifier = _build_readout(
				in_features=readout_dim,
				n_classes=n_classes,
				init_seed=readout_init_seed,
			)

	def _aggregate_per_node(self, msg: torch.Tensor) -> torch.Tensor:
		"""Aggregate [B, E, H] messages to [B, C, H] by destination."""
		batch_size, _, hidden_dim = msg.shape
		agg = torch.zeros(
			batch_size,
			self.n_channels,
			hidden_dim,
			device=msg.device,
			dtype=msg.dtype,
		)
		agg.index_add_(1, self.dst_idx, msg)
		return agg

	def configure_summary_context(
		self,
		*,
		batch_size: int,
		n_time: int,
		dtype: torch.dtype,
		n_samples: int | None = None,
	) -> None:
		self._summary_context = {
			"batch_size": int(batch_size),
			"n_time": int(n_time),
			"dtype": dtype,
			"n_samples": None if n_samples is None else int(n_samples),
		}

	def print_custom_summary(self, header: str = "Model") -> None:
		context = self._summary_context
		emit_initial_detail(
			f"[{header}] MSCEvidence config "
			f"nfreqs={self.nfreqs} hidden_dim={self.hidden_dim} "
			f"message_dim={self.message_dim} payload_dim={self.payload_dim} "
			f"readout_mode={self.readout_mode} evidence_norm={self.evidence_norm}"
		)
		emit_initial_detail(
			f"[{header}] MSCEvidence config "
			f"coherence_threshold={self.coherence_threshold:.4f} "
			f"phase_threshold_deg={math.degrees(self.phase_threshold_rad):.4f} "
			f"use_mag={self.use_mag} use_ang={self.use_ang} "
			f"use_raw={self.use_raw} use_freq={self.use_freq}"
		)
		self._print_selectable_message_mlp_summary(header)

		if context is None:
			emit_initial_detail(
				f"[{header}] MSCEvidence memory estimates unavailable: "
				"summary context was not configured."
			)
			return

		batch_size = int(context["batch_size"])
		n_time = int(context["n_time"])
		n_samples = context["n_samples"]
		dtype = context["dtype"]
		num_edges = self.src_idx.numel()
		dtype_bytes = _dtype_nbytes(dtype)
		emit_initial_detail(
			f"[{header}] MSCEvidence dimensions "
			f"B={batch_size} C={self.n_channels} E={num_edges} "
			f"T={n_time} F={self.nfreqs} D={self.feature_conv_feature_dim} "
			f"H={self.hidden_dim} dtype={dtype} bytes_per_elem={dtype_bytes} "
			f"n_samples={n_samples}"
		)
		emit_initial_detail(
			f"[{header}] MSCEvidence memory estimates "
			"approximate tensor payloads only; autograd, optimizer state, "
			"allocator fragmentation, and convolution workspace are excluded."
		)
		for label, shape, copies in self._critical_tensor_estimates(
			batch_size=batch_size,
			n_time=n_time,
		):
			numel = _shape_numel(shape) * copies
			copies_prefix = f"{copies} x " if copies != 1 else ""
			emit_initial_detail(
				f"[{header}] MSCEvidence tensor {label}: "
				f"shape={copies_prefix}{shape} "
				f"elements={numel} "
				f"approx_memory={_format_bytes(numel * dtype_bytes)}"
			)

	def _print_selectable_message_mlp_summary(self, header: str) -> None:
		if not isinstance(self.message_mlp, SelectPath):
			return

		gate = self.message_mlp.gate
		probs = gate.probabilities().detach().float().cpu()
		probs_by_candidate = probs.reshape(-1, int(probs.shape[-1])).mean(dim=0)
		entropy = -(
			probs
			* probs.clamp_min(torch.finfo(probs.dtype).tiny).log()
		).sum(dim=-1).mean()
		candidate_params = [
			sum(int(param.numel()) for param in candidate.parameters())
			for candidate in self.message_mlp.choice.candidates
		]
		selector_mode = _selector_mode_from_gate(gate)
		emit_initial_detail(
			f"[{header}] MSCEvidence selectable message_mlp "
			f"candidates={len(candidate_params)} "
			f"selector_optimizer_mode={selector_mode} "
			f"alpha={gate.alpha_optim}/{gate.alpha_update_split} "
			f"mode={gate.mode} eval_mode={gate.eval_mode} "
			f"gradient_mode={gate.gradient_mode} "
			f"temperature={gate.temperature:.4g} "
			f"exploration_epsilon={gate.exploration_epsilon} "
			f"initial_probs={_format_number_list(probs_by_candidate.tolist())} "
			f"entropy={float(entropy.item()):.4f} "
			f"candidate_params={candidate_params}"
		)

	def _critical_tensor_estimates(
		self,
		*,
		batch_size: int,
		n_time: int,
	) -> list[tuple[str, tuple[int, ...], int]]:
		num_edges = self.src_idx.numel()
		return [
			(
				"feature_conv_output",
				(
					batch_size,
					self.nfreqs * self.feature_conv_feature_dim,
					self.n_channels,
					1,
				),
				1,
			),
			(
				"edge_conv_src_dst",
				(
					batch_size,
					num_edges,
					self.nfreqs,
					self.feature_conv_feature_dim,
				),
				2,
			),
			(
				"full_edge_maps",
				(batch_size, num_edges, n_time, self.nfreqs),
				5 if self.use_mag else 4,
			),
			(
				"message_payload",
				(batch_size, num_edges, self.nfreqs, self.payload_dim),
				1,
			),
			(
				"messages",
				(batch_size, num_edges, self.nfreqs, self.hidden_dim),
				1,
			),
			("evidence", (batch_size, self.n_channels, self.hidden_dim), 1),
		]

	def _batched_freqs(self, freqs: torch.Tensor, batch_size: int) -> torch.Tensor:
		if freqs.ndim == 1:
			freqs = freqs.view(1, -1).expand(batch_size, -1)
		if freqs.shape != (batch_size, self.nfreqs):
			raise ValueError(
				f"Expected freqs shape {(batch_size, self.nfreqs)} or "
				f"{(self.nfreqs,)}, got {tuple(freqs.shape)}."
			)
		return freqs

	def _full_edge_wct_maps(
		self,
		w_real: torch.Tensor,
		w_imag: torch.Tensor,
		freqs: torch.Tensor,
		*,
		compute_mag: bool,
	) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		num_edges = self.src_idx.numel()
		real_edges = w_real.index_select(1, self.edge_pair_idx)
		imag_edges = w_imag.index_select(1, self.edge_pair_idx)
		src_r, dst_r = real_edges.split(num_edges, dim=1)
		src_i, dst_i = imag_edges.split(num_edges, dim=1)

		xwt_real = src_r * dst_r + src_i * dst_i
		xwt_imag = src_i * dst_r - src_r * dst_i
		mag = None
		if compute_mag:
			mag = torch.sqrt(xwt_real * xwt_real + xwt_imag * xwt_imag + 1e-12)
		auto1 = src_r * src_r + src_i * src_i
		auto2 = dst_r * dst_r + dst_i * dst_i

		inv_scale = freqs.view(freqs.shape[0], 1, 1, self.nfreqs)
		return (
			mag,
			xwt_real * inv_scale,
			xwt_imag * inv_scale,
			auto1 * inv_scale,
			auto2 * inv_scale,
		)

	def _compute_msc_features(
		self,
		w_real: torch.Tensor,
		w_imag: torch.Tensor,
		freqs: torch.Tensor,
	) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
		mag, xwt_real, xwt_imag, auto1, auto2 = self._full_edge_wct_maps(
			w_real,
			w_imag,
			freqs,
			compute_mag=self.use_mag,
		)
		mean_cross = torch.complex(xwt_real, xwt_imag).mean(dim=2)
		mean_auto1 = auto1.mean(dim=2)
		mean_auto2 = auto2.mean(dim=2)
		coh = (mean_cross.abs() ** 2) / (mean_auto1 * mean_auto2 + 1e-12)
		coh = coh.clamp(min=0.0, max=1.0)
		mean_phase = torch.angle(mean_cross)
		mean_mag = mag.mean(dim=2) if mag is not None else None
		return mean_mag, mean_phase, coh

	def _readout(self, evidence: torch.Tensor) -> torch.Tensor:
		batch_size = evidence.shape[0]
		readout = (
			evidence.reshape(batch_size, self.n_channels * self.hidden_dim)
			if self.readout_mode == "flatten"
			else evidence.mean(dim=1)
		)
		return self.classifier(readout)

	def forward(
		self,
		raw_x: torch.Tensor,
		w_real: torch.Tensor,
		w_imag: torch.Tensor,
		freqs: torch.Tensor,
	) -> tuple[torch.Tensor, float]:
		batch_size, n_channels, _ = raw_x.shape
		if n_channels != self.n_channels:
			raise ValueError(f"Expected {self.n_channels} channels, got {n_channels}.")

		device = raw_x.device
		conv_features = self.feature_conv(raw_x.unsqueeze(1)).squeeze(-1)
		conv_by_freq = conv_features.view(
			batch_size,
			self.nfreqs,
			self.feature_conv_feature_dim,
			n_channels,
		).permute(0, 3, 1, 2)

		edge_src_conv = conv_by_freq.index_select(1, self.src_idx)
		edge_dst_conv = conv_by_freq.index_select(1, self.dst_idx)

		freqs = self._batched_freqs(freqs, batch_size).to(
			device=raw_x.device,
			dtype=raw_x.dtype,
		)
		mean_mag, mean_phase, coh = self._compute_msc_features(
			w_real=w_real,
			w_imag=w_imag,
			freqs=freqs,
		)
		gate_mask = (coh > self.coherence_threshold) & (
			mean_phase > self.phase_threshold_rad
		)
		gate_sum = float(gate_mask.sum().item())
		gate_count = float(gate_mask.numel())

		features = [edge_src_conv, edge_dst_conv]
		if self.use_freq:
			inv_freq = (1.0 / freqs).view(batch_size, 1, self.nfreqs, 1)
			features.append(
				inv_freq.expand(
					batch_size,
					self.src_idx.numel(),
					self.nfreqs,
					1,
				)
			)
		if self.use_mag:
			if mean_mag is None:
				raise RuntimeError("Magnitude payload is enabled but mean_mag is missing.")
			features.append(
				torch.nan_to_num(mean_mag, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(-1)
			)
		if self.use_ang:
			features.append(
				torch.nan_to_num(mean_phase, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(-1)
			)
		if self.use_raw:
			raw_trial = torch.nan_to_num(raw_x.mean(dim=2), nan=0.0, posinf=0.0, neginf=0.0)
			src_raw = raw_trial[:, self.src_idx].unsqueeze(-1).unsqueeze(-1)
			dst_raw = raw_trial[:, self.dst_idx].unsqueeze(-1).unsqueeze(-1)
			features.extend(
				[
					src_raw.expand(batch_size, self.src_idx.numel(), self.nfreqs, 1),
					dst_raw.expand(batch_size, self.src_idx.numel(), self.nfreqs, 1),
				]
			)

		msg = self.message_mlp(torch.cat(features, dim=-1))
		msg = msg * gate_mask.to(dtype=msg.dtype).unsqueeze(-1)
		evidence = self._aggregate_per_node(msg.sum(dim=2))

		if self.evidence_norm == "all_slots":
			slots_per_destination = max((self.n_channels - 1) * self.nfreqs, 1)
			evidence = evidence / float(slots_per_destination)
		elif self.evidence_norm == "active_slots":
			active_per_edge = gate_mask.to(dtype=torch.float32).sum(dim=2)
			active_slots_per_node = torch.zeros(
				batch_size,
				self.n_channels,
				device=device,
				dtype=torch.float32,
			)
			active_slots_per_node.index_add_(1, self.dst_idx, active_per_edge)
			evidence = evidence / active_slots_per_node.clamp_min(1.0).unsqueeze(-1)

		edge_density = (gate_sum / gate_count) if gate_count > 0 else 0.0
		return self._readout(evidence), edge_density


class MSCEvidenceGNNClassifier(_BaseCWTGNNClassifier):
	"""sklearn/MOABB wrapper around the static MSC evidence GNN."""

	model_label = "MSC-Evidence"

	def __init__(
		self,
		sampling_rate: int = 250,
		lowest: float = 8.0,
		highest: float = 35.0,
		nfreqs: int = 16,
		cwt_resample_n_time: int | None = None,
		coherence_threshold: float = 0.7,
		phase_threshold_deg: float = 30.0,
		use_mag: bool = True,
		use_ang: bool = False,
		use_raw: bool = True,
		use_freq: bool = True,
		readout_mode: str = "mean",
		evidence_norm: str = "all_slots",
		hidden_dim: int = 8,
		message_dim: int = 8,
		epochs: int = 20,
		batch_size: int = 8,
		learning_rate: float = 1e-3,
		weight_decay: float = 1e-4,
		grad_clip_norm: float | None = 0.1,
		normalize_input: bool = True,
		noise_augmentation_enabled: bool = False,
		noise_apply_prob: float = 0.0,
		noise_strength: float = 0.0,
		noise_bank_size: int = 128,
		noise_bank_seed: int | None = None,
		validation_split: float | list | tuple | None = 0.2,
		validation_group_column: str | None = None,
		early_stopping_patience: int | None = None,
		device: str = "auto",
		seed: int = 42,
		last_batch_min_ratio: float = 0.0,
		selector_alpha_val_update_rate: float = 1.0,
		optimizer_step_batch_size: int | None = None,
		optimizer_step_batch_mode: Literal["credit", "split"] = "credit",
		optimizer_step_remainder_policy: Literal["flush", "drop", "carry"] = "flush",
		component_profile: str = "legacy",
		message_layer_norm: bool = False,
		message_init_seed: int | None = None,
		readout_init_seed: int | None = None,
		select_message_mlp: list[dict] | None = None,
		select_message_mlp_gate: dict | None = None,
		message_mlp_selector_mode: str = "separate_train",
		feature_conv_kernel_size: int = 5,
		feature_conv_pool_size: int = 4,
		feature_conv_intermediate_channels: int | None = None,
		feature_conv_intermediate_channels_reduced: int | None = None,
		feature_conv_feature_dim: int = 4,
		channel_subset: list[int] | list[str] | None = None,
		verbose: int = 0,
	) -> None:
		self.coherence_threshold = coherence_threshold
		self.phase_threshold_deg = phase_threshold_deg
		self.use_mag = use_mag
		self.use_ang = use_ang
		self.use_raw = use_raw
		self.use_freq = use_freq
		self.readout_mode = readout_mode
		self.evidence_norm = evidence_norm
		self.hidden_dim = hidden_dim
		self.message_dim = message_dim
		self.component_profile = component_profile
		self.message_layer_norm = message_layer_norm
		self.message_init_seed = message_init_seed
		self.readout_init_seed = readout_init_seed
		self.select_message_mlp = select_message_mlp
		self.select_message_mlp_gate = select_message_mlp_gate
		self.message_mlp_selector_mode = message_mlp_selector_mode
		self.feature_conv_kernel_size = feature_conv_kernel_size
		self.feature_conv_pool_size = feature_conv_pool_size
		self.feature_conv_intermediate_channels = feature_conv_intermediate_channels
		self.feature_conv_intermediate_channels_reduced = (
			feature_conv_intermediate_channels_reduced
		)
		self.feature_conv_feature_dim = feature_conv_feature_dim
		self._init_cwt_gnn_classifier(
			sampling_rate=sampling_rate,
			lowest=lowest,
			highest=highest,
			nfreqs=nfreqs,
			cwt_resample_n_time=cwt_resample_n_time,
			epochs=epochs,
			batch_size=batch_size,
			learning_rate=learning_rate,
			weight_decay=weight_decay,
			grad_clip_norm=grad_clip_norm,
			normalize_input=normalize_input,
			noise_augmentation_enabled=noise_augmentation_enabled,
			noise_apply_prob=noise_apply_prob,
			noise_strength=noise_strength,
			noise_bank_size=noise_bank_size,
			noise_bank_seed=noise_bank_seed,
			validation_split=validation_split,
			validation_group_column=validation_group_column,
			early_stopping_patience=early_stopping_patience,
			device=device,
			seed=seed,
			last_batch_min_ratio=last_batch_min_ratio,
			selector_alpha_val_update_rate=selector_alpha_val_update_rate,
			optimizer_step_batch_size=optimizer_step_batch_size,
			optimizer_step_batch_mode=optimizer_step_batch_mode,
			optimizer_step_remainder_policy=optimizer_step_remainder_policy,
			channel_subset=channel_subset,
			verbose=verbose,
		)

	def _build_model_from_features(self, features, n_classes: int, **kwargs) -> MSCEvidenceGNNCore:
		raw_x = features[0] if isinstance(features, tuple) else features
		model = self._build_model(
			n_channels=int(raw_x.shape[1]),
			n_classes=n_classes,
			**kwargs,
		)
		model.configure_summary_context(
			batch_size=int(self.batch_size),
			n_time=int(raw_x.shape[2]),
			dtype=raw_x.dtype,
			n_samples=int(raw_x.shape[0]),
		)
		return model

	def _build_model(self, n_channels: int, n_classes: int, **kwargs) -> MSCEvidenceGNNCore:
		return MSCEvidenceGNNCore(
			n_channels=n_channels,
			nfreqs=self.nfreqs,
			n_classes=n_classes,
			hidden_dim=self.hidden_dim,
			message_dim=self.message_dim,
			coherence_threshold=self.coherence_threshold,
			phase_threshold_deg=self.phase_threshold_deg,
			use_mag=self.use_mag,
			use_ang=self.use_ang,
			use_raw=self.use_raw,
			use_freq=self.use_freq,
			readout_mode=self.readout_mode,
			evidence_norm=self.evidence_norm,
			component_profile=self.component_profile,
			message_layer_norm=self.message_layer_norm,
			model_init_seed=self.seed,
			message_init_seed=self.message_init_seed,
			readout_init_seed=self.readout_init_seed,
			select_message_mlp=self.select_message_mlp,
			select_message_mlp_gate=self.select_message_mlp_gate,
			message_mlp_selector_mode=self.message_mlp_selector_mode,
			feature_conv_kernel_size=self.feature_conv_kernel_size,
			feature_conv_pool_size=self.feature_conv_pool_size,
			feature_conv_intermediate_channels=self.feature_conv_intermediate_channels,
			feature_conv_intermediate_channels_reduced=self.feature_conv_intermediate_channels_reduced,
			feature_conv_feature_dim=self.feature_conv_feature_dim,
			**kwargs,
		)
