import argparse
import ast
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import logging
import math
import numbers
import os
from pathlib import Path
import re
import sys

from mne import get_config
from mne.decoding import CSP
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from moabb.datasets import BNCI2014_001
from moabb.evaluations import CrossSessionEvaluation
try:
    from moabb.evaluations import GlobalFutureSessionEvaluation
except ImportError:
    GlobalFutureSessionEvaluation = None
from moabb.paradigms import LeftRightImagery

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_path = str(REPO_ROOT)
if repo_root_path not in sys.path:
    sys.path.insert(0, repo_root_path)

from coheriqs_contributions import run_wct_gnn as _wct_run
from coheriqs_contributions.experiment_logging import (
    EventCategory,
    add_console_arguments,
    configure_experiment_logging,
    console_policy_from_args,
    log_event,
    resolve_experiment_log_path,
)
from coheriqs_contributions.moabb_pipelines.msc_evidence_gnn import (
    MSCEvidenceGNNClassifier,
)


log = logging.getLogger(__name__)

RunConfigurationError = _wct_run.RunConfigurationError


def _make_msc_evidence_gnn():
    return MSCEvidenceGNNClassifier(
        sampling_rate=250,
        lowest=8.0,
        highest=35.0,
        nfreqs=16,
        cwt_resample_n_time=200,
        coherence_threshold=0.5,
        phase_threshold_deg=30.0,
        use_mag=True,
        use_ang=False,
        use_raw=False,
        use_freq=True,
        readout_mode="flatten",
        evidence_norm="none",
        hidden_dim=8,
        message_dim=8,
        epochs=50,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=1e-4,
        grad_clip_norm=0.1,
        normalize_input=True,
        noise_augmentation_enabled=False,
        noise_apply_prob=0.0,
        noise_strength=0.0,
        noise_bank_size=128,
        noise_bank_seed=None,
        validation_split=0.2,
        validation_group_column=None,
        early_stopping_patience=None,
        device="auto",
        seed=42,
        component_profile="legacy",
        message_layer_norm=False,
        message_init_seed=None,
        readout_init_seed=None,
        feature_conv_kernel_size=5,
        feature_conv_pool_size=4,
        feature_conv_intermediate_channels=None,
        feature_conv_intermediate_channels_reduced=4,
        feature_conv_feature_dim=2,
        select_message_mlp=None,
        select_message_mlp_gate=None,
        message_mlp_selector_mode="shared_train",
        selector_alpha_val_update_rate=0.5,
        last_batch_min_ratio=0.5,
        optimizer_step_batch_size=None,
        optimizer_step_batch_mode="credit",
        optimizer_step_remainder_policy="carry",
        channel_subset=[1, 5, 7, 8, 9, 10, 11, 13, 17],
        verbose=3,
    )


PIPELINE_BUILDERS = dict(_wct_run.PIPELINE_BUILDERS)
PIPELINE_BUILDERS["MSC-Evidence-GNN"] = _make_msc_evidence_gnn
DEFAULT_PIPELINES = ["MSC-Evidence-GNN"]
PIPELINE_PARAM_GRIDS = deepcopy(_wct_run.PIPELINE_PARAM_GRIDS)
PIPELINE_PARAM_GRIDS["MSC-Evidence-GNN"] = {
    "batch_size": [32],
    "readout_mode": ["flatten"],
    "evidence_norm": ["active_slots"],
    "message_layer_norm": [False],
    "seed": [42],
    "select_message_mlp": [
        [
            {"init_seed": 101},
            {"init_seed": 103},
            {"init_seed": 104},
        ],
    ],
    "select_message_mlp_gate": [
        {"mode": "gumbel_hard"},
    ],
    "message_mlp_selector_mode": [
        "separate_val",
    ],
    "selector_alpha_val_update_rate": [0.5],
    "last_batch_min_ratio": [0.5],
    "optimizer_step_batch_size": [None],
    "optimizer_step_batch_mode": ["credit"],
    "optimizer_step_remainder_policy": ["carry"],
    "epochs": [50],
    "normalize_input": [True],
    "learning_rate": [1.0e-3],
    "weight_decay": [1.0e-2],
    "noise_augmentation_enabled": [True],
    "noise_apply_prob": [1.0],
    "noise_strength": [0.15],
    "noise_bank_size": [20000],
    "noise_bank_seed": [33],
    "use_raw": [False],
    "use_freq": [True],
    "use_mag": [False],
    "use_ang": [False],
    "verbose": [2],
    "device": ["auto"],
    "channel_subset": [[1, 5, 7, 8, 9, 10, 11, 13, 17]],
}


def _parse_param_value(raw_value):
    normalized = raw_value.strip()
    named_literals = {"true": True, "false": False, "none": None, "null": None}
    if normalized.lower() in named_literals:
        return named_literals[normalized.lower()]
    try:
        return ast.literal_eval(normalized)
    except (SyntaxError, ValueError):
        return normalized


def _normalize_param_names(param_names):
    if param_names is None:
        return None
    normalized = [name.strip() for token in param_names for name in token.split(",")]
    if any(not name for name in normalized):
        raise ValueError("--param-names contains an empty parameter name.")
    return normalized


def _normalize_param_values(raw_param_values):
    if raw_param_values is None:
        return None
    if len(raw_param_values) == 1:
        parsed = _parse_param_value(raw_param_values[0])
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        return [parsed]
    return [_parse_param_value(raw_value) for raw_value in raw_param_values]


def _values_are_type_compatible(value, reference):
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if not math.isfinite(value):
            return False
    if isinstance(reference, bool) or isinstance(value, bool):
        return isinstance(value, bool) and isinstance(reference, bool)
    if isinstance(reference, numbers.Integral):
        return isinstance(value, numbers.Integral) and not isinstance(value, bool)
    if isinstance(reference, numbers.Real):
        return isinstance(value, numbers.Real) and not isinstance(value, bool)
    return isinstance(value, type(reference))


def _validate_param_overrides(pipeline_names, param_names, raw_param_values):
    if param_names is None and raw_param_values is None:
        return {}
    normalized_names = _normalize_param_names(param_names)
    parsed_values = _normalize_param_values(raw_param_values)
    if not normalized_names or not parsed_values:
        raise ValueError(
            "--param-names and --param-values must both be provided with at least "
            "one value."
        )
    if len(normalized_names) != len(parsed_values):
        raise ValueError(
            "--param-names and --param-values must contain the same number of entries."
        )
    duplicates = sorted(
        name for name in set(normalized_names) if normalized_names.count(name) > 1
    )
    if duplicates:
        raise ValueError(
            "--param-names must not contain duplicates: " + ", ".join(duplicates)
        )

    overrides = dict(zip(normalized_names, parsed_values))
    for pipeline_name in dict.fromkeys(pipeline_names):
        estimator = PIPELINE_BUILDERS[pipeline_name]()
        supported_params = estimator.get_params(deep=True)
        unsupported = sorted(set(overrides).difference(supported_params))
        if unsupported:
            raise ValueError(
                f"Pipeline '{pipeline_name}' does not support requested parameter(s): "
                + ", ".join(unsupported)
            )
        grid = PIPELINE_PARAM_GRIDS[pipeline_name]
        for name, value in overrides.items():
            references = [supported_params[name], *grid.get(name, [])]
            typed_references = [reference for reference in references if reference is not None]
            is_compatible = (
                any(reference is None for reference in references)
                if value is None
                else not typed_references
                or any(
                    _values_are_type_compatible(value, reference)
                    for reference in typed_references
                )
            )
            if not is_compatible:
                expected_types = sorted(
                    {
                        type(reference).__name__
                        for reference in references
                        if reference is not None
                    }
                )
                raise ValueError(
                    f"Value {value!r} for parameter '{name}' is not type-compatible "
                    f"with pipeline '{pipeline_name}' (expected one of: "
                    f"{', '.join(expected_types)})."
                )
    return overrides


def _write_group_artifacts(
    *,
    evaluation,
    group_results,
    group_id,
    run_id,
    subjects,
    eval_mode,
    inner_group,
    run_param_grid,
    singleton_applied,
    data_root,
    fixed_overrides=None,
    effective_params=None,
    experiment_log_path=None,
    console_policy=None,
    overwrite=True,
):
    hdf5_path = Path(evaluation.results.filepath).resolve()
    artifact_dir = hdf5_path.parent
    scores_path = artifact_dir / f"scores_{group_id}.csv"
    summary_path = artifact_dir / f"summary_{group_id}.md"
    if not overwrite:
        existing_paths = [path for path in (scores_path, summary_path) if path.exists()]
        if existing_paths:
            raise FileExistsError(
                "Run artifact already exists; refusing to overwrite: "
                + ", ".join(str(path) for path in existing_paths)
            )
    group_results.to_csv(scores_path, index=False)

    outer_columns = ["subject", "session", "pipeline", "score"]
    if "best_params" in group_results.columns:
        outer_columns.append("best_params")
    means = (
        group_results.groupby(["subject", "pipeline"], as_index=False)["score"]
        .mean()
        .rename(columns={"score": "mean_score"})
    )
    lines = [
        "# MSC run summary",
        "",
        f"- Run ID: `{run_id}`",
        f"- Group: `{group_id}`",
        f"- Evaluation mode: `{eval_mode}`",
        f"- Inner grouping: `{inner_group or 'none'}`",
        f"- Subjects: `{', '.join(str(subject) for subject in subjects)}`",
        f"- Configured data root: `{data_root}`",
        f"- HDF5 store: `{hdf5_path}`",
        f"- Scores CSV: `{scores_path}`",
    ]
    if experiment_log_path is not None:
        lines.append(f"- Experiment log: `{experiment_log_path}`")
    if console_policy is not None:
        lines.append(f"- Console output policy: `{console_policy!r}`")
    lines.append(f"- Overwrite existing outputs: `{overwrite}`")
    lines.append(f"- Fixed parameter overrides: `{fixed_overrides or {}}`")
    lines.extend(["", "## Outer-CV rows", ""])
    lines.extend(_wct_run._markdown_table(group_results, outer_columns))
    lines.extend(["", "## Subject/pipeline means", ""])
    lines.extend(_wct_run._markdown_table(means, ["subject", "pipeline", "mean_score"]))
    lines.extend(["", "## Run-grid configuration", ""])
    for label, configured_grid in run_param_grid.items():
        lines.append(f"### {label}")
        lines.append("")
        if label in singleton_applied:
            lines.append("Singleton parameters applied directly:")
            values = singleton_applied[label]
        else:
            lines.append("Search grid passed to MOABB:")
            values = configured_grid
        lines.append("")
        for name in sorted(values):
            lines.append(f"- `{name}`: `{values[name]!r}`")
        lines.append("")
        if effective_params is not None and label in effective_params:
            lines.append("Effective estimator parameters:")
            lines.append("")
            for name in sorted(effective_params[label]):
                lines.append(f"- `{name}`: `{effective_params[label][name]!r}`")
            lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    log_event(log, EventCategory.ARTIFACT, f"[Artifact] HDF5: {hdf5_path}")
    log_event(log, EventCategory.ARTIFACT, f"[Artifact] scores CSV: {scores_path}")
    log_event(log, EventCategory.ARTIFACT, f"[Artifact] summary: {summary_path}")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--pipeline",
        action="append",
        default=[],
        choices=sorted(PIPELINE_BUILDERS.keys()),
        metavar="PIPELINE",
        help=(
            "Pipeline/model to run. Repeat for multiple pipelines. "
            f"If omitted, defaults to {', '.join(DEFAULT_PIPELINES)}."
        ),
    )
    parser.add_argument(
        "--inner-group-mode",
        default="none",
        choices=["none", "run", "both"],
        help=(
            "Global inner-CV grouping applied to all selected pipelines: "
            "none => inner_cv_groups disabled, "
            "run => inner_cv_groups='run', "
            "both => run each selected pipeline twice (none + run)."
        ),
    )
    parser.add_argument(
        "--global-hyperparam-fit",
        default="false",
        choices=["false", "true", "both"],
        help=(
            "Evaluation selector: "
            "'false' => CrossSessionEvaluation, "
            "'true' => GlobalFutureSessionEvaluation, "
            "'both' => run both evaluation types."
        ),
    )
    parser.add_argument(
        "--show-inner-results",
        action="store_true",
        help="Print collected inner cv_results_ summaries and split scores.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Identifier used in result filenames. Managed execution supplies its "
            "execution ID automatically; manual runs receive a timestamped ID."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Replace outputs for an existing run ID (enabled by default). "
            "--no-overwrite preserves MOABB results and rejects existing logs "
            "and human-readable artifacts."
        ),
    )
    parser.add_argument(
        "--experiment-log",
        default=None,
        help=(
            "Durable UTF-8 experiment log path. Defaults to "
            "MOABB_RESULTS/<run-id>/experiment.log."
        ),
    )
    add_console_arguments(parser)
    parser.add_argument(
        "--param-names",
        "--param_names",
        nargs="+",
        default=None,
        metavar="PARAM",
        help=(
            "Estimator parameters to fix instead of searching. Accepts either "
            "one token per name or a quoted comma-separated list."
        ),
    )
    parser.add_argument(
        "--param-values",
        "--param_values",
        nargs="+",
        default=None,
        metavar="VALUE",
        help=(
            "Safe literal values paired with --param-names. Accepts either one "
            "token per value or one quoted Python literal list; bare values are "
            "strings."
        ),
    )
    return parser


def parse_parameters(arguments: list[str] | None = None) -> argparse.Namespace:
    return _build_argument_parser().parse_args(arguments)


def main(parameters: argparse.Namespace) -> None:
    run_id = _wct_run._safe_run_id(parameters.run_id)
    selected_pipelines = parameters.pipeline if parameters.pipeline else DEFAULT_PIPELINES
    try:
        fixed_overrides = _validate_param_overrides(
            selected_pipelines,
            parameters.param_names,
            parameters.param_values,
        )
    except ValueError as exc:
        raise RunConfigurationError(str(exc)) from exc
    pipeline_runs = _wct_run._build_pipeline_runs(
        pipeline_names=selected_pipelines,
        inner_group_mode=parameters.inner_group_mode,
        global_hyperparam_fit_mode=parameters.global_hyperparam_fit,
    )
    console_policy = console_policy_from_args(parameters)
    moabb_results_root = _wct_run._configured_moabb_result_root()
    try:
        experiment_log_path = resolve_experiment_log_path(
            parameters.experiment_log,
            run_id=run_id,
            moabb_results_root=moabb_results_root,
            overwrite=parameters.overwrite,
        )
    except FileExistsError as exc:
        raise RunConfigurationError(str(exc)) from exc
    configure_experiment_logging(
        experiment_log_path,
        console_policy=console_policy,
        overwrite=parameters.overwrite,
    )
    log_event(
        log,
        EventCategory.ARTIFACT,
        f"[Artifact] experiment log: {experiment_log_path}",
    )
    log_event(
        log,
        EventCategory.CONFIG,
        f"Console output policy: {console_policy!r}",
    )
    log_event(
        log,
        EventCategory.CONFIG,
        f"Overwrite existing run outputs: {parameters.overwrite}",
    )
    _wct_run._print_run_plan(parameters.subjects, selected_pipelines, pipeline_runs)
    log_event(log, EventCategory.STATUS, f"Run ID: {run_id}")
    if fixed_overrides:
        log_event(
            log,
            EventCategory.CONFIG,
            f"Fixed parameter overrides: {fixed_overrides}",
        )

    dataset = BNCI2014_001()
    dataset.subject_list = parameters.subjects
    data_root = _wct_run._configured_data_root()
    log_event(
        log,
        EventCategory.CONFIG,
        f"[Results] MOABB root: {moabb_results_root}",
    )
    paradigm = LeftRightImagery(fmin=8, fmax=35)

    base_eval_kwargs = dict(
        paradigm=paradigm,
        datasets=[dataset],
        overwrite=parameters.overwrite,
        n_jobs=1,
        random_state=42,
        progress_bar=console_policy.moabb_progress,
    )

    grouped_runs = defaultdict(list)
    for run_cfg in pipeline_runs:
        grouped_runs[(run_cfg["eval_mode"], run_cfg["inner_group"])].append(run_cfg)

    results_chunks = []
    inner_chunks = []
    for (eval_mode, inner_group), run_cfgs in grouped_runs.items():
        log_event(
            log,
            EventCategory.STATUS,
            "=== Starting group: "
            f"eval={eval_mode}, inner_group={inner_group or 'None'} ===",
        )
        eval_kwargs = dict(base_eval_kwargs)
        group_id = (
            f"{run_id}__{eval_mode}__inner-"
            f"{str(inner_group or 'none').replace('_', '-')}"
        )
        eval_kwargs["suffix"] = group_id
        if inner_group is not None:
            eval_kwargs.update(
                inner_cv_class=StratifiedGroupKFold,
                inner_cv_kwargs={"n_splits": 3, "shuffle": True, "random_state": 42},
                inner_cv_groups=inner_group,
            )

        pipelines = {
            cfg["label"]: PIPELINE_BUILDERS[cfg["base_name"]]()
            for cfg in run_cfgs
        }
        log_event(
            log,
            EventCategory.CONFIG,
            f"Group pipelines: {list(pipelines)}",
        )
        run_param_grid = {
            cfg["label"]: deepcopy(PIPELINE_PARAM_GRIDS[cfg["base_name"]])
            for cfg in run_cfgs
        }
        run_param_grid = _wct_run._apply_fixed_param_overrides(
            pipelines, run_param_grid, fixed_overrides
        )
        param_grid, singleton_applied = _wct_run._prepare_param_grid_for_run(
            pipelines, run_param_grid
        )

        if singleton_applied:
            for label, params in singleton_applied.items():
                log_event(
                    log,
                    EventCategory.CONFIG,
                    f"[Grid] singleton combo for '{label}' applied directly "
                    f"({len(params)} parameters).",
                )
                log_event(
                    log,
                    EventCategory.INITIAL_DETAILS,
                    f"[Grid] singleton parameters for '{label}': {params}",
                )

        for label, estimator in pipelines.items():
            log_event(
                log,
                EventCategory.INITIAL_DETAILS,
                f"Effective estimator parameters for '{label}': "
                f"{estimator.get_params(deep=True)}",
            )
        if param_grid:
            log_event(
                log,
                EventCategory.INITIAL_DETAILS,
                f"Search grid passed to MOABB: {param_grid}",
            )

        if not param_grid:
            param_grid = None
        grid_status = (
            "disabled after singleton application" if param_grid is None else "enabled"
        )
        log_event(
            log,
            EventCategory.CONFIG,
            f"Grid search: {grid_status}",
        )

        if eval_mode == "global":
            if GlobalFutureSessionEvaluation is None:
                raise ValueError(
                    "GlobalFutureSessionEvaluation not available in this moabb version. "
                    "Use --global-hyperparam-fit false instead."
                )
            evaluation = GlobalFutureSessionEvaluation(**eval_kwargs)
        elif eval_mode == "cross":
            evaluation = CrossSessionEvaluation(**eval_kwargs)
        else:
            raise ValueError(f"Unsupported eval_mode='{eval_mode}'.")
        group_results = evaluation.process(pipelines, param_grid=param_grid)
        log_event(
            log,
            EventCategory.STATUS,
            f"Completed group rows: {len(group_results)}",
        )
        effective_params = {
            label: estimator.get_params(deep=True)
            for label, estimator in pipelines.items()
        }
        _write_group_artifacts(
            evaluation=evaluation,
            group_results=group_results,
            group_id=group_id,
            run_id=run_id,
            subjects=parameters.subjects,
            eval_mode=eval_mode,
            inner_group=inner_group,
            run_param_grid=run_param_grid,
            singleton_applied=singleton_applied,
            fixed_overrides=fixed_overrides,
            effective_params=effective_params,
            data_root=data_root,
            experiment_log_path=experiment_log_path,
            console_policy=console_policy,
            overwrite=parameters.overwrite,
        )
        results_chunks.append(group_results)

        try:
            group_inner = evaluation.get_inner_cv_results()
            if not group_inner.empty:
                inner_chunks.append(group_inner)
        except (AttributeError, TypeError):
            pass

    results = pd.concat(results_chunks, ignore_index=True)
    inner = pd.concat(inner_chunks, ignore_index=True) if inner_chunks else pd.DataFrame()

    log_event(log, EventCategory.FINAL_RESULTS, "=== Outer CV results ===")
    outer_cols = ["subject", "session", "pipeline", "score"]
    if "best_params" in results.columns:
        outer_cols.append("best_params")
    log_event(
        log,
        EventCategory.FINAL_RESULTS,
        results[outer_cols].to_string(index=False),
    )

    log_event(
        log,
        EventCategory.FINAL_RESULTS,
        "=== Per subject/pipeline mean scores ===",
    )
    per_subject_pipeline = (
        results.groupby(["subject", "pipeline"], as_index=False)["score"].mean()
        .rename(columns={"score": "mean_score"})
        .sort_values(["subject", "pipeline"])
    )
    log_event(
        log,
        EventCategory.FINAL_RESULTS,
        per_subject_pipeline.to_string(index=False),
    )

    log_event(
        log,
        EventCategory.FINAL_RESULTS,
        "=== Per pipeline mean scores ===",
    )
    per_pipeline = (
        results.groupby(["pipeline"], as_index=False)["score"].mean()
        .rename(columns={"score": "mean_score"})
        .sort_values(["pipeline"])
    )
    log_event(
        log,
        EventCategory.FINAL_RESULTS,
        per_pipeline.to_string(index=False),
    )

    if parameters.show_inner_results:
        _wct_run._print_inner_results(inner)


if __name__ == "__main__":
    main(parse_parameters())