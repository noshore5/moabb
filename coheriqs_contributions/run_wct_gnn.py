import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
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

import moabb
from moabb.datasets import BNCI2014_001
from moabb.evaluations import CrossSessionEvaluation
try:
    from moabb.evaluations import GlobalFutureSessionEvaluation
except ImportError:
    # Fallback to CrossSessionEvaluation if GlobalFutureSessionEvaluation not available
    GlobalFutureSessionEvaluation = None
from moabb.paradigms import LeftRightImagery

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_path = str(REPO_ROOT)
if repo_root_path not in sys.path:
    sys.path.insert(0, repo_root_path)

from coheriqs_contributions.moabb_pipelines.coherence_cnn_classifier import (
    CoherenceCNNClassifier,
)
from coheriqs_contributions.moabb_pipelines.custom_classifiers import (
    WaveletTransformClassifier,
)
from coheriqs_contributions.moabb_pipelines.CWT_CNN import CWTCNNClassifier
from coheriqs_contributions.moabb_pipelines.EEGNet import EEGNetClassifier
from coheriqs_contributions.moabb_pipelines.wct_phase_gnn_classifier import (
    WCTPhaseGNNClassifier,
    WCTPhaseGNNV2Classifier,
)
from coheriqs_contributions.moabb_pipelines.wct_evidence_gnn_classifier import (
    WCTEvidenceGNNClassifier,
)
from coheriqs_contributions.moabb_pipelines.xwt_phase_gnn_classifier import (
    XWTPhaseGNNClassifier,
    XWTPhaseGNNV2Classifier,
)


def _make_csp_lda():
    return make_pipeline(
        CSP(n_components=4, log=True, norm_trace=False, reg=None),
        LinearDiscriminantAnalysis(solver="eigen"),
    )


def _make_wct_phase_gnn():
    return WCTPhaseGNNClassifier(
        sampling_rate=250,
        lowest=8.0,
        highest=35.0,
        nfreqs=16,
        cwt_resample_n_time=100,
        coherence_threshold=0.7,
        phase_threshold_deg=30.0,
        window_size=25,
        state_mode="per_node",
        use_mag=True,
        use_ang=False,
        use_raw=True,
        use_state_src=True,
        use_state_dst=False,
        hidden_dim=16,
        message_dim=16,
        epochs=20,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=1e-4,
        grad_clip_norm=0.1,
        normalize_input=True,
        validation_split=0.2,
        validation_group_column=None,
        early_stopping_patience=None,
        device="auto",
        seed=42,
        verbose=2,
    )


def _make_wct_evidence_gnn():
    return WCTEvidenceGNNClassifier(
        sampling_rate=250,
        lowest=8.0,
        highest=35.0,
        nfreqs=16,
        cwt_resample_n_time=200,
        coherence_threshold=0.5,
        phase_threshold_deg=30.0,
        window_size=25,
        use_mag=True,
        use_ang=False,
        use_raw=False,
        use_freq=True,
        use_time=True,
        readout_mode="flatten",
        evidence_norm="none",
        hidden_dim=8,
        message_dim=8,
        epochs=200,
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
        padding_time_dim=False,
        padding_mode="reflect",
        smooth_kernel_size=(None, 3),
        smooth_kernel_sigma=(None, None),
        window_compute_mode="sequential",
        max_windows_per_chunk=None,
        select_message_mlp=None,
        select_message_mlp_gate=None,
        message_mlp_selector_mode="shared_train",
        selector_alpha_val_update_rate=1.0,
        last_batch_min_ratio=0.0,
        optimizer_step_batch_size=None,
        optimizer_step_batch_mode="credit",
        optimizer_step_remainder_policy="flush",
        verbose=3,
    )


def _make_xwt_phase_gnn():
    return XWTPhaseGNNClassifier(
        sampling_rate=250,
        lowest=8.0,
        highest=35.0,
        nfreqs=32,
        cwt_resample_n_time=100,
        time_stride=1,
        theta_dead_deg=30.0,
        state_mode="per_node",
        use_mag=True,
        use_ang=False,
        use_raw=True,
        use_state_src=True,
        use_state_dst=False,
        hidden_dim=16,
        message_dim=16,
        epochs=10,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=1e-4,
        grad_clip_norm=0.1,
        normalize_input=True,
        validation_split=0.2,
        validation_group_column=None,
        early_stopping_patience=None,
        device="auto",
        seed=42,
        verbose=2,
    )


def _make_wavelet_rf():
    return WaveletTransformClassifier(
        lowest=4,
        highest=40,
        nfreqs=50,
        sampling_rate=250,
        verbose=2,
    )


def _make_coherence_cnn():
    return CoherenceCNNClassifier(
        lowest=4,
        highest=40,
        nfreqs=50,
        sampling_rate=250,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        device="cpu",
        use_class_weights=False,
        verbose=2,
    )


def _make_cwt_cnn():
    return CWTCNNClassifier(
        lowest=4,
        highest=40,
        nfreqs=50,
        sampling_rate=250,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        device="cpu",
        use_class_weights=False,
        verbose=2,
    )


def _make_eegnet():
    return EEGNetClassifier(
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        dropout_rate=0.5,
        device="cpu",
        verbose=1,
    )


def _make_wct_phase_gnn_v2():
    return WCTPhaseGNNV2Classifier(
        sampling_rate=250,
        lowest=8.0,
        highest=35.0,
        nfreqs=8,
        cwt_resample_n_time=100,
        coherence_threshold=0.7,
        phase_threshold_deg=30.0,
        window_size=25,
        message_dim=3,
        hidden_state_dim=8,
        encoder_dim=3,
        use_encoder_batch_norm=True,
        encoder_dropout=0.5,
        use_local_residual=True,
        use_prev_state_mean=True,
        gru_input_dropout=0.35,
        readout_dropout=0.2,
        use_raw_in_message=False,
        epochs=50,
        batch_size=16,
        learning_rate=1e-3,
        weight_decay=2e-4,
        grad_clip_norm=0.1,
        normalize_input=True,
        validation_split=0.2,
        validation_group_column=None,
        early_stopping_patience=None,
        device="auto",
        seed=42,
        verbose=2,
    )


def _make_xwt_phase_gnn_v2():
    return XWTPhaseGNNV2Classifier(
        sampling_rate=250,
        lowest=8.0,
        highest=35.0,
        nfreqs=32,
        cwt_resample_n_time=100,
        time_stride=1,
        theta_dead_deg=25.0,
        message_dim=3,
        hidden_state_dim=16,
        encoder_dim=3,
        use_encoder_batch_norm=True,
        encoder_dropout=0.4,
        use_local_residual=False,
        use_prev_state_mean=True,
        gru_input_dropout=0.35,
        readout_dropout=0.25,
        use_raw_in_message=True,
        epochs=50,
        batch_size=1,
        learning_rate=3e-3,
        weight_decay=2e-4,
        grad_clip_norm=0.1,
        normalize_input=True,
        validation_split=0.2,
        validation_group_column=None,
        early_stopping_patience=None,
        device="auto",
        seed=42,
        verbose=2,
    )


PIPELINE_BUILDERS = {
    "Coherence-CNN": _make_coherence_cnn,
    "CSP+LDA": _make_csp_lda,
    "CWT-CNN": _make_cwt_cnn,
    "EEGNet": _make_eegnet,
    "WCT-Evidence-GNN": _make_wct_evidence_gnn,
    "WCT-Phase-GNN": _make_wct_phase_gnn,
    "WCT-Phase-GNN-V2": _make_wct_phase_gnn_v2,
    "Wavelet-RF": _make_wavelet_rf,
    "XWT-Phase-GNN": _make_xwt_phase_gnn,
    "XWT-Phase-GNN-V2": _make_xwt_phase_gnn_v2,
}
DEFAULT_PIPELINES = ["WCT-Evidence-GNN",]
PIPELINE_PARAM_GRIDS = {
    "CSP+LDA": {
        "csp__n_components": [5, 6, 7],
        "csp__log": [
            True,
        ],
        "lineardiscriminantanalysis__shrinkage": [
            None,
            "auto",
        ],
    },
    "EEGNet": {
        "epochs": [100],
        "batch_size": [32],
        "learning_rate": [0.001],
        "dropout_rate": [0.5],
        "validation_split": [0.0],
        "weight_decay": [0.0],
    },
    "Wavelet-RF": {
        "lowest": [4],
        "highest": [40],
        "nfreqs": [50],
        "sampling_rate": [250],
    },
    "Coherence-CNN": {
        "lowest": [4],
        "highest": [40],
        "nfreqs": [50],
        "sampling_rate": [250],
        "epochs": [100],
        "batch_size": [32],
        "learning_rate": [0.001],
        "device": ["cpu"],
        "use_class_weights": [False],
    },
    "CWT-CNN": {
        "lowest": [4],
        "highest": [40],
        "nfreqs": [50],
        "sampling_rate": [250],
        "epochs": [100],
        "batch_size": [32],
        "learning_rate": [0.001],
        "device": ["cpu"],
        "use_class_weights": [False],
    },
    "WCT-Phase-GNN": {
        "batch_size": [32],
    },
    "WCT-Evidence-GNN": {
        "batch_size": [32],
        "readout_mode": ["flatten"],
        "evidence_norm": ["active_slots"],
        "message_layer_norm": [False],
        "seed": [42],
        # "message_init_seed": [43],
        # "readout_init_seed": [44],
        "select_message_mlp": [
            # None,  # To enable selectable message MLP candidates, replace None with:
            [
                {"init_seed": 101},
                {
                    "init_seed": 103,
                    # "message_dim": 16,
                    # "message_layer_norm": True,
                },
                {"init_seed": 104},

            ],
        ],
        "select_message_mlp_gate": [
            # None,
            {"mode": "gumbel_hard"}
        ],
        "message_mlp_selector_mode": [
            # "shared_train",
            # "separate_train",
            "separate_val",
        ],
        "selector_alpha_val_update_rate": [0.5],

        "last_batch_min_ratio": [0.5],
        "optimizer_step_batch_size": [None],
        "optimizer_step_batch_mode": ["credit"],
        "optimizer_step_remainder_policy": ["carry"],

        "epochs": [180],
        "normalize_input": [True],
        "learning_rate": [1.0e-3],
        "weight_decay": [1.0e-2],
        "noise_augmentation_enabled": [True],
        "noise_apply_prob": [1.0],
        "noise_strength": [0.15],
        "noise_bank_size": [20000],
        "noise_bank_seed": [33],
        "use_raw": [False],
        "use_time": [True],
        "use_freq": [True],
        "use_mag": [False],
        "use_ang": [False],
        "max_windows_per_chunk": [1],
        "window_compute_mode": ["single_pass_continuous"],
        "verbose": [2],
        "device": ["auto"],

    },
    "WCT-Phase-GNN-V2": {
        "batch_size": [32],
    },
    "XWT-Phase-GNN": {
        "batch_size": [32],
    },
    "XWT-Phase-GNN-V2": {
        "batch_size": [32],
    },
}


def _resolve_eval_modes(global_hyperparam_fit_mode):
    mode = str(global_hyperparam_fit_mode).lower()
    if mode == "false":
        return ["cross"]
    if mode == "true":
        return ["global"]
    if mode == "both":
        return ["cross", "global"]
    raise ValueError(
        f"Unsupported global_hyperparam_fit='{global_hyperparam_fit_mode}'. "
        "Expected one of: false, true, both."
    )


def _build_pipeline_runs(
    pipeline_names, inner_group_mode, global_hyperparam_fit_mode
):
    if inner_group_mode == "none":
        inner_groups = [None]
    elif inner_group_mode == "run":
        inner_groups = ["run"]
    elif inner_group_mode == "both":
        inner_groups = [None, "run"]
    else:
        raise ValueError(
            f"Unsupported inner_group_mode='{inner_group_mode}'. "
            "Expected one of: none, run, both."
        )

    eval_modes = _resolve_eval_modes(global_hyperparam_fit_mode)
    deduped_pipelines = list(dict.fromkeys(pipeline_names))
    runs = []
    for name in deduped_pipelines:
        for eval_mode in eval_modes:
            for inner_group in inner_groups:
                label = (
                    f"{name} [eval={eval_mode}] [inner_group={inner_group or 'None'}]"
                )
                runs.append(
                    {
                        "base_name": name,
                        "eval_mode": eval_mode,
                        "inner_group": inner_group,
                        "label": label,
                    }
                )
    return runs


def _prepare_param_grid_for_run(pipelines, run_param_grid):
    """Apply singleton grids directly and keep only multi-combo grids for search."""
    effective_param_grid = {}
    singleton_applied = {}

    for label, estimator in pipelines.items():
        if label not in run_param_grid:
            continue

        grid = run_param_grid[label]
        combos = ParameterGrid(grid)
        n_combos = len(combos)

        if n_combos <= 1:
            # Equivalent to GridSearchCV with one candidate + refit, but without inner CV.
            params = next(iter(combos), {})
            if params:
                estimator.set_params(**params)
            singleton_applied[label] = params
        else:
            effective_param_grid[label] = grid

    return effective_param_grid, singleton_applied


def _print_run_plan(subjects, selected_pipelines, pipeline_runs):
    print("\n=== Run plan ===", flush=True)
    print(f"Subjects: {subjects}", flush=True)
    print(f"Pipelines: {selected_pipelines}", flush=True)
    print(f"Evaluation runs: {len(pipeline_runs)}", flush=True)
    for run_cfg in pipeline_runs:
        print(f"  - {run_cfg['label']}", flush=True)


def _safe_run_id(value):
    """Return a filesystem-safe run identifier without hiding user mistakes."""
    run_id = value or os.environ.get("LOCAL_ORCHESTRATION_EXECUTION_ID")
    if run_id is None:
        run_id = datetime.now().strftime("manual-%Y%m%d-%H%M%S-%f")
    run_id = str(run_id)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError(
            "Run ID must start with an alphanumeric character and contain only "
            "letters, digits, dot, underscore, or hyphen."
        )
    return run_id


def _configured_data_root():
    """Report the configured root; MOABB reports each resolved dataset file."""
    configured = (
        os.environ.get("MNE_DATASETS_BNCI_PATH")
        or get_config("MNE_DATASETS_BNCI_PATH")
        or os.environ.get("MNE_DATA")
        or get_config("MNE_DATA")
    )
    if configured is None:
        configured = Path.home() / "mne_data"
    data_root = Path(configured).expanduser().resolve()
    print(f"[Data] configured MOABB/MNE root: {data_root}", flush=True)
    return data_root


def _print_moabb_result_root():
    configured = (
        os.environ.get("MOABB_RESULTS")
        or get_config("MOABB_RESULTS")
        or os.environ.get("MNE_DATA")
        or get_config("MNE_DATA")
        or str(Path.home() / "mne_data")
    )
    print(f"[Results] MOABB root: {Path(configured).expanduser().resolve()}", flush=True)


def _markdown_table(frame, columns):
    def clean(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(clean(value) for value in row) + " |")
    return lines


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
):
    """Write compact human-readable companions beside MOABB's HDF5 store."""
    hdf5_path = Path(evaluation.results.filepath).resolve()
    artifact_dir = hdf5_path.parent
    scores_path = artifact_dir / f"scores_{group_id}.csv"
    summary_path = artifact_dir / f"summary_{group_id}.md"
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
        "# WCT run summary",
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
    lines.extend(["", "## Outer-CV rows", ""])
    lines.extend(_markdown_table(group_results, outer_columns))
    lines.extend(["", "## Subject/pipeline means", ""])
    lines.extend(_markdown_table(means, ["subject", "pipeline", "mean_score"]))
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
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Artifact] HDF5: {hdf5_path}", flush=True)
    print(f"[Artifact] scores CSV: {scores_path}", flush=True)
    print(f"[Artifact] summary: {summary_path}", flush=True)


def _print_inner_results(inner):
    print("\n=== Inner cv_results_ ===")
    if inner.empty:
        print("No inner cv_results_ collected.")
        return

    summary_cols = [
        c
        for c in [
            "eval_type",
            "dataset",
            "subject",
            "session",
            "pipeline",
            "outer_fold",
            "params",
            "mean_test_score",
            "mean_train_score",
            "rank_test_score",
        ]
        if c in inner.columns
    ]
    print(inner[summary_cols].to_string(index=False))

    split_cols = [
        c
        for c in [
            "subject",
            "session",
            "pipeline",
            "outer_fold",
            "params",
            "mean_test_score",
            "mean_train_score",
        ]
        + [f"split{i}_{t}_score" for t in ["train", "test"] for i in range(3)]
        if c in inner.columns
    ]
    if split_cols != summary_cols:
        print("\n=== Inner CV split scores ===")
        print(inner[split_cols].to_string(index=False))


def main():
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
    args = parser.parse_args()
    run_id = _safe_run_id(args.run_id)
    selected_pipelines = args.pipeline if args.pipeline else DEFAULT_PIPELINES
    pipeline_runs = _build_pipeline_runs(
        pipeline_names=selected_pipelines,
        inner_group_mode=args.inner_group_mode,
        global_hyperparam_fit_mode=args.global_hyperparam_fit,
    )
    _print_run_plan(args.subjects, selected_pipelines, pipeline_runs)
    print(f"Run ID: {run_id}", flush=True)

    moabb.set_log_level("info")

    dataset = BNCI2014_001()
    dataset.subject_list = args.subjects
    data_root = _configured_data_root()
    _print_moabb_result_root()
    paradigm = LeftRightImagery(fmin=8, fmax=35)

    base_eval_kwargs = dict(
        paradigm=paradigm,
        datasets=[dataset],
        overwrite=True,
        n_jobs=1,
        random_state=42,
    )

    grouped_runs = defaultdict(list)
    for run_cfg in pipeline_runs:
        grouped_runs[(run_cfg["eval_mode"], run_cfg["inner_group"])].append(run_cfg)

    results_chunks = []
    inner_chunks = []
    for (eval_mode, inner_group), run_cfgs in grouped_runs.items():
        print(
            "\n=== Starting group: "
            f"eval={eval_mode}, inner_group={inner_group or 'None'} ===",
            flush=True,
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
        print(f"Group pipelines: {list(pipelines)}", flush=True)
        run_param_grid = {
            cfg["label"]: deepcopy(PIPELINE_PARAM_GRIDS[cfg["base_name"]])
            for cfg in run_cfgs
        }
        param_grid, singleton_applied = _prepare_param_grid_for_run(
            pipelines, run_param_grid
        )

        if singleton_applied:
            for label, params in singleton_applied.items():
                print(
                    f"[Grid] singleton combo for '{label}' applied directly: {params}",
                    flush=True,
                )

        if not param_grid:
            param_grid = None
        grid_status = (
            "disabled after singleton application" if param_grid is None else "enabled"
        )
        print(
            f"Grid search: {grid_status}",
            flush=True,
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
        print(f"Completed group rows: {len(group_results)}", flush=True)
        _write_group_artifacts(
            evaluation=evaluation,
            group_results=group_results,
            group_id=group_id,
            run_id=run_id,
            subjects=args.subjects,
            eval_mode=eval_mode,
            inner_group=inner_group,
            run_param_grid=run_param_grid,
            singleton_applied=singleton_applied,
            data_root=data_root,
        )
        results_chunks.append(group_results)

        try:
            group_inner = evaluation.get_inner_cv_results()
            if not group_inner.empty:
                inner_chunks.append(group_inner)
        except (AttributeError, TypeError):
            # Inner CV results not available in this moabb version
            pass

    results = pd.concat(results_chunks, ignore_index=True)
    inner = pd.concat(inner_chunks, ignore_index=True) if inner_chunks else pd.DataFrame()

    print("\n=== Outer CV results ===")
    outer_cols = ["subject", "session", "pipeline", "score"]
    if "best_params" in results.columns:
        outer_cols.append("best_params")
    print(results[outer_cols].to_string(index=False))

    print("\n=== Per subject/pipeline mean scores")
    per_subject_pipeline = (
        results.groupby(["subject", "pipeline"], as_index=False)["score"].mean()
        .rename(columns={"score": "mean_score"})
        .sort_values(["subject", "pipeline"])
    )
    print(per_subject_pipeline.to_string(index=False))

    print("\n=== Per pipeline mean scores")
    per_pipeline = (
        results.groupby(["pipeline"], as_index=False)["score"].mean()
        .rename(columns={"score": "mean_score"})
        .sort_values(["pipeline"])
    )
    print(per_pipeline.to_string(index=False))

    if args.show_inner_results:
        _print_inner_results(inner)


if __name__ == "__main__":
    main()
