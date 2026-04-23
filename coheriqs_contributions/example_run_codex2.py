import argparse
from collections import defaultdict
from copy import deepcopy

from mne.decoding import CSP
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

import moabb
from moabb.datasets import BNCI2014_001
from moabb.evaluations import CrossSessionEvaluation, GlobalFutureSessionEvaluation
from moabb.paradigms import LeftRightImagery
try:
    from coheriqs_contributions.moabb_pipelines.EEGNet import EEGNetClassifier
    from coheriqs_contributions.moabb_pipelines.xwt_phase_gnn_classifier import (
        XWTPhaseGNNClassifier,
        XWTPhaseGNNV2Classifier,
    )
except ModuleNotFoundError:
    # Support direct script execution from my_contributions/ directory.
    from moabb_pipelines.EEGNet import EEGNetClassifier
    from moabb_pipelines.xwt_phase_gnn_classifier import (
        XWTPhaseGNNClassifier,
        XWTPhaseGNNV2Classifier,
    )


def _make_csp_lda():
    return make_pipeline(
        CSP(n_components=None, log=True, norm_trace=False, reg=None),
        LinearDiscriminantAnalysis(solver="eigen"),
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
        coi_mode="ignore",
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
        readout_mode="trial",
        verbose=2,
    )


def _make_eegnet():
    return EEGNetClassifier(
        n_channels=22,
        n_timepoints=1001,
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        dropout_rate=0.5,
        device="cpu",
        verbose=1,
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
        coi_mode="ignore",
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
    "CSP+LDA": _make_csp_lda,
    "EEGNet": _make_eegnet,
    "XWT-Phase-GNN": _make_xwt_phase_gnn,
    "XWT-Phase-GNN-V2": _make_xwt_phase_gnn_v2,
}
PIPELINE_PARAM_GRIDS = {
    "CSP+LDA": {
        "csp__n_components": [5, 6, 7],
        "csp__log": [
            # False,
            True,
        ],
        # "csp__norm_trace": [False, True],
        # "csp__reg": [None, 0.001, 0.01, 0.1],
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
    },
    "XWT-Phase-GNN": {
        "time_stride": [1],
    },
    "XWT-Phase-GNN-V2": {
        "time_stride": [1],
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
            "If omitted, defaults to XWT-Phase-GNN and EEGNet."
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
        default="true",
        choices=["false", "true", "both"],
        help=(
            "Evaluation selector: "
            "'false' => CrossSessionEvaluation, "
            "'true' => GlobalFutureSessionEvaluation, "
            "'both' => run both evaluation types."
        ),
    )
    args = parser.parse_args()
    selected_pipelines = (
        args.pipeline if args.pipeline else ["XWT-Phase-GNN", "EEGNet"]
    )
    pipeline_runs = _build_pipeline_runs(
        pipeline_names=selected_pipelines,
        inner_group_mode=args.inner_group_mode,
        global_hyperparam_fit_mode=args.global_hyperparam_fit,
    )

    moabb.set_log_level("info")

    dataset = BNCI2014_001()
    dataset.subject_list = args.subjects
    paradigm = LeftRightImagery(fmin=8, fmax=35, scorer="roc_auc")

    base_eval_kwargs = dict(
        paradigm=paradigm,
        datasets=[dataset],
        overwrite=True,
        n_jobs=1,
        random_state=42,
        save_inner_cv_results=True,
    )

    grouped_runs = defaultdict(list)
    for run_cfg in pipeline_runs:
        grouped_runs[(run_cfg["eval_mode"], run_cfg["inner_group"])].append(run_cfg)

    results_chunks = []
    inner_chunks = []
    for (eval_mode, inner_group), run_cfgs in grouped_runs.items():
        eval_kwargs = dict(base_eval_kwargs)
        if inner_group is not None:
            eval_kwargs.update(
                inner_cv_class=StratifiedGroupKFold,
                inner_cv_kwargs={"n_splits": 3, "shuffle": True, "random_state": 42},
                inner_cv_groups=inner_group,
            )

        pipelines = {cfg["label"]: PIPELINE_BUILDERS[cfg["base_name"]]() for cfg in run_cfgs}
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

        if eval_mode == "global":
            evaluation = GlobalFutureSessionEvaluation(**eval_kwargs)
        elif eval_mode == "cross":
            evaluation = CrossSessionEvaluation(**eval_kwargs)
        else:
            raise ValueError(f"Unsupported eval_mode='{eval_mode}'.")
        group_results = evaluation.process(pipelines, param_grid=param_grid)
        results_chunks.append(group_results)

        group_inner = evaluation.get_inner_cv_results()
        if not group_inner.empty:
            inner_chunks.append(group_inner)

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

    # print("\n=== Inner cv_results_ ===")
    # if inner.empty:
    #     print("No inner cv_results_ collected.")
    #     return
    #
    # cols = [
    #     c
    #     for c in [
    #         "eval_type",
    #         "dataset",
    #         "subject",
    #         "session",
    #         "pipeline",
    #         "outer_fold",
    #         "params",
    #         "mean_test_score",
    #         "mean_train_score",
    #         "rank_test_score",
    #     ]
    #     if c in inner.columns
    # ]
    # print(inner[cols].to_string(index=False))
    #
    # cols = [
    #     c
    #     for c in [
    #         "subject",
    #         "session",
    #         "pipeline",
    #         "outer_fold",
    #         "params",
    #         "mean_test_score",
    #         "mean_train_score",
    #     ]
    #     + [f"split{i}_{t}_score" for t in ["train", "test"] for i in range(3)]
    # ]
    #
    # print(inner[cols].to_string(index=False))


if __name__ == "__main__":
    main()
