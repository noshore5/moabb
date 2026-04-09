import argparse
from collections import defaultdict
from copy import deepcopy

from mne.decoding import CSP
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

import moabb
from moabb.datasets import BNCI2014_001
from moabb.evaluations import CrossSessionEvaluation, GlobalFutureSessionEvaluation
from moabb.paradigms import LeftRightImagery
try:
    from my_contributions.moabb_pipelines.xwt_phase_gnn_classifier import (
        XWTPhaseGNNClassifier,
    )
except ModuleNotFoundError:
    # Support direct script execution from my_contributions/ directory.
    from moabb_pipelines.xwt_phase_gnn_classifier import XWTPhaseGNNClassifier


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
        cwt_resample_n_time=None,
        time_stride=1,
        theta_dead_deg=45.0,
        coi_mode="ignore",
        state_mode="per_node",
        use_mag=True,
        use_ang=True,
        use_raw=True,
        use_state_src=True,
        use_state_dst=True,
        hidden_dim=16,
        message_dim=16,
        epochs=30,
        batch_size=4,
        learning_rate=1e-3,
        device="auto",
        seed=42,
        readout_mode="trial",
        verbose=2,
    )


PIPELINE_BUILDERS = {
    "CSP+LDA": _make_csp_lda,
    "XWT-Phase-GNN": _make_xwt_phase_gnn,
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
    "XWT-Phase-GNN": {
        "hidden_dim": [16],
        "message_dim": [16],
        "theta_dead_deg": [45.0],
        # "epochs": [30],
        "batch_size": [4],
    },
}


def _parse_pipeline_group_specs(parser, raw_specs):
    if not raw_specs:
        raw_specs = ["XWT-Phase-GNN=None"]

    parsed = []
    for spec in raw_specs:
        if "=" in spec:
            name, group = spec.split("=", 1)
        else:
            name, group = spec, "None"
        name = name.strip()
        group = group.strip()

        if not name:
            parser.error("Invalid --inner-group-column entry: pipeline name is empty.")
        if name not in PIPELINE_BUILDERS:
            available = ", ".join(sorted(PIPELINE_BUILDERS))
            parser.error(
                f"Unknown pipeline '{name}' in --inner-group-column. Available: {available}."
            )

        parsed.append((name, None if group in {"", "None", "none"} else group))

    label_counts = defaultdict(int)
    configs = []
    for name, inner_group in parsed:
        base_label = f"{name} [inner_group={inner_group or 'None'}]"
        label_counts[base_label] += 1
        suffix = f" #{label_counts[base_label]}" if label_counts[base_label] > 1 else ""
        configs.append(
            {"base_name": name, "inner_group": inner_group, "label": f"{base_label}{suffix}"}
        )
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--inner-group-column",
        action="append",
        default=[],
        metavar="PIPELINE=GROUP",
        help=(
            "Per-pipeline inner CV grouping. Repeat for multiple runs, e.g. "
            "'--inner-group-column XWT-Phase-GNN=None --inner-group-column CSP+LDA=run'. "
            "If omitted, defaults to 'XWT-Phase-GNN=None'."
        ),
    )
    args = parser.parse_args()
    pipeline_runs = _parse_pipeline_group_specs(parser, args.inner_group_column)

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
        grouped_runs[run_cfg["inner_group"]].append(run_cfg)

    results_chunks = []
    inner_chunks = []
    for inner_group, run_cfgs in grouped_runs.items():
        eval_kwargs = dict(base_eval_kwargs)
        if inner_group is not None:
            eval_kwargs.update(
                inner_cv_class=StratifiedGroupKFold,
                inner_cv_kwargs={"n_splits": 3, "shuffle": True, "random_state": 42},
                inner_cv_groups=inner_group,
            )

        pipelines = {cfg["label"]: PIPELINE_BUILDERS[cfg["base_name"]]() for cfg in run_cfgs}
        param_grid = {
            cfg["label"]: deepcopy(PIPELINE_PARAM_GRIDS[cfg["base_name"]])
            for cfg in run_cfgs
        }

        evaluation = GlobalFutureSessionEvaluation(**eval_kwargs)
        # evaluation = CrossSessionEvaluation(**eval_kwargs)
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
