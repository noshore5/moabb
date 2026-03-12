

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP

from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from moabb.evaluations import CrossSessionEvaluation

import moabb
from my_contributions.moabb_pipelines.EEGNet import EEGNetClassifier

moabb.set_log_level("info")

# --------------------------------------------------
# Dataset and paradigm
# --------------------------------------------------

dataset = BNCI2014_001()
dataset.subject_list = [1, 2]   # only subject 1
paradigm = LeftRightImagery(
    fmin=8,
    fmax=35,
    scorer="accuracy",

)
# paradigm.scorer = {
#     "roc_auc": "roc_auc",
#     "accuracy": "accuracy",
#     "f1": "f1",
# }

datasets = [dataset]

# --------------------------------------------------
# Example pipeline: CSP + LDA
# --------------------------------------------------

pipeline = make_pipeline(
    CSP(n_components=None, log=True, norm_trace=False, reg=None),
    LinearDiscriminantAnalysis()
)
print(pipeline.get_params().keys())
# -----------------------------
# Custom pipeline
# -----------------------------
eegnet = EEGNetClassifier(
    n_channels=22,
    #n_timepoints=None,   # infer inside fit(X, y) if your class supports it
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    dropout_rate=0.5,
    device="cpu",
)

param_grid = {
    "EEGNet": {
        "epochs": [50, 100],
        "batch_size": [16, 32],
        "learning_rate": [1e-3, 5e-4],
        "dropout_rate": [0.25, 0.5],
    },
    "CSP+LDA": {
        "csp__n_components": [5,6],
        # "csp__log": [False, True],
        # "csp__norm_trace": [False, True],
        # "csp__reg": [None, 0.001, 0.01, 0.1],
    }
}

pipelines = {
    "CSP+LDA": pipeline,
    # "EEGNet": eegnet,
}


# --------------------------------------------------
# Cross-session evaluation
# --------------------------------------------------
# Custom evaluation that logs inner searches
class DebugCrossSession(CrossSessionEvaluation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_searches = []

    def _grid_search(self, param_grid, name, grid_clf, inner_cv):
        # Only run grid search if param_grid[name] is provided
        if param_grid and name in param_grid:
            search = GridSearchCV(
                grid_clf,
                param_grid=param_grid[name],
                refit="roc_auc",
                cv=inner_cv,
                n_jobs=self.n_jobs,
                scoring=self.paradigm.scoring,
                return_train_score=True,
                verbose=0,  # adjust if you want live output
            )
            # save the search object for later inspection
            self.all_searches.append((name, search))
            self.search = search  # maintain MOABB API compatibility
            return search
        # otherwise, skip search
        self.search = False
        return grid_clf



evaluation = DebugCrossSession(
    paradigm=paradigm,
    datasets=[dataset],
    overwrite=True,
    n_jobs=1,
    verbose=2,

)

results = evaluation.process(pipelines, param_grid=param_grid)


# --------------------------------------------------
# Inspect results
# --------------------------------------------------

print(results[["subject", "session", "pipeline", "score"]])


# --------------------------------------------------
# Average score across sessions
# --------------------------------------------------

mean_score = results["score"].mean()

print("\nAverage cross-session score:", mean_score)

# Inspect each inner search and print best parameters and scores
for i, (p_name, search) in enumerate(evaluation.all_searches):
    print(f"\n=== Outer fold {i+1} — pipeline {p_name} ===")
    df = pd.DataFrame(search.cv_results_)
    # Best params and mean validation score
    print("Best parameters:", search.best_params_)
    print("Best inner score:", search.best_score_)
    # Show all tested hyperparameters with their mean validation scores
    print(df[["params", "mean_test_score", "rank_test_score"]])