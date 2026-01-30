import logging
from copy import deepcopy
from time import perf_counter
from typing import Optional
from uuid import uuid4

import numpy as np
from mne.epochs import BaseEpochs
from sklearn.base import clone
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    StratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from moabb.evaluations.base import BaseEvaluation
from moabb.evaluations.splitters import (
    CrossSessionSplitter,
    CrossSubjectSplitter,
    LearningCurveSplitter,
    WithinSessionSplitter,
)
from moabb.evaluations.utils import (
    _average_scores,
    _create_save_path,
    _create_scorer,
    _ensure_fitted,
    _save_model_cv,
    _score_and_update,
    _update_result_with_scores,
)
from moabb.pipelines.classification import SSVEP_CCA, SSVEP_TRCA, SSVEP_MsetCCA


def _pipeline_requires_epochs(pipeline):
    """Check if any step in the pipeline requires MNE Epochs objects."""
    # Handle non-pipeline classifiers (like DummyClassifier)
    if not hasattr(pipeline, "steps"):
        return isinstance(pipeline, (SSVEP_CCA, SSVEP_TRCA, SSVEP_MsetCCA))

    for name, step in pipeline.steps:
        if isinstance(step, (SSVEP_CCA, SSVEP_TRCA, SSVEP_MsetCCA)):
            return True
    return False


try:
    from codecarbon import EmissionsTracker

    _carbonfootprint = True
except ImportError:
    _carbonfootprint = False


log = logging.getLogger(__name__)


class WithinSessionEvaluation(BaseEvaluation):
    """Performance evaluation within session (k-fold cross-validation)

    Within-session evaluation uses k-fold cross_validation to determine train
    and test sets on separate session for each subject. You can customize the
    cross-validation strategy.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits/folds for cross-validation.
    cv_class : sklearn cross-validator class, default=StratifiedKFold
        The cross-validation class to use. Examples: StratifiedKFold,
        StratifiedShuffleSplit, TimeSeriesSplit, etc.
    cv_params : dict, optional
        Additional parameters to pass to the cv_class constructor.
        Example: {'test_size': 0.2} for StratifiedShuffleSplit.
    return_fold_results : bool, default=False
        If True, return results for each fold separately instead of averaging.
        Useful for learning curves (with TimeSeriesSplit) or when you need
        per-fold statistics.
    paradigm : Paradigm instance
        The paradigm to use.
    datasets : List of Dataset instance
        The list of dataset to run the evaluation. If none, the list of
        compatible dataset will be retrieved from the paradigm instance.
    random_state: int, RandomState instance, default=None
        If not None, can guarantee same seed for shuffling examples.
    n_jobs: int, default=1
        Number of jobs for fitting of pipeline.
    overwrite: bool, default=False
        If true, overwrite the results.
    error_score: "raise" or numeric, default="raise"
        Value to assign to the score if an error occurs in estimator fitting.
    suffix: str
        Suffix for the results file.
    hdf5_path: str
        Specific path for storing the results and models.
    additional_columns: None
        Adding information to results.
    return_epochs: bool, default=False
        use MNE epoch to train pipelines.
    return_raws: bool, default=False
        use MNE raw to train pipelines.
    mne_labels: bool, default=False
        if returning MNE epoch, use original dataset label if True

    Examples
    --------
    Standard 5-fold cross-validation (default):

    >>> evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=[dataset])

    Using TimeSeriesSplit for learning curves:

    >>> from sklearn.model_selection import TimeSeriesSplit
    >>> evaluation = WithinSessionEvaluation(
    ...     paradigm=paradigm,
    ...     datasets=[dataset],
    ...     cv_class=TimeSeriesSplit,
    ...     n_splits=5,
    ...     return_fold_results=True,
    ... )

    Using StratifiedShuffleSplit with multiple permutations:

    >>> from sklearn.model_selection import StratifiedShuffleSplit
    >>> evaluation = WithinSessionEvaluation(
    ...     paradigm=paradigm,
    ...     datasets=[dataset],
    ...     cv_class=StratifiedShuffleSplit,
    ...     n_splits=10,
    ...     cv_params={'test_size': 0.2},
    ...     return_fold_results=True,
    ... )
    """

    def __init__(
        self,
        n_splits: int = 5,
        cv_class: type = None,
        cv_params: Optional[dict] = None,
        return_fold_results: bool = False,
        **kwargs,
    ):
        # Store values before calling super().__init__ which may overwrite them
        _n_splits = n_splits
        self.cv_class = cv_class if cv_class is not None else StratifiedKFold
        self.cv_params = cv_params if cv_params is not None else {}
        self.return_fold_results = return_fold_results

        # Check if using LearningCurveSplitter
        self._is_learning_curve = self.cv_class is LearningCurveSplitter

        # Determine additional columns
        if self._is_learning_curve:
            # Learning curve mode - always return fold results with data_size and permutation
            self.return_fold_results = True
            add_cols = ["data_size", "permutation"]
            super().__init__(additional_columns=add_cols, **kwargs)
        elif self.return_fold_results:
            add_cols = ["fold"]
            super().__init__(additional_columns=add_cols, **kwargs)
        else:
            super().__init__(**kwargs)

        # Set n_splits after super().__init__ to avoid being overwritten
        self.n_splits = _n_splits

    # flake8: noqa: C901
    def _evaluate(
        self,
        dataset,
        pipelines,
        param_grid,
        process_pipeline,
        postprocess_pipeline,
    ):
        # Progress Bar at subject level
        for subject in tqdm(dataset.subject_list, desc=f"{dataset.code}-WithinSession"):
            # check if we already have result for this subject/pipeline
            # we might need a better granularity, if we query the DB
            run_pipes = self.results.not_yet_computed(
                pipelines, dataset, subject, process_pipeline
            )
            if len(run_pipes) == 0:
                continue

            # get the data
            # Force return_epochs=True if any pipeline requires MNE Epochs objects
            requires_epochs = any(
                _pipeline_requires_epochs(clf) for clf in run_pipes.values()
            )
            return_epochs = True if requires_epochs else self.return_epochs
            # For pipelines requiring epochs, don't pass process_pipeline to ensure it's created
            # with return_epochs=True
            X, y, metadata = self.paradigm.get_data(
                dataset=dataset,
                subjects=[subject],
                return_epochs=return_epochs,
                return_raws=self.return_raws,
                cache_config=self.cache_config,
                postprocess_pipeline=postprocess_pipeline,
                process_pipelines=None if requires_epochs else [process_pipeline],
            )
            # iterate over sessions
            for session in np.unique(metadata.session):
                ix = metadata.session == session

                for name, clf in run_pipes.items():
                    # Create WithinSessionSplitter with configurable cv_class
                    # Determine if we need shuffle based on cv_class
                    shuffle = self.random_state is not None

                    # Prepare cv_params for inner cv_class, filtering out random_state
                    # since it's handled separately
                    cv_params = {
                        k: v for k, v in self.cv_params.items() if k != "random_state"
                    }

                    if self._is_learning_curve:
                        # For LearningCurveSplitter, pass random_state to control the
                        # StratifiedShuffleSplit randomness
                        lc_random_state = self.cv_params.get(
                            "random_state", self.random_state
                        )
                        cv_params["random_state"] = lc_random_state
                        # Don't shuffle subjects/sessions for learning curve
                        # (we want deterministic session order)
                        self.cv = WithinSessionSplitter(
                            n_folds=self.n_splits,
                            shuffle=False,
                            cv_class=self.cv_class,
                            **cv_params,
                        )
                    else:
                        self.cv = WithinSessionSplitter(
                            n_folds=self.n_splits,
                            shuffle=shuffle,
                            random_state=self.random_state,
                            cv_class=self.cv_class,
                            **cv_params,
                        )
                    inner_cv = StratifiedKFold(
                        3, shuffle=True, random_state=self.random_state
                    )

                    # Implement Grid Search
                    grid_clf = clone(clf)
                    grid_clf = self._grid_search(
                        param_grid=param_grid,
                        name=name,
                        grid_clf=grid_clf,
                        inner_cv=inner_cv,
                    )

                    le = LabelEncoder()
                    y_cv = le.fit_transform(y[ix])
                    X_ = X[ix]
                    y_ = y[ix] if self.mne_labels else y_cv
                    meta_ = metadata[ix].reset_index(drop=True)
                    acc = list()

                    # Initialize carbon tracking for this session
                    duration = 0.0
                    emissions = np.nan
                    task_name = ""
                    if _carbonfootprint:
                        tracker = EmissionsTracker(**self.codecarbon_config)
                        tracker.start()

                    # Create scorer once before CV loop
                    scorer = _create_scorer(grid_clf, self.paradigm.scoring)

                    for cv_ind, (train, test) in enumerate(self.cv.split(y_, meta_)):
                        cvclf = clone(grid_clf)

                        # For learning curve: check if training data has enough classes
                        not_enough_data = False
                        if self._is_learning_curve and len(np.unique(y_[train])) < 2:
                            log.warning(
                                "For current data size, only one class would remain."
                            )
                            not_enough_data = True

                        if not_enough_data:
                            # Skip training and scoring, return NaN
                            duration = 0.0
                            score = {"score": np.nan}
                            if _carbonfootprint:
                                emissions = np.nan
                                task_name = ""
                        else:
                            # Fit classifier with tracking
                            if _carbonfootprint:
                                task_name = str(uuid4())
                                tracker.start_task(task_name)
                            t_start = perf_counter()
                            cvclf.fit(X_[train], y_[train])
                            duration = perf_counter() - t_start
                            if _carbonfootprint:
                                emissions_data = tracker.stop_task()
                                emissions = (
                                    emissions_data.emissions if emissions_data else np.nan
                                )

                            if self.hdf5_path is not None and self.save_model:
                                model_save_path = _create_save_path(
                                    self.hdf5_path,
                                    dataset.code,
                                    subject,
                                    session,
                                    name,
                                    grid=self.search,
                                    eval_type="WithinSession",
                                )
                                _save_model_cv(
                                    model=cvclf,
                                    save_path=model_save_path,
                                    cv_index=cv_ind,
                                )

                            _ensure_fitted(cvclf)
                            # scorer always returns dict
                            score = scorer(cvclf, X_[test], y_[test])

                        if self.return_fold_results:
                            # Return each fold separately
                            nchan = (
                                X_.info["nchan"]
                                if isinstance(X_, BaseEpochs)
                                else X_.shape[1]
                            )
                            res = {
                                "time": duration,
                                "dataset": dataset,
                                "subject": subject,
                                "session": session,
                                "n_samples": len(train),
                                "n_channels": nchan,
                                "pipeline": name,
                            }

                            # Add learning curve metadata if available
                            if self._is_learning_curve:
                                lc_metadata = self.cv.get_inner_splitter_metadata()
                                if lc_metadata:
                                    res["data_size"] = lc_metadata["data_size"]
                                    res["permutation"] = lc_metadata["permutation"]
                            else:
                                res["fold"] = cv_ind

                            _update_result_with_scores(res, score)
                            if _carbonfootprint:
                                res["carbon_emission"] = (1000 * emissions,)
                                res["codecarbon_task_name"] = task_name
                            yield res
                        else:
                            acc.append(score)

                    if _carbonfootprint:
                        tracker.stop()

                    # For standard evaluation (not return_fold_results), yield averaged result
                    if not self.return_fold_results:
                        nchan = (
                            X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                        )
                        res = {
                            "time": duration / self.n_splits,
                            "dataset": dataset,
                            "subject": subject,
                            "session": session,
                            "n_samples": len(y_cv),
                            "n_channels": nchan,
                            "pipeline": name,
                        }

                        mean_scores = _average_scores(acc)
                        _update_result_with_scores(res, mean_scores)

                        if _carbonfootprint:
                            res["carbon_emission"] = (1000 * emissions,)
                            res["codecarbon_task_name"] = task_name

                        yield res

    def evaluate(
        self, dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline=None
    ):
        yield from self._evaluate(
            dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline
        )

    def is_valid(self, dataset):
        return True


class CrossSessionEvaluation(BaseEvaluation):
    """Cross-session performance evaluation.

    Evaluate performance of the pipeline across sessions but for a single
    subject. Verifies that there is at least two sessions before starting
    the evaluation.

    Parameters
    ----------
    paradigm : Paradigm instance
        The paradigm to use.
    datasets : List of Dataset instance
        The list of dataset to run the evaluation. If none, the list of
        compatible dataset will be retrieved from the paradigm instance.
    random_state: int, RandomState instance, default=None
        If not None, can guarantee same seed for shuffling examples.
    n_jobs: int, default=1
        Number of jobs for fitting of pipeline.
    overwrite: bool, default=False
        If true, overwrite the results.
    error_score: "raise" or numeric, default="raise"
        Value to assign to the score if an error occurs in estimator fitting. If set to
        'raise', the error is raised.
    suffix: str
        Suffix for the results file.
    hdf5_path: str
        Specific path for storing the results and models.
    additional_columns: None
        Adding information to results.
    return_epochs: bool, default=False
        use MNE epoch to train pipelines.
    return_raws: bool, default=False
        use MNE raw to train pipelines.
    mne_labels: bool, default=False
        if returning MNE epoch, use original dataset label if True
    save_model: bool, default=False
        Save model after training, for each fold of cross-validation if needed
    cache_config: bool, default=None
        Configuration for caching of datasets. See :class:`moabb.datasets.base.CacheConfig` for details.

    Notes
    -----
    .. versionadded:: 1.1.0
       Add save_model and cache_config parameters.
    """

    # flake8: noqa: C901
    def evaluate(
        self, dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline=None
    ):
        if not self.is_valid(dataset):
            reason = self._get_incompatibility_reason(dataset)
            raise AssertionError(
                f"Dataset '{dataset.code}' is not appropriate for {self.__class__.__name__}: {reason}"
            )
            # Progressbar at subject level
        for subject in tqdm(dataset.subject_list, desc=f"{dataset.code}-CrossSession"):
            # check if we already have result for this subject/pipeline
            # we might need a better granularity, if we query the DB
            run_pipes = self.results.not_yet_computed(
                pipelines, dataset, subject, process_pipeline
            )
            if len(run_pipes) == 0:
                log.info(f"Subject {subject} already processed")
                continue

            # get the data
            # Force return_epochs=True if any pipeline requires MNE Epochs objects
            requires_epochs = any(
                _pipeline_requires_epochs(clf) for clf in run_pipes.values()
            )
            return_epochs = True if requires_epochs else self.return_epochs
            # For pipelines requiring epochs, don't pass process_pipeline to ensure it's created
            # with return_epochs=True
            X, y, metadata = self.paradigm.get_data(
                dataset=dataset,
                subjects=[subject],
                return_epochs=return_epochs,
                return_raws=self.return_raws,
                cache_config=self.cache_config,
                postprocess_pipeline=postprocess_pipeline,
                process_pipelines=None if requires_epochs else [process_pipeline],
            )
            le = LabelEncoder()
            y = y if self.mne_labels else le.fit_transform(y)
            groups = metadata.session.values

            for name, clf in run_pipes.items():
                # we want to store a results per session
                self.cv = CrossSessionSplitter(random_state=self.random_state)
                inner_cv = StratifiedKFold(
                    3, shuffle=True, random_state=self.random_state
                )

                # Implement Grid Search
                grid_clf = clone(clf)
                grid_clf = self._grid_search(
                    param_grid=param_grid, name=name, grid_clf=grid_clf, inner_cv=inner_cv
                )

                if _carbonfootprint:
                    # Initialise CodeCarbon per cross-validation
                    tracker = EmissionsTracker(**self.codecarbon_config)
                    tracker.start()

                # Create scorer once before CV loop
                scorer = _create_scorer(grid_clf, self.paradigm.scoring)

                for cv_ind, (train, test) in enumerate(self.cv.split(y, metadata)):
                    model_list = []
                    cvclf = clone(grid_clf)

                    # Fit classifier with tracking
                    if _carbonfootprint:
                        task_name = str(uuid4())
                        tracker.start_task(task_name)
                    t_start = perf_counter()
                    cvclf.fit(X[train], y[train])
                    duration = perf_counter() - t_start
                    if _carbonfootprint:
                        emissions_data = tracker.stop_task()
                        emissions = emissions_data.emissions if emissions_data else np.nan

                    if self.hdf5_path is not None and self.save_model:
                        model_save_path = _create_save_path(
                            hdf5_path=self.hdf5_path,
                            code=dataset.code,
                            subject=subject,
                            session="",
                            name=name,
                            grid=self.search,
                            eval_type="CrossSession",
                        )
                        _save_model_cv(
                            model=cvclf,
                            save_path=model_save_path,
                            cv_index=str(cv_ind),
                        )

                    _ensure_fitted(cvclf)
                    model_list.append(cvclf)
                    nchan = X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]

                    res = {
                        "time": duration,
                        "dataset": dataset,
                        "subject": subject,
                        "session": groups[test][0],
                        "n_samples": len(train),
                        "n_channels": nchan,
                        "pipeline": name,
                    }

                    _score_and_update(res, scorer, cvclf, X[test], y[test])

                    if _carbonfootprint:
                        res["carbon_emission"] = (1000 * emissions,)
                        res["codecarbon_task_name"] = task_name

                    yield res

                if _carbonfootprint:
                    tracker.stop()

    def is_valid(self, dataset):
        return dataset.n_sessions > 1

    def _get_incompatibility_reason(self, dataset):
        """Get specific reason for dataset incompatibility."""
        n_sessions = dataset.n_sessions
        if n_sessions <= 1:
            return (
                f"dataset has only {n_sessions} session(s), "
                f"but {self.__class__.__name__} requires at least 2 sessions"
            )
        return "requirements not met"


class CrossSubjectEvaluation(BaseEvaluation):
    """Cross-subject evaluation performance.

    Evaluate performance of the pipeline trained on all subjects but one,
    concatenating sessions.

    Parameters
    ----------
    paradigm : Paradigm instance
        The paradigm to use.
    datasets : List of Dataset instance
        The list of dataset to run the evaluation. If none, the list of
        compatible dataset will be retrieved from the paradigm instance.
    random_state: int, RandomState instance, default=None
        If not None, can guarantee same seed for shuffling examples.
    n_jobs: int, default=1
        Number of jobs for fitting of pipeline.
    overwrite: bool, default=False
        If true, overwrite the results.
    error_score: "raise" or numeric, default="raise"
        Value to assign to the score if an error occurs in estimator fitting. If set to
        'raise', the error is raised.
    suffix: str
        Suffix for the results file.
    hdf5_path: str
        Specific path for storing the results and models.
    additional_columns: None
        Adding information to results.
    return_epochs: bool, default=False
        use MNE epoch to train pipelines.
    return_raws: bool, default=False
        use MNE raw to train pipelines.
    mne_labels: bool, default=False
        if returning MNE epoch, use original dataset label if True
    save_model: bool, default=False
        Save model after training, for each fold of cross-validation if needed
    cache_config: bool, default=None
        Configuration for caching of datasets. See :class:`moabb.datasets.base.CacheConfig` for details.
    n_splits: int, default=None
        Number of splits for cross-validation. If None, the number of splits
        is equal to the number of subjects.

    Notes
    -----
    .. versionadded:: 1.1.0
         Add save_model, cache_config and n_splits parameters
    """

    # flake8: noqa: C901
    def evaluate(
        self, dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline=None
    ):
        if not self.is_valid(dataset):
            reason = self._get_incompatibility_reason(dataset)
            raise AssertionError(
                f"Dataset '{dataset.code}' is not appropriate for {self.__class__.__name__}: {reason}"
            )
        # this is a bit awkward, but we need to check if at least one pipe
        # have to be run before loading the data. If at least one pipeline
        # need to be run, we have to load all the data.
        # we might need a better granularity, if we query the DB
        run_pipes = {}
        for subject in dataset.subject_list:
            run_pipes.update(
                self.results.not_yet_computed(
                    pipelines, dataset, subject, process_pipeline
                )
            )
        if len(run_pipes) == 0:
            return

        # Force return_epochs=True if any pipeline requires MNE Epochs objects
        requires_epochs = any(
            _pipeline_requires_epochs(clf) for clf in run_pipes.values()
        )
        return_epochs = True if requires_epochs else self.return_epochs
        # For pipelines requiring epochs, don't pass process_pipeline to ensure it's created
        # with return_epochs=True
        X, y, metadata = self.paradigm.get_data(
            dataset=dataset,
            return_epochs=return_epochs,
            return_raws=self.return_raws,
            cache_config=self.cache_config,
            postprocess_pipeline=postprocess_pipeline,
            process_pipelines=None if requires_epochs else [process_pipeline],
        )
        le = LabelEncoder()
        y = y if self.mne_labels else le.fit_transform(y)

        # extract metadata
        groups = metadata.subject.values
        sessions = metadata.session.values
        n_subjects = len(dataset.subject_list)

        # perform leave one subject out CV
        if self.n_splits is None:
            cv_class = LeaveOneGroupOut
            cv_kwargs = {}
        else:
            cv_class = GroupKFold
            cv_kwargs = {"n_splits": self.n_splits}
            n_subjects = self.n_splits

        self.cv = CrossSubjectSplitter(
            cv_class=cv_class, random_state=self.random_state, **cv_kwargs
        )

        inner_cv = StratifiedKFold(3, shuffle=True, random_state=self.random_state)

        if _carbonfootprint:
            # Initialise CodeCarbon per cross-validation
            tracker = EmissionsTracker(**self.codecarbon_config)
            tracker.start()

        # Progressbar at subject level
        for cv_ind, (train, test) in enumerate(
            tqdm(
                self.cv.split(y, metadata),
                total=n_subjects,
                desc=f"{dataset.code}-CrossSubject",
            )
        ):
            subject = groups[test[0]]
            # now we can check if this subject has results
            run_pipes = self.results.not_yet_computed(
                pipelines, dataset, subject, process_pipeline
            )
            # iterate over pipelines
            for name, clf in run_pipes.items():
                clf = self._grid_search(
                    param_grid=param_grid, name=name, grid_clf=clf, inner_cv=inner_cv
                )
                cvclf = deepcopy(clf)

                # Fit classifier with tracking
                if _carbonfootprint:
                    task_name = str(uuid4())
                    tracker.start_task(task_name)
                t_start = perf_counter()
                cvclf.fit(X[train], y[train])
                duration = perf_counter() - t_start
                if _carbonfootprint:
                    emissions_data = tracker.stop_task()
                    emissions = emissions_data.emissions if emissions_data else np.nan

                if self.hdf5_path is not None and self.save_model:
                    model_save_path = _create_save_path(
                        hdf5_path=self.hdf5_path,
                        code=dataset.code,
                        subject=subject,
                        session="",
                        name=name,
                        grid=self.search,
                        eval_type="CrossSubject",
                    )
                    _save_model_cv(
                        model=cvclf, save_path=model_save_path, cv_index=str(cv_ind)
                    )

                _ensure_fitted(cvclf)

                # Create scorer once per pipeline
                scorer = _create_scorer(cvclf, self.paradigm.scoring)

                # Evaluate on each session
                for session in np.unique(sessions[test]):
                    ix = sessions[test] == session
                    nchan = X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]

                    res = {
                        "time": duration,
                        "dataset": dataset,
                        "subject": subject,
                        "session": session,
                        "n_samples": len(train),
                        "n_channels": nchan,
                        "pipeline": name,
                    }

                    _score_and_update(res, scorer, cvclf, X[test[ix]], y[test[ix]])

                    if _carbonfootprint:
                        res["carbon_emission"] = (1000 * emissions,)
                        res["codecarbon_task_name"] = task_name

                    yield res

        if _carbonfootprint:
            tracker.stop()

    def is_valid(self, dataset):
        return len(dataset.subject_list) > 1

    def _get_incompatibility_reason(self, dataset):
        """Get specific reason for dataset incompatibility."""
        n_subjects = len(dataset.subject_list)
        if n_subjects <= 1:
            return (
                f"dataset has only {n_subjects} subject(s), "
                f"but {self.__class__.__name__} requires at least 2 subjects"
            )
        return "requirements not met"
