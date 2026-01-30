import numpy as np
import pytest
from sklearn.model_selection import (
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    LeavePGroupsOut,
    LeavePOut,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)
from sklearn.utils import check_random_state

from moabb.datasets.fake import FakeDataset
from moabb.evaluations.splitters import (
    CrossSessionSplitter,
    CrossSubjectSplitter,
    LearningCurveSplitter,
    WithinSessionSplitter,
    WithinSubjectSplitter,
)
from moabb.paradigms.motor_imagery import FakeImageryParadigm


@pytest.fixture
def data():
    dataset = FakeDataset(
        ["left_hand", "right_hand"], n_subjects=5, seed=12, n_sessions=5
    )
    paradigm = FakeImageryParadigm()
    return paradigm.get_data(dataset=dataset)


# Split done for the Within Session evaluation
def eval_split_within_session(shuffle, random_state, data):
    _, y, metadata = data
    rng = check_random_state(random_state) if shuffle else None

    all_index = metadata.index.values
    subjects = metadata["subject"].unique()
    if shuffle:
        rng.shuffle(subjects)

    for i, subject in enumerate(subjects):
        subject_mask = metadata["subject"] == subject

        subject_indices = all_index[subject_mask]
        subject_metadata = metadata[subject_mask]
        sessions = subject_metadata["session"].unique()
        y_subject = y[subject_mask]

        if shuffle:
            rng.shuffle(sessions)

        for session in sessions:
            session_mask = subject_metadata["session"] == session
            indices = subject_indices[session_mask]
            metadata_ = subject_metadata[session_mask]
            y_ = y_subject[session_mask]

            cv = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=rng)

            for idx_train, idx_test in cv.split(metadata_, y_):
                yield indices[idx_train], indices[idx_test]


def eval_split_within_subject(shuffle, random_state, data):
    _, y, metadata = data
    rng = check_random_state(random_state) if shuffle else None

    all_index = metadata.index.values
    subjects = metadata["subject"].unique()
    if shuffle:
        rng.shuffle(subjects)

    for subject in subjects:
        subject_mask = metadata["subject"] == subject
        subject_indices = all_index[subject_mask]
        y_subject = y[subject_mask]

        cv = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=rng)

        for idx_train, idx_test in cv.split(subject_indices, y_subject):
            yield subject_indices[idx_train], subject_indices[idx_test]


def eval_split_cross_subject(shuffle, random_state, data):
    rng = check_random_state(random_state) if shuffle else None

    _, y, metadata = data
    subjects = metadata["subject"].unique()

    if shuffle:
        splitter = GroupShuffleSplit(random_state=rng)
    else:
        splitter = LeaveOneGroupOut()

    for train_subj_idx, test_subj_idx in splitter.split(
        X=np.zeros(len(subjects)), y=None, groups=subjects
    ):
        train_mask = metadata["subject"].isin(subjects[train_subj_idx])
        test_mask = metadata["subject"].isin(subjects[test_subj_idx])

        yield metadata.index[train_mask].values, metadata.index[test_mask].values


def eval_split_cross_session(shuffle, random_state, data):
    _, y, metadata = data

    rng = check_random_state(random_state) if shuffle else None

    subjects = metadata["subject"].unique()

    for subject in subjects:
        subject_mask = metadata["subject"] == subject
        subject_metadata = metadata[subject_mask]
        subject_sessions = subject_metadata["session"].unique()

        if shuffle:
            splitter = GroupShuffleSplit(n_splits=len(subject_sessions), random_state=rng)
        else:
            splitter = LeaveOneGroupOut()

        for train_ix, test_ix in splitter.split(
            X=subject_metadata, y=y[subject_mask], groups=subject_metadata["session"]
        ):
            yield subject_metadata.index[train_ix], subject_metadata.index[test_ix]


@pytest.mark.parametrize("shuffle, random_state", [(True, 0), (True, 42), (False, None)])
def test_within_session_compatibility(shuffle, random_state, data):
    _, y, metadata = data

    split = WithinSessionSplitter(n_folds=5, shuffle=shuffle, random_state=random_state)

    for (idx_train, idx_test), (idx_train_splitter, idx_test_splitter) in zip(
        eval_split_within_session(shuffle=shuffle, random_state=random_state, data=data),
        split.split(y, metadata),
    ):
        # Check if the output is the same as the input
        assert np.array_equal(idx_train, idx_train_splitter)
        assert np.array_equal(idx_test, idx_test_splitter)


@pytest.mark.parametrize("shuffle, random_state", [(True, 0), (True, 42), (False, None)])
def test_within_subject_compatibility(shuffle, random_state, data):
    _, y, metadata = data

    split = WithinSubjectSplitter(n_folds=5, shuffle=shuffle, random_state=random_state)

    for (idx_train, idx_test), (idx_train_splitter, idx_test_splitter) in zip(
        eval_split_within_subject(shuffle=shuffle, random_state=random_state, data=data),
        split.split(y, metadata),
    ):
        # Check if the output is the same as the input
        assert np.array_equal(idx_train, idx_train_splitter)
        assert np.array_equal(idx_test, idx_test_splitter)


def test_is_shuffling(data):
    X, y, metadata = data

    split = WithinSessionSplitter(n_folds=5, shuffle=False)
    split_shuffle = WithinSessionSplitter(n_folds=5, shuffle=True, random_state=3)

    for (train, test), (train_shuffle, test_shuffle) in zip(
        split.split(y, metadata), split_shuffle.split(y, metadata)
    ):
        # Check if the output is the same as the input
        assert not np.array_equal(train, train_shuffle)
        assert not np.array_equal(test, test_shuffle)


@pytest.mark.parametrize(
    "splitter",
    [
        WithinSessionSplitter,
        WithinSubjectSplitter,
        CrossSessionSplitter,
        CrossSubjectSplitter,
    ],
)
def test_custom_inner_cv(
    splitter,
    data,
):
    X, y, metadata = data
    # Use a custom inner cv
    split = splitter(cv_class=TimeSeriesSplit, max_train_size=2)

    for train, test in split.split(y, metadata):
        # Check if the output is the same as the input
        assert len(train) <= 2  # Due to TimeSeriesSplit constraints
        assert len(test) >= 20


def test_custom_shuffle_group(data):
    _, y, metadata = data

    n_splits = 5
    splitter = CrossSubjectSplitter(
        random_state=42,
        cv_class=GroupShuffleSplit,
        n_splits=n_splits,
    )

    splits = list(splitter.split(y, metadata))

    assert len(splits) == n_splits, f"Expected {n_splits} splits, got {len(splits)}"

    for train, test in splits:
        train_subjects = metadata.iloc[train]["subject"].unique()
        test_subjects = metadata.iloc[test]["subject"].unique()

        # Assert no overlap between train and test subjects
        assert len(set(train_subjects) & set(test_subjects)) == 0

    # Check if shuffling produces different splits
    splitter_different_seed = CrossSubjectSplitter(
        cv_class=GroupShuffleSplit,
        n_splits=n_splits,
    )
    splits_different_seed = list(splitter_different_seed.split(y, metadata))

    assert not all(
        np.array_equal(train, train_alt) and np.array_equal(test, test_alt)
        for (train, test), (train_alt, test_alt) in zip(splits, splits_different_seed)
    )


@pytest.mark.parametrize("shuffle, random_state", [(True, 0), (True, 42), (False, None)])
def test_cross_session(shuffle, random_state, data):
    _, y, metadata = data

    params = {"random_state": random_state}
    params["shuffle"] = shuffle
    if shuffle:
        params["cv_class"] = GroupShuffleSplit

    split = CrossSessionSplitter(**params)

    for idx_train_splitter, idx_test_splitter in split.split(y, metadata):
        # Check if the output is the same as the input
        session_train = metadata.iloc[idx_train_splitter]["session"].unique()
        session_test = metadata.iloc[idx_test_splitter]["session"].unique()
        assert np.intersect1d(session_train, session_test).size == 0
        assert (
            np.union1d(session_train, session_test).size
            == metadata["session"].unique().size
        )


@pytest.mark.parametrize("splitter", [CrossSessionSplitter, CrossSubjectSplitter])
@pytest.mark.parametrize("shuffle, random_state", [(False, None), (True, 0), (True, 42)])
def test_cross_compatibility(splitter, shuffle, random_state, data):
    _, y, metadata = data

    if splitter == CrossSessionSplitter:
        function_split = eval_split_cross_session
    else:
        function_split = eval_split_cross_subject

    params = {"random_state": random_state}
    if splitter == CrossSessionSplitter:
        params["shuffle"] = shuffle
    if shuffle:
        params["cv_class"] = GroupShuffleSplit

    split = splitter(**params)

    for (idx_train, idx_test), (idx_train_splitter, idx_test_splitter) in zip(
        function_split(shuffle=shuffle, random_state=random_state, data=data),
        split.split(y, metadata),
    ):
        assert np.array_equal(idx_train, idx_train_splitter)
        assert np.array_equal(idx_test, idx_test_splitter)


def test_cross_session_is_shuffling_and_order(data):
    _, y, metadata = data

    splitter_no_shuffle = CrossSessionSplitter(shuffle=False)
    splitter_shuffle = CrossSessionSplitter(
        shuffle=True, random_state=3, cv_class=GroupShuffleSplit
    )

    splits_no_shuffle = list(splitter_no_shuffle.split(y, metadata))
    splits_shuffle = list(splitter_shuffle.split(y, metadata))

    train_diff = []
    test_diff = []

    # For tracking session order differences
    session_orders_no_shuffle = []
    session_orders_shuffle = []

    for i, ((train_ns, test_ns), (train_s, test_s)) in enumerate(
        zip(splits_no_shuffle, splits_shuffle)
    ):
        print(f"\nFold {i}:")

        # Get session ordering for non-shuffled and shuffled
        train_ns_sessions = metadata.iloc[train_ns]["session"].unique()
        test_ns_sessions = metadata.iloc[test_ns]["session"].unique()
        train_s_sessions = metadata.iloc[train_s]["session"].unique()
        test_s_sessions = metadata.iloc[test_s]["session"].unique()

        print(f"Train no shuffle sessions: {train_ns_sessions}")
        print(f"Test no shuffle sessions : {test_ns_sessions}")
        print(f"Train shuffled sessions  : {train_s_sessions}")
        print(f"Test shuffle sessions    : {test_s_sessions}")

        # Track if indices are the same
        train_diff.append(np.array_equal(train_ns, train_s))
        test_diff.append(np.array_equal(test_ns, test_s))

        # Track session orders
        session_orders_no_shuffle.append(
            (list(train_ns_sessions), list(test_ns_sessions))
        )
        session_orders_shuffle.append((list(train_s_sessions), list(test_s_sessions)))

    # Check if indices are different in at least some folds
    assert not all(train_diff), "All train indices are identical despite shuffle"
    assert not all(test_diff), "All test indices are identical despite shuffle"

    # Check if session ordering is different
    session_order_differences = [
        not (
            np.array_equal(no_shuffle[0], shuffle[0])
            and np.array_equal(no_shuffle[1], shuffle[1])
        )
        for no_shuffle, shuffle in zip(session_orders_no_shuffle, session_orders_shuffle)
    ]

    assert any(session_order_differences), (
        "Session ordering is identical in all folds despite shuffle. "
        "When shuffle=True, we expect some difference in the session ordering."
    )


def test_cross_session_unique_subjects(data):
    _, y, metadata = data

    splitter_shuffle = CrossSessionSplitter(
        shuffle=True, random_state=3, cv_class=GroupShuffleSplit
    )
    splits_shuffle = list(splitter_shuffle.split(y, metadata))

    # Check if session splits are different across subjects
    subject_session_patterns = {}
    for i, (train_idx, test_idx) in enumerate(splits_shuffle):
        subject = metadata.iloc[train_idx]["subject"].iloc[
            0
        ]  # Get the subject for this fold
        if subject not in subject_session_patterns:
            subject_session_patterns[subject] = []

        train_sessions = set(metadata.iloc[train_idx]["session"].unique())
        test_sessions = set(metadata.iloc[test_idx]["session"].unique())
        subject_session_patterns[subject].append((train_sessions, test_sessions))

    # Verify that at least some subjects have different session splitting patterns
    pattern_differences = []
    subjects = list(subject_session_patterns.keys())
    for sub1, sub2 in zip(subjects, subjects[1:]):
        # Compare patterns for each subject pair
        patterns_differ = False
        for (train1, test1), (train2, test2) in zip(
            subject_session_patterns[sub1], subject_session_patterns[sub2]
        ):
            if train1 != train2 or test1 != test2:
                patterns_differ = True
                break
        pattern_differences.append(patterns_differ)

    assert any(
        pattern_differences
    ), "Session splitting patterns are identical across all subjects"


@pytest.mark.parametrize("shuffle, random_state", [(True, 0), (True, 42), (False, None)])
def test_cross_session_unique_sessions(shuffle, random_state, data):
    _, y, metadata = data
    if shuffle:
        split = CrossSessionSplitter(
            shuffle=shuffle, random_state=random_state, cv_class=GroupShuffleSplit
        )
    else:
        split = CrossSessionSplitter(shuffle=shuffle, random_state=random_state)

    splits = list(split.split(y, metadata))

    for i, (train, test) in enumerate(splits):
        train_sessions = metadata.iloc[train]["session"].unique()
        test_sessions = metadata.iloc[test]["session"].unique()
        assert not np.intersect1d(
            train_sessions, test_sessions
        ).size, f"Fold {i} train and test sessions overlap"


@pytest.mark.parametrize("shuffle", [True, False])
def test_cross_session_get_n_splits(data, shuffle):
    _, y, metadata = data
    if shuffle:
        split = CrossSessionSplitter(shuffle=shuffle, cv_class=GroupShuffleSplit)
    else:
        split = CrossSessionSplitter()

    n_splits = split.get_n_splits(metadata)
    assert n_splits == 5 * 5  # 5 subjects, 5 sessions each


def test_cross_subject_get_n_splits(data):
    _, y, metadata = data

    split = CrossSubjectSplitter()

    n_splits = split.get_n_splits(metadata)
    assert n_splits == 5  # 5 subjects


def test_within_subject_get_n_splits(data):
    _, y, metadata = data

    split = WithinSubjectSplitter()

    n_splits = split.get_n_splits(metadata)
    assert n_splits == 5 * 5  # 5 subjects, 5 folds each


@pytest.mark.parametrize("splitter", [CrossSessionSplitter, CrossSubjectSplitter])
def test_if_split_is_not_random(data, splitter):
    _, y, metadata = data

    if splitter == CrossSessionSplitter:
        split = splitter(shuffle=True, random_state=42, cv_class=GroupShuffleSplit)
    else:
        split = splitter(random_state=42, cv_class=GroupShuffleSplit)

    splits = list(split.split(y, metadata))
    splits_2 = list(split.split(y, metadata))

    for (train, test), (train_2, test_2) in zip(splits, splits_2):
        print(f"Train: {train}")
        print(f"Test: {test}")
        print(f"Train 2: {train_2}")
        print(f"Test 2: {test_2}")
        assert np.array_equal(train, train_2)
        assert np.array_equal(test, test_2)


@pytest.mark.parametrize(
    "cv_class",
    [
        LeaveOneGroupOut,
        TimeSeriesSplit,
        # GroupKFold, changed behavior within scikit-learn 1.6
        LeaveOneOut,
        LeavePGroupsOut,
        LeavePOut,
    ],
)
def test_raise_error_on_invalid_cv_class(cv_class):
    with pytest.raises(ValueError):
        CrossSessionSplitter(shuffle=True, cv_class=cv_class)


@pytest.mark.parametrize(
    "cv_class",
    [
        GroupShuffleSplit,
        StratifiedKFold,
        KFold,
        RepeatedKFold,
        RepeatedStratifiedKFold,
        ShuffleSplit,
        StratifiedGroupKFold,
        StratifiedShuffleSplit,
    ],
)
def test_cross_session_splitter_without_error(
    cv_class,
):
    splitter = CrossSessionSplitter(shuffle=True, cv_class=cv_class)
    assert splitter is not None
    assert isinstance(splitter, CrossSessionSplitter)


# ============================================================================
# LearningCurveSplitter tests
# ============================================================================


class TestLearningCurveSplitter:
    """Tests for LearningCurveSplitter."""

    def test_basic_ratio_policy(self):
        """Test basic split with ratio policy."""
        X = np.arange(100)
        y = np.array([0] * 50 + [1] * 50)

        splitter = LearningCurveSplitter(
            data_size={"policy": "ratio", "value": np.array([0.5, 1.0])},
            n_perms=2,
            test_size=0.2,
            random_state=42,
        )

        splits = list(splitter.split(X, y))

        # 2 data sizes * 2 permutations = 4 splits
        assert len(splits) == 4

        # Check that all splits have valid indices
        for train, test in splits:
            assert len(train) > 0
            assert len(test) > 0
            # No overlap between train and test
            assert len(np.intersect1d(train, test)) == 0

    def test_per_class_policy(self):
        """Test split with per_class policy."""
        X = np.arange(100)
        y = np.array([0] * 50 + [1] * 50)

        splitter = LearningCurveSplitter(
            data_size={"policy": "per_class", "value": np.array([5, 10, 20])},
            n_perms=3,
            test_size=0.2,
            random_state=42,
        )

        splits = list(splitter.split(X, y))

        # 3 data sizes * 3 permutations = 9 splits
        assert len(splits) == 9

    def test_variable_n_perms(self):
        """Test with variable number of permutations per data size."""
        X = np.arange(100)
        y = np.array([0] * 50 + [1] * 50)

        # More perms for smaller data, fewer for larger
        n_perms = np.array([5, 3, 2])

        splitter = LearningCurveSplitter(
            data_size={"policy": "ratio", "value": np.array([0.2, 0.5, 1.0])},
            n_perms=n_perms,
            test_size=0.2,
            random_state=42,
        )

        splits = list(splitter.split(X, y))

        # 5 + 3 + 2 = 10 total splits
        assert len(splits) == 10

    def test_get_n_splits(self):
        """Test get_n_splits returns correct count."""
        splitter = LearningCurveSplitter(
            data_size={"policy": "ratio", "value": np.array([0.25, 0.5, 0.75, 1.0])},
            n_perms=5,
            test_size=0.2,
            random_state=42,
        )

        # 4 data sizes * 5 permutations = 20
        assert splitter.get_n_splits() == 20

    def test_metadata_tracking(self):
        """Test that metadata is tracked correctly during iteration."""
        X = np.arange(100)
        y = np.array([0] * 50 + [1] * 50)

        splitter = LearningCurveSplitter(
            data_size={"policy": "ratio", "value": np.array([0.5, 1.0])},
            n_perms=2,
            test_size=0.2,
            random_state=42,
        )

        metadata_list = []
        for train, test in splitter.split(X, y):
            metadata = splitter.get_metadata()
            metadata_list.append(metadata)

        # Check we have 4 metadata entries
        assert len(metadata_list) == 4

        # Check that permutation values are correct (1-indexed)
        perms = [m["permutation"] for m in metadata_list]
        assert 1 in perms and 2 in perms

        # Check that data_size values vary
        data_sizes = set(m["data_size"] for m in metadata_list)
        assert len(data_sizes) == 2  # Two different data sizes

    def test_reproducibility(self):
        """Test that same random_state produces same splits."""
        X = np.arange(100)
        y = np.array([0] * 50 + [1] * 50)

        kwargs = {
            "data_size": {"policy": "ratio", "value": np.array([0.5, 1.0])},
            "n_perms": 3,
            "test_size": 0.2,
            "random_state": 42,
        }

        splitter1 = LearningCurveSplitter(**kwargs)
        splitter2 = LearningCurveSplitter(**kwargs)

        splits1 = list(splitter1.split(X, y))
        splits2 = list(splitter2.split(X, y))

        for (train1, test1), (train2, test2) in zip(splits1, splits2):
            assert np.array_equal(train1, train2)
            assert np.array_equal(test1, test2)

    def test_test_set_fixed_per_permutation(self):
        """Test that test set is fixed within each permutation."""
        X = np.arange(100)
        y = np.array([0] * 50 + [1] * 50)

        splitter = LearningCurveSplitter(
            data_size={"policy": "ratio", "value": np.array([0.25, 0.5, 0.75, 1.0])},
            n_perms=2,
            test_size=0.2,
            random_state=42,
        )

        # Consume splits first to verify we get them all
        splits = list(splitter.split(X, y))
        assert len(splits) > 0

        # Re-run to check test set consistency per permutation
        test_sets_by_perm = {}
        for train, test in splitter.split(X, y):
            meta = splitter.get_metadata()
            perm = meta["permutation"]
            if perm not in test_sets_by_perm:
                test_sets_by_perm[perm] = set(test)
            else:
                # Test set should be the same within the same permutation
                assert test_sets_by_perm[perm] == set(test)

    def test_invalid_policy_raises(self):
        """Test that invalid policy raises ValueError."""
        with pytest.raises(ValueError, match="not valid"):
            LearningCurveSplitter(
                data_size={"policy": "invalid", "value": np.array([0.5])},
                n_perms=1,
            )

    def test_missing_data_size_raises(self):
        """Test that missing data_size raises ValueError."""
        with pytest.raises(ValueError, match="data_size must be provided"):
            LearningCurveSplitter(data_size=None, n_perms=1)

    def test_missing_n_perms_raises(self):
        """Test that missing n_perms raises ValueError."""
        with pytest.raises(ValueError, match="n_perms must be provided"):
            LearningCurveSplitter(
                data_size={"policy": "ratio", "value": np.array([0.5])},
                n_perms=None,
            )

    def test_non_monotonic_data_size_raises(self):
        """Test that non-monotonically increasing data_size raises ValueError."""
        with pytest.raises(ValueError, match="monotonically increasing"):
            LearningCurveSplitter(
                data_size={"policy": "ratio", "value": np.array([1.0, 0.5])},
                n_perms=1,
            )

    def test_non_monotonic_n_perms_raises(self):
        """Test that non-monotonically decreasing n_perms raises ValueError."""
        with pytest.raises(ValueError, match="monotonically decreasing"):
            LearningCurveSplitter(
                data_size={"policy": "ratio", "value": np.array([0.5, 1.0])},
                n_perms=np.array([2, 5]),  # Should decrease, not increase
            )

    def test_ratio_out_of_range_raises(self):
        """Test that ratio values outside [0, 1] raise ValueError."""
        splitter = LearningCurveSplitter(
            data_size={"policy": "ratio", "value": np.array([1.5])},
            n_perms=1,
        )
        X = np.arange(100)
        y = np.array([0] * 50 + [1] * 50)

        with pytest.raises(ValueError, match="range"):
            list(splitter.split(X, y))

    def test_per_class_too_large_raises(self):
        """Test that per_class value larger than smallest class raises ValueError."""
        splitter = LearningCurveSplitter(
            data_size={"policy": "per_class", "value": np.array([100])},
            n_perms=1,
            test_size=0.2,
        )
        X = np.arange(100)
        y = np.array([0] * 50 + [1] * 50)

        with pytest.raises(ValueError, match="too large"):
            list(splitter.split(X, y))


class TestLearningCurveSplitterWithinSession:
    """Test LearningCurveSplitter integration with WithinSessionSplitter."""

    def test_within_session_integration(self, data):
        """Test that LearningCurveSplitter works as cv_class for WithinSessionSplitter."""
        _, y, metadata = data

        splitter = WithinSessionSplitter(
            n_folds=1,  # LearningCurveSplitter handles its own "folds"
            shuffle=True,
            random_state=42,
            cv_class=LearningCurveSplitter,
            data_size={"policy": "ratio", "value": np.array([0.5, 1.0])},
            n_perms=2,
            test_size=0.2,
        )

        splits = list(splitter.split(y, metadata))

        # Should have splits for each session/subject combination
        # 5 subjects * 5 sessions * 4 splits (2 data sizes * 2 perms) = 100
        assert len(splits) == 100

        # Check all splits are valid
        for train, test in splits:
            assert len(train) > 0
            assert len(test) > 0
            assert len(np.intersect1d(train, test)) == 0

    def test_within_session_metadata_access(self, data):
        """Test that metadata can be accessed via WithinSessionSplitter."""
        _, y, metadata = data

        splitter = WithinSessionSplitter(
            n_folds=1,
            shuffle=True,
            random_state=42,
            cv_class=LearningCurveSplitter,
            data_size={"policy": "ratio", "value": np.array([0.5, 1.0])},
            n_perms=2,
            test_size=0.2,
        )

        # Iterate and check metadata access
        for train, test in splitter.split(y, metadata):
            lc_meta = splitter.get_inner_splitter_metadata()
            assert lc_meta is not None
            assert "permutation" in lc_meta
            assert "data_size" in lc_meta
            break  # Just check first split
