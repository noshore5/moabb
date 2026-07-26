import numpy as np
import pytest

from coheriqs_contributions.moabb_pipelines.common import (
    resolve_train_val_indices,
    validation_split_summary,
)


def _grouped_split(groups, labels, fraction, seed=7):
    return resolve_train_val_indices(
        len(labels),
        np.asarray(labels),
        seed,
        fraction,
        "run",
        np.asarray(groups),
    )


def test_default_validation_split_remains_stratified_random():
    labels = np.repeat([0, 1], 10)

    train_idx, val_idx, chosen = resolve_train_val_indices(
        len(labels), labels, 4, 0.2, None, None
    )

    assert chosen is None
    assert len(val_idx) == 4
    assert set(labels[train_idx]) == {0, 1}
    assert set(labels[val_idx]) == {0, 1}


@pytest.mark.parametrize(
    ("fraction", "expected_groups"),
    [(0.01, 1), (0.25, 2), (0.99, 5)],
)
def test_grouped_fraction_snaps_half_up_and_leaves_a_train_group(
    fraction, expected_groups
):
    groups = np.repeat(np.arange(6), 2)
    labels = np.tile([0, 1], 6)

    _, val_idx, chosen = _grouped_split(groups, labels, fraction)

    assert len(chosen) == expected_groups
    assert set(groups[val_idx]) == set(chosen)
    assert set(groups) - set(chosen)


def test_grouped_selection_is_seeded_and_prefers_class_coverage():
    groups = np.repeat(["a", "b", "c", "d"], 2)
    labels = np.repeat([0, 1, 0, 1], 2)

    first = _grouped_split(groups, labels, 0.5, seed=11)
    second = _grouped_split(groups, labels, 0.5, seed=11)
    train_idx, val_idx, chosen = first

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], second[2])
    assert set(labels[val_idx]) == {0, 1}
    assert set(labels[train_idx]) == {0, 1}
    assert set(groups[train_idx]).isdisjoint(set(groups[val_idx]))
    assert set(groups[val_idx]) == set(chosen)


def test_explicit_group_selection_rejects_every_absent_requested_value():
    groups = np.repeat(["run-1", "run-2"], 2)
    labels = np.tile([0, 1], 2)

    with pytest.raises(ValueError, match=r"absent: \['missing'\]"):
        _grouped_split(groups, labels, ["run-1", "missing"])

    with pytest.raises(ValueError, match="requires validation_group_column"):
        resolve_train_val_indices(len(labels), labels, 7, ["run-1"], None, None)


def test_grouped_summary_reports_requested_and_actual_fractions():
    groups = np.repeat(["run-1", "run-2", "run-3"], [1, 2, 3])
    labels = np.array([0, 0, 1, 0, 1, 1])
    _, val_idx, chosen = _grouped_split(groups, labels, 0.5, seed=3)

    summary = validation_split_summary(
        n_samples=len(labels),
        validation_split=0.5,
        validation_group_column="run",
        validation_groups=groups,
        chosen_groups=chosen,
        val_idx=val_idx,
        seed=3,
    )

    assert summary == {
        "group_column": "run",
        "requested_fraction": 0.5,
        "actual_group_fraction": pytest.approx(len(chosen) / 3),
        "actual_sample_fraction": pytest.approx(len(val_idx) / len(labels)),
        "total_groups": 3,
        "selected_values": chosen.tolist(),
        "seed": 3,
    }
