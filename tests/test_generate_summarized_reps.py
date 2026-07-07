import numpy as np
import polars as pl
from scipy.sparse import coo_array

from MEDS_tabular_automl.generate_summarized_reps import aggregate_matrix, sparse_aggregate


def test_sparse_aggregate_last_returns_most_recent_value_per_column():
    # Rows are in chronological order. Column 0's largest value (5) occurs in the earliest
    # row, so 'last' must differ from 'max' to prove it isn't just an alias for max.
    matrix = coo_array(([5.0, 1.0, 3.0, 2.0], ([0, 1, 2, 1], [0, 0, 0, 1])), shape=(3, 2))
    assert matrix.toarray().tolist() == [[5.0, 0.0], [1.0, 2.0], [3.0, 0.0]]

    last = sparse_aggregate(matrix, "last")
    max_ = sparse_aggregate(matrix, "max")

    np.testing.assert_array_equal(last, [3.0, 2.0])
    np.testing.assert_array_equal(np.asarray(max_.todense()).ravel(), [5.0, 2.0])


def test_sparse_aggregate_last_single_entry_column():
    # A column with only one populated row should return that row's value as 'last'.
    matrix = coo_array(([7.0], ([1], [0])), shape=(3, 1))
    last = sparse_aggregate(matrix, "last")
    np.testing.assert_array_equal(last, [7.0])


def test_sparse_aggregate_last_empty_column_is_zero():
    matrix = coo_array(([1.0], ([0], [0])), shape=(3, 2))
    last = sparse_aggregate(matrix, "last")
    np.testing.assert_array_equal(last, [1.0, 0.0])


def test_aggregate_matrix_last_across_multiple_windows():
    # Each window should independently pick the most-recent value per feature, and results
    # must land in their own row (regression test for the row-index bug where 'min'/'max'/'last'
    # all collapsed into row 0 because scipy's 1-D coo_array.row is a dummy placeholder).
    matrix = coo_array(
        ([3.0, 1.0, 2.0, 5.0], ([0, 1, 2, 1], [0, 0, 0, 1])),
        shape=(3, 2),
    )
    assert matrix.toarray().tolist() == [[3.0, 0.0], [1.0, 5.0], [2.0, 0.0]]

    windows = pl.DataFrame({"min_index": [0, 0, 1], "max_index": [1, 3, 3]})
    result = aggregate_matrix(windows, matrix, "last", num_features=2).toarray()

    expected = [
        [3.0, 0.0],  # window covering only row 0
        [2.0, 5.0],  # window covering rows 0-2: last value per column is from the highest row index
        [2.0, 5.0],  # window covering rows 1-2
    ]
    np.testing.assert_array_equal(result, expected)

    # 'max' on the same data should differ in column 0 (3.0, from the earliest row), proving
    # 'last' is not just reusing the max implementation.
    max_result = aggregate_matrix(windows, matrix, "max", num_features=2).toarray()
    expected_max = [
        [3.0, 0.0],
        [3.0, 5.0],
        [2.0, 5.0],
    ]
    np.testing.assert_array_equal(max_result, expected_max)


def test_aggregate_matrix_last_preserves_float_values():
    # Guards against a prior bug where aggregated data was unconditionally cast to int64,
    # truncating fractional values (e.g. z-scored numeric values) to zero.
    matrix = coo_array(([0.25, 0.75], ([0, 1], [0, 0])), shape=(2, 1))
    windows = pl.DataFrame({"min_index": [0], "max_index": [2]})
    result = aggregate_matrix(windows, matrix, "last", num_features=1).toarray()
    np.testing.assert_allclose(result, [[0.75]])
