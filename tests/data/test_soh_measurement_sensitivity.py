import pytest

from bafuse.labeling.measurement_semantics_audit import (
    _coalesce_duplicate_times,
    _integrate,
)


def test_duplicate_timestamp_policies_preserve_order_and_expose_ambiguity():
    time = [0.0, 1.0, 1.0, 2.0]
    current = [1.0, 2.0, 4.0, 3.0]

    original, zero_intervals, negative_intervals = _integrate(time, current)
    first_time, first_current = _coalesce_duplicate_times(time, current, "first")
    last_time, last_current = _coalesce_duplicate_times(time, current, "last")
    mean_time, mean_current = _coalesce_duplicate_times(time, current, "mean")

    assert (zero_intervals, negative_intervals) == (1, 0)
    assert original == pytest.approx(5.0)
    assert _integrate(first_time, first_current)[0] == pytest.approx(4.0)
    assert _integrate(last_time, last_current)[0] == pytest.approx(6.0)
    assert _integrate(mean_time, mean_current)[0] == pytest.approx(original)


def test_duplicate_timestamp_policy_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown duplicate-time policy"):
        _coalesce_duplicate_times([0.0], [1.0], "sort")
