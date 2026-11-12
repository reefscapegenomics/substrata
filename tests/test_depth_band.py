"""substrata intercepts --depth-range: one point per (across, depth) cell of a depth band (depth_band_sample)."""

import numpy as np
import pytest

from substrata.measurements import depth_band_sample


def _slope(n=200_000, seed=1):
    """A 40 m wide slope going down along +x (1 m deeper per 2 m), depths 10-60 m; contours along y."""
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(20, 120, n), rng.uniform(0, 40, n)
    return np.c_[x, y, -(x / 2)]


def test_cells_cover_the_band_across_its_width():
    got = depth_band_sample(_slope(), 30, 40, width_step=2, depth_step=1, seed=0)
    assert got["rows"] == 10 and got["columns"] == 20  # (40 m / 2, 10 m / 1)
    assert got["cells_with_cloud"] == 200 and len(got["points"]) == 200
    assert abs(abs(got["across"][1]) - 1) < 0.01  # (across the slope: along y)
    d = -got["points"][:, 2]
    assert d.min() >= 30 and d.max() < 40
    rows = np.floor(d - 30).astype(int)
    assert np.all(rows == got["cells"][:, 1])  # each point in its own cell's depth interval
    assert len({tuple(c) for c in got["cells"]}) == 200  # one per cell
    assert 39 < got["width"] <= 40


def test_order_and_sign_of_the_range_do_not_matter_and_seed_repeats():
    a = depth_band_sample(_slope(), 40, 30, 2, 1, seed=3)
    b = depth_band_sample(_slope(), -30, -40, 2, 1, seed=3)
    assert np.array_equal(a["points"], b["points"])


def test_empty_cells_are_left_out():
    pts = _slope()
    pts = pts[~((pts[:, 1] < 10) & (-pts[:, 2] < 35))]  # a hole: no cloud there
    got = depth_band_sample(pts, 30, 40, 2, 1, seed=0)
    assert got["cells_with_cloud"] == 200 - 5 * 5


def test_no_cloud_in_the_band():
    with pytest.raises(ValueError, match="no cloud between 100 and 110"):
        depth_band_sample(_slope(), 100, 110, 1, 1)


def test_empty_range():
    with pytest.raises(ValueError, match="empty depth band"):
        depth_band_sample(_slope(), 30, -30, 1, 1)


def test_vertical_wall_across_is_along_the_wall():
    rng = np.random.default_rng(2)
    s, z = rng.uniform(0, 60, 100_000), rng.uniform(-80, -20, 100_000)
    off = rng.normal(0, 0.3, len(s))  # a rough wall along the xy diagonal
    pts = np.c_[s / np.sqrt(2) - off / np.sqrt(2), s / np.sqrt(2) + off / np.sqrt(2), z]
    got = depth_band_sample(pts, 30, 40, 2, 1, seed=0)
    assert abs(abs(got["across"] @ np.array([1, 1]) / np.sqrt(2)) - 1) < 0.01
    assert 59 < got["width"] < 61
