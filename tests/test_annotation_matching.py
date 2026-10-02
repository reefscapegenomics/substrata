"""Tests for matching random point annotations to manual annotations.

Covers the pure helpers behind ``Annotations.match_annotations`` (candidate
search, decision rule, in-mask test) and ``save_annotation_matches_csv``.
``annotations.py`` imports the heavy conda deps at module load, so these tests
are skipped when they are not installed.
"""

# Standard Library
import importlib
import os
import tempfile
import types
import unittest

import numpy as np

try:  # Heavy deps (open3d, cv2, …) are only present in the conda env.
    a = importlib.import_module("substrata.annotations")
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    a = None


@unittest.skipUnless(a is not None, "requires the open3d/substrata environment")
class TestFindMatchCandidates(unittest.TestCase):
    def test_sorted_within_range_and_capped(self):
        query = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        targets = np.array(
            [[0.5, 0, 0], [0.1, 0, 0], [0.3, 0, 0], [2.0, 0, 0], [0.2, 0, 0]]
        )
        idx, dist = a._find_match_candidates(query, targets, 1.0, 3)
        np.testing.assert_array_equal(idx[0], [1, 4, 2])
        np.testing.assert_allclose(dist[0], [0.1, 0.2, 0.3])
        self.assertEqual(len(idx[1]), 0)  # nothing within 1 m

    def test_chunking_gives_same_result(self):
        rng = np.random.default_rng(0)
        query, targets = rng.random((50, 3)), rng.random((20, 3))
        full = a._find_match_candidates(query, targets, 0.5, 5)
        chunked = a._find_match_candidates(query, targets, 0.5, 5, chunk_size=7)
        for x, y in zip(full[0], chunked[0]):
            np.testing.assert_array_equal(x, y)


@unittest.skipUnless(a is not None, "requires the open3d/substrata environment")
class TestDecideMatch(unittest.TestCase):
    # fwd/rev per candidate: tuple of in-mask results per mask rank (top, 2nd),
    # or None when not tested (no view / out of view).
    def test_edge_of_large_colony_beats_nearer_small_colony(self):
        idx, conf = a._decide_match(
            [0.05, 0.40], [(False, False), (True, True)],
            [(False, False), (True, True)], 0.05,
        )
        self.assertEqual((idx, conf), (1, "high"))

    def test_two_way_beats_other_one_way_votes(self):
        # Random 471: nearest agrees both ways, another only fwd -> still high
        self.assertEqual(
            a._decide_match([0.03, 0.16], [(True,), (True,)], [(True,), (False,)],
                            0.05),
            (0, "high"),
        )

    def test_second_mask_two_way_is_medium(self):
        # Random 2153: candidate's top mask is a tiny structure, its 2nd mask
        # contains the random point
        self.assertEqual(
            a._decide_match([0.1, 0.3], [(False, False), (False, True)],
                            [(False, False), (True, True)], 0.05),
            (1, "medium"),
        )

    def test_second_masks_on_both_sides_do_not_count(self):
        # Random 1261: both 2nd masks span several colonies
        self.assertEqual(
            a._decide_match([0.08, 0.15], [(False, False), (False, True)],
                            [(False, True), (False, True)], 0.05),
            (None, ""),
        )

    def test_top_mask_one_way_beats_second_mask_one_way(self):
        # Random 1045: nearest only inside via its 2nd mask, 2nd candidate via
        # its top mask -> the latter
        self.assertEqual(
            a._decide_match([0.07, 0.3], [(False, True), (True, True)],
                            [(False, False), (False, False)], 0.05),
            (1, "medium"),
        )

    def test_one_way_is_medium(self):
        self.assertEqual(
            a._decide_match([0.1, 0.3], [None, (True, False)],
                            [None, (False, False)], 0.05),
            (1, "medium"),
        )
        # rev-only counts when fwd could not be tested ...
        self.assertEqual(
            a._decide_match([0.1, 0.3], [None, None], [(False,), (True,)], 0.05),
            (1, "medium"),
        )

    def test_rev_only_contradicted_by_fwd_is_ignored(self):
        # ... but not when the candidate's own masks exclude the random point
        # (random 960: candidate hidden behind the colony) -> unmatched
        self.assertEqual(
            a._decide_match([0.1, 0.3], [(False, False), (False, False)],
                            [(False, False), (True, True)], 0.05),
            (None, ""),
        )

    def test_several_agree_takes_nearest_low(self):
        self.assertEqual(
            a._decide_match([0.1, 0.2, 0.3], [(False,), (True,), (True,)],
                            [(False,), (True,), (True,)], 0.05),
            (1, "low"),
        )
        self.assertEqual(
            a._decide_match([0.1, 0.2], [(True,), (True,)], [None, None], 0.05),
            (0, "low"),
        )

    def test_fallback_and_unmatched(self):
        self.assertEqual(
            a._decide_match([0.03, 0.3], [(False,), (False,)], [(False,), (False,)],
                            0.05),
            (0, "low"),
        )
        self.assertEqual(
            a._decide_match([0.2, 0.3], [None, ()], [None, ()], 0.05), (None, "")
        )
        self.assertEqual(a._decide_match([], [], [], 0.05), (None, ""))


class _FakeCam:
    """Projects every point to fixed pixels (for in-mask tests)."""

    def __init__(self, xs, ys, depth, in_view):
        self.xs, self.ys = np.array(xs, float), np.array(ys, float)
        self.depth, self.in_view = np.array(depth, float), np.array(in_view)

    def project_points(self, pts, use_orig_coords=True):
        n = len(np.atleast_2d(pts))
        return (self.xs[:n], self.ys[:n], self.depth[:n], np.arange(n, dtype=float),
                self.in_view[:n])


@unittest.skipUnless(a is not None, "requires the open3d/substrata environment")
class TestPointsInMasks(unittest.TestCase):
    def test_tolerance_scale_ranks_and_out_of_view(self):
        small = np.zeros((10, 10), dtype=np.uint8)
        small[4:6, 4:6] = 1
        big = np.ones((10, 10), dtype=np.uint8)
        masks = [types.SimpleNamespace(vals=small, scale=2.0),
                 types.SimpleNamespace(vals=big, scale=2.0)]  # 20x20 full-res
        cam = _FakeCam([9, 15, 30, 9], [9, 15, 30, 9], [1, 1, 1, -1],
                       [True, True, False, True])
        pts = [np.zeros(3)] * 4
        self.assertEqual(
            a._points_in_masks(cam, pts, masks, 0.0),
            [(True, True), (False, True), None, None],
        )
        # 15 px full-res = 7.5 mask px; 3 mask px tolerance reaches the mask
        self.assertEqual(a._points_in_masks(cam, pts[:2], masks[:1], 6.0),
                         [(True,), (True,)])
        self.assertEqual(a._points_in_masks(cam, pts[:2], [], 6.0), [(), ()])


@unittest.skipUnless(a is not None, "requires the open3d/substrata environment")
class TestBestCameraForPoints(unittest.TestCase):
    def test_prefers_more_leading_points_then_relevance(self):
        cams = [
            _FakeCam([0] * 4, [0] * 4, [1] * 4, [True, True, False, True]),
            _FakeCam([0] * 4, [0] * 4, [1] * 4, [True, True, True, False]),
            _FakeCam([0] * 4, [0] * 4, [1] * 4, [False, True, True, True]),
        ]
        pts = np.zeros((4, 3))
        best = a._best_camera_for_points(cams, pts, prefix=True)
        self.assertIs(best[0], cams[1])  # random + nearest 2 candidates
        self.assertIsNone(a._best_camera_for_points(cams, pts, min_visible=4))
        best = a._best_camera_for_points(cams, pts[:2], min_visible=2)
        self.assertIs(best[0], cams[0])


@unittest.skipUnless(a is not None, "requires the open3d/substrata environment")
class TestSaveAnnotationMatchesCsv(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "rand.csv")
        self.out = os.path.join(self.tmp.name, "rand_matched.csv")
        with open(self.src, "w", newline="") as f:
            f.write("id,orig_x,orig_y,orig_z,label,extra\r\n")
            f.write("0,1,2,3,CSF,a\r\n1,4,5,6,MAF_T,\r\n")
        self.records = [
            {"id": "0", "manual_id": "m_001", "manual_label": "POR",
             "manual_label_conf": "0.9", "match_conf": "high"},
            {"id": "1", "manual_id": "", "manual_label": "", "manual_label_conf": "",
             "match_conf": ""},
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def test_appends_columns_and_keeps_original(self):
        a.save_annotation_matches_csv(self.src, self.records, self.out)
        with open(self.out, newline="") as f:
            text = f.read()
        self.assertEqual(
            text,
            "id,orig_x,orig_y,orig_z,label,extra,manual_id,manual_label,"
            "manual_label_conf,match_conf\r\n0,1,2,3,CSF,a,m_001,POR,0.9,high\r\n"
            "1,4,5,6,MAF_T,,,,,\r\n",
        )

    def test_rerun_overwrites_existing_match_columns(self):
        a.save_annotation_matches_csv(self.src, self.records, self.out)
        self.records[0].update(manual_id="m_002", match_conf="low")
        rerun = os.path.join(self.tmp.name, "rerun.csv")
        a.save_annotation_matches_csv(self.out, self.records, rerun)
        with open(rerun, newline="") as f:
            lines = f.read().splitlines()
        self.assertEqual(lines[0].count("manual_id"), 1)
        self.assertEqual(lines[1], "0,1,2,3,CSF,a,m_002,POR,0.9,low")


if __name__ == "__main__":
    unittest.main()
