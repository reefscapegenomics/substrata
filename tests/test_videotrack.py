"""substrata.videotrack: the parts of track-video that need no video, GPU or cloud (stops, quarter turns, the
north-up turn of a photo, outlier positions, stop rows, video names). The module is loaded from its file (it
imports only numpy at the top). The matching itself was checked on SAV03 (see the changelog)."""

# Standard Library
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

# Third-Party Libraries
import numpy as np

_FILE = Path(__file__).resolve().parents[1] / "src" / "substrata" / "videotrack.py"
_spec = importlib.util.spec_from_file_location("substrata._videotrack_under_test", _FILE)  # (.sidecars resolves)
vt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vt)


class TestStops(unittest.TestCase):
    def test_runs_of_stillness(self):
        t = np.arange(60.0)
        m = np.full(60, 5.0)
        m[5:20] = 0.2  # 14 s still
        m[30:36] = 0.2  # too short
        m[40:48] = 0.2
        m[50:58] = 0.3  # 40-57 with a 2 s move between: one stop
        stops = vt.find_stops(t, m)
        self.assertEqual(stops, [(5.0, 19.0), (40.0, 57.0)])

    def test_nan_motion_is_not_still(self):
        self.assertEqual(vt.find_stops([0, 1, 2], [np.nan, np.nan, np.nan]), [])


class TestTurns(unittest.TestCase):
    def test_unturn_inverts_rot90(self):
        h, w = 6, 10
        img = np.arange(h * w).reshape(h, w)
        px = np.array([[0, 0], [9, 0], [3, 5], [7, 2]])
        for k in range(4):
            r = np.rot90(img, k)
            # the pixel of the turned image that holds each original pixel's value
            for x, y in px:
                yy, xx = np.argwhere(r == img[y, x])[0]
                back = vt.unturn(np.array([[xx, yy]]), k, w, h)[0]
                self.assertEqual(tuple(back), (x, y), f"k={k}")

    def test_north_turn(self):
        """A camera looking straight down, its image turned by a, gets the quarter turn that brings world +y up."""

        class Cam:
            def __init__(self, a_deg):
                self.sensor = type("S", (), {"width": 400, "height": 300})()
                self.a = np.radians(a_deg)

            def pixel_to_ray(self, x, y):
                # image x -> world (cos a, sin a), image y (down) -> world (sin a, -cos a): image up is world
                # (-sin a, cos a), i.e. +y when a = 0
                u, v = (x - 200) / 1000, (y - 150) / 1000
                c, s = np.cos(self.a), np.sin(self.a)
                return np.array([u * c + v * s, u * s - v * c, -1.0])

        # a = 0: up is +y (0 turns); a = 90: up is -x, right is +y: one turn (rot90 brings right to the top)
        self.assertEqual(vt.north_turn(Cam(0)), 0)
        self.assertEqual(vt.north_turn(Cam(90)), 1)
        self.assertEqual(vt.north_turn(Cam(180)), 2)
        self.assertEqual(vt.north_turn(Cam(-90)), 3)
        self.assertEqual(vt.north_turn(Cam(30)), 0)


class TestRows(unittest.TestCase):
    def test_despike(self):
        rows = [{"t_s": float(t), "how": "track", "x": 1.0 + 0.01 * t, "y": 2.0, "z": -40.0} for t in range(10)]
        rows[5].update(x=6.0)  # 5 m off its neighbours 1 s away
        rows.insert(6, {"t_s": 5.5, "how": "still", "x": 6.0, "y": 2.0, "z": -40.0})  # copied it
        out = vt.despike(rows)
        self.assertEqual(out[5]["how"], "outlier")
        self.assertNotIn("x", out[5])
        self.assertEqual(out[6]["how"], "lost")
        self.assertEqual(sum("x" in r for r in out), 9)
        self.assertIn("x", rows[5])  # (copies: the input is kept)

    def test_stop_rows(self):
        rows = [{"t_s": 10.0, "how": "seed", "x": 1.0, "y": 1.0, "z": -40.0},
                {"t_s": 11.0, "how": "still", "x": 1.0, "y": 1.0, "z": -40.0},
                {"t_s": 12.0, "how": "track", "x": 1.2, "y": 1.0, "z": -40.0},
                {"t_s": 13.0, "how": "track", "x": 1.4, "y": 1.0, "z": -40.0},
                {"t_s": 40.0, "how": "lost"}]
        st = vt.stop_rows("v", "p", rows, [(10.0, 13.0), (39.0, 52.0)])
        self.assertEqual(st[0]["n_matched"], 3)
        self.assertAlmostEqual(st[0]["x"], 1.2)
        self.assertAlmostEqual(st[0]["spread_m"], 0.2)
        self.assertEqual(st[0]["start"], "0:10.0")
        self.assertEqual(st[1]["x"], "")
        self.assertEqual(vt._timecode(3725.25), "1:02:05.2")


class TestSequences(unittest.TestCase):
    def _setup(self):
        a, b = "/d/GRAV02/GX01.MP4", "/d/GRAV02/GX02.MP4"
        infos = {a: {"duration": 100.0, "creation_time": "2025-09-28T18:00:00Z"},
                 b: {"duration": 100.0, "creation_time": "2025-09-28T18:02:00Z"}}  # b starts 120 s after a
        track = lambda ts, x0: [{"t_s": float(t), "x": x0 + t / 10, "y": 0.0, "z": -40.0, "how": "track"} for t in ts]
        tracks = {(a, "p"): track([10, 20, 80, 90], 0.0), (b, "p"): track([5, 30], 20.0)}

        def stop(v, n, t0, t1, x=""):
            return {"video": os.path.basename(v), "video_path": v, "stop": n, "t_start_s": t0, "t_end_s": t1,
                    "plot": "p" if x != "" else "", "placed": "matched" if x != "" else "", "x": x,
                    "y": 0.0 if x != "" else "", "z": -40.0 if x != "" else ""}

        stops = [stop(a, 1, 12, 25, 1.5), stop(a, 2, 40, 60), stop(b, 1, 10, 20), stop(a, 3, 95, 99)]
        return a, b, infos, tracks, stops

    def test_order_and_estimates(self):
        a, b, infos, tracks, stops = self._setup()
        vt.sequence_stops(stops, tracks, [a, b], infos)
        self.assertEqual([s["seq_stop"] for s in stops], [1, 2, 4, 3])  # by time over both videos
        self.assertTrue(all(s["sequence"] == "GRAV02" and s["plot"] == "p" for s in stops))
        mid = stops[1]  # between frames at 20 s (x 2) and 80 s (x 8), at its middle (50 s): x 5
        self.assertEqual(mid["placed"], "estimated")
        self.assertAlmostEqual(mid["x"], 5.0)
        self.assertAlmostEqual(mid["est_gap_s"], 20.0)
        across = stops[2]  # b's stop at 130-140 s on the clock: between a's 90 s (x 9) and b's 125 s (x 20.5)
        self.assertEqual(across["placed"], "estimated")
        self.assertEqual(stops[3]["placed"], "estimated")  # after a's last frame (90 s) by 5 s, before b's
        self.assertEqual(stops[0]["placed"], "matched")

    def test_one_sided_limit(self):
        a, b, infos, tracks, stops = self._setup()
        tracks = {(a, "p"): tracks[(a, "p")][:1]}  # one frame at 10 s, nothing after
        s = [st for st in stops if st["video_path"] == a and st["stop"] == 2]
        vt.sequence_stops(s, tracks, [a, b], infos)
        self.assertEqual(s[0]["placed"], "estimated")  # 30 s after the frame, nothing after it: within 60 s
        far = [{"video": "GX01.MP4", "video_path": a, "stop": 9, "t_start_s": 80.0, "t_end_s": 95.0, "plot": "",
                "placed": "", "x": "", "y": "", "z": ""}]
        vt.sequence_stops(far, tracks, [a, b], infos)
        self.assertEqual(far[0]["placed"], "")  # 70 s after it: too far

    def test_starts_without_creation_time(self):
        a, b = "/d/c/GX01.MP4", "/d/c/GX02.MP4"
        st = vt.video_starts([b, a], {a: {"duration": 50.0}, b: {"duration": 10.0}})
        self.assertEqual(st[a], 0.0)
        self.assertEqual(st[b], 51.0)


class TestShapes(unittest.TestCase):
    def test_smooth_path_through_points(self):
        P = np.array([[0, 0], [1, 0], [2, 1], [4, 1]], float)
        L = vt.smooth_path(P, per=8)
        for p in P:
            self.assertLess(np.min(np.linalg.norm(L - p, axis=1)), 1e-9)

    def test_circular_mean(self):
        self.assertAlmostEqual(vt.circular_mean_deg([350, 10]) % 360, 0.0, places=6)
        self.assertAlmostEqual(vt.circular_mean_deg([80, 100, None]), 90.0, places=6)
        self.assertIsNone(vt.circular_mean_deg([]))


class TestStopsAnnotations(unittest.TestCase):
    def test_add_keep_edits_never_readd(self):
        import csv
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as d:
            pr = SimpleNamespace(id="plot", path=d, world_transform=np.eye(4))

            def stop(n, x, placed="matched"):
                return {"video": "V", "stop": n, "x": x, "y": 0.0, "z": -40.0, "placed": placed, "sequence": "S",
                        "seq_stop": n, "t_start_s": 0, "t_end_s": 10, "duration_s": 10, "n_matched": 1}

            path = vt.write_stops_annotations(pr, [stop(1, 1.0), stop(2, 2.0), stop(3, "")], d)
            rows = list(csv.DictReader(open(path)))
            self.assertEqual([r["id"] for r in rows], ["plot_0", "plot_1"])
            # the annotator moves and labels point 0 and deletes point 1
            rows[0].update(orig_x="1.25", label="ACR")
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerow(rows[0])
            st = [stop(1, 1.0), stop(2, 2.0), stop(3, 3.0, "estimated")]
            vt.write_stops_annotations(pr, st, d)
            rows = list(csv.DictReader(open(path)))
            self.assertEqual([r["id"] for r in rows], ["plot_0", "plot_2"])  # 1 not added again; 3 new
            self.assertEqual((rows[0]["orig_x"], rows[0]["label"]), ("1.25", "ACR"))
            self.assertEqual((rows[0]["is_extrapolated"], rows[1]["is_extrapolated"]), ("False", "True"))
            self.assertNotIn("placed", rows[0])
            self.assertEqual([s["annotation_id"] for s in st], ["plot_0", "", "plot_2"])
            self.assertTrue(os.listdir(os.path.join(d, "plot.annos", "backup")))
            self.assertIsNone(vt.write_stops_annotations(pr, st, d))  # nothing new: not written


class TestVideos(unittest.TestCase):
    def test_find_and_name(self):
        with tempfile.TemporaryDirectory() as d:
            for sub, name in (("A", "GX01.MP4"), ("B", "GX01.MP4"), ("B", "GX02.mov"), ("B", ".GX03.MP4"),
                              ("B", "notes.txt"), (".hidden", "GX04.MP4")):
                os.makedirs(os.path.join(d, sub), exist_ok=True)
                open(os.path.join(d, sub, name), "w").close()
            vids = vt.find_videos([d])
            self.assertEqual([os.path.relpath(v, d) for v in vids], ["A/GX01.MP4", "B/GX01.MP4", "B/GX02.mov"])
            keys = vt.video_keys(vids)
            self.assertEqual(sorted(keys.values()), ["A_GX01", "B_GX01", "GX02"])


if __name__ == "__main__":
    unittest.main()
