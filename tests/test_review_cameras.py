"""Review fixes in cameras.py (and the light modules around it).

Containers (nested iteration, subsets that keep their cameras' parent), the
in-view convention, EXIF times, time matching, meta JSON edge cases and
the shared helpers (``frames.uniform_scale``, ``fileio``). Modules are loaded
from this checkout into a private ``substrata`` package (open3d-based modules
stubbed), as in ``test_crops.py``.
"""

# Standard Library
import importlib.util
import json
import logging
import os
import stat
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

# Third-Party Libraries
import numpy as np
from PIL import Image

_SUB = Path(__file__).resolve().parents[1] / "src" / "substrata"
_MODULES = (
    "settings", "logging", "geom", "fileio", "frames", "pathrepair", "sidecars", "imagecrops",
    "visualizations", "measurements", "pointclouds", "cameras", "annotations",
    "imagesync", "framepose", "firefish", "initializer", "frameretrieval", "videopose",
)
_KEYS = ["substrata"] + [f"substrata.{name}" for name in _MODULES]


def _load_isolated():
    saved = {key: sys.modules.get(key) for key in _KEYS}
    try:
        pkg = types.ModuleType("substrata")
        pkg.__path__ = [str(_SUB)]
        sys.modules["substrata"] = pkg
        for key in _KEYS[1:]:
            sys.modules.pop(key, None)

        def install(name, mod):
            sys.modules[f"substrata.{name}"] = mod
            setattr(pkg, name, mod)
            return mod

        def load(name):
            spec = importlib.util.spec_from_file_location(
                f"substrata.{name}", _SUB / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            install(name, mod)
            spec.loader.exec_module(mod)
            return mod

        @contextmanager
        def tqdm_joblib(_tqdm_object):
            yield

        load("settings")
        load("geom")
        log_mod = install("logging", types.ModuleType("substrata.logging"))
        log_mod.logger = logging.getLogger("substrata-test")
        log_mod.tqdm_joblib = tqdm_joblib
        for name in ("visualizations", "measurements"):
            install(name, types.ModuleType(f"substrata.{name}"))
        pcd_mod = install("pointclouds", types.ModuleType("substrata.pointclouds"))
        pcd_mod.SimplePointCloud = pcd_mod.PointCloud = object
        ff = install("firefish", types.ModuleType("substrata.firefish"))
        import datetime as _dt
        ff.get_unix_time = lambda s: _dt.datetime.strptime(
            s, "%Y:%m:%d %H:%M:%S").timestamp()
        mods = (load("cameras"), load("annotations"), load("frames"),
                load("imagesync"), load("framepose"), load("initializer"),
                load("pathrepair"), load("frameretrieval"), load("videopose"))
        mine = {key: sys.modules[key] for key in _KEYS if key in sys.modules}
    finally:
        for key, mod in saved.items():
            if mod is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = mod
    return mods, mine


(cm, an, frames, isy, fp, ini, pr, fr, vp), _MINE = _load_isolated()
_SAVED = {}


def setUpModule():
    _SAVED.update({key: sys.modules.get(key) for key in _MINE})
    sys.modules.update(_MINE)


def tearDownModule():
    for key, mod in _SAVED.items():
        if mod is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = mod


class _Sensor:
    """A pinhole 2000 x 1000 sensor, f 1000, centred."""
    k1 = k2 = k3 = k4 = p1 = p2 = p3 = p4 = b1 = b2 = 0.0
    f = fx = fy = 1000.0
    width, height = 2000, 1000
    cx, cy = 1000.0, 500.0
    r2_max = 10.0


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def meta(self, n=3, reference=None, nested=True):
        cams = {}
        for i in range(n):
            T = np.eye(4)
            T[:3, 3] = [i, 0, 0]
            folder = "a.auv" if i % 2 == 0 else "b.closeup"
            e = {"path": f"/x/{folder}/IMG_{i}.jpg", "center": [float(i), 0.0, 0.0],
                 "transform": T.tolist() if nested else T.ravel().tolist()}
            if reference is not None:
                e["reference"] = reference(i)
            cams[str(i)] = e
        path = os.path.join(self.tmp, "m.json")
        with open(path, "w") as fh:
            json.dump({"cameras": cams}, fh)
        return path


class TestContainer(_Tmp):
    """A14 nested iteration, A6 subsets, A13, A12/A35 removals."""

    def test_nested_iteration(self):
        cams = cm.Cameras(self.meta())
        self.assertEqual(len([(a, b) for a in cams for b in cams]), 9)

    def test_subsets_do_not_reparent(self):
        cams = cm.Cameras(self.meta(4))
        groups = {c.cam_id: c.group for c in cams}
        cams.projection_arrays()  # (fills the private stack cache)
        for sub in (cams.subset(2), cams.filter_by_ids(["1"]),
                    cams.subset_by_group(groups["0"])):
            self.assertTrue(len(sub))
            for cam in sub:
                self.assertIs(cam.parent, cams)
            self.assertNotIn("_proj_stacks", sub.__dict__)
        self.assertEqual({c.cam_id: c.group for c in cams}, groups)

    def test_get_original_coords_sets_world_transform(self):
        cams = cm.Cameras(self.meta())
        T = np.diag([2.0, 2.0, 2.0, 1.0])
        cams.reverse_transform(T)
        np.testing.assert_allclose(cams.world_transform, T)
        np.testing.assert_allclose(cams["2"].orig_coords, [1, 0, 0])

    def test_removed_api(self):
        for name in ("reset_transform", "get_average_vector", "filter_by_cams",
                     "set_base_path", "set_filename_prefix",
                     "get_camera_by_filename_partial_match",
                     "subset_by_filepath_postfix"):
            self.assertFalse(hasattr(cm.Cameras, name), name)
        self.assertFalse(hasattr(cm, "Video") or hasattr(cm, "Frame"))
        self.assertFalse(hasattr(cm, "_property_reduce"))


class TestMetaJson(_Tmp):
    """A21 null reference z; A22 nested-list poses in camsync; A10 cache."""

    def test_reference_without_z(self):
        cams = cm.Cameras(self.meta(reference=lambda i: [1, 2, None if i == 1 else -5]))
        self.assertIsNone(cams["1"].depth_sensor_m)
        self.assertEqual(cams["0"].depth_sensor_m, -5)

    def test_timematch_offset_with_list_pose(self):
        a, b = cm.Cameras(self.meta()), cm.Cameras(self.meta())
        for cam in list(a) + list(b):
            cam.datetime = "2024:01:01 00:00:0" + cam.cam_id
        a.get_centers_and_transforms_based_on_timematch(b, offset_xyz=[0, 0, 0.1])
        np.testing.assert_allclose(a["2"].coords, [2, 0, 0.1])
        np.testing.assert_allclose(np.asarray(a["2"].camera_transform)[:3, 3], [2, 0, 0.1])
        # without an offset the two cameras do not share arrays
        c = cm.Cameras(self.meta())
        for cam in c:
            cam.datetime = "2024:01:01 00:00:0" + cam.cam_id
        c.get_centers_and_transforms_based_on_timematch(b)
        c["1"].camera_transform[0, 3] = 99
        self.assertEqual(np.asarray(b["1"].camera_transform)[0][3], 1)

    def test_camera_attributes_round_trip(self):
        cams = cm.Cameras(self.meta())
        for cam in cams:
            cam.datetime, cam.camdist = "2024:01:01 00:00:00", 1.5
        out = os.path.join(self.tmp, "attrs.csv")
        cams.save_camera_attributes(out)
        cams["0"].set_filepath("/repaired/a.auv/IMG_0.jpg")
        again = cm.Cameras(self.meta())
        again["0"].set_filepath("/repaired/a.auv/IMG_0.jpg")
        again.load_camera_attributes(out)
        self.assertEqual(again["0"].camdist, 1.5)
        self.assertEqual(again["0"].filepath, "/repaired/a.auv/IMG_0.jpg")

    def test_saved_json_keeps_mode(self):
        path = self.meta()
        os.chmod(path, 0o664)
        cams = cm.Cameras(path)
        cams.save(path)
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o664)


class TestProjection(_Tmp):
    """A16 in-view convention; A11 Camera.get_image_matches; A36 stack cache."""

    def cam(self):
        c = cm.Camera(None, "c", np.eye(4).tolist(), [0.0, 0, 0], "/x/a/IMG.jpg")
        c.sensor = _Sensor()
        return c

    def test_right_border(self):
        c = self.cam()
        # x = 1999.7: in view, rounds to 2000 -> clamped to the last pixel 1999
        x, *_ = c.get_pixel_coords(np.array([0.9997, 0.0, 1.0]))
        self.assertEqual(x, 1999)
        # x = 2000 exactly: out of view (0 <= x < width)
        self.assertIsNone(c.get_pixel_coords(np.array([1.0, 0.0, 1.0]))[0])
        hits = cm.pixel_hits([c], np.array([0.9997, 0.0, 1.0]), use_orig_coords=True)
        self.assertEqual(hits[0][1], 1999)

    def test_camera_image_matches_world_frame(self):
        c = self.cam()
        W = np.eye(4)
        W[:3, 3] = [5, 0, 0]
        c.transform_coords(W)
        ann = an.Annotation([0.1, 0.0, 2.0], id="a")
        ann.transform_coords(W)
        m = c.get_image_matches([ann], use_orig_coords=False)
        self.assertEqual((m[0].x, m[0].y), (1050, 500))
        self.assertEqual(c.get_image_matches([an.Annotation([0, 0, -5.0], id="b")]), [])

    def test_stack_cache_checks_the_whole_list(self):
        cams = [self.cam() for _ in range(200)]
        for i, c in enumerate(cams):
            c.cam_id = i
        other = list(cams)
        other[5] = self.cam()  # (same length, same sampled cameras, one differs)
        other[5].orig_camera_transform = np.diag([1.0, 1, 1, 1])
        other[5].orig_camera_transform[:3, 3] = [100, 0, 0]
        a = cm._stack_for(cams)
        b = cm._stack_for(other)
        self.assertIsNot(a, b)
        self.assertEqual(b.centres.shape, (200, 3))

    def test_image_match_filepath_follows_camera(self):
        c = self.cam()
        m = cm.ImageMatch(c, 1, 1, 1.0, 1.0)
        c.set_filepath("/new/a/IMG.jpg")
        self.assertEqual(m.filepath, "/new/a/IMG.jpg")


class TestTimes(_Tmp):
    """A18 same-second photos; A19 PIL fallback reads the Exif IFD; A20."""

    def photo(self, name, original, modified="2030:01:01 00:00:00", subsec=None):
        path = os.path.join(self.tmp, name)
        img = Image.new("RGB", (8, 8))
        exif = img.getexif()
        exif[0x0132] = modified
        ifd = exif.get_ifd(0x8769)
        ifd[0x9003] = original
        if subsec is not None:
            ifd[0x9291] = subsec
        img.save(path, exif=exif)
        return path

    def test_datetime_original_and_subsec(self):
        path = self.photo("a.jpg", "2024:05:06 07:08:09", subsec="25")
        cam = cm.Camera(None, "a", np.eye(4), [0, 0, 0], path)
        self.assertEqual(cam.get_datetime_original(), "2024:05:06 07:08:09")
        cams = cm.Cameras()
        cams.append(cam)
        cams.get_datetime_originals()
        self.assertEqual(cam.datetime_subsec, 0.25)

    def test_pil_fallback_uses_exif_ifd(self):
        path = self.photo("b.jpg", "2024:05:06 07:08:09")
        cam = cm.Camera(None, "b", np.eye(4), [0, 0, 0], path)
        orig = cm.exifread.process_file
        cm.exifread.process_file = lambda *a, **k: {}
        try:
            self.assertEqual(cam.get_datetime_original(), "2024:05:06 07:08:09")
        finally:
            cm.exifread.process_file = orig

    def test_same_second_uses_subsec(self):
        cams = cm.Cameras(self.meta())
        for cam, sub in zip(cams, (0.1, 0.5, 0.9)):
            cam.datetime, cam.datetime_subsec = "2024:01:01 00:00:00", sub
        self.assertEqual(cams.get_camera_by_datetime("2024:01:01 00:00:00", 0.55).cam_id, "1")
        with self.assertLogs("substrata.cameras", level="WARNING"):
            self.assertEqual(cams.get_camera_by_datetime("2024:01:01 00:00:00").cam_id, "0")

    def test_time_delta_uses_min_max(self):
        cams = cm.Cameras(self.meta())
        for cam, t in zip(cams, ("00:00:05", "00:00:01", "00:00:09")):
            cam.datetime = "2024:01:01 " + t
        self.assertEqual(cams.get_time_delta_between_first_and_last_photo(), 8)


class TestErrors(_Tmp):
    """A34: a cams.xml without sensors raises (no sys.exit in the library)."""

    def test_no_sensors_raises(self):
        xml = os.path.join(self.tmp, "c.xml")
        with open(xml, "w") as fh:
            fh.write("<document><chunk><cameras/></chunk></document>")
        cams = cm.Cameras(self.meta())
        with self.assertRaises(ValueError):
            cams.get_cam_sensor_parameters_from_file(xml)

    def test_filepath_replace_with_empty(self):
        cams = cm.Cameras(self.meta())
        cams.set_filepath_replace("/x/", "/")
        self.assertEqual(cams["0"].filepath, os.path.join("/", "a.auv", "IMG_0.jpg"))


class TestUniformScale(unittest.TestCase):
    """C5: one world-scale definition (cbrt |det|) for cameras/imagesync/framepose."""

    def test_similarity_and_reflection(self):
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        W = np.eye(4)
        W[:3, :3] = 2.5 * R
        W[:3, 3] = [3, 4, 5]
        self.assertAlmostEqual(frames.uniform_scale(W), 2.5)
        W[:3, 0] *= -1
        self.assertAlmostEqual(frames.uniform_scale(W), 2.5)

    def test_metres_per_unit(self):
        W = np.diag([0.5, 0.5, 0.5, 1.0])
        self.assertAlmostEqual(frames.metres_per_unit(W, 3.0), 0.5)  # oriented: W
        self.assertAlmostEqual(frames.metres_per_unit(np.eye(4), 3.0), 3.0)
        self.assertAlmostEqual(frames.metres_per_unit(None, 3.0), 3.0)
        self.assertEqual(frames.metres_per_unit(np.eye(4), None), 1.0)
        self.assertIsNone(frames.metres_per_unit(None, None))
        self.assertIsNone(frames.metres_per_unit(np.zeros((4, 4)), None))

    def test_geocentric_chunk_scale(self):
        geom = sys.modules["substrata.geom"]
        chunk = np.eye(4)
        chunk[:3, :3] *= 2.0
        chunk[:3, 3] = geom.geodetic_to_ecef(-13.8, -171.8, 0.0)
        _T, scale, _ = geom.chunk_transform_to_enu(chunk, [0.0, 0.0, 0.0])
        self.assertAlmostEqual(scale, 2.0)
        chunk[:3, 0] *= -1  # a reflection is refused
        with self.assertRaises(ValueError):
            geom.chunk_transform_to_enu(chunk, [0.0, 0.0, 0.0])

    def test_camera_depth_in_m_with_vector_up(self):
        geom = sys.modules["substrata.geom"]
        parent = types.SimpleNamespace(up_vector=geom.Vector([0.0, 0.0, 2.0]),
                                       depth_offset=-1.0)
        cam = cm.Camera(parent=parent, cam_id="0", coords=np.array([1.0, 2.0, 3.0]),
                        path="/x/IMG_0.jpg")
        self.assertAlmostEqual(cam.depth_in_m, 5.0)

    def test_one_unrotate(self):
        self.assertIs(sys.modules["substrata.imagesync"].unrotate, isy.unrotate)
        k, w, h = 1, 60, 40
        self.assertEqual(isy.unrotate([[11, 7]], 0, w, h).tolist(), [[11.0, 7.0]])


class TestInitializerYaml(_Tmp):
    """A31 project folder; A32 save_config_to_yaml patches the file in place."""

    def project(self, path_line):
        folder = os.path.join(self.tmp, "cur_sna_10m_20200101")
        os.makedirs(folder)
        y = os.path.join(folder, "cur_sna_10m_20200101.yaml")
        with open(y, "w") as fh:
            fh.write("# my project\n" + path_line + "id: cur_sna_10m_20200101\n"
                     "cams_xml: cams.xml  # relative\n"
                     "my_note: keep me\n"
                     "scale_factor: 1.0\n")
        return folder, y

    def test_without_path_uses_yaml_folder(self):
        folder, y = self.project("")
        init = ini.ProjectInitializer(yaml=y)
        self.assertEqual(init.cams_xml_filepath, os.path.join(folder, "cams.xml"))

    def test_stale_path_uses_yaml_folder(self):
        folder, y = self.project("path: /no/such/folder\n")
        with self.assertLogs("substrata-test", level="WARNING"):
            init = ini.ProjectInitializer(yaml=y)
        self.assertEqual(init.cams_xml_filepath, os.path.join(folder, "cams.xml"))

    def test_save_keeps_unknown_keys_comments_relative_paths(self):
        folder, y = self.project("")
        os.chmod(y, 0o664)
        init = ini.ProjectInitializer(yaml=y)
        init.scale_factor = 2.0
        init.save_config_to_yaml()
        with open(y) as fh:
            text = fh.read()
        self.assertIn("# my project", text)
        self.assertIn("cams_xml: cams.xml  # relative", text)
        self.assertIn("my_note: keep me", text)
        self.assertNotIn("path:", text)
        import yaml
        self.assertEqual(yaml.safe_load(text)["scale_factor"], 2.0)
        self.assertEqual(stat.S_IMODE(os.stat(y).st_mode), 0o664)
        again = ini.ProjectInitializer(yaml=y)
        self.assertEqual(again.cams_xml_filepath, os.path.join(folder, "cams.xml"))

    def _with_colour_fit(self):
        folder, y = self.project("")
        with open(y, "a") as fh:
            fh.write("color_correction:\n"
                     "  matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]\n"
                     "  offset: [0.0, 0.0, 0.0]\n"
                     "  fit_rmse: 3.2\n"
                     "  fit_patches: 22/24\n")
        return y

    def test_save_keeps_colour_fit_quality(self):
        import yaml
        y = self._with_colour_fit()
        init = ini.ProjectInitializer(yaml=y)
        init.scale_factor = 2.0
        init.save_config_to_yaml()
        with open(y) as fh:
            cc = yaml.safe_load(fh)["color_correction"]
        self.assertEqual(cc["fit_rmse"], 3.2)
        self.assertEqual(cc["fit_patches"], "22/24")

    def test_new_colour_fit_drops_stale_quality(self):
        import yaml
        y = self._with_colour_fit()
        init = ini.ProjectInitializer(yaml=y)
        init.color_correction["offset"] = np.array([1.0, 2.0, 3.0])
        init.save_config_to_yaml()
        with open(y) as fh:
            cc = yaml.safe_load(fh)["color_correction"]
        self.assertEqual(cc["offset"], [1.0, 2.0, 3.0])
        self.assertNotIn("fit_rmse", cc)

    def test_metres_per_unit_propagated(self):
        folder, y = self.project("")
        init = ini.ProjectInitializer(yaml=y)
        init.scale_factor = 0.25  # unoriented project scaled by its scale bars
        init.cams = types.SimpleNamespace()
        init.annotations = types.SimpleNamespace()
        init.propagate_metres_per_unit()
        self.assertEqual(init.cams.metres_per_unit, 0.25)
        self.assertEqual(init.annotations.metres_per_unit, 0.25)
        init.world_transform = np.diag([2.0, 2.0, 2.0, 1.0])
        init.propagate_metres_per_unit()
        self.assertAlmostEqual(init.cams.metres_per_unit, 2.0)


class TestPathRepairRule3(_Tmp):
    """A36: a single file of the name in another folder is asked about."""

    def test_other_folder_is_asked(self):
        found = os.path.join(self.tmp, "dive2", "GOPR0001.JPG")
        index = {"gopr0001.jpg": [found]}
        asked = []

        def on_duplicate(cam_id, stored, candidates):
            asked.append(candidates)
            return ("skip", None)

        plan = pr.resolve_image_paths({"1": "/old/dive1/GOPR0001.JPG"}, index,
                                      on_duplicate=on_duplicate)
        self.assertEqual(asked, [[found]])
        self.assertIn("1", plan.skipped)
        same = pr.resolve_image_paths({"1": "/old/dive2/GOPR0001.JPG"}, index,
                                      on_duplicate=on_duplicate)
        self.assertEqual(len(asked), 1)
        self.assertNotIn("1", same.skipped)


_CAL = {k: 0.0 for k in ("cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2", "p3", "p4")}


class TestLightModules(_Tmp):
    """A16 keypoints to sensor pixels; A33 NaN rays; frame calibration size;
    videopose.save_json with a bare file name."""

    def test_to_sensor_px(self):
        np.testing.assert_allclose(isy.to_sensor_px([[0, 0], [9, 4]], 4.0, (10, 20)),
                                   [[12, 22], [48, 38]])

    def test_pose_from_points_skips_unreachable_pixels(self):
        s = cm.Sensor(0, "s", {"width": 2000, "height": 1000},
                      dict(_CAL, f=1000.0))
        rng = np.random.default_rng(1)
        obj = np.c_[rng.uniform(-1, 1, (40, 2)), rng.uniform(4, 6, 40)]
        img = np.c_[1000 + 1000 * obj[:, 0] / obj[:, 2], 500 + 1000 * obj[:, 1] / obj[:, 2]]
        obj[3] = np.nan
        got = isy.pose_from_points(obj, img, s)
        self.assertIsNotNone(got)
        np.testing.assert_allclose(got[1], [0, 0, 0], atol=1e-3)

    def test_borrowed_calibration_size_checked(self):
        cams = cm.Cameras(self.meta(2))
        for cam in cams:
            cam.sensor_id = 5
        cams.sensors[5] = cm.Sensor(5, "gopro", {"width": 3840, "height": 2160}, dict(_CAL, f=1000.0))
        other = cm.Sensor(5, "gopro", {"width": 4000, "height": 3000}, dict(_CAL, f=1000.0))
        with self.assertRaises(ValueError):
            fr.give_sensor(cams, other)

    def test_save_json_bare_name(self):
        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            vp.save_json("x.json", {"a": 1})
        finally:
            os.chdir(cwd)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "x.json")))


class TestViewRanking(unittest.TestCase):
    """View relevance in consistent units."""

    def test_relevance_independent_of_units(self):
        dist_sq = np.array([1.0, 4.0, 2.0, 1.2])
        x = np.array([1000.0, 1500.0, 1010.0, 1900.0])
        y = np.full(4, 750.0)
        metres = cm._relevance(dist_sq, x, y, 1000.0, 1000.0, 2000.0, 1500.0)
        units = cm._relevance(dist_sq / 0.19 ** 2, x, y, 1000.0, 1000.0, 2000.0, 1500.0)
        np.testing.assert_array_equal(np.argsort(metres), np.argsort(units))
        np.testing.assert_allclose(metres.min() / metres, units.min() / units)  # predict's weights

    def test_relevance_prefers_close_and_central(self):
        near, far = cm._relevance(np.array([1.0, 4.0]), 1000.0, 750.0, 1000.0, 1000.0, 2000.0, 1500.0)
        centre, edge = cm._relevance(1.0, np.array([1000.0, 1900.0]), 750.0, 1000.0, 1000.0, 2000.0, 1500.0)
        self.assertLess(near, far)
        self.assertLess(centre, edge)


if __name__ == "__main__":
    unittest.main()
