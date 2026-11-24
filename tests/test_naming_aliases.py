"""The names renamed for consistency (2026-11-25) keep working under their old names.

Renamed functions, classes and methods keep the old name as an alias (the same
object); renamed keyword parameters still accept the old keyword, with a
DeprecationWarning (:func:`substrata.compat.renamed_kwargs`). substrata-explore,
the notebooks and scripts use several of the old names.

The package modules import open3d/cv2/…, so the module is skipped when those
heavy deps are not installed.
"""

# Standard Library
import importlib
import inspect
import os
import sys
import tempfile
import unittest
import warnings

import numpy as np

try:  # Heavy deps (open3d, cv2, …) are only present in the conda env.
    import open3d  # noqa: F401
    import cv2  # noqa: F401

    HAVE_DEPS = True
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    HAVE_DEPS = False

substrata = compat = None
_SAVED = {}


def _is_substrata(name):
    return name == "substrata" or name.startswith("substrata.")


def setUpModule():
    """Import the real package: other test modules leave stubs of substrata.* in sys.modules."""
    global substrata, compat
    if not HAVE_DEPS:
        return
    _SAVED.update({k: v for k, v in sys.modules.items() if _is_substrata(k)})
    for k in _SAVED:
        del sys.modules[k]
    substrata = importlib.import_module("substrata")
    compat = importlib.import_module("substrata.compat")


def tearDownModule():
    for k in [k for k in sys.modules if _is_substrata(k)]:
        del sys.modules[k]
    sys.modules.update(_SAVED)

# (module, old name, new name): module-level functions / classes / constants
FUNCTION_ALIASES = [
    ("measurements", "conduct_PCA", "conduct_pca"),
    ("measurements", "conduct_xy_PCA", "conduct_xy_pca"),
    ("measurements", "get_best_fit_plane_PCA", "get_best_fit_plane_pca"),
    ("firefish", "get_time_diff_in_secs", "get_time_diff_s"),
    ("cameras", "_photo_folder_name", "photo_folder_name"),
    ("cameras", "_undistort", "pixels_to_normalized"),
    ("cameras", "_distort", "camera_points_to_pixels"),
    ("cameras", "_lens", "lens_distortion"),
    ("cameras", "_projection_changed", "invalidate_projection_cache"),
    ("cameras", "_int_pixel", "int_pixel"),
    ("cameras", "_sensor_offset_units", "sensor_offset_units"),
    ("cameras", "_count_statuses", "count_crop_statuses"),
    ("cameras", "_update_crop_metadata", "update_crop_metadata"),
    ("annotations", "_read_csv_rows", "read_csv_rows"),
    ("annotations", "_write_csv_rows", "write_csv_rows"),
    ("imagesync", "_gray", "read_gray"),
    ("imagesync", "_resize_width", "resize_to_width"),
    ("imagesync", "_reduce_for", "jpeg_reduction_for"),
    ("pointfeatures", "_image_widths", "image_widths"),
    ("pointfeatures", "_train_offsets", "enabled_training_offsets"),
    ("pointfeatures", "_median_image_width", "median_image_width"),
    ("pointfeatures", "_cache_file", "feature_cache_path"),
    ("segmentation", "_build_label_colors", "build_label_colors"),
    ("segmentation", "_match_points_to_cameras", "match_points_to_cameras"),
    ("classification", "_result_from_probs", "result_from_probs"),
    ("classification", "_write_stats_pdf", "write_classifier_stats_pdf"),
    ("stratified", "_PLY_TYPES", "PLY_TYPES"),
    ("cli", "_classifier_crop_size", "resolve_crop_size"),
]

# (module, class, old method, new method)
METHOD_ALIASES = [
    ("annotations", "Annotations", "get_radius_from_2D_surface_area",
     "get_radius_from_2d_surface_area"),
    ("annotations", "Annotation", "get_radius_from_2D_surface_area",
     "get_radius_from_2d_surface_area"),
    ("annotations", "Scalebars", "calc_scalefactor", "calc_scale_factor"),
    ("annotations", "Scalebar", "calc_scalefactor", "calc_scale_factor"),
    ("annotations", "Annotations", "transform_coords", "apply_transform"),
    ("annotations", "Annotations", "get_original_coords", "reverse_transform"),
    ("cameras", "Cameras", "transform_coords", "apply_transform"),
    ("cameras", "Cameras", "get_original_coords", "reverse_transform"),
    ("pointclouds", "PointCloud", "principal_axis_xy_2D", "principal_axis_xy"),
]

# Old names that substrata.* (the flat namespace) exported before the rename
FLAT_ALIASES = [
    ("conduct_PCA", "conduct_pca"),
    ("conduct_xy_PCA", "conduct_xy_pca"),
    ("get_best_fit_plane_PCA", "get_best_fit_plane_pca"),
]

# (module, qualified name, old keyword, new keyword)
KEYWORD_ALIASES = [
    ("annotationpages", "save_pages_pdf", "output_filepath", "output_path"),
    ("annotations", "save_annotation_matches_csv", "output_filepath", "output_path"),
    ("annotations", "save_joined_manual_annotations", "output_filepath", "output_path"),
    ("annotations", "Annotations.reverse_transform", "transform_matrix", "transform"),
    ("annotations", "Annotations.load_undecimated_neighborhoods", "show_progress",
     "progress"),
    ("cameras", "Cameras.apply_transform", "transform_matrix", "transform"),
    ("cameras", "Cameras.reverse_transform", "transform_matrix", "transform"),
    ("cameras", "Cameras.missing_image_paths", "use_orig", "use_orig_filepath"),
    ("cameras", "Cameras.save_camera_attributes", "output_filepath", "output_path"),
    ("cameras", "Cameras.get_datetime_originals", "offset_secs", "offset_s"),
    ("cameras", "Camera.transform_coords", "transform_matrix", "transform"),
    ("cameras", "Camera.sensor_coords", "orig", "use_orig_coords"),
    ("cameras", "Camera.get_datetime_original", "offset_secs", "offset_s"),
    ("classification", "convert_coralnet_csv", "out_path", "output_path"),
    ("classification", "generate_crop", "out_path", "output_path"),
    ("classification", "write_training_summary_pdf", "pdf_path", "output_path"),
    ("classification", "write_classifier_stats_pdf", "pdf_path", "output_path"),
    ("grouppredict", "write_results", "out_csv", "output_path"),
    ("measurements", "calc_gap_fraction", "output_filename", "output_path"),
    ("measurements", "get_vector_dispersion", "geom", "pcd"),
    ("ortho", "OrthoMap.show", "point_size_metres", "point_size_m"),
    ("ortho", "OrthoMapGroup.show", "point_size_metres", "point_size_m"),
    ("ortho", "OrthoGrid.show", "point_size_metres", "point_size_m"),
    ("pathrepair", "atomic_write_text", "out_path", "output_path"),
    ("pointclouds", "PointCloud.read_point_cloud", "show_progress", "progress"),
    ("pointclouds", "PointCloud.apply_transform", "transform_matrix", "transform"),
    ("pointclouds", "PointCloud.apply_transforms", "transform_matrices", "transforms"),
    ("pointclouds", "SimplePointCloud.transform", "transform_matrix", "transform"),
    ("pointclouds", "transform_normals", "transform_matrix", "transform"),
    ("pointclouds", "stream_extract_neighborhoods_ply", "show_progress", "progress"),
    ("pointclouds", "decimate_ply_file", "show_progress", "progress"),
    ("pointclouds", "repair_ply_for_open3d", "show_progress", "progress"),
    ("pointfeatures", "save_dense_masks", "out_dir", "output_dir"),
    ("stratified", "Occluder.line_of_sight_margin", "min_metres", "min_m"),
    ("videotrack", "extract_keyframes", "out_dir", "output_dir"),
    ("videotrack", "make_proxy", "out", "output_path"),
    ("videotrack", "track_videos", "out_dir", "output_dir"),
    ("videotrack", "read_divers", "out_dir", "output_dir"),
    ("videotrack", "write_outputs", "out_dir", "output_dir"),
]


def _module(name):
    return importlib.import_module(f"substrata.{name}")


def _resolve(module, qualname):
    obj = _module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _deprecations(caught):
    return [w for w in caught if issubclass(w.category, DeprecationWarning)]


@unittest.skipIf(not HAVE_DEPS, "substrata's heavy dependencies are not installed")
class TestNameAliases(unittest.TestCase):
    def test_function_aliases_are_the_new_objects(self):
        for module, old, new in FUNCTION_ALIASES:
            with self.subTest(f"{module}.{old}"):
                mod = _module(module)
                self.assertIs(getattr(mod, old), getattr(mod, new))

    def test_method_aliases_are_the_new_methods(self):
        for module, cls, old, new in METHOD_ALIASES:
            with self.subTest(f"{cls}.{old}"):
                klass = getattr(_module(module), cls)
                self.assertIs(klass.__dict__[old], klass.__dict__[new])

    def test_flat_namespace_keeps_old_names(self):
        for old, new in FLAT_ALIASES:
            with self.subTest(old):
                self.assertIs(getattr(substrata, old), getattr(substrata, new))
                self.assertIn(old, _module("measurements").__all__)

    def test_explore_imports(self):
        # the old private names substrata-explore imports
        from substrata.classification import _result_from_probs  # noqa: F401
        from substrata.cli import _classifier_crop_size  # noqa: F401
        from substrata.measurements import conduct_PCA  # noqa: F401
        from substrata import cameras, pointfeatures

        self.assertTrue(callable(cameras._photo_folder_name))
        self.assertTrue(callable(pointfeatures._median_image_width))
        self.assertTrue(callable(pointfeatures._cache_file))
        self.assertEqual(cameras._photo_folder_name("/a/ton_tof.auv/x.jpg"), "ton_tof.auv")


@unittest.skipIf(not HAVE_DEPS, "substrata's heavy dependencies are not installed")
class TestKeywordAliases(unittest.TestCase):
    def setUp(self):
        compat.reset_warnings()

    def test_every_renamed_keyword_is_declared(self):
        for module, qualname, old, new in KEYWORD_ALIASES:
            with self.subTest(f"{module}.{qualname}({old}=)"):
                func = _resolve(module, qualname)
                self.assertEqual(getattr(func, "__renamed_kwargs__", {}).get(old), new)
                params = inspect.signature(func).parameters
                self.assertIn(new, params)
                self.assertNotIn(old, params)

    def test_old_keyword_works_and_warns_once(self):
        pc = _module("pointclouds")
        normals = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        T = np.diag([2.0, 1.0, 1.0, 1.0])
        want = pc.transform_normals(normals, transform=T)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = pc.transform_normals(normals, transform_matrix=T)
            again = pc.transform_normals(normals, transform_matrix=T)
        np.testing.assert_allclose(got, want)
        np.testing.assert_allclose(again, want)
        dep = _deprecations(caught)
        self.assertEqual(len(dep), 1)
        self.assertIn("transform_matrix", str(dep[0].message))
        self.assertIn("'transform'", str(dep[0].message))
        self.assertEqual(os.path.basename(dep[0].filename), os.path.basename(__file__))

    def test_old_and_new_keyword_together_is_an_error(self):
        pc = _module("pointclouds")
        with self.assertRaises(TypeError):
            pc.transform_normals(np.eye(3), transform=np.eye(4), transform_matrix=np.eye(4))

    def test_simple_point_cloud_transform(self):
        pc = _module("pointclouds")
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        cloud = pc.SimplePointCloud(np.zeros((2, 3)))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cloud.transform(transform_matrix=T)
        np.testing.assert_allclose(cloud.points, [[1, 2, 3], [1, 2, 3]])
        self.assertEqual(len(_deprecations(caught)), 1)

    def test_cameras_and_annotations_transforms(self):
        cm, an = _module("cameras"), _module("annotations")
        T = np.eye(4)
        T[:3, 3] = [5.0, 0.0, 0.0]
        for coll in (cm.Cameras(), an.Annotations()):
            with self.subTest(type(coll).__name__):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    coll.get_original_coords(transform_matrix=T)
                np.testing.assert_allclose(coll.world_transform, T)
                coll.transform_coords(T)  # (old method name, positional)
                np.testing.assert_allclose(coll.world_transform, T @ T)
                if isinstance(coll, cm.Cameras):
                    with warnings.catch_warnings(record=True) as caught2:
                        warnings.simplefilter("always")
                        coll.transform_coords(transform_matrix=np.eye(4))
                    self.assertEqual(len(_deprecations(caught2)), 1)
                self.assertEqual(len(_deprecations(caught)), 1)

    def test_atomic_write_text(self):
        pr = _module("pathrepair")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.txt")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pr.atomic_write_text("hello", out_path=path)
            with open(path) as f:
                self.assertEqual(f.read(), "hello")
            self.assertEqual(len(_deprecations(caught)), 1)

    def test_build_image_index_quiet(self):
        pr = _module("pathrepair")
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "IMG_1.JPG"), "wb").close()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                index = pr.build_image_index(d, quiet=True)
            self.assertIn("img_1.jpg", index)
            dep = _deprecations(caught)
            self.assertEqual(len(dep), 1)
            self.assertIn("quiet", str(dep[0].message))
            self.assertEqual(os.path.basename(dep[0].filename), os.path.basename(__file__))

    def test_vector_dispersion_geom(self):
        m, pc = _module("measurements"), _module("pointclouds")
        n = np.array([[0, 0, 1.0], [0, 0, -1.0]])
        cloud = pc.SimplePointCloud(np.zeros((2, 3)), None, n)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            disp = m.get_vector_dispersion(geom=cloud, generate_image=False)[0]
        self.assertAlmostEqual(disp, 2.0)
        self.assertEqual(len(_deprecations(caught)), 1)
        # (Annotation.measure dispatches on the function's name)
        self.assertEqual(m.get_vector_dispersion.__name__, "get_vector_dispersion")
        self.assertEqual(m.calc_gap_fraction.__name__, "calc_gap_fraction")

    def test_line_of_sight_margin(self):
        st = _module("stratified")
        occ = st.Occluder.__new__(st.Occluder)
        occ.spacing = np.array([0.001])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                got = occ.line_of_sight_margin(scale_factor=2.0, min_metres=0.04)
            except AttributeError:
                self.skipTest("Occluder needs more state for line_of_sight_margin")
        self.assertEqual(got, occ.line_of_sight_margin(scale_factor=2.0, min_m=0.04))
        self.assertEqual(len(_deprecations(caught)), 1)


if __name__ == "__main__":
    unittest.main()
