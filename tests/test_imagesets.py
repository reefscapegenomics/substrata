"""Image sets (substrata.imagesets) and the camera domain a set gives its points in training."""

import csv
import os
import tempfile
import unittest

try:
    from substrata import classification, imagesets, sidecars
except ImportError:  # pragma: no cover - outside the substrata environment
    imagesets = None


@unittest.skipUnless(imagesets is not None, "requires the substrata environment")
class TestImageSets(unittest.TestCase):
    def test_layout(self):
        d = os.path.join(tempfile.mkdtemp(), "reef_auv")
        os.makedirs(os.path.join(d, "reef_auv.photos"))
        self.assertIsNone(imagesets.image_set_yaml(d))
        with self.assertRaises(ValueError):
            imagesets.read_image_set(d)
        y = imagesets.write_image_set_yaml(d, camera="auv")
        self.assertEqual(imagesets.image_set_yaml(d), y)
        info = imagesets.read_image_set(d)
        self.assertEqual((info["id"], info["camera"], info["gsd_mm"]), ("reef_auv", "auv", None))
        self.assertEqual(info["photos"], os.path.join(d, "reef_auv.photos"))
        self.assertEqual(info["annos"], os.path.join(d, "reef_auv.annos"))
        plot = os.path.join(os.path.dirname(d), "plot_1")  # a plot's <id>.yaml has no type: images
        os.makedirs(plot)
        with open(os.path.join(plot, "plot_1.yaml"), "w") as f:
            f.write("cams_xml: plot_1.cams.xml\n")
        self.assertIsNone(imagesets.image_set_yaml(plot))

    def test_sample_points_refuses_a_plot(self):
        from argparse import Namespace
        from substrata.cli import handle_sample_points
        plot = os.path.join(tempfile.mkdtemp(), "plot_1")  # a plot has <id>.photos/ too
        os.makedirs(os.path.join(plot, "plot_1.photos"))
        with open(os.path.join(plot, "plot_1.yaml"), "w") as f:
            f.write("cams_xml: plot_1.cams.xml\n")
        cwd = os.getcwd()
        os.chdir(plot)
        try:
            with self.assertRaises(SystemExit) as e:
                handle_sample_points(Namespace(tag=None, overwrite=False))
        finally:
            os.chdir(cwd)
        self.assertIn("is a plot", str(e.exception.code))
        with open(os.path.join(plot, "plot_1.yaml")) as f:
            self.assertNotIn("type", f.read())  # (left as it was)

    def test_set_domain_in_training(self):
        d = tempfile.mkdtemp()
        photo = os.path.join(d, "IMG_1.jpg")
        open(photo, "wb").close()
        p = os.path.join(d, "reef_auv_points.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "orig_x", "orig_y", "orig_z", "label", "cam_filepath", "cam_x", "cam_y"])
            w.writerow(["a", "nan", "nan", "nan", "CST", photo, "100", "100"])
        sidecars.write_annotation_sidecar(p, {"type": "image_points", "domain": "auv"})
        out = os.path.join(d, "training.csv")
        classification.collate_training_annotations([p], "*.csv", {"CST"}, out, d, prompt=False)
        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["domain"] for r in rows], ["auv"])


if __name__ == "__main__":
    unittest.main()
