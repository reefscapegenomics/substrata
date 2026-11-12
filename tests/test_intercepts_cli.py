"""substrata intercepts' new-set helpers: --tag file names, and the labelled-set guard (no overwrite)."""

import os
import tempfile
import unittest

try:
    from substrata.cli import _intercepts_tag, _labelled_rows
except ImportError:  # pragma: no cover - outside the substrata environment
    _intercepts_tag = _labelled_rows = None


@unittest.skipUnless(_labelled_rows is not None, "requires the substrata environment")
class TestInterceptsNewSets(unittest.TestCase):
    def test_tag(self):
        self.assertEqual(_intercepts_tag(None), "")
        self.assertEqual(_intercepts_tag("v2"), "_v2")
        self.assertEqual(_intercepts_tag("my pool #1"), "_my-pool-1")
        self.assertEqual(_intercepts_tag("training_set"), "_training_set")
        with self.assertRaises(SystemExit):
            _intercepts_tag("##")

    def test_labelled_rows(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "p_slope_intercepts.csv")
        self.assertEqual(_labelled_rows(p), 0)  # no file: nothing to protect
        with open(p, "w") as f:
            f.write("id,orig_x,orig_y,orig_z,label\np_0,1,2,3,\np_1,1,2,3,CST\np_2,1,2,3, \n")
        self.assertEqual(_labelled_rows(p), 1)
        with open(p, "w") as f:
            f.write("id,orig_x,orig_y,orig_z\np_0,1,2,3\n")
        self.assertEqual(_labelled_rows(p), 0)  # no label column

    def test_depth_range_arguments_checked_first(self):
        from argparse import Namespace
        from substrata.cli import handle_intercepts
        base = dict(depth_range=[30.0, 40.0], slope=False, position=None, width_step=1.0, depth_step=1.0)
        for bad, msg in ((dict(slope=True), "no --slope"), (dict(width_step=0.0), "positive"),
                         (dict(depth_range=[30.0, -30.0]), "empty band")):
            with self.assertRaises(SystemExit) as e:  # (before any project is loaded)
                handle_intercepts(Namespace(**{**base, **bad}))
            self.assertIn(msg, str(e.exception.code))


if __name__ == "__main__":
    unittest.main()
