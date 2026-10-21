"""Tests for ``classification.aggregate_predictions``."""

# Standard Library
import importlib.util
import logging
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SUB = _SRC / "substrata"


def _load_classification():
    """Load ``classification.py`` from this checkout, isolated from sys.modules.

    The module (and the real ``settings`` it needs) are loaded into a private
    ``substrata`` package; ``sys.modules`` is restored afterwards, so neither
    other test modules nor an installed ``substrata`` are affected.
    """
    keys = [
        "substrata", "substrata.settings", "substrata.logging",
        "substrata.classification",
    ]
    saved = {key: sys.modules.get(key) for key in keys}
    try:
        pkg = types.ModuleType("substrata")
        pkg.__path__ = [str(_SUB)]
        sys.modules["substrata"] = pkg
        for name in ("settings", "classification"):
            sys.modules.pop(f"substrata.{name}", None)

        spec = importlib.util.spec_from_file_location(
            "substrata.settings", _SUB / "settings.py"
        )
        settings = importlib.util.module_from_spec(spec)
        sys.modules["substrata.settings"] = pkg.settings = settings
        spec.loader.exec_module(settings)

        log_mod = types.ModuleType("substrata.logging")
        log_mod.logger = logging.getLogger("substrata-test")

        @contextmanager
        def tqdm_joblib(_tqdm_object):
            yield

        log_mod.tqdm_joblib = tqdm_joblib
        sys.modules["substrata.logging"] = pkg.logging = log_mod

        spec = importlib.util.spec_from_file_location(
            "substrata.classification", _SUB / "classification.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["substrata.classification"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for key, mod_saved in saved.items():
            if mod_saved is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = mod_saved


cl = _load_classification()


def _match(relevance, obstructed=False):
    return types.SimpleNamespace(
        relevance=relevance, potentially_obstructed=obstructed
    )


def _result(probs):
    label = max(probs, key=probs.get)
    return {"label": label, "confidence": probs[label], "probs": probs}


class TestAggregatePredictions(unittest.TestCase):
    def test_no_results(self):
        self.assertIsNone(cl.aggregate_predictions([], []))
        self.assertIsNone(cl.aggregate_predictions([_match(1.0)], [None]))

    def test_first_uses_top_ranked_successful_result(self):
        matches = [_match(1.0), _match(2.0), _match(3.0)]
        results = [None, _result({"C": 0.6, "S": 0.4}), _result({"S": 0.9})]
        agg = cl.aggregate_predictions(matches, results, "first")
        self.assertEqual(agg, {"label": "C", "confidence": 0.6, "n": 1})

    def test_overconfident_outlier_does_not_win(self):
        # Two consistent crops vs one very confident (but less relevant) one;
        # picking the single most confident crop would choose "S".
        matches = [_match(1.0), _match(1.2), _match(1.5)]
        results = [
            _result({"C": 0.7, "S": 0.3}),
            _result({"C": 0.7, "S": 0.3}),
            _result({"C": 0.01, "S": 0.99}),
        ]
        agg = cl.aggregate_predictions(matches, results, "weighted")
        self.assertEqual(agg["label"], "C")
        self.assertEqual(agg["n"], 3)

    def test_relevance_weighting(self):
        matches = [_match(1.0), _match(4.0)]
        results = [_result({"C": 0.6, "S": 0.4}), _result({"C": 0.0, "S": 1.0})]
        agg = cl.aggregate_predictions(matches, results)
        # weights 1 and 0.25 -> C = 0.6/1.25 = 0.48, S = 0.65/1.25 = 0.52
        self.assertEqual(agg["label"], "S")
        self.assertAlmostEqual(agg["confidence"], 0.52)

    def test_obstructed_crops_excluded_unless_all_are(self):
        matches = [_match(1.0), _match(1.0, obstructed=True)]
        results = [_result({"C": 0.6, "S": 0.4}), _result({"C": 0.0, "S": 1.0})]
        agg = cl.aggregate_predictions(matches, results)
        self.assertEqual((agg["label"], agg["n"]), ("C", 1))

        matches = [_match(1.0, obstructed=True), _match(1.0, obstructed=True)]
        agg = cl.aggregate_predictions(matches, results)
        self.assertEqual((agg["label"], agg["n"]), ("S", 2))

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            cl.aggregate_predictions([_match(1.0)], [_result({"C": 1.0})], "max")


if __name__ == "__main__":
    unittest.main()
