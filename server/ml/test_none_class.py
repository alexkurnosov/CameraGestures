"""
Tests for the synthetic _none class generation used to reject unknown gestures.

Run from the server directory:
    python -m pytest ml/test_none_class.py -v
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

# Stub heavy transitive dependencies so we can import just the function
# under test without requiring the full server stack.
_server_dir = str(Path(__file__).resolve().parent.parent)
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

# config.settings
_config_mod = types.ModuleType("config")
_config_mod.settings = types.SimpleNamespace(models_dir=Path("/tmp"))  # type: ignore[attr-defined]
sys.modules.setdefault("config", _config_mod)

# ml.preprocessor (needs py_mini_racer which isn't installed in test env)
_preproc_mod = types.ModuleType("ml.preprocessor")
_preproc_mod.summary_features = lambda hf: np.zeros(256, dtype=np.float32)  # type: ignore[attr-defined]
sys.modules.setdefault("ml.preprocessor", _preproc_mod)

from ml.trainer_rf_mlp import NONE_GESTURE_ID, _generate_none_examples


class TestGenerateNoneExamples:
    """Tests for _generate_none_examples."""

    def _make_data(self, n_per_class: int = 20, n_classes: int = 2, n_features: int = 256):
        """Create dummy feature matrix and labels."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n_per_class * n_classes, n_features)).astype(np.float32)
        y = np.repeat(np.arange(n_classes), n_per_class)
        return X, y

    def test_output_shape(self):
        X, y = self._make_data()
        rng = np.random.default_rng(42)
        result = _generate_none_examples(X, y, per_class_count=15, rng=rng)
        assert result.shape == (15, X.shape[1])

    def test_output_dtype(self):
        X, y = self._make_data()
        rng = np.random.default_rng(42)
        result = _generate_none_examples(X, y, per_class_count=10, rng=rng)
        assert result.dtype == np.float32

    def test_no_nans_or_infs(self):
        X, y = self._make_data()
        rng = np.random.default_rng(42)
        result = _generate_none_examples(X, y, per_class_count=30, rng=rng)
        assert np.all(np.isfinite(result))

    def test_different_from_originals(self):
        """Synthetic examples should not be exact copies of any real example."""
        X, y = self._make_data()
        rng = np.random.default_rng(42)
        result = _generate_none_examples(X, y, per_class_count=20, rng=rng)
        for row in result:
            assert not any(np.allclose(row, x) for x in X)

    def test_small_request(self):
        """Even a very small request (3 samples) should work."""
        X, y = self._make_data(n_per_class=5)
        rng = np.random.default_rng(42)
        result = _generate_none_examples(X, y, per_class_count=3, rng=rng)
        assert result.shape[0] == 3

    def test_none_gesture_id_constant(self):
        assert NONE_GESTURE_ID == "_none"
