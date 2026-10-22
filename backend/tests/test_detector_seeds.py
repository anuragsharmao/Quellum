from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import get_settings
from app.services.detector import Detector, ModelLoadError


def test_model_ids_support_both_weight_naming_styles(tmp_path):
    detector = Detector(replace(get_settings(), models_dir=tmp_path))
    (tmp_path / "seed_0.weights.h5").touch()
    (tmp_path / "seed5.weights.h5").touch()

    assert detector._weights_path(0).name == "seed_0.weights.h5"
    assert detector._weights_path(5).name == "seed5.weights.h5"


@pytest.mark.parametrize("seed_ids", [(), (0, 0), (-1,), (10,)])
def test_invalid_model_ids_fail_before_loading_artifacts(tmp_path, seed_ids):
    detector = Detector(replace(get_settings(), artifacts_dir=tmp_path, detector_seeds=seed_ids))
    with pytest.raises(ModelLoadError, match="invalid_model_seeds"):
        detector.load()
