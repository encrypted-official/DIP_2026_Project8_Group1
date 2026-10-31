import numpy as np

from src.estimate_moments import suppress_one_sided_shadow
from src.front_back import axis_profile_features

def test_shadow_suppression_returns_binary_mask():
    gray = np.full((40, 60), 150, dtype=np.uint8)
    component = np.zeros((40, 60), dtype=bool)
    component[12:28, 8:52] = True
    gray[12:28, 8:20] = 20
    result = suppress_one_sided_shadow(gray, component)
    assert result.shape == component.shape
    assert set(np.unique(result)).issubset({0, 255})

def test_axis_profile_features_are_finite():
    image = np.full((50, 90, 3), 100, dtype=np.uint8)
    image[:, 55:, :] = 180
    features = axis_profile_features(image, 0)
    assert features is not None
    assert np.isfinite(list(features.values())).all()
