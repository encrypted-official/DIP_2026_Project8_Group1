"""Basic tests for the Week 5 ablation configuration grid."""

from src.ablation import build_configurations


def test_ablation_grid_has_expected_number_of_configs():
    configs = build_configurations()
    # PCA and Hough: 3 blur sizes x 2 CLAHE states x 3 Canny pairs each.
    # Moments: 3 blur sizes x 2 CLAHE states.
    assert len(configs) == 42


def test_configurations_are_unique():
    configs = build_configurations()
    normalized = [repr(sorted(c.items())) for c in configs]
    assert len(normalized) == len(set(normalized))
