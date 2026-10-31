"""Week 6 ablation grid tests."""

from src.ablation import build_configurations

def test_week6_grid_has_48_configurations():
    configs = build_configurations()
    assert len(configs) == 48

def test_configurations_are_unique():
    configs = build_configurations()
    normalized = [repr(sorted(config.items())) for config in configs]
    assert len(normalized) == len(set(normalized))

def test_shadow_toggle_only_applies_to_moments():
    configs = build_configurations()
    shadow_configs = [c for c in configs if c["shadow_check"]]
    assert len(shadow_configs) == 6
    assert all(c["method"] == "otsu_moments" for c in shadow_configs)
