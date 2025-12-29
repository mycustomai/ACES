from experiments.analysis.common import REPO_ROOT, load_model_display_names


def test_load_model_display_names_length():
    """Verify dict length matches the number of model config files."""
    display_names = load_model_display_names()

    config_dir = REPO_ROOT / "config" / "models"
    config_files = [
        f for f in config_dir.glob("*.yaml")
        if not f.name.startswith("_") and not f.name.startswith(".")
    ]

    assert len(display_names) == len(config_files)


def test_load_model_display_names_all_have_values():
    """Verify all entries have non-empty keys and display names."""
    display_names = load_model_display_names()

    assert len(display_names) > 0

    for model_name, display_name in display_names.items():
        assert model_name, "Model name should not be empty"
        assert isinstance(model_name, str)
        assert display_name, f"Display name for {model_name} should not be empty"
        assert isinstance(display_name, str)
