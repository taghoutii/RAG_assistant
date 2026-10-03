import pytest

from ragpapers.config import load_config


def test_default_config_loads():
    config = load_config()
    assert config["embedding"]["model"] == "BAAI/bge-small-en-v1.5"
    assert "paths" in config


def test_non_mapping_config_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)
