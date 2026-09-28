import os
from pathlib import Path
import pytest
import yaml

from utils.common_utils import (
    load_yaml,
    load_config,
    resolve_config_path,
    resolve_output_path,
    deep_merge,
    raise_and_log,
    Registry,
    load_env,
    CONFIG_ROOT,
)


def test_load_yaml_success(tmp_path: Path) -> None:
    yaml_file = tmp_path / "test.yaml"
    data = {"key": "value", "number": 42}
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

    loaded = load_yaml(yaml_file)
    assert loaded == data


def test_load_yaml_file_not_found(tmp_path: Path) -> None:
    non_existent = tmp_path / "non_existent.yaml"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_yaml(non_existent)


def test_load_yaml_empty_content(tmp_path: Path) -> None:
    yaml_file = tmp_path / "empty.yaml"
    yaml_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to load config"):
        load_yaml(yaml_file)


def test_deep_merge() -> None:
    base = {
        "a": 1,
        "b": {"x": 10, "y": 20},
        "c": [1, 2, 3],
    }
    override = {
        "b": {"y": 25, "z": 30},
        "c": [4, 5],
        "d": 99,
    }
    merged = deep_merge(base, override)
    assert merged["a"] == 1
    assert merged["b"] == {"x": 10, "y": 25, "z": 30}
    assert merged["c"] == [4, 5]  # lists are replaced, not merged
    assert merged["d"] == 99


def test_resolve_config_path_variations(tmp_path: Path) -> None:
    # 1. Existing absolute file
    abs_file = tmp_path / "custom_cfg.yaml"
    abs_file.write_text("test: 1", encoding="utf-8")
    resolved = resolve_config_path(str(abs_file))
    assert resolved == abs_file

    # 2. Config name relative to CONFIG_ROOT
    resolved_config = resolve_config_path("config")
    assert resolved_config == CONFIG_ROOT / "config.yaml"

    # 3. Invalid extension error
    with pytest.raises(ValueError, match="unexpected extension"):
        resolve_config_path("config.json")

    # 4. Non-existent file in CONFIG_ROOT
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_config_path("non_existent_config_12345")


def test_load_config_with_defaults() -> None:
    # Load default config.yaml which includes defaults: model: cnn, data: dummy
    cfg = load_config(resolve_config_path("config"))
    assert "model" in cfg
    assert "data" in cfg
    assert cfg["model"]["name"] == "cnn"
    assert cfg["data"].get("name", cfg["data"].get("type")) == "dummy"
    assert "epochs" in cfg


def test_all_existing_yaml_configs_parseable() -> None:
    """Validate that every YAML file in the configs/ directory can be successfully loaded."""
    yaml_files = list(CONFIG_ROOT.rglob("*.yaml")) + list(CONFIG_ROOT.rglob("*.yml"))
    assert len(yaml_files) > 0, "No YAML config files found in CONFIG_ROOT"

    for yaml_path in yaml_files:
        content = load_yaml(yaml_path)
        assert isinstance(content, dict), f"Config {yaml_path} did not parse as a dictionary"


def test_resolve_output_path() -> None:
    out_path = resolve_output_path("runs", "exp1")
    assert out_path == CONFIG_ROOT / "runs" / "exp1"

    with pytest.raises(ValueError, match="Run name cannot be empty"):
        resolve_output_path("runs", "   ")


def test_raise_and_log() -> None:
    with pytest.raises(ValueError, match="Test error msg"):
        raise_and_log("Test error msg", ValueError)

    with pytest.raises(TypeError, match="Type error msg"):
        raise_and_log("Type error msg", TypeError)


def test_registry_class() -> None:
    reg: Registry[int] = Registry("test_kind")

    @reg.register("item1")
    class Item1:
        pass

    assert "item1" in reg
    assert reg.get("item1") == Item1
    assert reg.names() == ["item1"]

    # Test duplicate registration error
    with pytest.raises(ValueError, match="already registered"):
        @reg.register("item1")
        class Item1Duplicate:
            pass

    # Test non-registered lookup error
    with pytest.raises(ValueError, match="is not registered"):
        reg.get("unknown_item")

    # Test build error
    class ItemWithError:
        def __init__(self, required_arg: int) -> None:
            self.val = required_arg

    reg2: Registry[ItemWithError] = Registry("err_kind")
    reg2.register("err_item")(ItemWithError)

    with pytest.raises(ValueError, match="has invalid configuration"):
        reg2.build("err_item")  # missing required_arg


def test_load_env(tmp_path: Path) -> None:
    load_env()
