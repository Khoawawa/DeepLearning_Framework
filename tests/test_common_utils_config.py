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


def test_utils_enum() -> None:
    from engines.enums.utils_enum import VerboseEnum
    assert VerboseEnum.NO.value == 0
    assert VerboseEnum.LOW.value == 1
    assert VerboseEnum.MEDIUM.value == 2
    assert VerboseEnum.HIGH.value == 3


def test_engine_utils_branches() -> None:
    import numpy as np
    import torch
    from engines.engine_utils import resolve_factory, to_var
    from engines.factory import Factory

    fact = resolve_factory("general")
    assert isinstance(fact, Factory)

    arr = np.array([1.0, 2.0, 3.0])
    res_arr = to_var(arr, torch.device("cpu"))
    assert isinstance(res_arr, torch.Tensor)

    with pytest.raises(ValueError, match="to_var util doesnt support"):
        to_var(object(), torch.device("cpu"))


def test_torch_utils_resolve_device() -> None:
    from unittest.mock import patch
    import torch
    from utils.torch_utils import resolve_device

    dev = resolve_device("cpu")
    assert dev.type == "cpu"

    with patch("torch.cuda.is_available", return_value=True):
        dev_cuda = resolve_device(None)
        assert dev_cuda.type == "cuda"


def test_initializer_decorator() -> None:
    from utils.common_utils import initializer

    class SampleClass:
        @initializer
        def __init__(self, a: int, b: str = "default") -> None:
            pass

    obj = SampleClass(42)
    assert obj.a == 42
    assert obj.b == "default"


def test_main_py_full_coverage(tmp_path: Path) -> None:
    import sys
    from unittest.mock import MagicMock, patch
    from torch.utils.data import DataLoader
    from tests.dummy_datasets import DummyImageDataset
    from main import configure_logging, main, parse_args, resolve_profile, run

    with patch.dict(os.environ, {"PROFILE": "invalid_profile"}):
        prof = resolve_profile()
        assert prof == "prod"

    configure_logging(log_dir=tmp_path / "logs", profile="dev")

    test_args = ["main.py", "-c", "configs/model/cnn.yaml", "-m", "train", "-e", "1"]
    with patch.object(sys, "argv", test_args):
        args = parse_args()
        assert args.config == "configs/model/cnn.yaml"

        bad_args = parse_args()
        bad_args.epochs = 0
        with patch("main.load_config", return_value={"model": {"name": "cnn"}, "data": {}}):
            with pytest.raises(ValueError, match="Please define the number of epochs"):
                run(bad_args)

        invalid_mode_args = parse_args()
        invalid_mode_args.mode = "unknown_mode"
        with patch("main.load_config", return_value={"model": {"name": "cnn"}}):
            with pytest.raises(ValueError, match="Unknown mode"):
                run(invalid_mode_args)

        test_run_args = parse_args()
        test_run_args.mode = "test"
        test_run_args.config = "configs/model/cnn.yaml"

        mock_factory = MagicMock()
        mock_factory.build_dataloader.return_value = DataLoader(DummyImageDataset(num_samples=4), batch_size=2)
        mock_tester = MagicMock()
        mock_tester.to.return_value = mock_tester
        mock_factory.build_tester.return_value = mock_tester
        mock_inferencer = MagicMock()
        mock_inferencer.to.return_value = mock_inferencer
        mock_factory.build_inferencer.return_value = mock_inferencer

        with patch("main.resolve_factory", return_value=mock_factory):
            with patch("main.load_config", return_value={"model": {"name": "cnn"}, "data": {}}):
                run(test_run_args)
                assert mock_tester.test.called

                test_run_args.mode = "inference"
                run(test_run_args)
                assert mock_factory.build_inferencer.called

        with patch("main.parse_args") as mock_parse:
            mock_parse.return_value = test_run_args
            with patch("main.run", side_effect=RuntimeError("Test exception")):
                assert main() == 1
            with patch("main.run", return_value=None):
                assert main() == 0

