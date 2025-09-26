"""Tests for the unified ScreenshotRuntime implementation."""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from agent.src.typedefs import EngineParams, EngineType
from experiments.config import ExperimentData
from experiments.runners.screenshot_runtime import ScreenshotRuntime


@pytest.fixture
def mock_engine_params() -> list[EngineParams]:
    """Provide a couple of engine configurations for tests."""
    return [
        EngineParams(
            engine_type=EngineType.OPENAI,
            model="gpt-4-vision-preview",
            config_name="openai",
            api_key="key-openai",
            temperature=0.1,
            max_new_tokens=1000,
        ),
        EngineParams(
            engine_type=EngineType.ANTHROPIC,
            model="claude-3-sonnet-20240229",
            config_name="anthropic",
            api_key="key-anthropic",
            temperature=0.1,
            max_new_tokens=1000,
        ),
    ]


@pytest.fixture
def sample_experiment() -> ExperimentData:
    """Return a representative experiment row."""
    experiment_df = Mock()
    experiment_df.copy.return_value = experiment_df
    return ExperimentData(
        query="wireless mouse",
        experiment_label="test",
        experiment_number=1,
        prompt_template="Find a wireless mouse",
        experiment_df=experiment_df,
        dataset_name="test_dataset",
        screenshot=None,
    )


def _make_loader(
    *,
    dataset_name: str = "test_dataset",
    dataset_path: str | None = None,
    screenshots_dir: Path | None = None,
    experiments: list[ExperimentData] | None = None,
) -> Mock:
    loader = Mock()
    loader.dataset_name = dataset_name
    loader.dataset_path = dataset_path
    loader.screenshots_dir = screenshots_dir
    container: dict[str, list[ExperimentData]] = {"value": list(experiments or [])}

    def _iter():
        return iter(container["value"])

    loader.experiments_iter.side_effect = _iter
    loader.container = container  # Allow tests to mutate experiment list if needed
    return loader


@contextmanager
def _runtime_context(loader: Mock, mock_engine_params, **runtime_kwargs):
    with patch(
        "experiments.runners.screenshot_runtime.base.ExperimentLoader",
        return_value=loader,
    ) as loader_cls, patch(
        "experiments.runners.screenshot_runtime.base.ExperimentWorkerService"
    ) as worker_cls, patch(
        "experiments.runners.screenshot_runtime.base.ScreenshotValidationService"
    ) as validation_cls:
        worker_cls.return_value = Mock()
        validation_instance = Mock()
        validation_cls.return_value = validation_instance

        runtime = ScreenshotRuntime(
            engine_params_list=mock_engine_params,
            **runtime_kwargs,
        )

        yield runtime, loader_cls, validation_cls, validation_instance


class TestScreenshotRuntime:
    def test_local_dataset_initialization_sets_validation_service(
        self, mock_engine_params, tmp_path
    ):
        dataset_path = tmp_path / "dataset.csv"
        screenshots_dir = tmp_path / "screenshots"
        loader = _make_loader(
            dataset_path=str(dataset_path),
            screenshots_dir=screenshots_dir,
        )

        with _runtime_context(
            loader,
            mock_engine_params,
            local_dataset_path=str(dataset_path),
            remote=False,
        ) as (runtime, loader_cls, validation_cls, validation_instance):
            expected_call = dict(
                engine_params=mock_engine_params,
                experiment_count_limit=None,
                local_dataset_path=str(dataset_path),
                hf_dataset_name=None,
                hf_subset="all",
            )
            loader_cls.assert_called_once_with(**expected_call)
            validation_cls.assert_called_once_with(
                screenshots_dir,
                str(dataset_path.parent),
                debug_mode=False,
            )
            assert runtime.validation_service is validation_instance
            assert runtime.screenshots_dir == screenshots_dir

    def test_hf_dataset_initialization_skips_validation(self, mock_engine_params):
        loader = _make_loader(dataset_name="hf_dataset")

        with _runtime_context(
            loader,
            mock_engine_params,
            hf_dataset_name="my/dataset",
            hf_subset="subset",
            remote=False,
        ) as (runtime, loader_cls, validation_cls, _):
            expected_call = dict(
                engine_params=mock_engine_params,
                experiment_count_limit=None,
                local_dataset_path=None,
                hf_dataset_name="my/dataset",
                hf_subset="subset",
            )
            loader_cls.assert_called_once_with(**expected_call)
            assert not validation_cls.called
            assert runtime.validation_service is None
            assert runtime.screenshots_dir is None

    def test_experiments_iter_delegates_to_loader(
        self, mock_engine_params, sample_experiment
    ):
        loader = _make_loader(experiments=[sample_experiment])

        with _runtime_context(
            loader,
            mock_engine_params,
            hf_dataset_name="my/dataset",
            remote=False,
        ) as (runtime, _, _, _):
            experiments = list(runtime.experiments_iter)
            assert experiments == [sample_experiment]
            assert loader.experiments_iter.call_count == 1

    def test_create_environment_local_dataset(
        self, mock_engine_params, sample_experiment, tmp_path
    ):
        dataset_path = tmp_path / "dataset.csv"
        screenshots_dir = tmp_path / "screenshots"
        loader = _make_loader(
            dataset_path=str(dataset_path),
            screenshots_dir=screenshots_dir,
        )

        with patch(
            "experiments.runners.screenshot_runtime.base.FilesystemShoppingEnvironment"
        ) as fs_env:
            fs_env_instance = Mock()
            fs_env.return_value = fs_env_instance

            with _runtime_context(
                loader,
                mock_engine_params,
                local_dataset_path=str(dataset_path),
                remote=True,
            ) as (runtime, _, _, _):
                environment = runtime.create_shopping_environment(sample_experiment)

            fs_env.assert_called_once_with(
                screenshots_dir=screenshots_dir,
                query=sample_experiment.query,
                experiment_label=sample_experiment.experiment_label,
                experiment_number=sample_experiment.experiment_number,
                dataset_name=loader.dataset_name,
                remote=True,
            )
            assert environment is fs_env_instance

    def test_create_environment_hf_dataset(
        self, mock_engine_params, sample_experiment
    ):
        screenshot = Mock()
        sample_experiment.screenshot = screenshot
        loader = _make_loader()

        with patch(
            "experiments.runners.screenshot_runtime.base.DatasetShoppingEnvironment"
        ) as dataset_env:
            dataset_env_instance = Mock()
            dataset_env.return_value = dataset_env_instance

            with _runtime_context(
                loader,
                mock_engine_params,
                hf_dataset_name="my/dataset",
                remote=False,
            ) as (runtime, _, _, _):
                environment = runtime.create_shopping_environment(sample_experiment)

            dataset_env.assert_called_once_with(screenshot_image=screenshot)
            assert environment is dataset_env_instance

    def test_create_environment_hf_dataset_missing_screenshot_raises(
        self, mock_engine_params, sample_experiment
    ):
        loader = _make_loader()

        with _runtime_context(
            loader,
            mock_engine_params,
            hf_dataset_name="my/dataset",
            remote=False,
        ) as (runtime, _, _, _):
            with pytest.raises(ValueError, match="No screenshot found"):
                runtime.create_shopping_environment(sample_experiment)

    def test_validate_prerequisites_uses_validation_service(
        self, mock_engine_params, sample_experiment, tmp_path
    ):
        dataset_path = tmp_path / "dataset.csv"
        screenshots_dir = tmp_path / "screenshots"
        loader = _make_loader(
            dataset_path=str(dataset_path),
            screenshots_dir=screenshots_dir,
            experiments=[sample_experiment],
        )

        with _runtime_context(
            loader,
            mock_engine_params,
            local_dataset_path=str(dataset_path),
            remote=False,
        ) as (runtime, _, _, validation_instance):
            validation_instance.validate_all_screenshots.return_value = True
            assert runtime.validate_prerequisites() is True
            args, _ = validation_instance.validate_all_screenshots.call_args
            passed_experiments, passed_path = args
            assert passed_path == str(dataset_path)
            assert passed_experiments == [sample_experiment]

    def test_get_dataset_path_delegates_to_loader(
        self, mock_engine_params, tmp_path
    ):
        dataset_path = tmp_path / "dataset.csv"
        loader = _make_loader(dataset_path=str(dataset_path))

        with _runtime_context(
            loader,
            mock_engine_params,
            local_dataset_path=str(dataset_path),
            remote=False,
        ) as (runtime, _, _, _):
            assert runtime.get_dataset_path() == str(dataset_path)


class TestApiKeyLoading:
    def test_single_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "single-key")
        keys = ScreenshotRuntime.load_api_keys_for_provider(EngineType.OPENAI)
        assert keys == ["single-key"]

    def test_multiple_keys(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "key-1")
        monkeypatch.setenv("OPENAI_API_KEY_2", "key-2")
        monkeypatch.setenv("OPENAI_API_KEY_3", "key-3")
        keys = ScreenshotRuntime.load_api_keys_for_provider(EngineType.OPENAI)
        assert keys == ["key-1", "key-2", "key-3"]

    def test_no_keys(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY_2", raising=False)
        keys = ScreenshotRuntime.load_api_keys_for_provider(EngineType.OPENAI)
        assert keys == []
