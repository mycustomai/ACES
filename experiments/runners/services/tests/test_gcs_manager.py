import os
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from experiments.config import ExperimentData, InstructionConfig
from experiments.runners.services.gcs_manager import GCSManager, _UploadTask


class TestGCSManager:
    @pytest.fixture
    def mock_storage_client(self):
        """Mock Google Cloud Storage client."""
        with patch(
            "experiments.runners.services.gcs_manager.storage.Client"
        ) as mock_client:
            mock_bucket = Mock()
            mock_client.return_value.bucket.return_value = mock_bucket
            yield mock_client, mock_bucket

    @pytest.fixture
    def mock_experiment_data(self):
        """Create mock experiment data."""
        exp1 = Mock(spec=ExperimentData)
        exp1.get_gcs_screenshot_path.return_value = (
            "screenshots/mousepad/bias_test/mousepad_bias_test_1.png"
        )
        exp1.get_local_screenshot_path.return_value = Path(
            "/local/screenshots/mousepad/bias_test/mousepad_bias_test_1.png"
        )

        exp2 = Mock(spec=ExperimentData)
        exp2.get_gcs_screenshot_path.return_value = (
            "screenshots/mousepad/bias_test/mousepad_bias_test_2.png"
        )
        exp2.get_local_screenshot_path.return_value = Path(
            "/local/screenshots/mousepad/bias_test/mousepad_bias_test_2.png"
        )

        return [exp1, exp2]

    @pytest.fixture
    def env_vars(self):
        """Mock environment variables."""
        with patch.dict(
            os.environ,
            {"GCS_BUCKET_NAME": "test-bucket", "GOOGLE_CLOUD_PROJECT": "test-project"},
        ):
            yield

    @pytest.fixture
    def gcs_manager(self, env_vars, mock_storage_client):
        """Create GCSManager instance with mocked dependencies."""
        mock_client, mock_bucket = mock_storage_client
        with patch(
            "experiments.runners.services.gcs_manager.get_dataset_name",
            return_value="mousepad",
        ):
            manager = GCSManager(
                local_dataset_path="/path/to/mousepad_dataset.csv", max_workers=2
            )
            manager._bucket = mock_bucket
            return manager

    def test_init_success(self, env_vars, mock_storage_client):
        """Test successful initialization."""
        mock_client, mock_bucket = mock_storage_client

        with patch(
            "experiments.runners.services.gcs_manager.get_dataset_name",
            return_value="mousepad",
        ):
            manager = GCSManager(
                local_dataset_path="/path/to/mousepad_dataset.csv", max_workers=8
            )

            assert manager.dataset_name == "mousepad"
            assert manager.local_dataset_path == "/path/to/mousepad_dataset.csv"
            assert manager.max_workers == 8
            assert manager.bucket_name == "test-bucket"
            assert manager.project_id == "test-project"
            mock_client.assert_called_once_with(project="test-project")

    def test_init_missing_bucket_name(self):
        """Test initialization fails when GCS_BUCKET_NAME is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GCS bucket name must be provided"):
                GCSManager(local_dataset_path="/path/to/dataset.csv")

    def test_init_with_default_project(self, mock_storage_client):
        """Test initialization with default project (None)."""
        mock_client, mock_bucket = mock_storage_client

        with patch.dict(os.environ, {"GCS_BUCKET_NAME": "test-bucket"}):
            with patch(
                "experiments.runners.services.gcs_manager.get_dataset_name",
                return_value="mousepad",
            ):
                manager = GCSManager(local_dataset_path="/path/to/mousepad_dataset.csv")

                assert manager.project_id is None
                mock_client.assert_called_once_with(project=None)

    def test_upload_no_new_screenshots(self, gcs_manager, mock_experiment_data):
        """Test upload when all screenshots are already uploaded."""
        # Mock that all screenshots are already uploaded
        gcs_manager._list_uploaded_screenshots = Mock(
            return_value=[
                "screenshots/mousepad/bias_test/mousepad_bias_test_1.png",
                "screenshots/mousepad/bias_test/mousepad_bias_test_2.png",
            ]
        )
        gcs_manager._upload_screenshots = Mock()

        gcs_manager.upload([mock_experiment_data], verbose=True)

        # Should not call upload since all are already uploaded
        gcs_manager._upload_screenshots.assert_called_once_with([])

    def test_upload_with_new_screenshots(self, gcs_manager, mock_experiment_data):
        """Test upload with new screenshots to upload."""
        # Mock that no screenshots are uploaded yet
        gcs_manager._list_uploaded_screenshots = Mock(return_value=[])
        gcs_manager._prepare_upload_tasks = Mock(
            return_value=[
                _UploadTask(local_path="/local/path1.png", gcs_path="gcs/path1.png"),
                _UploadTask(local_path="/local/path2.png", gcs_path="gcs/path2.png"),
            ]
        )
        gcs_manager._upload_screenshots = Mock()

        gcs_manager.upload([mock_experiment_data], verbose=True)

        gcs_manager._prepare_upload_tasks.assert_called_once_with(mock_experiment_data)
        gcs_manager._upload_screenshots.assert_called_once()

    def test_upload_partial_existing_screenshots(
        self, gcs_manager, mock_experiment_data
    ):
        """Test upload with some screenshots already uploaded."""
        # Mock that only first screenshot is already uploaded
        gcs_manager._list_uploaded_screenshots = Mock(
            return_value=["screenshots/mousepad/bias_test/mousepad_bias_test_1.png"]
        )
        gcs_manager._prepare_upload_tasks = Mock(
            return_value=[
                _UploadTask(local_path="/local/path2.png", gcs_path="gcs/path2.png")
            ]
        )
        gcs_manager._upload_screenshots = Mock()

        gcs_manager.upload([mock_experiment_data], verbose=False)

        # Should only prepare upload task for the second experiment
        expected_not_uploaded = [mock_experiment_data[1]]
        gcs_manager._prepare_upload_tasks.assert_called_once_with(expected_not_uploaded)

    def test_prepare_upload_tasks_success(self, gcs_manager, mock_experiment_data):
        """Test _prepare_upload_tasks with existing files."""
        # Mock that local files exist
        for exp in mock_experiment_data:
            mock_path = Mock()
            mock_path.exists.return_value = True
            exp.get_local_screenshot_path.return_value = mock_path

        result = gcs_manager._prepare_upload_tasks(mock_experiment_data)

        assert len(result) == 2
        assert isinstance(result[0], _UploadTask)
        assert isinstance(result[1], _UploadTask)
        assert result[0].local_path == str(
            mock_experiment_data[0].get_local_screenshot_path.return_value
        )
        assert (
            result[0].gcs_path
            == mock_experiment_data[0].get_gcs_screenshot_path.return_value
        )

    def test_prepare_upload_tasks_missing_file(self, gcs_manager, mock_experiment_data):
        """Test _prepare_upload_tasks raises error when local file doesn't exist."""
        # Mock that first file doesn't exist
        mock_path1 = Mock()
        mock_path1.exists.return_value = False
        mock_experiment_data[0].get_local_screenshot_path.return_value = mock_path1

        mock_path2 = Mock()
        mock_path2.exists.return_value = True
        mock_experiment_data[1].get_local_screenshot_path.return_value = mock_path2

        with pytest.raises(FileNotFoundError, match="Screenshot file not found"):
            gcs_manager._prepare_upload_tasks(mock_experiment_data)

    def test_upload_single_screenshot_success(self, gcs_manager):
        """Test successful single screenshot upload."""
        task = _UploadTask(local_path="/local/test.png", gcs_path="gcs/test.png")
        mock_blob = Mock()
        gcs_manager._bucket.blob.return_value = mock_blob

        gcs_manager._upload_single_screenshot(task)

        gcs_manager._bucket.blob.assert_called_once_with("gcs/test.png")
        mock_blob.upload_from_filename.assert_called_once_with("/local/test.png")

    @patch("experiments.runners.services.gcs_manager.ThreadPoolExecutor")
    @patch("experiments.runners.services.gcs_manager.Progress")
    def test_upload_screenshots_success(
        self, mock_progress, mock_executor, gcs_manager
    ):
        """Test successful screenshot upload with progress tracking."""
        # Setup mocks
        mock_progress_instance = Mock()
        mock_progress.return_value.__enter__.return_value = mock_progress_instance
        mock_progress_instance.add_task.return_value = "task_id"

        mock_executor_instance = Mock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        # Mock futures
        future1 = Mock(spec=Future)
        future2 = Mock(spec=Future)
        mock_executor_instance.submit.side_effect = [future1, future2]

        # Mock as_completed to return futures in order
        with patch(
            "experiments.runners.services.gcs_manager.as_completed",
            return_value=[future1, future2],
        ):
            tasks = [
                _UploadTask(local_path="/local/1.png", gcs_path="gcs/1.png"),
                _UploadTask(local_path="/local/2.png", gcs_path="gcs/2.png"),
            ]

            gcs_manager._upload_screenshots(tasks)

            # Verify progress tracking
            mock_progress_instance.add_task.assert_called_once_with(
                "Uploading screenshots", total=2
            )
            assert mock_progress_instance.advance.call_count == 2

            # Verify executor usage
            mock_executor.assert_called_once_with(max_workers=2)
            assert mock_executor_instance.submit.call_count == 2

    def test_list_uploaded_screenshots_success(self, gcs_manager):
        """Test successful listing of uploaded screenshots."""
        mock_blob1 = Mock()
        mock_blob1.name = "screenshots/mousepad/test1.png"
        mock_blob2 = Mock()
        mock_blob2.name = "screenshots/mousepad/test2.png"

        gcs_manager._bucket.list_blobs.return_value = [mock_blob1, mock_blob2]

        result = gcs_manager._list_uploaded_screenshots()

        assert result == [
            "screenshots/mousepad/test1.png",
            "screenshots/mousepad/test2.png",
        ]
        gcs_manager._bucket.list_blobs.assert_called_once_with(
            prefix="screenshots/mousepad"
        )

    def test_list_uploaded_screenshots_empty(self, gcs_manager):
        """Test listing uploaded screenshots when none exist."""
        gcs_manager._bucket.list_blobs.return_value = []

        result = gcs_manager._list_uploaded_screenshots()

        assert result == []
        gcs_manager._bucket.list_blobs.assert_called_once_with(
            prefix="screenshots/mousepad"
        )

    def test_upload_with_multiple_experiment_batches(self, gcs_manager):
        """Test upload with multiple batches of experiments."""
        # Create multiple batches
        batch1 = [Mock(spec=ExperimentData) for _ in range(2)]
        batch2 = [Mock(spec=ExperimentData) for _ in range(3)]

        for i, exp in enumerate(batch1):
            exp.get_gcs_screenshot_path.return_value = (
                f"screenshots/mousepad/batch1_{i}.png"
            )
        for i, exp in enumerate(batch2):
            exp.get_gcs_screenshot_path.return_value = (
                f"screenshots/mousepad/batch2_{i}.png"
            )

        gcs_manager._list_uploaded_screenshots = Mock(return_value=[])
        gcs_manager._prepare_upload_tasks = Mock(return_value=[])
        gcs_manager._upload_screenshots = Mock()

        gcs_manager.upload([batch1, batch2])

        # Should call with flattened list of all experiments
        expected_experiments = batch1 + batch2
        gcs_manager._prepare_upload_tasks.assert_called_once_with(expected_experiments)

    def test_upload_task_model(self):
        """Test _UploadTask pydantic model."""
        task = _UploadTask(local_path="/local/test.png", gcs_path="gcs/test.png")

        assert task.local_path == "/local/test.png"
        assert task.gcs_path == "gcs/test.png"

        # Test validation
        with pytest.raises(Exception):  # Pydantic validation error
            _UploadTask(local_path="/local/test.png")  # Missing gcs_path

    @patch("experiments.runners.services.gcs_manager.ThreadPoolExecutor")
    def test_upload_screenshots_handles_exceptions(self, mock_executor, gcs_manager):
        """Test that upload handles exceptions from individual upload tasks."""
        mock_executor_instance = Mock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        # Mock a future that raises an exception
        future = Mock(spec=Future)
        future.result.side_effect = Exception("Upload failed")
        mock_executor_instance.submit.return_value = future

        with patch(
            "experiments.runners.services.gcs_manager.as_completed",
            return_value=[future],
        ):
            with patch("experiments.runners.services.gcs_manager.Progress"):
                tasks = [_UploadTask(local_path="/local/1.png", gcs_path="gcs/1.png")]

                # Should raise the exception from the failed upload
                with pytest.raises(Exception, match="Upload failed"):
                    gcs_manager._upload_screenshots(tasks)

    def test_screenshots_dir_calculation(self, env_vars, mock_storage_client):
        """Test that screenshots directory is calculated correctly."""
        with patch(
            "experiments.runners.services.gcs_manager.get_dataset_name",
            return_value="mousepad",
        ):
            manager = GCSManager(local_dataset_path="/path/to/mousepad_dataset.csv")

            expected_screenshots_dir = Path("/path/to/screenshots")
            assert manager.screenshots_dir == expected_screenshots_dir
