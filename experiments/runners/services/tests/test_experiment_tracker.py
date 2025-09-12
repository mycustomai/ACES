import json
import tempfile
from pathlib import Path

import pytest
from pydantic import SecretStr

from agent.src.typedefs import EngineConfigName, EngineParams, EngineType
from experiments.config import ExperimentId
from experiments.runners.batch_runtime.typedefs import (BatchStatus,
                                                        ExperimentSubmissionRecord,
                                                        ProviderBatchId)
from experiments.runners.services.experiment_tracker import ExperimentTracker


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_engine_params():
    """Create mock EngineParams for testing."""
    return [
        EngineParams(
            engine_type=EngineType.OPENAI,
            model="gpt-4",
            api_key=SecretStr("test-openai-key"),
            temperature=0.7,
        ),
        EngineParams(
            engine_type=EngineType.ANTHROPIC,
            model="claude-3",
            api_key=SecretStr("test-anthropic-key"),
            temperature=0.5,
        ),
        EngineParams(
            engine_type=EngineType.GEMINI,
            model="gemini-pro",
            api_key=SecretStr("test-gemini-key"),
            temperature=0.3,
        ),
    ]


@pytest.fixture
def sample_records(mock_engine_params):
    """Create sample ExperimentSubmissionRecord objects for testing."""
    # Use the actual config names from mock_engine_params to ensure consistency
    return [
        ExperimentSubmissionRecord(
            experiment_id=ExperimentId("exp-001"),
            batch_id=ProviderBatchId("batch-001"),
            config_name=mock_engine_params[0].config_name,  # OPENAI
        ),
        ExperimentSubmissionRecord(
            experiment_id=ExperimentId("exp-002"),
            batch_id=ProviderBatchId("batch-002"),
            config_name=mock_engine_params[1].config_name,  # ANTHROPIC
        ),
        ExperimentSubmissionRecord(
            experiment_id=ExperimentId("exp-003"),
            batch_id=ProviderBatchId("batch-003"),
            config_name=mock_engine_params[2].config_name,  # GEMINI
        ),
        ExperimentSubmissionRecord(
            experiment_id=ExperimentId("exp-004"),
            batch_id=ProviderBatchId("batch-004"),
            config_name=mock_engine_params[0].config_name,  # OPENAI again
        ),
    ]


class TestExperimentTracker:
    """Test cases for ExperimentTracker class."""

    def test_init_creates_directory_structure(self, temp_dir, mock_engine_params):
        """Test that initialization creates the correct directory structure."""
        tracker = ExperimentTracker(mock_engine_params, temp_dir)

        # Check that batch_metadata directory exists
        assert (temp_dir / "batch_metadata").exists()
        assert (temp_dir / "batch_metadata").is_dir()

        # Check that subdirectories for each engine config exist
        for engine in mock_engine_params:
            config_dir = temp_dir / "batch_metadata" / engine.config_name
            assert config_dir.exists()
            assert config_dir.is_dir()

            # Check in_progress and completed subdirectories
            assert (config_dir / "in_progress").exists()
            assert (config_dir / "in_progress").is_dir()
            assert (config_dir / "completed").exists()
            assert (config_dir / "completed").is_dir()

        # Check that completed and in_progress dicts are initialized correctly
        assert len(tracker.completed) == len(mock_engine_params)
        assert len(tracker.in_progress) == len(mock_engine_params)

        for engine in mock_engine_params:
            assert engine.config_name in tracker.completed
            assert engine.config_name in tracker.in_progress
            assert tracker.completed[engine.config_name] == []
            assert tracker.in_progress[engine.config_name] == []

    def test_set_experiments_in_progress(
        self, temp_dir, mock_engine_params, sample_records
    ):
        """Test setting experiments as in progress."""
        tracker = ExperimentTracker(mock_engine_params, temp_dir)

        record = sample_records[0]  # OPENAI_gpt-4 record
        submitted_records = {record.config_name: [record]}
        tracker.set_experiments_in_progress(submitted_records)

        # Check in-memory tracking
        assert len(tracker.in_progress[record.config_name]) == 1
        assert tracker.in_progress[record.config_name][0] == record

        # Check file persistence
        expected_file = (
            temp_dir
            / "batch_metadata"
            / record.config_name
            / "in_progress"
            / f"{record.experiment_id}.json"
        )
        assert expected_file.exists()

        # Verify file contents
        with open(expected_file, "r") as f:
            saved_data = json.load(f)
        assert saved_data["experiment_id"] == record.experiment_id
        assert saved_data["batch_id"] == record.batch_id
        assert saved_data["config_name"] == record.config_name

    def test_set_multiple_experiments_in_progress(
        self, temp_dir, mock_engine_params, sample_records
    ):
        """Test setting multiple experiments as in progress."""
        tracker = ExperimentTracker(mock_engine_params, temp_dir)

        # Group records by config name
        submitted_records = {}
        for record in sample_records[:3]:
            if record.config_name not in submitted_records:
                submitted_records[record.config_name] = []
            submitted_records[record.config_name].append(record)

        tracker.set_experiments_in_progress(submitted_records)

        # Check that records are tracked correctly
        assert len(tracker.in_progress[mock_engine_params[0].config_name]) == 1
        assert len(tracker.in_progress[mock_engine_params[1].config_name]) == 1
        assert len(tracker.in_progress[mock_engine_params[2].config_name]) == 1

        # Add another OPENAI record
        additional_records = {mock_engine_params[0].config_name: [sample_records[3]]}
        tracker.set_experiments_in_progress(additional_records)
        assert len(tracker.in_progress[mock_engine_params[0].config_name]) == 2

    def test_load_submitted_experiments_empty(self, temp_dir, mock_engine_params):
        """Test loading experiments when directories are empty."""
        tracker = ExperimentTracker(mock_engine_params, temp_dir)
        result = tracker.load_submitted_experiments()

        # Should return empty lists for each config
        assert len(result) == len(mock_engine_params)
        for engine in mock_engine_params:
            assert engine.config_name in result
            assert result[engine.config_name] == []

    def test_load_submitted_experiments_with_data(
        self, temp_dir, mock_engine_params, sample_records
    ):
        """Test loading existing experiments from disk."""
        # Manually create some record files
        batch_metadata_dir = temp_dir / "batch_metadata"
        batch_metadata_dir.mkdir()

        # Create completed record
        completed_record = sample_records[0]
        completed_dir = batch_metadata_dir / completed_record.config_name / "completed"
        completed_dir.mkdir(parents=True)
        with open(completed_dir / f"{completed_record.experiment_id}.json", "w") as f:
            json.dump(completed_record.model_dump(), f)

        # Create in-progress records
        for record in sample_records[1:3]:
            in_progress_dir = batch_metadata_dir / record.config_name / "in_progress"
            in_progress_dir.mkdir(parents=True, exist_ok=True)
            with open(in_progress_dir / f"{record.experiment_id}.json", "w") as f:
                json.dump(record.model_dump(), f)

        # Create tracker and load experiments
        tracker = ExperimentTracker(mock_engine_params, temp_dir)
        result = tracker.load_submitted_experiments()

        # Verify results
        assert len(result[mock_engine_params[0].config_name]) == 1
        assert ExperimentId("exp-001") in result[mock_engine_params[0].config_name]

        assert len(result[mock_engine_params[1].config_name]) == 1
        assert ExperimentId("exp-002") in result[mock_engine_params[1].config_name]

        assert len(result[mock_engine_params[2].config_name]) == 1
        assert ExperimentId("exp-003") in result[mock_engine_params[2].config_name]

        # Check in-memory tracking
        assert len(tracker.completed[mock_engine_params[0].config_name]) == 1
        assert len(tracker.in_progress[mock_engine_params[1].config_name]) == 1
        assert len(tracker.in_progress[mock_engine_params[2].config_name]) == 1

    def test_set_experiment_complete(
        self, temp_dir, mock_engine_params, sample_records
    ):
        """Test moving an experiment from in-progress to completed."""
        tracker = ExperimentTracker(mock_engine_params, temp_dir)

        # Set a record as in progress
        record = sample_records[0]
        submitted_records = {record.config_name: [record]}
        tracker.set_experiments_in_progress(submitted_records)

        # Verify it's in progress
        assert len(tracker.in_progress[record.config_name]) == 1
        assert len(tracker.completed[record.config_name]) == 0

        in_progress_file = (
            temp_dir
            / "batch_metadata"
            / record.config_name
            / "in_progress"
            / f"{record.experiment_id}.json"
        )
        completed_file = (
            temp_dir
            / "batch_metadata"
            / record.config_name
            / "completed"
            / f"{record.experiment_id}.json"
        )

        assert in_progress_file.exists()
        assert not completed_file.exists()

        # Move to completed
        tracker.set_experiment_complete(record.batch_id, record.config_name)

        # Verify it's completed
        assert len(tracker.in_progress[record.config_name]) == 0
        assert len(tracker.completed[record.config_name]) == 1
        assert tracker.completed[record.config_name][0] == record

        # Verify files moved
        assert not in_progress_file.exists()
        assert completed_file.exists()

        # Verify file contents
        with open(completed_file, "r") as f:
            saved_data = json.load(f)
        assert saved_data["experiment_id"] == record.experiment_id
        assert saved_data["batch_id"] == record.batch_id
        assert saved_data["config_name"] == record.config_name

    def test_set_experiment_complete_not_found(self, temp_dir, mock_engine_params):
        """Test error handling when trying to complete a non-existent record."""
        tracker = ExperimentTracker(mock_engine_params, temp_dir)

        with pytest.raises(
            ValueError,
            match="No in-progress experiments found with batch_id: nonexistent-batch",
        ):
            tracker.set_experiment_complete(
                ProviderBatchId("nonexistent-batch"), mock_engine_params[0].config_name
            )

    def test_set_experiment_complete_multiple_records(
        self, temp_dir, mock_engine_params, sample_records
    ):
        """Test completing experiments when multiple are in progress."""
        tracker = ExperimentTracker(mock_engine_params, temp_dir)

        # Set multiple records in progress
        submitted_records = {}
        for record in sample_records[:3]:
            if record.config_name not in submitted_records:
                submitted_records[record.config_name] = []
            submitted_records[record.config_name].append(record)
        tracker.set_experiments_in_progress(submitted_records)

        # Complete the second record
        tracker.set_experiment_complete(
            sample_records[1].batch_id, sample_records[1].config_name
        )

        # Verify correct record was moved
        assert len(tracker.in_progress[mock_engine_params[0].config_name]) == 1
        assert len(tracker.in_progress[mock_engine_params[1].config_name]) == 0
        assert len(tracker.in_progress[mock_engine_params[2].config_name]) == 1
        assert len(tracker.completed[mock_engine_params[1].config_name]) == 1

    def test_set_experiment_complete_file_missing(
        self, temp_dir, mock_engine_params, sample_records
    ):
        """Test completing an experiment when the file is missing (edge case)."""
        tracker = ExperimentTracker(mock_engine_params, temp_dir)

        record = sample_records[0]
        # Manually add to in-progress without creating file
        tracker.in_progress[record.config_name].append(record)

        # Should raise error because file doesn't exist
        with pytest.raises(RuntimeError, match="In-progress file does not exist"):
            tracker.set_experiment_complete(record.batch_id, record.config_name)

        # Since it raised an error, file should not exist
        completed_file = (
            temp_dir
            / "batch_metadata"
            / record.config_name
            / "completed"
            / f"{record.experiment_id}.json"
        )
        assert not completed_file.exists()

    def test_load_experiment_dir_nonexistent(self, temp_dir):
        """Test that _load_experiment_dir raises error for non-existent directory."""
        nonexistent_dir = temp_dir / "nonexistent"

        with pytest.raises(
            ValueError, match=f"Directory does not exist: {nonexistent_dir}"
        ):
            list(ExperimentTracker._load_experiment_dir(nonexistent_dir))

    def test_load_experiment_dir_with_invalid_json(self, temp_dir, mock_engine_params):
        """Test handling of invalid JSON files."""
        tracker = ExperimentTracker(mock_engine_params, temp_dir)

        # Create an invalid JSON file
        invalid_dir = (
            temp_dir
            / "batch_metadata"
            / mock_engine_params[0].config_name
            / "completed"
        )
        invalid_file = invalid_dir / "invalid.json"
        with open(invalid_file, "w") as f:
            f.write("not valid json")

        # Should raise an error when trying to load
        with pytest.raises(json.JSONDecodeError):
            tracker.load_submitted_experiments()

    def test_concurrent_operations(self, temp_dir, mock_engine_params, sample_records):
        """Test that tracker handles concurrent operations correctly."""
        tracker = ExperimentTracker(mock_engine_params, temp_dir)

        # Set multiple experiments in progress
        submitted_records = {}
        for record in sample_records:
            if record.config_name not in submitted_records:
                submitted_records[record.config_name] = []
            submitted_records[record.config_name].append(record)
        tracker.set_experiments_in_progress(submitted_records)

        # Complete them in different order
        tracker.set_experiment_complete(
            sample_records[2].batch_id, sample_records[2].config_name
        )
        tracker.set_experiment_complete(
            sample_records[0].batch_id, sample_records[0].config_name
        )
        tracker.set_experiment_complete(
            sample_records[3].batch_id, sample_records[3].config_name
        )
        tracker.set_experiment_complete(
            sample_records[1].batch_id, sample_records[1].config_name
        )

        # All should be completed
        for config_name in tracker.completed:
            assert len(tracker.in_progress[config_name]) == 0

        assert len(tracker.completed[mock_engine_params[0].config_name]) == 2
        assert len(tracker.completed[mock_engine_params[1].config_name]) == 1
        assert len(tracker.completed[mock_engine_params[2].config_name]) == 1

    def test_persistence_across_instances(
        self, temp_dir, mock_engine_params, sample_records
    ):
        """Test that data persists across different tracker instances."""
        # First tracker instance
        tracker1 = ExperimentTracker(mock_engine_params, temp_dir)

        # Set some experiments
        submitted_records = {
            sample_records[0].config_name: [sample_records[0]],
            sample_records[1].config_name: [sample_records[1]],
        }
        tracker1.set_experiments_in_progress(submitted_records)
        tracker1.set_experiment_complete(
            sample_records[0].batch_id, sample_records[0].config_name
        )

        # Create new tracker instance with same directory
        tracker2 = ExperimentTracker(mock_engine_params, temp_dir)
        result = tracker2.load_submitted_experiments()

        # Verify data was persisted and loaded correctly
        assert len(tracker2.completed[mock_engine_params[0].config_name]) == 1
        assert len(tracker2.in_progress[mock_engine_params[1].config_name]) == 1

        # Verify experiment IDs
        assert ExperimentId("exp-001") in result[mock_engine_params[0].config_name]
        assert ExperimentId("exp-002") in result[mock_engine_params[1].config_name]
