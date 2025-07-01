from unittest.mock import Mock, patch

import pandas as pd
import pytest

from agent.src.typedefs import EngineConfigName, EngineParams, EngineType
from experiments.config import ExperimentData, ExperimentId
from experiments.runners.batch_new.common.encoded_id_mixin import \
    EncodedExperimentIdMixin
from experiments.runners.services.experiment_loader import ExperimentLoader


class TestExperimentLoader:
    @pytest.fixture
    def mock_engine_params(self):
        engine1 = Mock(spec=EngineParams)
        engine1.engine_type = EngineType.OPENAI
        engine1.model = "gpt-4"
        engine1.config_name = EngineConfigName("openai_gpt-4")

        engine2 = Mock(spec=EngineParams)
        engine2.engine_type = EngineType.ANTHROPIC
        engine2.model = "claude-3"
        engine2.config_name = EngineConfigName("anthropic_claude-3")

        return [engine1, engine2]

    @pytest.fixture
    def mock_experiment_data(self):
        # Create real ExperimentData objects with minimal required data
        mock_df = pd.DataFrame(
            {
                "product_name": ["Product A", "Product B", "Product C"],
                "price": [10.0, 20.0, 30.0],
            }
        )

        exp1 = ExperimentData(
            experiment_label="test",
            experiment_number=1,
            experiment_df=mock_df,
            query="mousepad",
            dataset_name="mousepad",
            prompt_template="Test prompt",
        )

        exp2 = ExperimentData(
            experiment_label="test",
            experiment_number=2,
            experiment_df=mock_df,
            query="mousepad",
            dataset_name="mousepad",
            prompt_template="Test prompt",
        )

        exp3 = ExperimentData(
            experiment_label="control",
            experiment_number=1,
            experiment_df=mock_df,
            query="mousepad",
            dataset_name="mousepad",
            prompt_template="Test prompt",
        )

        return [exp1, exp2, exp3]

    @pytest.fixture
    def experiment_loader(self, mock_engine_params):
        with patch(
            "experiments.runners.services.experiment_loader.ExperimentLoader._load_local_dataset"
        ) as mock_load:
            mock_load.return_value = []
            return ExperimentLoader(
                local_dataset_path="/path/to/mousepad_dataset.csv",
                engine_params=mock_engine_params,
            )

    @patch(
        "experiments.runners.services.experiment_loader.ExperimentLoader._load_local_dataset"
    )
    def test_init(self, mock_load_local_dataset, mock_engine_params):
        mock_load_local_dataset.return_value = []
        loader = ExperimentLoader(
            local_dataset_path="/path/to/dataset.csv", engine_params=mock_engine_params
        )
        assert loader.local_dataset_path == "/path/to/dataset.csv"
        assert loader.engine_params == mock_engine_params
        assert isinstance(loader.experiments, set)
        mock_load_local_dataset.assert_called_once()

    @patch("experiments.runners.services.experiment_loader.pd.read_csv")
    @patch("experiments.runners.services.experiment_loader.get_dataset_name")
    @patch("experiments.runners.services.experiment_loader.experiments_iter")
    def test_load_local_dataset(
        self,
        mock_experiments_iter,
        mock_get_dataset_name,
        mock_read_csv,
        experiment_loader,
        mock_experiment_data,
    ):
        mock_df = pd.DataFrame()
        mock_read_csv.return_value = mock_df
        mock_get_dataset_name.return_value = "mousepad"
        mock_experiments_iter.return_value = mock_experiment_data

        result = experiment_loader._load_local_dataset()

        mock_read_csv.assert_called_once_with("/path/to/mousepad_dataset.csv")
        mock_get_dataset_name.assert_called_once_with("/path/to/mousepad_dataset.csv")
        mock_experiments_iter.assert_called_once_with(mock_df, "mousepad")
        assert result == mock_experiment_data

    def test_load_outstanding_experiments_no_existing(
        self, experiment_loader, mock_experiment_data
    ):
        experiment_loader.experiments = set(mock_experiment_data)
        existing_experiments = {}

        result = experiment_loader.load_outstanding_experiments(existing_experiments)

        assert len(result) == 2
        assert len(result[EngineConfigName("openai_gpt-4")]) == 3
        assert len(result[EngineConfigName("anthropic_claude-3")]) == 3
        assert set(result[EngineConfigName("openai_gpt-4")]) == set(
            mock_experiment_data
        )
        assert set(result[EngineConfigName("anthropic_claude-3")]) == set(
            mock_experiment_data
        )

    def test_load_outstanding_experiments_with_some_existing(
        self, experiment_loader, mock_experiment_data
    ):
        experiment_loader.experiments = set(mock_experiment_data)
        existing_experiments = {
            EngineConfigName("openai_gpt-4"): [
                ExperimentId("mousepad_test_1"),
                ExperimentId("mousepad_control_1"),
            ]
        }

        result = experiment_loader.load_outstanding_experiments(existing_experiments)

        assert len(result) == 2
        assert len(result[EngineConfigName("openai_gpt-4")]) == 1
        assert result[EngineConfigName("openai_gpt-4")][
            0
        ].experiment_id == ExperimentId("mousepad_test_2")
        assert len(result[EngineConfigName("anthropic_claude-3")]) == 3
        assert set(result[EngineConfigName("anthropic_claude-3")]) == set(
            mock_experiment_data
        )

    def test_load_outstanding_experiments_all_existing(
        self, experiment_loader, mock_experiment_data
    ):
        experiment_loader.experiments = set(mock_experiment_data)
        existing_experiments = {
            EngineConfigName("openai_gpt-4"): [
                ExperimentId("mousepad_test_1"),
                ExperimentId("mousepad_test_2"),
                ExperimentId("mousepad_control_1"),
            ],
            EngineConfigName("anthropic_claude-3"): [
                ExperimentId("mousepad_test_1"),
                ExperimentId("mousepad_test_2"),
                ExperimentId("mousepad_control_1"),
            ],
        }

        result = experiment_loader.load_outstanding_experiments(existing_experiments)

        assert len(result) == 2
        assert len(result[EngineConfigName("openai_gpt-4")]) == 0
        assert len(result[EngineConfigName("anthropic_claude-3")]) == 0

    def test_load_outstanding_experiments_mixed_existing(
        self, experiment_loader, mock_experiment_data
    ):
        experiment_loader.experiments = set(mock_experiment_data)
        existing_experiments = {
            EngineConfigName("openai_gpt-4"): [ExperimentId("mousepad_test_1")]
        }

        result = experiment_loader.load_outstanding_experiments(existing_experiments)

        assert len(result) == 2
        assert len(result[EngineConfigName("openai_gpt-4")]) == 2
        outstanding_ids = [
            exp.experiment_id for exp in result[EngineConfigName("openai_gpt-4")]
        ]
        assert ExperimentId("mousepad_test_2") in outstanding_ids
        assert ExperimentId("mousepad_control_1") in outstanding_ids
        assert ExperimentId("mousepad_test_1") not in outstanding_ids
        assert len(result[EngineConfigName("anthropic_claude-3")]) == 3
        assert set(result[EngineConfigName("anthropic_claude-3")]) == set(
            mock_experiment_data
        )

    def test_hash_optimization_allows_set_operations(self, mock_experiment_data):
        """Test that ExperimentData objects can be used in sets due to __hash__ implementation."""
        exp_set = set(mock_experiment_data)
        assert len(exp_set) == 3

        # Test that we can check membership efficiently
        exp1 = mock_experiment_data[0]
        assert exp1 in exp_set

        # Test that duplicate experiment_ids are handled correctly
        mock_df = pd.DataFrame({"product_name": ["Product A"], "price": [10.0]})

        duplicate_exp = ExperimentData(
            experiment_label="test",
            experiment_number=1,
            experiment_df=mock_df,
            query="mousepad",
            dataset_name="mousepad",
            prompt_template="Test prompt",
        )

        exp_set.add(duplicate_exp)
        assert len(exp_set) == 3  # Should still be 3 due to same experiment_id

    def test_load_outstanding_experiments_empty_existing_for_engine(
        self, experiment_loader, mock_experiment_data
    ):
        """Test when some engines have no existing experiments."""
        experiment_loader.experiments = set(mock_experiment_data)
        existing_experiments = {
            EngineConfigName("openai_gpt-4"): [],
            EngineConfigName("anthropic_claude-3"): [ExperimentId("mousepad_test_1")],
        }

        result = experiment_loader.load_outstanding_experiments(existing_experiments)

        assert len(result) == 2
        assert len(result[EngineConfigName("openai_gpt-4")]) == 3
        assert len(result[EngineConfigName("anthropic_claude-3")]) == 2

        # Verify the correct experiments are filtered out
        anthropic_ids = [
            exp.experiment_id for exp in result[EngineConfigName("anthropic_claude-3")]
        ]
        assert ExperimentId("mousepad_test_1") not in anthropic_ids
        assert ExperimentId("mousepad_test_2") in anthropic_ids
        assert ExperimentId("mousepad_control_1") in anthropic_ids

    def test_load_outstanding_experiments_nonexistent_engine_in_existing(
        self, experiment_loader, mock_experiment_data
    ):
        """Test when existing experiments contain engines not in engine_params."""
        experiment_loader.experiments = set(mock_experiment_data)
        existing_experiments = {
            EngineConfigName("openai_gpt-4"): [ExperimentId("mousepad_test_1")],
            EngineConfigName("nonexistent_engine"): [ExperimentId("mousepad_test_2")],
        }

        result = experiment_loader.load_outstanding_experiments(existing_experiments)

        # Should only return results for engines in engine_params
        assert len(result) == 2
        assert EngineConfigName("openai_gpt-4") in result
        assert EngineConfigName("anthropic_claude-3") in result
        assert EngineConfigName("nonexistent_engine") not in result

    def test_get_experiment_by_id(self, experiment_loader, mock_experiment_data):
        """Test that get_experiment_by_id returns the correct experiment."""
        experiment_loader.experiments = set(mock_experiment_data)

        # Test finding existing experiment
        result = experiment_loader.get_experiment_by_id(ExperimentId("mousepad_test_1"))
        assert result.experiment_id == ExperimentId("mousepad_test_1")
        assert result.experiment_label == "test"
        assert result.experiment_number == 1

        # Test finding another experiment
        result = experiment_loader.get_experiment_by_id(
            ExperimentId("mousepad_control_1")
        )
        assert result.experiment_id == ExperimentId("mousepad_control_1")
        assert result.experiment_label == "control"
        assert result.experiment_number == 1

    def test_get_experiment_by_id_not_found(
        self, experiment_loader, mock_experiment_data
    ):
        """Test that get_experiment_by_id raises appropriate exception when experiment not found."""
        experiment_loader.experiments = set(mock_experiment_data)

        with pytest.raises(
            KeyError, match="No experiment found with id 'nonexistent_experiment'"
        ):
            experiment_loader.get_experiment_by_id(
                ExperimentId("nonexistent_experiment")
            )

    def test_experiments_property_initialization(
        self, mock_engine_params, mock_experiment_data
    ):
        """Test that experiments property is properly initialized during __init__."""
        with patch(
            "experiments.runners.services.experiment_loader.ExperimentLoader._load_local_dataset"
        ) as mock_load:
            mock_load.return_value = mock_experiment_data

            loader = ExperimentLoader(
                local_dataset_path="/path/to/dataset.csv",
                engine_params=mock_engine_params,
            )

            # Test that experiments is a set
            assert isinstance(loader.experiments, set)
            # Test that it contains the loaded experiments
            assert len(loader.experiments) == 3
            assert loader.experiments == set(mock_experiment_data)

    def test_experiments_set_behavior(self, experiment_loader, mock_experiment_data):
        """Test that experiments property behaves as a proper set."""
        experiment_loader.experiments = set(mock_experiment_data)

        # Test set membership
        exp1 = mock_experiment_data[0]
        assert exp1 in experiment_loader.experiments

        # Test set size
        assert len(experiment_loader.experiments) == 3

        # Test that adding duplicate doesn't change size
        original_size = len(experiment_loader.experiments)
        experiment_loader.experiments.add(exp1)
        assert len(experiment_loader.experiments) == original_size

        # Test set operations work
        other_set = {mock_experiment_data[0], mock_experiment_data[1]}
        intersection = experiment_loader.experiments.intersection(other_set)
        assert len(intersection) == 2

    def test_mixin_static_method_access(self, experiment_loader):
        """Test that ExperimentLoader can access mixin static methods."""
        # Test that we can access is_encoded_experiment_id as static method
        assert hasattr(EncodedExperimentIdMixin, "is_encoded_experiment_id")
        assert callable(EncodedExperimentIdMixin.is_encoded_experiment_id)

        # Test with some sample IDs
        normal_id = ExperimentId("mousepad_test_1")
        encoded_id = EncodedExperimentIdMixin.encode_custom_id(normal_id)

        # Normal ID should not be encoded
        assert not EncodedExperimentIdMixin.is_encoded_experiment_id(normal_id)

        # Encoded ID should be detected as encoded
        assert EncodedExperimentIdMixin.is_encoded_experiment_id(encoded_id)

    def test_get_experiment_by_id_encoded(
        self, experiment_loader, mock_experiment_data
    ):
        """Test retrieval using encoded experiment ID."""
        experiment_loader.experiments = set(mock_experiment_data)

        # Test with each experiment
        for exp_data in mock_experiment_data:
            # Encode the experiment ID
            encoded_id = EncodedExperimentIdMixin.encode_custom_id(
                exp_data.experiment_id
            )

            # Retrieve using encoded ID
            result = experiment_loader.get_experiment_by_id(encoded_id)

            # Should return the correct experiment
            assert result == exp_data
            assert result.experiment_id == exp_data.experiment_id

    def test_get_experiment_by_id_encoded_not_found(
        self, experiment_loader, mock_experiment_data
    ):
        """Test encoded ID that doesn't match any experiment."""
        experiment_loader.experiments = set(mock_experiment_data)

        # Create encoded ID for non-existent experiment
        nonexistent_id = ExperimentId("nonexistent_experiment_id")
        encoded_id = EncodedExperimentIdMixin.encode_custom_id(nonexistent_id)
        
        # Should raise ValueError when encoded ID cannot be decoded
        with pytest.raises(
                ValueError, match=f"Could not decode encoded ExperimentId: {encoded_id}"
        ):
            experiment_loader.get_experiment_by_id(encoded_id)

    def test_get_experiment_by_id_backward_compatibility(
        self, experiment_loader, mock_experiment_data
    ):
        """Test that normal experiment IDs still work after refactoring."""
        experiment_loader.experiments = set(mock_experiment_data)

        # Test that all normal IDs still work
        for exp_data in mock_experiment_data:
            result = experiment_loader.get_experiment_by_id(exp_data.experiment_id)
            assert result == exp_data
            assert result.experiment_id == exp_data.experiment_id

    def test_get_experiment_by_id_mixed_scenarios(
        self, experiment_loader, mock_experiment_data
    ):
        """Test mixed scenarios with both encoded and normal IDs."""
        experiment_loader.experiments = set(mock_experiment_data)

        # Test normal ID retrieval
        exp1 = mock_experiment_data[0]
        result1 = experiment_loader.get_experiment_by_id(exp1.experiment_id)
        assert result1 == exp1

        # Test encoded ID retrieval for same experiment
        encoded_id = EncodedExperimentIdMixin.encode_custom_id(exp1.experiment_id)
        result2 = experiment_loader.get_experiment_by_id(encoded_id)
        assert result2 == exp1

        # Both should return the same experiment
        assert result1 == result2

    def test_get_experiment_by_id_encoded_with_mocking(
        self, experiment_loader, mock_experiment_data
    ):
        """Test encoded ID retrieval with mocked mixin methods."""
        experiment_loader.experiments = set(mock_experiment_data)

        # Mock the mixin static methods to ensure they're called correctly
        with patch.object(
            EncodedExperimentIdMixin, "is_encoded_experiment_id", return_value=True
        ) as mock_is_encoded:
            with patch.object(
                EncodedExperimentIdMixin,
                "decode_custom_id",
                return_value="mousepad_test_1",
            ) as mock_decode:
                encoded_id = EncodedExperimentIdMixin.encode_custom_id(
                    ExperimentId("test_experiment")
                )
                result = experiment_loader.get_experiment_by_id(encoded_id)

                # Verify mixin methods were called
                # Note: is_encoded_experiment_id may be called multiple times (validation + lookup)
                assert mock_is_encoded.call_count >= 1
                mock_decode.assert_called_once()

                # Verify correct experiment returned
                assert result.experiment_id == ExperimentId("mousepad_test_1")

    def test_inheritance_from_anthropic_custom_id_mixin(self, experiment_loader):
        """Test that ExperimentLoader properly inherits from EncodedExperimentIdMixin."""
        # ExperimentLoader should be an instance of EncodedExperimentIdMixin
        assert isinstance(experiment_loader, EncodedExperimentIdMixin)

        # Should have access to mixin methods
        assert hasattr(experiment_loader, "encode_custom_id")
        assert hasattr(experiment_loader, "decode_custom_id")
        assert hasattr(experiment_loader, "is_encoded_experiment_id")
        assert callable(experiment_loader.encode_custom_id)
        assert callable(experiment_loader.decode_custom_id)
        assert callable(experiment_loader.is_encoded_experiment_id)

    def test_get_experiment_by_id_encoded_integration(
        self, experiment_loader, mock_experiment_data
    ):
        """Integration test using real mixin methods for encoding/decoding."""
        experiment_loader.experiments = set(mock_experiment_data)

        # Test full integration without mocking
        for exp_data in mock_experiment_data:
            # Use real mixin method to encode
            encoded_id = experiment_loader.encode_custom_id(exp_data.experiment_id)

            # Retrieve using encoded ID
            result = experiment_loader.get_experiment_by_id(encoded_id)

            # Should return the correct experiment
            assert result == exp_data
            assert result.experiment_id == exp_data.experiment_id
            assert result.experiment_label == exp_data.experiment_label
            assert result.experiment_number == exp_data.experiment_number
