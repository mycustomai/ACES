import pytest

from experiments.config import ExperimentId
from experiments.runners.batch_new.common.encoded_id_mixin import \
    EncodedExperimentIdMixin


class TestEncodedExperimentIdMixin:
    """Test suite for EncodedExperimentIdMixin encoding/decoding functionality."""

    def test_encode_custom_id_basic(self):
        """Test basic encoding functionality."""
        experiment_id = ExperimentId("stapler_sc_price_reduce_100_cent_0")
        encoded = EncodedExperimentIdMixin.encode_custom_id(experiment_id)

        assert isinstance(encoded, str)
        assert len(encoded) == 32
        # Check it's a valid hex string
        int(encoded, 16)  # Should not raise ValueError

    def test_encode_custom_id_deterministic(self):
        """Test that encoding is deterministic - same input produces same output."""
        experiment_id = ExperimentId("mousepad_test_1")
        encoded1 = EncodedExperimentIdMixin.encode_custom_id(experiment_id)
        encoded2 = EncodedExperimentIdMixin.encode_custom_id(experiment_id)

        assert encoded1 == encoded2

    def test_encode_custom_id_format(self):
        """Test that encoded output meets format requirements."""
        experiment_ids = [
            "short_id",
            "very_long_experiment_id_with_many_underscores_and_numbers_123",
            "mousepad_test_1",
            "stapler_sc_price_reduce_100_cent_0",
        ]

        for exp_id in experiment_ids:
            encoded = EncodedExperimentIdMixin.encode_custom_id(ExperimentId(exp_id))

            # Check format requirements
            assert len(encoded) == 32, f"Length {len(encoded)} != 32 for {exp_id}"
            # Check it's a valid hex string
            int(encoded, 16)  # Should not raise ValueError

    def test_encode_custom_id_different_inputs(self):
        """Test that different inputs produce different encoded outputs."""
        experiment_ids = [
            "mousepad_test_1",
            "mousepad_test_2",
            "mousepad_control_1",
            "stapler_sc_price_reduce_100_cent_0",
        ]

        encoded_ids = [
            EncodedExperimentIdMixin.encode_custom_id(ExperimentId(exp_id))
            for exp_id in experiment_ids
        ]

        # All encoded IDs should be unique
        assert len(set(encoded_ids)) == len(encoded_ids)

    def test_encode_custom_id_edge_cases(self):
        """Test encoding with edge cases."""
        edge_cases = [
            "a",  # Very short
            "experiment_with_special_chars_123",
            "UPPERCASE_EXPERIMENT_ID",
            "mixed_Case_Experiment_ID",
        ]

        for exp_id in edge_cases:
            encoded = EncodedExperimentIdMixin.encode_custom_id(ExperimentId(exp_id))
            assert len(encoded) == 32
            # Check it's a valid hex string
            int(encoded, 16)  # Should not raise ValueError

    def test_decode_custom_id_success(self):
        """Test successful decoding with matching experiment."""
        experiment_ids = [
            "mousepad_test_1",
            "mousepad_test_2",
            "stapler_sc_price_reduce_100_cent_0",
        ]

        for original_id in experiment_ids:
            encoded = EncodedExperimentIdMixin.encode_custom_id(
                ExperimentId(original_id)
            )
            decoded = EncodedExperimentIdMixin.decode_custom_id(encoded, experiment_ids)

            assert decoded == original_id

    def test_decode_custom_id_no_match(self):
        """Test that decode returns None when no match found."""
        experiment_ids = ["mousepad_test_1", "mousepad_test_2"]

        # Create encoded ID for an experiment not in the list
        other_id = ExperimentId("nonexistent_experiment")
        encoded = EncodedExperimentIdMixin.encode_custom_id(other_id)

        decoded = EncodedExperimentIdMixin.decode_custom_id(encoded, experiment_ids)
        assert decoded is None

    def test_decode_custom_id_empty_list(self):
        """Test decode with empty experiment_ids list."""
        encoded = EncodedExperimentIdMixin.encode_custom_id(
            ExperimentId("some_experiment")
        )
        decoded = EncodedExperimentIdMixin.decode_custom_id(encoded, [])

        assert decoded is None

    def test_encode_decode_roundtrip(self):
        """Test that encode followed by decode returns original for multiple IDs."""
        experiment_ids = [
            "mousepad_test_1",
            "mousepad_test_2",
            "mousepad_control_1",
            "stapler_sc_price_reduce_100_cent_0",
            "keyboard_bias_experiment_5",
            "short_id",
            "very_long_experiment_id_with_many_parts_and_numbers_999",
        ]

        for original_id in experiment_ids:
            # Encode
            encoded = EncodedExperimentIdMixin.encode_custom_id(
                ExperimentId(original_id)
            )

            # Decode (need to provide the original ID in the list for lookup)
            decoded = EncodedExperimentIdMixin.decode_custom_id(encoded, experiment_ids)

            # Should recover the original
            assert decoded == original_id

    def test_decode_custom_id_multiple_candidates(self):
        """Test decode works correctly when multiple experiment IDs are provided."""
        experiment_ids = [
            "mousepad_test_1",
            "mousepad_test_2",
            "mousepad_control_1",
            "stapler_sc_price_reduce_100_cent_0",
        ]

        # Test decoding for each ID
        for target_id in experiment_ids:
            encoded = EncodedExperimentIdMixin.encode_custom_id(ExperimentId(target_id))
            decoded = EncodedExperimentIdMixin.decode_custom_id(encoded, experiment_ids)

            assert decoded == target_id

    def test_decode_custom_id_hash_collision_resistance(self):
        """Test that hash collisions are handled correctly."""
        # Create a large set of similar experiment IDs
        experiment_ids = [f"mousepad_test_{i}" for i in range(100)]

        # Test that each can be encoded and decoded correctly
        for original_id in experiment_ids[:10]:  # Test first 10 to keep test reasonable
            encoded = EncodedExperimentIdMixin.encode_custom_id(
                ExperimentId(original_id)
            )
            decoded = EncodedExperimentIdMixin.decode_custom_id(encoded, experiment_ids)

            assert decoded == original_id

    def test_encode_custom_id_hash_length_validation(self):
        """Test that MD5 always produces 32-character output."""
        # Test with various experiment IDs - MD5 always produces 32 chars
        test_ids = [
            "a" * 10,  # Short repeated chars
            "z" * 50,  # Long repeated chars
            "experiment_" + "x" * 100,  # Very long ID
            "1" * 20,  # Numeric
            "special_chars_!@#$%^&*()",  # Special chars (though not used in practice)
        ]

        for exp_id in test_ids:
            encoded = EncodedExperimentIdMixin.encode_custom_id(
                ExperimentId(exp_id)
            )
            assert len(encoded) == 32
            # Check it's a valid hex string
            int(encoded, 16)  # Should not raise ValueError

    def test_encode_custom_id_validation(self):
        """Test that encode_custom_id validates its output."""
        # Normal cases should work fine
        experiment_id = ExperimentId("test_experiment")
        encoded = EncodedExperimentIdMixin.encode_custom_id(experiment_id)

        # Should be valid according to is_encoded_experiment_id
        assert EncodedExperimentIdMixin.is_encoded_experiment_id(encoded)

    def test_decode_custom_id_validation(self):
        """Test that decode_custom_id validates its input."""
        experiment_ids = ["test_1", "test_2"]

        # Valid encoded ID should work
        valid_encoded = EncodedExperimentIdMixin.encode_custom_id(
            ExperimentId("test_1")
        )
        result = EncodedExperimentIdMixin.decode_custom_id(
            valid_encoded, experiment_ids
        )
        assert result == "test_1"

        # Invalid encoded ID should raise ValueError
        invalid_encoded = ExperimentId("invalid_encoded_id")
        with pytest.raises(ValueError, match="Invalid encoded custom_id"):
            EncodedExperimentIdMixin.decode_custom_id(invalid_encoded, experiment_ids)

    def test_is_encoded_experiment_id(self):
        """Test the is_encoded_experiment_id validation method."""
        # Valid encoded IDs (generated by encode_custom_id)
        valid_ids = [
            EncodedExperimentIdMixin.encode_custom_id(ExperimentId("test_1")),
            EncodedExperimentIdMixin.encode_custom_id(ExperimentId("another_test")),
        ]

        for valid_id in valid_ids:
            assert EncodedExperimentIdMixin.is_encoded_experiment_id(valid_id)

        # Invalid encoded IDs
        invalid_ids = [
            ExperimentId("1234567890123456789012345678901"),  # 31 chars
            ExperimentId("123456789012345678901234567890123"),  # 33 chars
            ExperimentId("E"),  # Too short
            ExperimentId(""),  # Empty string
            ExperimentId(
                "g123456789012345678901234567890f"
            ),  # Contains invalid hex chars (g)
            ExperimentId(
                "GHIJKLMNOPQRSTUVWXYZ123456789012"
            ),  # Contains non-hex chars
            ExperimentId("normal_experiment_id"),  # Normal experiment ID
        ]

        for invalid_id in invalid_ids:
            assert not EncodedExperimentIdMixin.is_encoded_experiment_id(invalid_id)

    def test_decode_custom_id_hash_map_performance(self):
        """Test that decode_custom_id uses hash map for better performance."""
        # Create a large list of experiment IDs
        experiment_ids = [f"experiment_{i}" for i in range(1000)]

        # Test that decoding still works efficiently
        target_id = "experiment_500"
        encoded = EncodedExperimentIdMixin.encode_custom_id(ExperimentId(target_id))
        decoded = EncodedExperimentIdMixin.decode_custom_id(encoded, experiment_ids)

        assert decoded == target_id

    def test_md5_deterministic_behavior(self):
        """Test that MD5 hashing is deterministic and always produces valid hex."""
        # Test with various IDs to ensure MD5 behavior is consistent
        test_ids = [
            "test_hash_1",
            "test_hash_2",
            "aaaaaaaa",  # Short string
            "zzzzzzzzz",  # Another string pattern
        ]

        for exp_id in test_ids:
            experiment_id = ExperimentId(exp_id)
            encoded = EncodedExperimentIdMixin.encode_custom_id(experiment_id)

            # Should always be valid hex
            assert EncodedExperimentIdMixin.is_encoded_experiment_id(encoded)
            
            # Should only contain hex characters (0-9, a-f)
            assert all(c in "0123456789abcdef" for c in encoded)

            # Should decode correctly
            decoded = EncodedExperimentIdMixin.decode_custom_id(encoded, [exp_id])
            assert decoded == exp_id

    def test_error_handling_consistency(self):
        """Test consistent error handling across methods."""
        # MD5 handles any length input, so very long IDs should work fine
        very_long_id = ExperimentId("x" * 1000)
        encoded = EncodedExperimentIdMixin.encode_custom_id(very_long_id)
        # Encoding should always succeed with MD5
        assert len(encoded) == 32
        # Decoding should work
        decoded = EncodedExperimentIdMixin.decode_custom_id(encoded, ["x" * 1000])
        assert decoded == "x" * 1000
    
    def test_resolve_experiment_id(self):
        """Test the resolve_experiment_id method."""
        experiment_ids = [
            "mousepad_test_1",
            "mousepad_test_2", 
            "stapler_sc_price_reduce_100_cent_0"
        ]
        
        # Test with encoded ID
        original_id = "mousepad_test_1"
        encoded = EncodedExperimentIdMixin.encode_custom_id(ExperimentId(original_id))
        resolved = EncodedExperimentIdMixin.resolve_experiment_id(encoded, experiment_ids)
        assert resolved == original_id
        
        # Test with non-encoded ID
        non_encoded = ExperimentId("mousepad_test_2")
        resolved = EncodedExperimentIdMixin.resolve_experiment_id(non_encoded, experiment_ids)
        assert resolved == non_encoded
        
        # Test with encoded ID not in list (should raise ValueError)
        unknown_encoded = EncodedExperimentIdMixin.encode_custom_id(ExperimentId("unknown_experiment"))
        with pytest.raises(ValueError, match="Could not decode encoded ExperimentId"):
            EncodedExperimentIdMixin.resolve_experiment_id(unknown_encoded, experiment_ids)
