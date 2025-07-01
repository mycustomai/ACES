"""
Custom ID encoding/decoding mixin for Anthropic batch providers.

This mixin provides shared functionality for encoding experiment IDs to
Anthropic-compatible custom_ids and decoding them back without requiring
stored mappings.
"""

import hashlib

from experiments.config import ExperimentId


class EncodedExperimentIdMixin:
    """
    Mixin providing custom_id encoding/decoding for Anthropic batch API.

    Anthropic batch API requires custom_ids to be 32-character alphanumeric strings.
    This mixin provides deterministic encoding that can be reversed by recomputing
    hashes without needing to store mappings.
    """

    @classmethod
    def encode_custom_id(cls, experiment_id: ExperimentId) -> ExperimentId:
        """
        Encode experiment_id to a 32-character alphanumeric string using MD5.

        Args:
            experiment_id: The original experiment ID (e.g., "stapler_sc_price_reduce_100_cent_0")

        Returns:
            32-character hexadecimal string suitable for Anthropic custom_id

        Example:
            >>> cls.encode_custom_id(ExperimentId("stapler_sc_price_reduce_100_cent_0"))
            "a1b2c3d4e5f6789012345678901234567"
        """
        md5_hash = hashlib.md5(experiment_id.encode('utf-8')).hexdigest()
        
        encoded_id = ExperimentId(md5_hash)

        if not cls.is_encoded_experiment_id(encoded_id):
            raise ValueError(f"Invalid encoded custom_id: {encoded_id}")
        return encoded_id

    @classmethod
    def decode_custom_id(
        cls, encoded_custom_id: ExperimentId, experiment_ids: list[ExperimentId]
    ) -> ExperimentId | None:
        """
        Decode an encoded custom_id back to the original experiment_id.

        This method recomputes the MD5 hash for each experiment_id to find the match.
        Used to recover the original experiment_id without needing a stored mapping.

        Args:
            encoded_custom_id: The 32-character MD5 hash from batch results
            experiment_ids: List of possible experiment_ids to check against

        Returns:
            The original experiment_id if found, None otherwise

        Example:
            >>> cls.decode_custom_id(ExperimentId("a1b2c3d4e5f6789012345678901234567"),
            ...                      [ExperimentId("stapler_sc_price_reduce_100_cent_0"), ExperimentId("other_id")])
            "stapler_sc_price_reduce_100_cent_0"
        """
        if not cls.is_encoded_experiment_id(encoded_custom_id):
            raise ValueError(f"Invalid encoded custom_id: {encoded_custom_id}")
        
        # Create a mapping of MD5 hashes to experiment IDs
        hash_map = {
            hashlib.md5(eid.encode('utf-8')).hexdigest(): eid 
            for eid in experiment_ids
        }
        
        return hash_map.get(encoded_custom_id)

    @staticmethod
    def is_encoded_experiment_id(experiment_id: ExperimentId) -> bool:
        """Check if an experiment ID is an MD5 hash (32-character hex string)."""
        if len(experiment_id) != 32:
            return False
        
        # Check if all characters are valid hexadecimal (0-9, a-f)
        try:
            int(experiment_id, 16)
            return True
        except ValueError:
            return False

    @classmethod
    def resolve_experiment_id(
        cls, experiment_id: ExperimentId, experiment_ids: list[ExperimentId]
    ) -> ExperimentId:
        """
        Resolve an experiment ID by decoding it if encoded, otherwise return as-is.

        Args:
            experiment_id: The experiment ID to resolve (may be encoded)
            experiment_ids: List of possible experiment IDs to decode against

        Returns:
            The resolved experiment ID (decoded if it was encoded, original otherwise)
            
        Raises:
            ValueError: If the experiment ID is detected as encoded but cannot be decoded
        """
        if cls.is_encoded_experiment_id(experiment_id):
            decoded = cls.decode_custom_id(experiment_id, experiment_ids)
            if not decoded:
                raise ValueError(
                    f"Could not decode encoded ExperimentId: {experiment_id}"
                )
            return decoded
        return experiment_id
