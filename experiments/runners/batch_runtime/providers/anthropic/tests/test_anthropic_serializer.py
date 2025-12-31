"""
Tests for Anthropic batch provider serializer.

This test suite verifies that our serializer produces batch requests that comply with
the official Anthropic batch API specification as documented at:
https://docs.anthropic.com/en/docs/build-with-claude/message-batches
https://docs.anthropic.com/en/docs/tool-use

Key requirements tested:
1. Batch request structure with custom_id and params
2. Custom ID uniqueness and format
3. Message format compliance
4. Tool definition schema compliance
5. Parameter validation
"""

import json

import pandas as pd
import pytest

from agent.src.core.tools import AddToCartTool
from common.messages import RawMessageExchange
from experiments.config import ExperimentData
from experiments.runners.batch_runtime.providers.anthropic.serializer import \
    AnthropicBatchProviderSerializer
from experiments.runners.batch_runtime.typedefs import (BatchRequest,
                                                        SerializedBatchRequest)


@pytest.fixture
def anthropic_serializer(mock_anthropic_params):
    """Create serializer using shared mock_anthropic_params fixture."""
    return AnthropicBatchProviderSerializer(mock_anthropic_params)


class TestAnthropicBatchProviderSerializer:
    """Test Anthropic serializer implementation."""

    def test_serialize_basic_request(self, anthropic_serializer, mock_anthropic_params):
        """Test basic serialization without tools."""
        experiments = [
            ExperimentData(
                experiment_label="sc_price_reduce_100_cent",
                experiment_number=0,
                experiment_df=pd.DataFrame(
                    {"product": ["stapler"], "price": [9.00], "asin": ["B001234567"]}
                ),
                query="stapler",
                dataset_name="test_dataset",
                prompt_template="You are a helpful shopping assistant.",
            )
        ]

        raw_messages = [
            RawMessageExchange(
                [
                    {
                        "role": "system",
                        "content": "You are a helpful shopping assistant.",
                    },
                    {"role": "user", "content": "Find me a stapler"},
                ]
            )
        ]

        batch_request = BatchRequest(
            experiments=experiments,
            raw_messages=raw_messages,
            engine_params=mock_anthropic_params,
        )

        result = anthropic_serializer.serialize(batch_request)

        assert len(result) == 1
        assert isinstance(result[0], SerializedBatchRequest)
        assert result[0].experiment_id == experiments[0].experiment_id

        provider_request = result[0].provider_request
        assert "custom_id" in provider_request
        assert "params" in provider_request

        # Check custom_id encoding
        custom_id = provider_request["custom_id"]
        assert len(custom_id) == 32
        # Allow alphanumeric characters and hyphens for negative hash values
        assert all(c.isalnum() or c == "-" for c in custom_id)

        # Check params structure
        params = provider_request["params"]
        assert params["model"] == "claude-3-5-sonnet-20241022"
        assert params.get("temperature", 0.0) == 0.0
        assert params["max_tokens"] == 1000
        assert params["system"] == "You are a helpful shopping assistant."
        assert len(params["messages"]) == 1
        assert params["messages"][0]["role"] == "user"
        assert params["messages"][0]["content"] == "Find me a stapler"

    def test_serialize_with_tools(self, anthropic_serializer, mock_anthropic_params):
        """Test serialization with tool conversion."""
        experiments = [
            ExperimentData(
                experiment_label="bias_expensive",
                experiment_number=0,
                experiment_df=pd.DataFrame(
                    {"product": ["notebook"], "price": [4.50], "asin": ["B987654321"]}
                ),
                query="notebook",
                dataset_name="test_dataset",
            )
        ]

        raw_messages = [
            RawMessageExchange([{"role": "user", "content": "Buy a notebook"}])
        ]

        batch_request = BatchRequest(
            experiments=experiments,
            raw_messages=raw_messages,
            engine_params=mock_anthropic_params,
            tools=[AddToCartTool()],
        )

        result = anthropic_serializer.serialize(batch_request)

        assert len(result) == 1
        provider_request = result[0].provider_request
        params = provider_request["params"]

        # Check tools were converted
        assert "tools" in params
        assert len(params["tools"]) == 1

        tool = params["tools"][0]
        assert tool["name"] == "add_to_cart"
        assert "description" in tool
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"
        assert "properties" in tool["input_schema"]
        assert "product_title" in tool["input_schema"]["properties"]
        assert "price" in tool["input_schema"]["properties"]
        assert "rating" in tool["input_schema"]["properties"]
        assert "number_of_reviews" in tool["input_schema"]["properties"]

    def test_custom_id_encoding_decoding(self, anthropic_serializer, mock_anthropic_params):
        """Test custom_id encoding is deterministic and reversible."""
        experiment_ids = [
            "stapler_sc_price_reduce_100_cent_0",
            "stapler_sc_price_reduce_100_cent_1",
            "notebook_bias_expensive_0",
            "short_id",
            "very_long_experiment_id_that_exceeds_32_characters_0",
        ]

        # Test encoding
        encoded_ids = {}
        for exp_id in experiment_ids:
            encoded = AnthropicBatchProviderSerializer.encode_custom_id(exp_id)
            encoded_ids[exp_id] = encoded

            # Verify properties
            assert len(encoded) == 32
            # Allow alphanumeric characters and hyphens for negative hash values
            assert all(c.isalnum() or c == "-" for c in encoded)

        # Verify uniqueness
        unique_encoded = set(encoded_ids.values())
        assert len(unique_encoded) == len(experiment_ids)

        # Test decoding (now using deserializer)
        from experiments.runners.batch_runtime.providers.anthropic.deserializer import \
            AnthropicBatchProviderDeserializer

        for original_id, encoded_id in encoded_ids.items():
            decoded = AnthropicBatchProviderDeserializer.decode_custom_id(
                encoded_id, experiment_ids
            )
            assert decoded == original_id

        # Test unknown custom_id (using proper format but unknown hash)
        unknown = "E999999999999999999999999999999E"
        decoded = AnthropicBatchProviderDeserializer.decode_custom_id(
            unknown, experiment_ids
        )
        assert decoded is None

        # Test determinism
        test_id = "test_determinism"
        encoded1 = AnthropicBatchProviderSerializer.encode_custom_id(test_id)
        encoded2 = AnthropicBatchProviderSerializer.encode_custom_id(test_id)
        assert encoded1 == encoded2

    def test_serialize_multiple_experiments(self, anthropic_serializer, mock_anthropic_params):
        """Test serializing multiple experiments."""
        experiments = [
            ExperimentData(
                experiment_label="test",
                experiment_number=i,
                experiment_df=pd.DataFrame(
                    {"product": [f"item{i}"], "price": [i + 0.50], "asin": [f"B00{i}"]}
                ),
                query=f"item{i}",
                dataset_name="test_dataset",
            )
            for i in range(3)
        ]

        raw_messages = [
            RawMessageExchange([{"role": "user", "content": f"Buy item {i}"}])
            for i in range(3)
        ]

        batch_request = BatchRequest(
            experiments=experiments,
            raw_messages=raw_messages,
            engine_params=mock_anthropic_params,
        )

        result = anthropic_serializer.serialize(batch_request)

        assert len(result) == 3

        # Verify each experiment is properly serialized
        for i, serialized in enumerate(result):
            assert serialized.experiment_id == experiments[i].experiment_id
            provider_request = serialized.provider_request
            assert len(provider_request["custom_id"]) == 32
            assert (
                provider_request["params"]["messages"][0]["content"] == f"Buy item {i}"
            )

    def test_system_message_handling(self, anthropic_serializer, mock_anthropic_params):
        """Test that system messages are properly extracted and placed."""
        experiments = [
            ExperimentData(
                experiment_label="system_msg_test",
                experiment_number=0,
                experiment_df=pd.DataFrame(
                    {"product": ["test item"], "price": [0.90], "asin": ["B123"]}
                ),
                query="test",
                dataset_name="test_dataset",
            )
        ]

        # Test with system message first
        raw_messages = [
            RawMessageExchange(
                [
                    {"role": "system", "content": "You are a shopping assistant."},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "Find me something"},
                ]
            )
        ]

        batch_request = BatchRequest(
            experiments=experiments,
            raw_messages=raw_messages,
            engine_params=mock_anthropic_params,
        )

        result = anthropic_serializer.serialize(batch_request)
        params = result[0].provider_request["params"]

        # System message should be in params["system"]
        assert params["system"] == "You are a shopping assistant."

        # Messages should not include system message
        assert len(params["messages"]) == 3
        assert all(msg["role"] != "system" for msg in params["messages"])
        assert params["messages"][0]["role"] == "user"
        assert params["messages"][1]["role"] == "assistant"
        assert params["messages"][2]["role"] == "user"

    def test_batch_request_structure_compliance(self, anthropic_serializer, mock_anthropic_params):
        """
        Test that batch requests match the official Anthropic batch API structure.

        Per Anthropic docs, each request must contain:
        - A unique custom_id (string identifier)
        - A params object with standard Messages API parameters
        """
        experiment = ExperimentData(
            experiment_label="docs_compliance_test",
            experiment_number=0,
            experiment_df=pd.DataFrame(
                {"product": ["test_item"], "price": [19.99], "asin": ["B123TEST"]}
            ),
            query="test_query",
            dataset_name="compliance_test",
        )

        raw_message = RawMessageExchange([{"role": "user", "content": "Hello, world"}])

        batch_request = BatchRequest(
            experiments=[experiment],
            raw_messages=[raw_message],
            engine_params=mock_anthropic_params,
        )

        result = anthropic_serializer.serialize(batch_request)

        # Verify we get a SerializedBatchRequest
        assert len(result) == 1
        assert isinstance(result[0], SerializedBatchRequest)

        # Extract the provider request
        provider_request = result[0].provider_request

        # Verify required top-level structure per Anthropic docs
        assert "custom_id" in provider_request, (
            "custom_id is required per Anthropic batch API docs"
        )
        assert "params" in provider_request, (
            "params object is required per Anthropic batch API docs"
        )

        # Verify custom_id is a string
        assert isinstance(provider_request["custom_id"], str), (
            "custom_id must be a string"
        )

        # Verify params contains required Messages API parameters
        params = provider_request["params"]
        assert "model" in params, "model parameter is required"
        assert "messages" in params, "messages parameter is required"
        assert "max_tokens" in params, "max_tokens parameter is required"

        # Verify model value matches what we set
        assert params["model"] == "claude-3-5-sonnet-20241022"
        assert params["max_tokens"] == 1000
        assert params.get("temperature", 0.0) == 0.0

    def test_custom_id_uniqueness_and_format(self, anthropic_serializer, mock_anthropic_params):
        """
        Test custom_id uniqueness and format compliance.

        Per Anthropic docs:
        - Must be unique within the batch
        - Used for matching results back to original requests
        """
        experiments = [
            ExperimentData(
                experiment_label="unique_test",
                experiment_number=i,
                experiment_df=pd.DataFrame(
                    {
                        "product": [f"item_{i}"],
                        "price": [i * 10.0],
                        "asin": [f"B{i:03d}"],
                    }
                ),
                query=f"query_{i}",
                dataset_name="uniqueness_test",
            )
            for i in range(5)
        ]

        raw_messages = [
            RawMessageExchange([{"role": "user", "content": f"Find item {i}"}])
            for i in range(5)
        ]

        batch_request = BatchRequest(
            experiments=experiments,
            raw_messages=raw_messages,
            engine_params=mock_anthropic_params,
        )

        result = anthropic_serializer.serialize(batch_request)

        # Extract all custom_ids
        custom_ids = [req.provider_request["custom_id"] for req in result]

        # Verify uniqueness
        assert len(set(custom_ids)) == len(custom_ids), (
            "All custom_ids must be unique within batch"
        )

        # Verify all are strings and non-empty
        for custom_id in custom_ids:
            assert isinstance(custom_id, str), "custom_id must be string"
            assert len(custom_id) > 0, "custom_id must not be empty"
            # Allow alphanumeric characters and hyphens for negative hash values
            assert all(c.isalnum() or c == "-" for c in custom_id), (
                "custom_id should be alphanumeric or contain hyphens for negative hash values"
            )

    def test_message_format_compliance(self, anthropic_serializer, mock_anthropic_params):
        """
        Test that message format complies with Anthropic Messages API.

        Per Anthropic docs, messages should:
        - Have role and content fields
        - Support multi-turn conversations
        - Handle system messages properly
        """
        experiment = ExperimentData(
            experiment_label="message_format_test",
            experiment_number=0,
            experiment_df=pd.DataFrame(
                {"product": ["notebook"], "price": [15.99], "asin": ["B456NOTE"]}
            ),
            query="notebook",
            dataset_name="message_test",
        )

        # Test multi-turn conversation with system message
        raw_message = RawMessageExchange(
            [
                {"role": "system", "content": "You are a helpful shopping assistant."},
                {"role": "user", "content": "I need a notebook for school"},
                {
                    "role": "assistant",
                    "content": "I can help you find a suitable notebook.",
                },
                {"role": "user", "content": "What would you recommend?"},
            ]
        )

        batch_request = BatchRequest(
            experiments=[experiment],
            raw_messages=[raw_message],
            engine_params=mock_anthropic_params,
        )

        result = anthropic_serializer.serialize(batch_request)
        params = result[0].provider_request["params"]

        # Verify system message is handled separately (Anthropic best practice)
        assert "system" in params, "System message should be in params.system"
        assert params["system"] == "You are a helpful shopping assistant."

        # Verify messages array contains only user/assistant messages
        messages = params["messages"]
        assert len(messages) == 3, "Should have 3 non-system messages"

        # Verify message structure
        for message in messages:
            assert "role" in message, "Each message must have a role"
            assert "content" in message, "Each message must have content"
            assert message["role"] in ["user", "assistant"], (
                "Role must be user or assistant in messages array"
            )
            assert isinstance(message["content"], str), "Content must be string"

    def test_tool_definition_schema_compliance(self, anthropic_serializer, mock_anthropic_params):
        """
        Test that tool definitions comply with Anthropic tool use schema.

        Per Anthropic docs, tool definitions must have:
        - name: string
        - description: string
        - input_schema: object with type, properties, required fields
        """
        experiment = ExperimentData(
            experiment_label="tool_schema_test",
            experiment_number=0,
            experiment_df=pd.DataFrame(
                {"product": ["cart_item"], "price": [29.99], "asin": ["B789CART"]}
            ),
            query="shopping_cart",
            dataset_name="tool_test",
        )

        raw_message = RawMessageExchange(
            [{"role": "user", "content": "Add this item to my cart"}]
        )

        batch_request = BatchRequest(
            experiments=[experiment],
            raw_messages=[raw_message],
            engine_params=mock_anthropic_params,
            tools=[AddToCartTool()],
        )

        result = anthropic_serializer.serialize(batch_request)
        params = result[0].provider_request["params"]

        # Verify tools are present
        assert "tools" in params, "Tools should be included when provided"
        assert len(params["tools"]) == 1, "Should have one tool"

        tool = params["tools"][0]

        # Verify required tool fields per Anthropic docs
        assert "name" in tool, "Tool must have name field"
        assert "description" in tool, "Tool must have description field"
        assert "input_schema" in tool, "Tool must have input_schema field"

        # Verify field types
        assert isinstance(tool["name"], str), "Tool name must be string"
        assert isinstance(tool["description"], str), "Tool description must be string"
        assert isinstance(tool["input_schema"], dict), (
            "Tool input_schema must be object"
        )

        # Verify input_schema structure per Anthropic docs
        schema = tool["input_schema"]
        assert "type" in schema, "input_schema must have type field"
        assert "properties" in schema, "input_schema must have properties field"
        assert schema["type"] == "object", "input_schema type should be object"
        assert isinstance(schema["properties"], dict), "properties must be object"

        # Verify specific tool content
        assert tool["name"] == "add_to_cart"
        assert (
            "cart" in tool["description"].lower()
            or "add" in tool["description"].lower()
        )

        # Verify properties have proper structure
        properties = schema["properties"]
        for prop_name, prop_def in properties.items():
            assert isinstance(prop_def, dict), f"Property {prop_name} must be object"
            assert "type" in prop_def, f"Property {prop_name} must have type"
            assert "description" in prop_def, (
                f"Property {prop_name} must have description"
            )

    def test_parameter_validation_compliance(self, anthropic_serializer, mock_anthropic_params):
        """
        Test that all parameters comply with Anthropic API requirements.

        Verify that the serialized request contains valid parameter values
        and proper types as expected by the Anthropic API.
        """
        experiment = ExperimentData(
            experiment_label="param_validation_test",
            experiment_number=0,
            experiment_df=pd.DataFrame(
                {"product": ["validation_item"], "price": [9.99], "asin": ["B999VALID"]}
            ),
            query="validation",
            dataset_name="param_test",
        )

        raw_message = RawMessageExchange(
            [{"role": "user", "content": "Test parameter validation"}]
        )

        batch_request = BatchRequest(
            experiments=[experiment],
            raw_messages=[raw_message],
            engine_params=mock_anthropic_params,
        )

        result = anthropic_serializer.serialize(batch_request)
        params = result[0].provider_request["params"]

        # Verify required parameters are present and valid
        assert isinstance(params["model"], str), "model must be string"
        assert len(params["model"]) > 0, "model must not be empty"

        assert isinstance(params["max_tokens"], int), "max_tokens must be integer"
        assert params["max_tokens"] > 0, "max_tokens must be positive"

        if "temperature" in params:
            assert isinstance(params["temperature"], (int, float)), (
                "temperature must be numeric"
            )
            assert 0 <= params["temperature"] <= 1, "temperature must be between 0 and 1"

        assert isinstance(params["messages"], list), "messages must be array"
        assert len(params["messages"]) > 0, "messages must not be empty"

        # Verify the serialized request can be JSON serialized (important for API)
        try:
            json.dumps(result[0].provider_request)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Serialized request must be JSON serializable: {e}")

    def test_message_create_params_structure(self, anthropic_serializer, mock_anthropic_params):
        """
        Test that params conform to Anthropic's MessageCreateParamsNonStreaming structure.

        Verifies that the params object matches the expected structure from the
        Anthropic Messages API specification.
        """
        experiment = ExperimentData(
            experiment_label="params_structure_test",
            experiment_number=0,
            experiment_df=pd.DataFrame(
                {"product": ["test_product"], "price": [25.99], "asin": ["B555PARAM"]}
            ),
            query="test_params",
            dataset_name="params_test",
        )

        raw_message = RawMessageExchange(
            [
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": "Test message creation params"},
            ]
        )

        batch_request = BatchRequest(
            experiments=[experiment],
            raw_messages=[raw_message],
            engine_params=mock_anthropic_params,
            tools=[AddToCartTool()],
        )

        result = anthropic_serializer.serialize(batch_request)
        params = result[0].provider_request["params"]

        # Verify required fields per Anthropic MessageCreateParamsNonStreaming
        assert "model" in params, "model is required"
        assert "max_tokens" in params, "max_tokens is required"
        assert "messages" in params, "messages is required"

        # Verify field types
        assert isinstance(params["model"], str), "model must be string"
        assert isinstance(params["max_tokens"], int), "max_tokens must be integer"
        assert isinstance(params["messages"], list), "messages must be array"
        assert params["max_tokens"] >= 1, "max_tokens must be >= 1"

        # Verify optional fields have correct types when present
        if "system" in params:
            assert isinstance(params["system"], str), (
                "system must be string when present"
            )

        if "temperature" in params:
            assert isinstance(params["temperature"], (int, float)), (
                "temperature must be numeric"
            )
            assert 0 <= params["temperature"] <= 1, "temperature must be 0-1"

        if "tools" in params:
            assert isinstance(params["tools"], list), "tools must be array"
            for tool in params["tools"]:
                assert isinstance(tool, dict), "each tool must be object"
                assert "name" in tool, "tool must have name"
                assert "input_schema" in tool, "tool must have input_schema"

        # Verify message structure
        for message in params["messages"]:
            assert isinstance(message, dict), "each message must be object"
            assert "role" in message, "message must have role"
            assert "content" in message, "message must have content"
            assert message["role"] in ["user", "assistant"], (
                "role must be user or assistant"
            )
            assert isinstance(message["content"], str), "content must be string"

        # Verify no forbidden fields for batch processing
        forbidden_fields = ["stream"]  # stream must not be present in batch requests
        for field in forbidden_fields:
            assert field not in params, (
                f"{field} should not be present in batch requests"
            )
