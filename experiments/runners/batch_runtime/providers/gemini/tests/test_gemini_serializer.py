import pandas as pd
import pytest

from agent.src.core.tools import AddToCartTool
from experiments.config import ExperimentData
from experiments.runners.batch_runtime.providers.gemini.serializer import (
    GeminiBatchProviderSerializer,
)
from experiments.runners.batch_runtime.typedefs import BatchRequest, RawMessageExchange


@pytest.fixture
def batch_request(mock_gemini_params):
    # Create actual ExperimentData objects
    experiment1 = ExperimentData(
        experiment_label="test_exp1",
        experiment_number=1,
        experiment_df=pd.DataFrame({"product": ["mousepad"], "price": [25.99]}),
        query="mousepad",
        dataset_name="test_dataset",
        prompt_template="You are a shopping assistant.",
    )

    experiment2 = ExperimentData(
        experiment_label="test_exp2",
        experiment_number=2,
        experiment_df=pd.DataFrame({"product": ["keyboard"], "price": [99.99]}),
        query="keyboard",
        dataset_name="test_dataset",
        prompt_template="You are a shopping assistant.",
    )

    # Create raw messages
    raw_messages = [
        RawMessageExchange(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Find me a mousepad"},
            ]
        ),
        RawMessageExchange(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Find me a keyboard"},
            ]
        ),
    ]

    return BatchRequest(
        experiments=[experiment1, experiment2],
        raw_messages=raw_messages,
        engine_params=mock_gemini_params,
        tools=[AddToCartTool()],
    )


class TestGeminiBatchProviderSerializer:
    def test_serialize_basic(self, mock_gemini_params, batch_request):
        """Test basic serialization of batch requests."""
        serializer = GeminiBatchProviderSerializer(mock_gemini_params)
        result = serializer.serialize(batch_request)

        assert len(result) == 2
        for i, serialized in enumerate(result):
            assert (
                serialized.experiment_id == batch_request.experiments[i].experiment_id
            )
            assert isinstance(serialized.provider_request, dict)

            # Check the structure of the provider request (wrapped format)
            request = serialized.provider_request
            assert "request" in request
            assert "custom_id" in request

            # Check the inner request structure
            inner_request = request["request"]
            assert "contents" in inner_request
            assert "generationConfig" in inner_request
            assert "systemInstruction" in inner_request

    def test_system_message_handling(self, mock_gemini_params):
        """Test that system messages are properly extracted to system_instruction."""
        experiment = ExperimentData(
            experiment_label="test_system",
            experiment_number=1,
            experiment_df=pd.DataFrame({"product": ["item"], "price": [10.0]}),
            query="test",
            dataset_name="test_dataset",
        )

        raw_messages = [
            RawMessageExchange(
                [
                    {"role": "system", "content": "You are a shopping expert."},
                    {"role": "user", "content": "Help me find products"},
                    {"role": "assistant", "content": "I'll help you find products."},
                ]
            )
        ]

        batch_request = BatchRequest(
            experiments=[experiment],
            raw_messages=raw_messages,
            engine_params=mock_gemini_params,
        )

        serializer = GeminiBatchProviderSerializer(mock_gemini_params)
        result = serializer.serialize(batch_request)

        request = result[0].provider_request["request"]
        # System message should be in systemInstruction
        assert (
            request["systemInstruction"]["parts"][0]["text"]
            == "You are a shopping expert."
        )

        # Contents should only have user/assistant messages
        assert len(request["contents"]) == 2
        assert request["contents"][0]["role"] == "user"
        assert request["contents"][1]["role"] == "model"  # assistant mapped to model

    def test_multimodal_message_handling(self, mock_gemini_params):
        """Test handling of messages with images."""
        experiment = ExperimentData(
            experiment_label="test_multimodal",
            experiment_number=1,
            experiment_df=pd.DataFrame({"product": ["item"], "price": [10.0]}),
            query="test",
            dataset_name="test_dataset",
        )

        raw_messages = [
            RawMessageExchange(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://storage.googleapis.com/bucket/image.png"
                                },
                            },
                        ],
                    }
                ]
            )
        ]

        batch_request = BatchRequest(
            experiments=[experiment],
            raw_messages=raw_messages,
            engine_params=mock_gemini_params,
        )

        serializer = GeminiBatchProviderSerializer(mock_gemini_params)
        result = serializer.serialize(batch_request)

        request = result[0].provider_request["request"]
        content_parts = request["contents"][0]["parts"]

        assert len(content_parts) == 2
        assert content_parts[0]["text"] == "What's in this image?"
        assert content_parts[1]["fileData"]["fileUri"] == "gs://bucket/image.png"
        assert content_parts[1]["fileData"]["mimeType"] == "image/png"

    def test_gcs_url_conversion(self, mock_gemini_params):
        """Test conversion of different GCS URL formats."""
        experiment = ExperimentData(
            experiment_label="test_gcs",
            experiment_number=1,
            experiment_df=pd.DataFrame({"product": ["item"], "price": [10.0]}),
            query="test",
            dataset_name="test_dataset",
        )

        raw_messages = [
            RawMessageExchange(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Check these images"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://storage.googleapis.com/mybucket/img1.png"
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": "gs://mybucket/img2.png"},
                            },
                        ],
                    }
                ]
            )
        ]

        batch_request = BatchRequest(
            experiments=[experiment],
            raw_messages=raw_messages,
            engine_params=mock_gemini_params,
        )

        serializer = GeminiBatchProviderSerializer(mock_gemini_params)
        result = serializer.serialize(batch_request)

        request = result[0].provider_request["request"]
        parts = request["contents"][0]["parts"]

        # Both should be converted to gs:// format
        assert parts[1]["fileData"]["fileUri"] == "gs://mybucket/img1.png"
        assert parts[2]["fileData"]["fileUri"] == "gs://mybucket/img2.png"

    def test_tool_conversion(self, mock_gemini_params, batch_request):
        """Test conversion of OpenAI tools to Gemini format."""
        serializer = GeminiBatchProviderSerializer(mock_gemini_params)
        result = serializer.serialize(batch_request)

        request = result[0].provider_request["request"]
        assert "tools" in request

        # Check tool format
        tools = request["tools"]
        assert len(tools) == 1
        assert "functionDeclarations" in tools[0]

        func_decl = tools[0]["functionDeclarations"][0]
        assert func_decl["name"] == "add_to_cart"
        assert "description" in func_decl
        assert "parameters" in func_decl

    def test_generation_config(self, mock_gemini_params):
        """Test generation config is properly set."""
        experiment = ExperimentData(
            experiment_label="test_config",
            experiment_number=1,
            experiment_df=pd.DataFrame({"product": ["item"], "price": [10.0]}),
            query="test",
            dataset_name="test_dataset",
        )

        raw_messages = [RawMessageExchange([{"role": "user", "content": "Hello"}])]

        # Test with different engine param configurations
        mock_gemini_params.max_new_tokens = 500
        mock_gemini_params.temperature = 0.9

        batch_request = BatchRequest(
            experiments=[experiment],
            raw_messages=raw_messages,
            engine_params=mock_gemini_params,
        )

        serializer = GeminiBatchProviderSerializer(mock_gemini_params)
        result = serializer.serialize(batch_request)

        gen_config = result[0].provider_request["request"]["generationConfig"]
        assert gen_config["maxOutputTokens"] == 500
        assert gen_config["temperature"] == 0.9

    def test_complex_system_message(self, mock_gemini_params):
        """Test handling of complex system messages with multiple parts."""
        experiment = ExperimentData(
            experiment_label="test_complex_system",
            experiment_number=1,
            experiment_df=pd.DataFrame({"product": ["item"], "price": [10.0]}),
            query="test",
            dataset_name="test_dataset",
        )

        raw_messages = [
            RawMessageExchange(
                [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are a helpful assistant."},
                            {"type": "text", "text": "Always be polite."},
                        ],
                    },
                    {"role": "user", "content": "Hello"},
                ]
            )
        ]

        batch_request = BatchRequest(
            experiments=[experiment],
            raw_messages=raw_messages,
            engine_params=mock_gemini_params,
        )

        serializer = GeminiBatchProviderSerializer(mock_gemini_params)
        result = serializer.serialize(batch_request)

        request = result[0].provider_request["request"]
        # System messages should be combined
        expected_system = "You are a helpful assistant.\nAlways be polite."
        assert request["systemInstruction"]["parts"][0]["text"] == expected_system

    def test_non_gcs_images_raise_error(self, mock_gemini_params):
        """Test that non-GCS images raise ValueError."""
        experiment = ExperimentData(
            experiment_label="test_error_images",
            experiment_number=1,
            experiment_df=pd.DataFrame({"product": ["item"], "price": [10.0]}),
            query="test",
            dataset_name="test_dataset",
        )

        raw_messages = [
            RawMessageExchange(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Check this image"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/image.png"},
                            },
                        ],
                    }
                ]
            )
        ]

        batch_request = BatchRequest(
            experiments=[experiment],
            raw_messages=raw_messages,
            engine_params=mock_gemini_params,
        )

        serializer = GeminiBatchProviderSerializer(mock_gemini_params)

        # Should raise ValueError for non-GCS URLs
        with pytest.raises(ValueError, match="Invalid image URL format"):
            serializer.serialize(batch_request)

    def test_empty_tools(self, mock_gemini_params):
        """Test serialization without tools."""
        experiment = ExperimentData(
            experiment_label="test_no_tools",
            experiment_number=1,
            experiment_df=pd.DataFrame({"product": ["item"], "price": [10.0]}),
            query="test",
            dataset_name="test_dataset",
        )

        raw_messages = [RawMessageExchange([{"role": "user", "content": "Hello"}])]

        batch_request = BatchRequest(
            experiments=[experiment],
            raw_messages=raw_messages,
            engine_params=mock_gemini_params,
            tools=None,
        )

        serializer = GeminiBatchProviderSerializer(mock_gemini_params)
        result = serializer.serialize(batch_request)

        request = result[0].provider_request["request"]
        assert "tools" not in request

    def test_edge_case_empty_content(self, mock_gemini_params):
        """Test handling of empty content messages."""
        experiment = ExperimentData(
            experiment_label="test_empty",
            experiment_number=1,
            experiment_df=pd.DataFrame({"product": ["item"], "price": [10.0]}),
            query="test",
            dataset_name="test_dataset",
        )

        raw_messages = [RawMessageExchange([{"role": "user", "content": ""}])]

        batch_request = BatchRequest(
            experiments=[experiment],
            raw_messages=raw_messages,
            engine_params=mock_gemini_params,
        )

        serializer = GeminiBatchProviderSerializer(mock_gemini_params)
        result = serializer.serialize(batch_request)

        request = result[0].provider_request["request"]
        assert len(request["contents"]) == 1
        assert request["contents"][0]["parts"][0]["text"] == ""

    def test_edge_case_unicode_content(self, mock_gemini_params):
        """Test handling of unicode content in messages."""
        experiment = ExperimentData(
            experiment_label="test_unicode",
            experiment_number=1,
            experiment_df=pd.DataFrame({"product": ["item"], "price": [10.0]}),
            query="test",
            dataset_name="test_dataset",
        )

        unicode_content = "Test with emoji 🛒 and special chars: ñáéíóú"
        raw_messages = [
            RawMessageExchange([{"role": "user", "content": unicode_content}])
        ]

        batch_request = BatchRequest(
            experiments=[experiment],
            raw_messages=raw_messages,
            engine_params=mock_gemini_params,
        )

        serializer = GeminiBatchProviderSerializer(mock_gemini_params)
        result = serializer.serialize(batch_request)

        request = result[0].provider_request["request"]
        assert request["contents"][0]["parts"][0]["text"] == unicode_content
