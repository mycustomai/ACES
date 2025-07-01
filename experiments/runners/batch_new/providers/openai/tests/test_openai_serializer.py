from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pytest

from agent.src.core.tools import AddToCartTool
from agent.src.typedefs import EngineParams, EngineType
from experiments.config import ExperimentData, InstructionConfig
from experiments.runners.batch_new.providers.openai.serializer import \
    OpenAIBatchProviderSerializer
from experiments.runners.batch_new.typedefs import (BatchRequest,
                                                    RawMessageExchange)


@pytest.fixture
def engine_params():
    return EngineParams(
        engine_type=EngineType.OPENAI,
        model="test-model",
        max_new_tokens=100,
        temperature=0.7,
    )


@pytest.fixture
def batch_request(engine_params):
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

    experiment3 = ExperimentData(
        experiment_label="test_exp3",
        experiment_number=3,
        experiment_df=pd.DataFrame({"product": ["mouse"], "price": [39.99]}),
        query="mouse",
        dataset_name="test_dataset",
        prompt_template="You are a shopping assistant.",
    )

    # Create raw messages for each experiment - each experiment gets its own message exchange
    raw_messages = [
        # Experiment 1 - mousepad shopping
        RawMessageExchange(
            [
                {
                    "role": "system",
                    "content": "You are a shopping assistant helping with mousepad selection.",
                },
                {"role": "user", "content": "I need a good mousepad for gaming."},
                {
                    "role": "assistant",
                    "content": "I can help you find a gaming mousepad. Let me look at the available options.",
                },
            ]
        ),
        # Experiment 2 - keyboard shopping
        RawMessageExchange(
            [
                {
                    "role": "system",
                    "content": "You are a shopping assistant helping with keyboard selection.",
                },
                {"role": "user", "content": "I'm looking for a mechanical keyboard."},
                {
                    "role": "assistant",
                    "content": "Great! I'll help you find a mechanical keyboard that suits your needs.",
                },
            ]
        ),
        # Experiment 3 - mouse shopping
        RawMessageExchange(
            [
                {
                    "role": "system",
                    "content": "You are a shopping assistant helping with mouse selection.",
                },
                {"role": "user", "content": "I need a wireless mouse for my laptop."},
                {
                    "role": "assistant",
                    "content": "I'll help you find a wireless mouse. Let me check what's available.",
                },
            ]
        ),
    ]

    # Create actual BatchRequest
    return BatchRequest(
        experiments=[experiment1, experiment2, experiment3],
        raw_messages=raw_messages,
        engine_params=engine_params,
        tools=[AddToCartTool()],
    )


@pytest.fixture
def serializer(engine_params):
    return OpenAIBatchProviderSerializer(engine_params)


def test_serialize_returns_correct_provider_requests(serializer, batch_request):
    results = serializer.serialize(batch_request)
    # Should produce as many requests as there are batches/experiments
    assert len(results) == 3

    expected_exp_ids = [
        "mousepad_test_exp1_1",
        "keyboard_test_exp2_2",
        "mouse_test_exp3_3",
    ]

    for result, exp_id, messages in zip(
        results,
        expected_exp_ids,
        batch_request.raw_messages,
    ):
        # result is now a SerializedBatchRequest with experiment_id and provider_request
        assert result.experiment_id == exp_id
        d = result.provider_request  # This is the actual ProviderBatchRequest dict
        assert d["custom_id"] == exp_id
        assert d["method"] == "POST"
        assert d["url"] == "/v1/chat/completions"
        assert d["body"]["model"] == "test-model"
        assert d["body"]["messages"] == messages
        assert d["body"]["max_tokens"] == 100
        assert d["body"]["temperature"] == 0.7
        # Check that tools are serialized correctly according to OpenAI API format
        tools = d["body"]["tools"]
        assert len(tools) == 1
        assert d["body"]["tool_choice"] == "auto"

        # Verify tool follows OpenAI function calling format
        tool = tools[0]
        assert tool["type"] == "function"
        assert "function" in tool

        function_def = tool["function"]
        assert function_def["name"] == "add_to_cart"
        assert (
            function_def["description"]
            == "Add the selected item to cart. This is a special tool that is only used when you've selected a product item."
        )

        # Verify parameters follow JSON Schema format
        parameters = function_def["parameters"]
        assert parameters["type"] == "object"
        assert "properties" in parameters
        assert "required" in parameters

        # Check required fields
        assert set(parameters["required"]) == {
            "product_title",
            "price",
            "rating",
            "number_of_reviews",
        }

        # Check properties structure
        properties = parameters["properties"]
        assert "product_title" in properties
        assert "price" in properties
        assert "rating" in properties
        assert "number_of_reviews" in properties

        # Verify property types and descriptions
        assert properties["product_title"]["type"] == "string"
        assert (
            properties["product_title"]["description"]
            == "The exact title of the product as shown in the screenshot."
        )

        assert properties["price"]["type"] == "number"
        assert properties["price"]["description"] == "The price of the product."

        assert properties["rating"]["type"] == "number"
        assert properties["rating"]["description"] == "The rating of the product."

        assert properties["number_of_reviews"]["type"] == "integer"
        assert (
            properties["number_of_reviews"]["description"]
            == "The number of reviews of the product."
        )
