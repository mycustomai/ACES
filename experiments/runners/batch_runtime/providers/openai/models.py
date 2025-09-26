"""
Pydantic models for OpenAI Batch API response parsing.
Conforms to OpenAI Batch API specification.
"""

from typing import List, Optional

from openai.types.chat import (ChatCompletion, ChatCompletionMessage,
                               ChatCompletionMessageToolCall)
from pydantic import BaseModel


class OpenAIBatchError(BaseModel):
    """OpenAI batch error structure"""

    type: Optional[str] = None
    code: Optional[str] = None
    message: Optional[str] = None
    param: Optional[str] = None


class OpenAIResponse(BaseModel):
    """OpenAI batch response structure"""

    status_code: Optional[int] = None
    request_id: Optional[str] = None
    body: ChatCompletion | OpenAIBatchError

    @property
    def error(self) -> Optional[OpenAIBatchError]:
        """Check if this response contains an error"""
        return self.body if isinstance(self.body, OpenAIBatchError) else None


class OpenAIBatchResultLine(BaseModel):
    """Single line from OpenAI batch result JSONL"""

    custom_id: str
    response: Optional[OpenAIResponse] = None

    def has_error(self) -> bool:
        """Check if this result line contains an error"""
        if self.response is None:
            return True
        return self.response.error is not None or self.response.status_code != 200

    def get_message(self) -> ChatCompletionMessage:
        return self.response.body.choices[0].message

    def get_tool_calls(self) -> List[ChatCompletionMessageToolCall] | None:
        message = self.get_message()
        return message.tool_calls

    def get_content(self) -> str | None:
        message = self.get_message()
        return message.content


class OpenAIBatchResults(BaseModel):
    """Container for multiple OpenAI batch result lines"""

    results: List[OpenAIBatchResultLine]

    def get_successful_results(self) -> List[OpenAIBatchResultLine]:
        """Get only successful results (no errors)"""
        return [result for result in self.results if not result.has_error()]

    def get_failed_results(self) -> List[OpenAIBatchResultLine]:
        """Get only failed results (with errors)"""
        return [result for result in self.results if result.has_error()]
