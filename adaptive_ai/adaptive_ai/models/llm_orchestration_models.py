from typing import Any

from pydantic import BaseModel, Field, model_validator

from .llm_enums import ProtocolType  # Import enum


# Strict definition for OpenAI ChatCompletion parameters
class LogprobsContent(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[dict[str, Any]] | None = None


class Logprobs(BaseModel):
    content: list[LogprobsContent] | None = None


class OpenAIParameters(BaseModel):
    # Controls randomness: higher values mean more diverse completions.
    # Recommended range 0.2 to 0.8 as per your article.
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    # Nucleus sampling: only considers tokens whose cumulative probability exceeds top_p.
    # Recommended range 0.3 to 0.9 as per your article.
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, alias="top_p")
    # The maximum number of tokens to generate in the completion.
    max_tokens: int | None = Field(None, ge=1)
    # How many chat completion choices to generate for each input message.
    n: int = Field(default=1, ge=1)
    # Up to 4 sequences where the API will stop generating further tokens.
    stop: list[str] | str | None = None
    # Penalize new tokens based on their existing frequency in the text so far.
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    # Penalize new tokens based on whether they appear in the text so far.
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)


class Alternative(BaseModel):
    provider: str
    model: str


class GroqAlternative(BaseModel):
    model: str


class StandardLLMInfo(BaseModel):
    provider: str
    model: str
    parameters: OpenAIParameters
    alternatives: list[Alternative]


class MinionInfo(BaseModel):
    model: str
    parameters: OpenAIParameters
    alternatives: list[GroqAlternative]


# Extensible OrchestratorResponse for future protocols
class OrchestratorResponse(BaseModel):
    protocol: ProtocolType
    standard: StandardLLMInfo | None = None
    minion: MinionInfo | None = None
    # For future extensibility, add new protocol payloads as new fields
    # e.g., minions_protocol: MinionsProtocolInfo | None = None

    @model_validator(mode="after")
    def validate_exclusive_fields(self) -> "OrchestratorResponse":
        # Allow multiple protocol payloads for extensibility, but enforce at least one present
        if not any([self.standard, self.minion]):
            raise ValueError(
                "Must have at least one protocol payload (standard, minion, etc.)"
            )
        return self
