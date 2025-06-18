from typing import Literal

from pydantic import BaseModel

from adaptive_ai.models.parameters import OpenAIParameters


# --- Details for each protocol ---
class StandardLLMDetails(BaseModel):
    provider: str
    model: str


class MinionDetails(BaseModel):
    task_type: str


class RemoteLLM(BaseModel):
    provider: str
    model: str


class MinionsProtocolDetails(BaseModel):
    task_type: str
    remote_llm: RemoteLLM


# --- Response types for each protocol ---
class StandardLLMOrchestratorResponse(BaseModel):
    protocol: Literal["standard_llm"]
    standard_llm_data: StandardLLMDetails
    selected_model: str
    confidence: float
    parameters: OpenAIParameters


class MinionOrchestratorResponse(BaseModel):
    protocol: Literal["minion"]
    minion_data: MinionDetails


class MinionsProtocolOrchestratorResponse(BaseModel):
    protocol: Literal["minions_protocol"]
    minions_protocol_data: MinionsProtocolDetails


# --- Discriminated Union ---
OrchestratorResponse = (
    StandardLLMOrchestratorResponse
    | MinionOrchestratorResponse
    | MinionsProtocolOrchestratorResponse
)
