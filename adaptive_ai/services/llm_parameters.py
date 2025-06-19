from adaptive_ai.models.parameters import OpenAIParameters


class LLMParameterService:
    """Always uses OpenAIParameters under the hood."""

    def __init__(self) -> None:
        self._provider = OpenAIParameters

    def get_parameters(
        self,
        model: str,
        task_type: str,
        prompt_scores: dict[str, list[float]],
    ) -> dict[str, float | int]:
        params = self._provider(model=model)
        return params.adjust_parameters(task_type, prompt_scores)
