from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    task_type_1: list[str] = Field(alias="task_type_1")
    task_type_2: list[str] = Field(alias="task_type_2")
    task_type_prob: list[float] = Field(alias="task_type_prob")
    creativity_scope: list[float] = Field(alias="creativity_scope")
    reasoning: list[float] = Field(alias="reasoning")
    contextual_knowledge: list[float] = Field(alias="contextual_knowledge")
    prompt_complexity_score: list[float] = Field(alias="prompt_complexity_score")
    domain_knowledge: list[float] = Field(alias="domain_knowledge")
    number_of_few_shots: list[float] = Field(alias="number_of_few_shots")
    no_label_reason: list[float] = Field(alias="no_label_reason")
    constraint_ct: list[float] = Field(alias="constraint_ct")
