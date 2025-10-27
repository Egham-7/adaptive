"""Evaluation and testing models for model performance assessment.

This module contains models for evaluating model performance on test datasets,
including multiple choice questions and evaluation results.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class MCQAnswer(BaseModel):
    """Structured output for multiple choice question answers.

    Attributes:
        answer: Selected answer (A, B, C, D, E, F, G, H, I, or J)
        reasoning: Brief explanation for the choice
        confidence: Confidence in the answer (0.0-1.0)
    """

    answer: str = Field(
        ..., description="Selected answer: A, B, C, D, E, F, G, H, I, or J"
    )
    reasoning: str = Field(..., description="Brief explanation for the choice")
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in the answer (0.0-1.0)",
    )


class CodeQuestion(BaseModel):
    """A coding question from the dataset.

    Attributes:
        question_id: Unique question identifier
        question: Question text
        choices: List of answer choices
        answer: Correct answer (A, B, C, D, E, F, G, H, I, or J)
        category: Question category (optional)
        difficulty: Question difficulty (optional)
    """

    question_id: str
    question: str
    choices: List[str]
    answer: str  # A, B, C, D, E, F, G, H, I, or J
    category: Optional[str] = None
    difficulty: Optional[str] = None

    def format_for_llm(self) -> str:
        """Format question as MCQ prompt for LLM."""
        choices_text = "\n".join(
            [f"{chr(65+i)}) {choice}" for i, choice in enumerate(self.choices)]
        )

        # Generate valid answer options based on number of choices
        max_choice = chr(65 + len(self.choices) - 1)
        answer_range = f"A-{max_choice}" if len(self.choices) > 1 else "A"

        return f"""You are taking a multiple choice coding test. Read the question carefully and select the best answer.

Question: {self.question}

{choices_text}

Instructions:
- Select exactly one answer ({answer_range})
- Provide brief reasoning for your choice
- Be concise and accurate"""


class EvaluationResult(BaseModel):
    """Result of evaluating a model on a single question.

    Attributes:
        model_id: ID of the model that was evaluated
        question_id: ID of the question
        cluster_id: Cluster ID the question belongs to
        predicted_answer: Answer predicted by the model
        correct_answer: Correct answer
        is_correct: Whether the prediction was correct
        response_time_ms: Response time in milliseconds
        confidence: Model's confidence in the answer
        reasoning: Model's reasoning (optional)
    """

    model_id: str
    question_id: str
    cluster_id: int
    predicted_answer: str
    correct_answer: str
    is_correct: bool
    response_time_ms: float
    confidence: float = 0.5
    reasoning: Optional[str] = None
