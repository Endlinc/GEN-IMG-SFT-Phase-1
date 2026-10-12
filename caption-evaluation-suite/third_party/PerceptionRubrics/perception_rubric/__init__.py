"""Minimal Perception Rubric evaluation package."""

from .evaluator import (
    SCORER_PROMPT,
    EvaluationResult,
    OpenAICompatibleJudge,
    build_judge_payload,
    compute_score,
    evaluate_caption,
    evaluate_with_raw_judge,
    normalize_rubrics,
    parse_scorer_response,
    split_rubrics,
)

__all__ = [
    "SCORER_PROMPT",
    "EvaluationResult",
    "OpenAICompatibleJudge",
    "build_judge_payload",
    "compute_score",
    "evaluate_caption",
    "evaluate_with_raw_judge",
    "normalize_rubrics",
    "parse_scorer_response",
    "split_rubrics",
]
