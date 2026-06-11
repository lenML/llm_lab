"""Evaluation module — run inference and collect metrics."""
from .runner import evaluate, EvaluationReport

__all__ = ["evaluate", "EvaluationReport"]