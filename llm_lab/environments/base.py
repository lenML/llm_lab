"""Abstract base environment for multi-turn model interaction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class TurnResult:
    """Result of a single interaction turn."""

    action: str
    observation: str
    reward: float
    done: bool


class BaseEnvironment(ABC):
    """Environment that manages multi-turn interaction with the model.

    The environment holds the conversation state and determines when
    an episode ends. Each call to ``step`` processes the model's
    latest action and returns the next observation.
    """

    @abstractmethod
    def reset(self, item: Any) -> "BaseEnvironment":
        """Reset with a new problem item. Returns self for chaining."""
        ...

    @abstractmethod
    def step(self, action: str) -> TurnResult:
        """Process a model action, return turn result."""
        ...

    @abstractmethod
    def get_prompt(self) -> str:
        """Return the formatted prompt including conversation history."""
        ...

    @property
    @abstractmethod
    def done(self) -> bool:
        """Whether the episode has finished."""
        ...

    @property
    @abstractmethod
    def final_reward(self) -> float:
        """Final reward (0 if not done)."""
        ...