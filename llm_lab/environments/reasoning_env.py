"""reasoning-gym based multi-turn environment."""

from .base import BaseEnvironment, TurnResult


def _extract_answer(text: str) -> str | None:
    """Extract last <answer>...</answer> block from text."""
    import re
    m = re.findall(r"<answer>([\s\S]*?)</answer>", text, re.DOTALL)
    return m[-1].strip() if m else None


class ReasoningGymEnvironment(BaseEnvironment):
    """Multi-turn environment built on reasoning-gym datasets.

    The model is given a problem and can iterate for up to ``max_turns``
    steps. Each turn the model may produce reasoning + a final answer in
    ``<answer>...</answer>`` tags. Once a valid answer is found the
    episode ends and the score is returned as the final reward.

    Args:
        tokenizer: HuggingFace tokenizer (for ``apply_chat_template``).
        score_fn: ``dataset.score_answer(answer, entry=item)`` callable.
        system_prompt: Optional system prompt template.
        max_turns: Maximum number of interaction turns.
    """

    def __init__(
        self,
        tokenizer,
        score_fn,
        system_prompt: str = "",
        max_turns: int = 3,
    ):
        self.tokenizer = tokenizer
        self.score_fn = score_fn
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self._reset_state()

    # ── internal helpers ──────────────────────────────────────────

    def _reset_state(self):
        self.item = None
        self.turn = 0
        self._history: list[TurnResult] = []
        self._final_reward = 0.0
        self._messages: list[dict] = []

    # ── public API ────────────────────────────────────────────────

    def reset(self, item: dict) -> "ReasoningGymEnvironment":
        """Reset with a reasoning-gym item (must have a ``prompt`` key)."""
        self._reset_state()
        self.item = item
        if self.system_prompt:
            self._messages.append({"role": "system", "content": self.system_prompt})
        self._messages.append({"role": "user", "content": item["prompt"]})
        return self

    def step(self, action: str) -> TurnResult:
        """Process model action, return next state."""
        self.turn += 1

        answer = _extract_answer(action)
        if answer is not None:
            score = self.score_fn(answer, entry=self.item["metadata"])
            self._final_reward = score
            observation = (
                f"Answer score: {score:.2f}" if score < 1.0
                else "Correct! Episode complete."
            )
            result = TurnResult(action=action, observation=observation, reward=score, done=True)
        elif self.turn >= self.max_turns:
            result = TurnResult(
                action=action, observation="Max turns reached.", reward=0.0, done=True
            )
        else:
            result = TurnResult(
                action=action, observation="Keep reasoning.", reward=0.0, done=False
            )

        self._history.append(result)
        self._messages.append({"role": "assistant", "content": action})
        if not result.done:
            self._messages.append({"role": "user", "content": result.observation})

        return result

    def get_prompt(self) -> str:
        """Format current chat history as a model prompt string."""
        return self.tokenizer.apply_chat_template(
            self._messages,
            tokenize=False,
            add_generation_prompt=not self.done,
        )

    @property
    def done(self) -> bool:
        return any(r.done for r in self._history) or self.turn >= self.max_turns

    @property
    def final_reward(self) -> float:
        return self._final_reward