"""Tool-calling multi-turn environment.

The model can call tools in a structured format, receive results,
and iteratively work toward a solution::

     user:  What is 235 × 437?
  assistant: <tool_call>{"name":"calculator","arguments":{"expr":"235 * 437"}}</tool_call>
     user:  <tool_result>102695</tool_result>
  assistant: The answer is 102695. <answer>102695</answer>
     user:  Answer score: 1.00

Tools must be pre-registered.  The default set includes a calculator
and a web-search stub.
"""

import json
import re
from typing import Any, Callable

from .base import BaseEnvironment, TurnResult


# ── built-in tools ────────────────────────────────────────────────────


def _tool_calculator(expr: str) -> str:
    """Evaluate a mathematical expression safely."""
    allowed = set("0123456789+-*/.()% ")
    if not all(c in allowed for c in expr):
        return "Error: invalid characters in expression"
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


TOOL_REGISTRY: dict[str, Callable] = {
    "calculator": _tool_calculator,
}


# ── parsing helpers ────────────────────────────────────────────────────


_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)
_ANSWER_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>", re.DOTALL
)


def _parse_tool_calls(text: str) -> list[dict]:
    """Extract ``<tool_call>...json...</tool_call>`` blocks."""
    calls = []
    for match in _TOOL_CALL_RE.finditer(text):
        raw = match.group(1).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            calls.append(parsed)
        elif isinstance(parsed, list):
            calls.extend(parsed)
    return calls


def _extract_answer(text: str) -> str | None:
    m = _ANSWER_RE.findall(text)
    return m[-1].strip() if m else None


# ── environment ────────────────────────────────────────────────────────


class ToolUseEnvironment(BaseEnvironment):
    """Multi-turn environment with tool-execution support.

    The model may call zero or more tools per turn, receive results,
    and optionally emit a final answer inside ``<answer>...</answer>``.

    Args:
        tokenizer: HF tokenizer (for ``apply_chat_template``).
        tools: Dict of ``name → callable``.  Defaults to ``{"calculator": ...}``.
        answer_score_fn: ``(predicted: str, expected: str) → float``.
            If None, uses exact-match (1.0 / 0.0).
        expected_answer: The ground-truth answer string.
        system_prompt: System instruction string.
        max_turns: Maximum turns before forced termination.
    """

    def __init__(
        self,
        tokenizer: Any,
        tools: dict[str, Callable] | None = None,
        answer_score_fn: Callable | None = None,
        expected_answer: str = "",
        system_prompt: str = "",
        max_turns: int = 5,
    ):
        self.tokenizer = tokenizer
        self.tools = {**TOOL_REGISTRY, **(tools or {})}
        self.answer_score_fn = answer_score_fn or (
            lambda pred, exp: 1.0 if pred.strip() == exp.strip() else 0.0
        )
        self.expected_answer = expected_answer
        self._system_prompt = system_prompt
        self.max_turns = max_turns
        self._reset_state()

    # ── internal ────────────────────────────────────────────────────

    def _reset_state(self):
        self.turn = 0
        self._history: list[TurnResult] = []
        self._final_reward = 0.0
        self._messages: list[dict] = []

    def _execute_tool(self, call: dict) -> str:
        """Execute a tool call and return the result string."""
        name = call.get("name", "")
        args = call.get("arguments", {})
        fn = self.tools.get(name)
        if fn is None:
            return f"Error: unknown tool '{name}'"
        try:
            if isinstance(args, dict):
                result = fn(**args)
            else:
                result = fn(args)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    # ── public ──────────────────────────────────────────────────────

    def reset(self, item: dict) -> "ToolUseEnvironment":
        self._reset_state()
        self.expected_answer = item.get("answer", item.get("expected_answer", ""))
        prompt_text = item.get("prompt", item.get("question", ""))
        if self._system_prompt:
            self._messages.append({"role": "system", "content": self._system_prompt})
        self._messages.append({"role": "user", "content": prompt_text})
        return self

    def step(self, action: str) -> TurnResult:
        self.turn += 1

        # Check for final answer first
        answer = _extract_answer(action)
        if answer is not None:
            score = self.answer_score_fn(answer, self.expected_answer)
            self._final_reward = score
            observation = (
                f"Answer score: {score:.2f}" if score < 1.0 else "Correct!"
            )
            result = TurnResult(action=action, observation=observation, reward=score, done=True)
        else:
            # Parse and execute tool calls
            tool_calls = _parse_tool_calls(action)
            if tool_calls:
                results = []
                for call in tool_calls:
                    r = self._execute_tool(call)
                    results.append(
                        f'<tool_result name="{call.get("name", "")}">{r}</tool_result>'
                    )
                observation = "\n".join(results)
                result = TurnResult(
                    action=action,
                    observation=observation,
                    reward=0.0,
                    done=self.turn >= self.max_turns,
                )
            elif self.turn >= self.max_turns:
                result = TurnResult(
                    action=action, observation="Max turns reached.", reward=0.0, done=True
                )
            else:
                result = TurnResult(
                    action=action, observation="Continue.", reward=0.0, done=False
                )

        self._history.append(result)
        self._messages.append({"role": "assistant", "content": action})
        if not result.done:
            self._messages.append({"role": "user", "content": result.observation})

        return result

    def get_prompt(self) -> str:
        return self.tokenizer.apply_chat_template(
            self._messages,
            tokenize=False,
            add_generation_prompt=not self.done,
        )

    @property
    def done(self) -> bool:
        return (
            any(r.done for r in self._history) or self.turn >= self.max_turns
        )

    @property
    def final_reward(self) -> float:
        return self._final_reward