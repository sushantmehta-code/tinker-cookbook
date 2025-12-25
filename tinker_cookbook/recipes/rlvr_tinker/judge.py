"""
Tinker-based LLM Judge for RLVR

This module implements a batched rubric evaluator using Tinker's sampling client.
All rubrics for a given response are evaluated in a single API call, returning
binary Yes/No scores per rubric and a pass rate between 0 and 1.

Key differences from rlvr/judge.py:
- Uses TinkerMessageCompleter instead of OpenAI API
- Same batched prompt format (all rubrics in one call)
- Same binary Yes/No scoring with JSON response
- Raises JudgeError on failure (no fake rewards)
"""

import json
import logging
from dataclasses import dataclass

import tinker

from tinker_cookbook import model_info
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


class JudgeError(Exception):
    """Raised when judge evaluation fails after all retries or cannot parse response."""

    pass


# Judge prompt templates (same as rlvr/judge.py)
JUDGE_SYSTEM_PROMPT = """You are evaluating a response against multiple rubric criteria. Your task is to determine if the response satisfies each given criterion.

**CRITICAL: For each rubric item, you MUST output exactly one of these options: `Yes`, `No`**

Choose the option that best represents your evaluation of whether the response meets each rubric criterion."""


JUDGE_USER_TEMPLATE = """Prompt:
{prompt}

Completion:
{completion}

Rubric items:
{rubrics}

Respond as a JSON object with a "ratings" field containing a list of objects, where each object has:
- rating: str in ["Yes", "No"]
- rationale: str explaining the rating

The list should have one entry for each rubric item in the same order as provided above."""


@dataclass
class JudgeResult:
    """Result from the rubric judge."""

    pass_rate: float
    num_passed: int
    num_total: int
    ratings: list[dict]  # List of {rating: str, rationale: str}
    raw_response: str | None = None


class TinkerRubricJudge:
    """
    Batched rubric evaluator using Tinker sampling client.

    Evaluates all rubrics for a response in a single API call for efficiency.
    Uses binary Yes/No scoring per rubric and computes pass rate as the reward.
    """

    def __init__(
        self,
        grader_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
        max_retries: int = 3,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        base_url: str | None = None,
    ):
        """
        Initialize the Tinker rubric judge.

        Args:
            grader_model: Tinker model to use for judging
            max_retries: Maximum number of retries on API failure
            max_tokens: Maximum tokens for judge response
            temperature: Sampling temperature for judge
            base_url: Optional Tinker API base URL
        """
        self.grader_model = grader_model
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Create TinkerMessageCompleter
        tokenizer = get_tokenizer(grader_model)
        renderer_name = model_info.get_recommended_renderer_name(grader_model)
        renderer = get_renderer(name=renderer_name, tokenizer=tokenizer)

        service_client = tinker.ServiceClient(base_url=base_url)
        sampling_client = service_client.create_sampling_client(base_model=grader_model)

        self.completer = TinkerMessageCompleter(
            sampling_client=sampling_client,
            renderer=renderer,
            max_tokens=max_tokens,
        )

        logger.info(f"Initialized TinkerRubricJudge with model: {grader_model}")

    def _format_rubrics(self, rubrics: list[str]) -> str:
        """Format rubrics as a numbered list for the prompt."""
        return "\n".join(f"{i+1}. {rubric}" for i, rubric in enumerate(rubrics))

    def _parse_response(self, response_text: str, num_rubrics: int) -> list[dict]:
        """
        Parse the judge's JSON response into a list of ratings.

        Args:
            response_text: Raw response text from the judge
            num_rubrics: Expected number of rubric ratings

        Returns:
            List of rating dictionaries with 'rating' and 'rationale' keys

        Raises:
            JudgeError: If JSON parsing fails
        """
        try:
            # Try to extract JSON from the response
            # Handle case where response might have markdown code blocks
            text = response_text.strip()
            if text.startswith("```"):
                # Remove markdown code blocks
                lines = text.split("\n")
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block or not line.startswith("```"):
                        json_lines.append(line)
                text = "\n".join(json_lines)

            data = json.loads(text, strict=False)  # Allow control characters in strings
            ratings = data.get("ratings", [])

            # Validate and normalize ratings - strict mode: reject malformed responses
            normalized_ratings = []
            for i, r in enumerate(ratings):
                raw_rating = r.get("rating", "")
                rating = raw_rating.strip().lower() if raw_rating else ""
                
                # Strict validation: only accept valid ratings
                if rating in ("yes", "true", "1"):
                    normalized_rating = "Yes"
                elif rating in ("no", "false", "0"):
                    normalized_rating = "No"
                else:
                    # Reject unclear/invalid ratings - don't use this score for training
                    raise JudgeError(
                        f"Invalid rating '{raw_rating}' for rubric {i+1} - "
                        f"expected Yes/No/True/False/1/0"
                    )

                normalized_ratings.append(
                    {"rating": normalized_rating, "rationale": r.get("rationale", "")}
                )

            # Strict validation: rating count must exactly match rubric count
            if len(normalized_ratings) != num_rubrics:
                raise JudgeError(
                    f"Rating count mismatch: got {len(normalized_ratings)}, "
                    f"expected {num_rubrics}"
                )

            return normalized_ratings

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse judge response as JSON: {e}")
            raise JudgeError(f"Failed to parse judge response as JSON: {e}")

    async def evaluate(
        self,
        prompt: str,
        completion: str,
        rubrics: list[str],
    ) -> JudgeResult:
        """
        Evaluate a completion against all rubrics in a single API call.

        Args:
            prompt: The original prompt/instruction
            completion: The model's completion/response to evaluate
            rubrics: List of rubric criteria to check

        Returns:
            JudgeResult with pass rate and individual ratings

        Raises:
            JudgeError: If evaluation fails after all retries
        """
        if not rubrics:
            return JudgeResult(
                pass_rate=1.0,
                num_passed=0,
                num_total=0,
                ratings=[],
            )

        formatted_rubrics = self._format_rubrics(rubrics)
        user_message = JUDGE_USER_TEMPLATE.format(
            prompt=prompt,
            completion=completion,
            rubrics=formatted_rubrics,
        )

        # Build conversation for the judge
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Call the Tinker completer
                response = await self.completer(messages)
                raw_response = response.get("content", "")

                if not isinstance(raw_response, str):
                    raise JudgeError(f"Judge response content is not a string: {type(raw_response)}")

                ratings = self._parse_response(raw_response, len(rubrics))

                num_passed = sum(1 for r in ratings if r["rating"] == "Yes")
                pass_rate = num_passed / len(rubrics) if rubrics else 1.0

                return JudgeResult(
                    pass_rate=pass_rate,
                    num_passed=num_passed,
                    num_total=len(rubrics),
                    ratings=ratings,
                    raw_response=raw_response,
                )

            except JudgeError:
                # Re-raise JudgeError from parsing
                raise
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Judge API call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                # No sleep/delay needed - Tinker handles retries internally

        # All retries exhausted - raise exception instead of returning fake 0.0 reward
        logger.error(f"Judge failed after {self.max_retries} attempts: {last_error}")
        raise JudgeError(f"Judge API failed after {self.max_retries} retries: {last_error}")

