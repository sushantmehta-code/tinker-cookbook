"""
LLM Judge for RLVR (RL with Verifiable Rubrics)

This module implements a batched rubric evaluator using GPT-5-mini. For efficiency,
all rubrics for a given response are evaluated in a single API call, returning
a pass rate between 0 and 1.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class JudgeError(Exception):
    """Raised when judge evaluation fails after all retries or cannot parse response."""

    pass


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


class RubricJudge:
    """
    Batched rubric evaluator using GPT-5-mini.

    Evaluates all rubrics for a response in a single API call for efficiency.
    """

    def __init__(
        self,
        model: str = "gpt-5-mini",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 1.0,
        max_completion_tokens: int = 4096,
    ):
        """
        Initialize the rubric judge.

        Args:
            model: OpenAI model to use for judging (default: gpt-5-mini)
            max_retries: Maximum number of retries on API failure
            retry_delay: Initial delay between retries (exponential backoff)
            temperature: Sampling temperature for judge (0.0 for deterministic)
            max_completion_tokens: Maximum tokens for judge response
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens

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

            data = json.loads(text)
            ratings = data.get("ratings", [])

            # Validate and normalize ratings
            normalized_ratings = []
            for r in ratings:
                rating = r.get("rating", "").strip()
                # Normalize to Yes/No
                if rating.lower() in ("yes", "true", "1"):
                    rating = "Yes"
                elif rating.lower() in ("no", "false", "0"):
                    rating = "No"
                else:
                    rating = "No"  # Default to No for unclear responses

                normalized_ratings.append(
                    {"rating": rating, "rationale": r.get("rationale", "")}
                )

            # If we got fewer ratings than rubrics, pad with No
            while len(normalized_ratings) < num_rubrics:
                normalized_ratings.append(
                    {"rating": "No", "rationale": "No rating provided by judge"}
                )

            # If we got more ratings, truncate
            return normalized_ratings[:num_rubrics]

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

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    # temperature=self.temperature,
                    # max_completion_tokens=self.max_completion_tokens,
                    response_format={"type": "json_object"},
                )

                raw_response = response.choices[0].message.content or ""
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

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Judge API call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)

        # All retries exhausted - raise exception instead of returning fake 0.0 reward
        logger.error(f"Judge failed after {self.max_retries} attempts: {last_error}")
        raise JudgeError(f"Judge API failed after {self.max_retries} retries: {last_error}")


async def evaluate_batch(
    judge: RubricJudge,
    items: list[tuple[str, str, list[str]]],
    max_concurrent: int = 256,
) -> list[JudgeResult]:
    """
    Evaluate multiple (prompt, completion, rubrics) tuples concurrently.

    Args:
        judge: The RubricJudge instance to use
        items: List of (prompt, completion, rubrics) tuples
        max_concurrent: Maximum concurrent API calls (for rate limiting)

    Returns:
        List of JudgeResults in the same order as input items
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_with_semaphore(
        prompt: str, completion: str, rubrics: list[str]
    ) -> JudgeResult:
        async with semaphore:
            return await judge.evaluate(prompt, completion, rubrics)

    tasks = [
        evaluate_with_semaphore(prompt, completion, rubrics)
        for prompt, completion, rubrics in items
    ]

    return await asyncio.gather(*tasks)

