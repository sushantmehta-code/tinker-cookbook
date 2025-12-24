"""
RLVR Tinker Environment and Dataset

This module implements the RL environment for training with verifiable rubrics
using Tinker's sampling client for the judge model.

Key features:
- Single batched judge call per response (evaluates all rubrics at once)
- Binary Yes/No scoring per rubric with pass rate as reward
- Fault-tolerant rollouts that skip failed judge calls
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import chz
import tinker

from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import StopCondition, TokenCompleter
from tinker_cookbook.recipes.rlvr_tinker.judge import JudgeError, JudgeResult, TinkerRubricJudge
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_TRAIN_PATH = Path(__file__).parent / "data" / "train.jsonl"
DEFAULT_TEST_PATH = Path(__file__).parent / "data" / "test.jsonl"


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class RLVRTinkerSample:
    """A single RLVR sample with prompt and rubrics."""

    task_id: str
    prompt: str
    rubrics: list[str]


def load_jsonl_data(path: Path) -> list[RLVRTinkerSample]:
    """Load RLVR data from JSONL file."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            samples.append(
                RLVRTinkerSample(
                    task_id=item.get("task_id", ""),
                    prompt=item["prompt"],
                    rubrics=item["rubrics"],
                )
            )
    return samples


# =============================================================================
# Fault-Tolerant Rollout Functions
# =============================================================================


async def safe_single_rollout(policy: TokenCompleter, env: Env) -> Trajectory | None:
    """
    Run a single rollout with exception handling.

    If the rollout fails (e.g., due to JudgeError), returns None instead of
    crashing the entire training job.

    Args:
        policy: The token completer (policy) to use for actions
        env: The environment to roll out in

    Returns:
        Trajectory if successful, None if the rollout failed
    """
    try:
        return await do_single_rollout(policy, env)
    except JudgeError as e:
        logger.warning(f"Rollout failed due to judge error (skipping sample): {e}")
        return None
    except Exception as e:
        # Catch any other unexpected errors to prevent training crash
        logger.error(f"Unexpected error during rollout (skipping sample): {e}")
        return None


@logtree.scope_header_decorator
async def do_fault_tolerant_group_rollout(
    env_group_builder: EnvGroupBuilder, policy: TokenCompleter
) -> TrajectoryGroup | None:
    """
    Run group rollout with fault tolerance for failed individual rollouts.

    If some rollouts fail (e.g., due to judge API errors), they are filtered out
    and training continues with the successful ones. If all rollouts in the group
    fail, the entire group is skipped.

    Args:
        env_group_builder: Builder for the group of environments
        policy: The token completer (policy) to use for actions

    Returns:
        TrajectoryGroup with only successful trajectories, or None if all failed
    """
    envs_G: Sequence[Env] = await env_group_builder.make_envs()

    # Run all rollouts, collecting results (including None for failures)
    results = await asyncio.gather(
        *[safe_single_rollout(policy, env) for env in envs_G],
        return_exceptions=True,  # Don't fail fast on first exception
    )

    # Filter out failures (None values and any uncaught exceptions)
    valid_pairs: list[tuple[Trajectory, Env]] = []
    num_failed = 0
    for result, env in zip(results, envs_G, strict=True):
        if isinstance(result, Exception):
            logger.warning(f"Rollout raised exception: {result}")
            num_failed += 1
        elif result is None:
            num_failed += 1
        else:
            valid_pairs.append((result, env))

    if num_failed > 0:
        logger.info(f"Group rollout: {len(valid_pairs)} succeeded, {num_failed} failed")

    if not valid_pairs:
        logger.warning("All rollouts in group failed, skipping entire group")
        return None

    # Unzip the valid pairs
    valid_trajectories, valid_envs = zip(*valid_pairs, strict=True)
    trajectories_G = list(valid_trajectories)

    # Compute rewards only for valid trajectories
    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(
        trajectories_G, list(valid_envs)
    )
    rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)

    # Log trajectory tables with final rewards
    with logtree.scope_header("Trajectory Summary"):
        for i, (traj, final_reward) in enumerate(
            zip(trajectories_G, rewards_G, strict=True)
        ):
            rows = []
            step_reward_sum = 0.0
            for t_idx, t in enumerate(traj.transitions):
                step_reward_sum += t.reward
                rows.append(
                    {
                        "step": t_idx,
                        "ob_len": t.ob.length,
                        "ac_len": len(t.ac.tokens),
                        "reward": f"{t.reward:.3f}",
                    }
                )
            # Add final row with final observation and computed reward
            rows.append(
                {
                    "step": "final",
                    "ob_len": traj.final_ob.length,
                    "ac_len": "-",
                    "reward": f"{final_reward:.3f}",
                }
            )
            # Add total reward row
            rows.append(
                {
                    "step": "total",
                    "ob_len": "-",
                    "ac_len": "-",
                    "reward": f"{step_reward_sum + final_reward:.3f}",
                }
            )
            logtree.table(rows, caption=f"Trajectory {i}")

    return TrajectoryGroup(trajectories_G, list(rewards_G), list(metrics_G))


# =============================================================================
# Environment Classes
# =============================================================================


class RLVRTinkerEnv(Env):
    """
    Single-turn environment for RLVR training with Tinker judge.

    The environment presents a prompt to the model, collects its response,
    and uses a Tinker-based judge to evaluate the response against all rubrics
    in a single batched call.
    """

    def __init__(
        self,
        sample: RLVRTinkerSample,
        renderer: renderers.Renderer,
        judge: TinkerRubricJudge,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        """
        Initialize the RLVR Tinker environment.

        Args:
            sample: The RLVR sample with prompt and rubrics
            renderer: Renderer for converting messages to model input
            judge: The Tinker rubric judge for evaluation
            convo_prefix: Optional conversation prefix (e.g., system message)
        """
        self.sample = sample
        self.renderer = renderer
        self.judge = judge
        self.convo_prefix = convo_prefix or []
        self._response: str | None = None
        self._judge_result: JudgeResult | None = None

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Return the initial observation (the prompt) and stop condition."""
        convo = self.convo_prefix + [
            {"role": "user", "content": self.sample.prompt},
        ]
        return self.renderer.build_generation_prompt(convo), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        """
        Process the model's response and compute reward via batched judge call.

        Args:
            action: Token IDs of the model's response

        Returns:
            StepResult with reward based on rubric pass rate

        Raises:
            JudgeError: If judge evaluation fails (will be caught by fault-tolerant rollout)
        """
        # Parse the response
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.ensure_text(message["content"])
        self._response = content

        # Evaluate with the judge (single batched call for all rubrics)
        # This may raise JudgeError, which will be caught by safe_single_rollout
        self._judge_result = await self.judge.evaluate(
            prompt=self.sample.prompt,
            completion=content,
            rubrics=self.sample.rubrics,
        )

        # Reward is the rubric pass rate (0 to 1)
        reward = self._judge_result.pass_rate

        # Log the attempt
        logtree.log_text(f"Prompt: {self.sample.prompt[:200]}...")
        logtree.log_text(f"Response: {content[:500]}...")
        logtree.log_text(
            f"Rubrics: {self._judge_result.num_passed}/{self._judge_result.num_total} passed"
        )
        logtree.log_text(f"Reward (pass rate): {reward:.3f}")

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "pass_rate": reward,
                "num_rubrics": self._judge_result.num_total,
                "num_passed": self._judge_result.num_passed,
            },
        )


@dataclass(frozen=True)
class RLVRTinkerEnvGroupBuilder(EnvGroupBuilder):
    """
    Builds a group of RLVR Tinker environments for the same prompt.

    Multiple environments allow GRPO to center advantages within the group.
    """

    sample: RLVRTinkerSample
    renderer: renderers.Renderer
    judge: TinkerRubricJudge
    num_envs: int
    convo_prefix: list[renderers.Message] | None = None

    async def make_envs(self) -> Sequence[Env]:
        """Create multiple environments for the same prompt."""
        return [
            RLVRTinkerEnv(
                sample=self.sample,
                renderer=self.renderer,
                judge=self.judge,
                convo_prefix=self.convo_prefix,
            )
            for _ in range(self.num_envs)
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """No additional group-level rewards needed (using per-step rewards)."""
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        """Tags for logging/aggregation."""
        return ["rlvr_tinker", "instruction_following"]


# =============================================================================
# Dataset Classes
# =============================================================================


class RLVRTinkerDataset(RLDataset):
    """
    Dataset that produces batches of RLVR Tinker environment groups.
    """

    def __init__(
        self,
        samples: list[RLVRTinkerSample],
        renderer: renderers.Renderer,
        judge: TinkerRubricJudge,
        groups_per_batch: int,
        group_size: int,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        """
        Initialize the RLVR Tinker dataset.

        Args:
            samples: List of RLVR samples
            renderer: Renderer for model input conversion
            judge: Tinker rubric judge instance
            groups_per_batch: Number of environment groups per batch
            group_size: Number of environments per group (for GRPO)
            convo_prefix: Optional conversation prefix
        """
        self.samples = samples
        self.renderer = renderer
        self.judge = judge
        self.groups_per_batch = groups_per_batch
        self.group_size = group_size
        self.convo_prefix = convo_prefix

    def __len__(self) -> int:
        """Number of batches in the dataset."""
        return (len(self.samples) + self.groups_per_batch - 1) // self.groups_per_batch

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Get a batch of environment group builders."""
        start_idx = index * self.groups_per_batch
        end_idx = min(start_idx + self.groups_per_batch, len(self.samples))

        builders = []
        for i in range(start_idx, end_idx):
            sample = self.samples[i]
            builder = RLVRTinkerEnvGroupBuilder(
                sample=sample,
                renderer=self.renderer,
                judge=self.judge,
                num_envs=self.group_size,
                convo_prefix=self.convo_prefix,
            )
            builders.append(builder)

        return builders


@chz.chz
class RLVRTinkerDatasetBuilder(RLDatasetBuilder):
    """
    Builder for RLVR Tinker datasets with configurable parameters.
    """

    # Model configuration
    model_name_for_tokenizer: str
    renderer_name: str

    # Dataset paths
    train_jsonl_path: str | None = None
    test_jsonl_path: str | None = None

    # Batch configuration
    groups_per_batch: int = 32
    train_group_size: int = 8
    test_group_size: int = 1

    # Judge configuration
    grader_llm_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    judge_max_retries: int = 3
    judge_max_tokens: int = 4096
    judge_temperature: float = 0.7

    # Optional conversation prefix
    system_message: str | None = None

    # Tinker configuration
    base_url: str | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        """
        Build train and test datasets.

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        # Determine paths
        train_path = Path(self.train_jsonl_path) if self.train_jsonl_path else DEFAULT_TRAIN_PATH
        test_path = Path(self.test_jsonl_path) if self.test_jsonl_path else DEFAULT_TEST_PATH

        # Check that data exists
        if not train_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {train_path}\n"
                "Run data conversion first:\n"
                "  python -m tinker_cookbook.recipes.rlvr_tinker.convert_data"
            )

        # Load data
        logger.info(f"Loading training data from {train_path}")
        train_samples = load_jsonl_data(train_path)
        logger.info(f"Loaded {len(train_samples)} training samples")

        test_samples = None
        if test_path.exists():
            logger.info(f"Loading test data from {test_path}")
            test_samples = load_jsonl_data(test_path)
            logger.info(f"Loaded {len(test_samples)} test samples")

        # Create tokenizer and renderer for the policy model
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        # Create judge (shared across train and test)
        judge = TinkerRubricJudge(
            grader_model=self.grader_llm_name,
            max_retries=self.judge_max_retries,
            max_tokens=self.judge_max_tokens,
            temperature=self.judge_temperature,
            base_url=self.base_url,
        )

        # Create conversation prefix if system message provided
        convo_prefix: list[renderers.Message] | None = None
        if self.system_message:
            convo_prefix = [{"role": "system", "content": self.system_message}]

        # Build train dataset
        train_dataset = RLVRTinkerDataset(
            samples=train_samples,
            renderer=renderer,
            judge=judge,
            groups_per_batch=self.groups_per_batch,
            group_size=self.train_group_size,
            convo_prefix=convo_prefix,
        )

        # Build test dataset if available
        test_dataset = None
        if test_samples:
            test_dataset = RLVRTinkerDataset(
                samples=test_samples,
                renderer=renderer,
                judge=judge,
                groups_per_batch=len(test_samples),  # All test samples in one batch
                group_size=self.test_group_size,
                convo_prefix=convo_prefix,
            )

        return train_dataset, test_dataset

