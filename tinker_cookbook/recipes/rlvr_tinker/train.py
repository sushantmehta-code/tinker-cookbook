"""
RLVR Tinker Training Script

CLI entry point for training models with RL using verifiable rubrics,
with Tinker's sampling client for the judge model.

Key features:
- Uses TinkerMessageCompleter for fast judge inference
- Batched judge calls (1 call per response for all rubrics)
- Binary Yes/No scoring with pass rate as reward
- Fault-tolerant rollouts that skip failed judge calls

Usage:
    python -m tinker_cookbook.recipes.rlvr_tinker.train \
        model_name="Qwen/Qwen3-30B-Instruct-2507" \
        grader_llm_name="Qwen/Qwen3-235B-A22B-Instruct-2507" \
        groups_per_batch=16 \
        group_size=8 \
        learning_rate=1e-5 \
        max_tokens=8192
"""

import asyncio
import logging
from datetime import datetime

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.rlvr_tinker.env import (
    RLVRTinkerDatasetBuilder,
    do_fault_tolerant_group_rollout,
)
from tinker_cookbook.rl import rollouts, train

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """CLI configuration for RLVR Tinker training."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    lora_rank: int = 32
    renderer_name: str | None = None

    # Dataset configuration
    train_jsonl_path: str | None = None  # Defaults to bundled data
    test_jsonl_path: str | None = None

    # Training hyperparameters
    groups_per_batch: int = 32
    train_group_size: int = 8
    test_group_size: int = 1
    num_substeps: int = 1
    learning_rate: float = 1e-5
    max_tokens: int = 2048
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0

    # Judge configuration
    grader_llm_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    judge_max_retries: int = 3
    judge_max_tokens: int = 4096
    judge_temperature: float = 0.0

    # Optional system message for the policy model
    system_message: str | None = None

    # Logging configuration
    eval_every: int = 5
    save_every: int = 5
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Resume from checkpoint
    load_checkpoint_path: str | None = None

    # Service configuration
    base_url: str | None = None

    # Random seed
    seed: int = 0


async def main(cli_config: CLIConfig):
    """Main training entry point."""
    # Generate run name and log path
    model_name_short = cli_config.model_name.replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"rlvr-tinker-{model_name_short}-"
        f"gs{cli_config.train_group_size}-"
        f"gpb{cli_config.groups_per_batch}-"
        f"lr{cli_config.learning_rate}-"
        f"{date_and_time}"
    )

    log_path = cli_config.log_path or f"/tmp/tinker-examples/rlvr_tinker/{run_name}"
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Get recommended renderer for the model
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    logger.info(f"Using renderer: {renderer_name}")

    # Build dataset configuration
    dataset_builder = RLVRTinkerDatasetBuilder(
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        train_jsonl_path=cli_config.train_jsonl_path,
        test_jsonl_path=cli_config.test_jsonl_path,
        groups_per_batch=cli_config.groups_per_batch,
        train_group_size=cli_config.train_group_size,
        test_group_size=cli_config.test_group_size,
        grader_llm_name=cli_config.grader_llm_name,
        judge_max_retries=cli_config.judge_max_retries,
        judge_max_tokens=cli_config.judge_max_tokens,
        judge_temperature=cli_config.judge_temperature,
        system_message=cli_config.system_message,
        base_url=cli_config.base_url,
    )

    # Build training configuration
    cfg = train.Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        lora_rank=cli_config.lora_rank,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name or run_name,
        log_path=log_path,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    logger.info("Starting RLVR Tinker training")
    logger.info(f"Policy model: {cli_config.model_name}")
    logger.info(f"Grader model: {cli_config.grader_llm_name}")
    logger.info(f"Groups per batch: {cli_config.groups_per_batch}")
    logger.info(f"Group size: {cli_config.train_group_size}")
    logger.info(f"Learning rate: {cli_config.learning_rate}")
    logger.info(f"Max tokens: {cli_config.max_tokens}")
    logger.info(f"Log path: {log_path}")

    # Monkey-patch do_group_rollout to use the fault-tolerant version
    # This ensures that failed judge calls skip samples rather than crashing
    # We must patch BOTH the module AND the reference in rl/train.py since it imports directly
    from tinker_cookbook.rl import train as rl_train
    rollouts.do_group_rollout = do_fault_tolerant_group_rollout
    rl_train.do_group_rollout = do_fault_tolerant_group_rollout
    logger.info("Using fault-tolerant rollouts (failed judge calls will skip samples)")

    await train.main(cfg)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(main(cli_config))

