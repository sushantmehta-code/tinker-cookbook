"""
RLVR Training Script

CLI entry point for training models with RL using verifiable rubrics.
Uses GPT-5-mini as a judge to evaluate responses against instruction-following rubrics.
"""

import asyncio
import logging
from datetime import datetime

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.rlvr.rlvr_env import (
    RLVRDatasetBuilder,
    do_fault_tolerant_group_rollout,
)
from tinker_cookbook.rl import rollouts, train

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """CLI configuration for RLVR training."""

    # Model configuration
    model_name: str = "meta-llama/Llama-3.1-8B"
    lora_rank: int = 32

    # Dataset configuration
    data_path: str | None = None  # Defaults to bundled dataset
    train_ratio: float = 0.9
    data_seed: int = 42

    # Training hyperparameters
    group_size: int = 8
    groups_per_batch: int = 32
    num_substeps: int = 1
    learning_rate: float = 4e-5
    max_tokens: int = 2048
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0

    # Judge configuration
    judge_model: str = "gpt-5-mini"
    judge_max_retries: int = 5
    judge_temperature: float = 1.0

    # Optional system message for the model
    system_message: str | None = None

    # Logging configuration
    eval_every: int = 7
    save_every: int = 7
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Resume from checkpoint
    load_checkpoint_path: str | None = None


async def main(cli_config: CLIConfig):
    """Main training entry point."""
    # Generate run name and log path
    model_name_short = cli_config.model_name.replace("/", "_")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"rlvr-{model_name_short}-"
        f"gs{cli_config.group_size}-"
        f"gpb{cli_config.groups_per_batch}-"
        f"lr{cli_config.learning_rate}-"
        f"{date_and_time}"
    )

    log_path = cli_config.log_path or f"/tmp/tinker-examples/rlvr/{run_name}"
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Get recommended renderer for the model
    renderer_name = model_info.get_recommended_renderer_name(cli_config.model_name)
    logger.info(f"Using renderer: {renderer_name}")

    # Build dataset configuration
    dataset_builder = RLVRDatasetBuilder(
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        data_path=cli_config.data_path,
        train_ratio=cli_config.train_ratio,
        seed=cli_config.data_seed,
        groups_per_batch=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        judge_model=cli_config.judge_model,
        judge_max_retries=cli_config.judge_max_retries,
        judge_temperature=cli_config.judge_temperature,
        system_message=cli_config.system_message,
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
    )

    logger.info("Starting RLVR training")
    logger.info(f"Model: {cli_config.model_name}")
    logger.info(f"Judge: {cli_config.judge_model}")
    logger.info(f"Group size: {cli_config.group_size}")
    logger.info(f"Groups per batch: {cli_config.groups_per_batch}")
    logger.info(f"Learning rate: {cli_config.learning_rate}")
    logger.info(f"Max tokens: {cli_config.max_tokens}")
    logger.info(f"Log path: {log_path}")

    # Use fault-tolerant rollouts to handle judge failures gracefully
    # This ensures failed judge calls skip samples rather than using fake 0.0 rewards
    rollouts.do_group_rollout = do_fault_tolerant_group_rollout
    logger.info("Using fault-tolerant rollouts (failed judge calls will skip samples)")

    await train.main(cfg)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(main(cli_config))

