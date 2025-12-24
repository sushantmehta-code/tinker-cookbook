# RLVR with Tinker Sampling Client for Judge

This recipe implements RL with Verifiable Rubrics (RLVR) using **Tinker's sampling client** for the judge model instead of external APIs like OpenAI.

## Key Features

- **Batched judge calls**: 1 API call per response evaluates all rubrics at once (not per-rubric)
- **Binary Yes/No scoring**: Each rubric gets a binary pass/fail, and the pass rate is used as the reward
- **Fault-tolerant rollouts**: Failed judge calls skip samples instead of crashing the training job
- **Fast inference**: Uses `TinkerMessageCompleter` for judge inference on Tinker infrastructure

## Comparison with Other Approaches

| Aspect | `rlvr/` (OpenAI) | `rubric/` (upstream) | `rlvr_tinker/` (this) |
|--------|-----------------|---------------------|----------------------|
| Judge API | OpenAI `gpt-5-mini` | Tinker (per-rubric) | Tinker (batched) |
| Calls per response | 1 (batched) | N (one per rubric) | 1 (batched) |
| Scoring | Binary Yes/No | Continuous 0-1 | Binary Yes/No |
| External dependency | OpenAI API key | None | None |
| Latency | ~500ms+ per call | ~50ms × N rubrics | ~50ms per call |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Loop                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   Dataset    │───▶│ Policy Model │───▶│ Model Response   │  │
│  │  (prompts)   │    │   (Tinker)   │    │                  │  │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘  │
│                                                    │             │
│                                                    ▼             │
│                                          ┌──────────────────┐   │
│                                          │  Tinker Judge    │   │
│                                          │ (1 batched call) │   │
│                                          │                  │   │
│                                          │ Rubric 1: Yes ✓  │   │
│                                          │ Rubric 2: No ✗   │   │
│                                          │ Rubric 3: Yes ✓  │   │
│                                          │ ...              │   │
│                                          └────────┬─────────┘   │
│                                                    │             │
│                                                    ▼             │
│                                          ┌──────────────────┐   │
│                                          │   Pass Rate      │   │
│                                          │ = 2/3 = 0.667    │   │
│                                          │ (used as reward) │   │
│                                          └────────┬─────────┘   │
│                                                    │             │
│                                                    ▼             │
│                                          ┌──────────────────┐   │
│                                          │  Policy Update   │   │
│                                          │   (GRPO/PPO)     │   │
│                                          └──────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Files

- [`convert_data.py`](./convert_data.py) - Converts RLVR JSON to JSONL format
- [`judge.py`](./judge.py) - `TinkerRubricJudge` for batched rubric evaluation
- [`env.py`](./env.py) - Environment classes and fault-tolerant rollouts
- [`train.py`](./train.py) - CLI entry point for training
- [`data/train.jsonl`](./data/train.jsonl) - Training data (generated)
- [`data/test.jsonl`](./data/test.jsonl) - Test data (generated)

## Quick Start

### 1. Convert Data (one-time)

First, convert the RLVR dataset to the JSONL format:

```bash
python -m tinker_cookbook.recipes.rlvr_tinker.convert_data
```

This creates `data/train.jsonl` (900 samples) and `data/test.jsonl` (100 samples).

### 2. Run Training

```bash
python -m tinker_cookbook.recipes.rlvr_tinker.train \
    model_name="Qwen/Qwen3-30B-Instruct-2507" \
    grader_llm_name="Qwen/Qwen3-235B-A22B-Instruct-2507" \
    groups_per_batch=32 \
    train_group_size=8 \
    learning_rate=1e-5 \
    max_tokens=2048 \
    wandb_project="rlvr-tinker"
```

### 3. Monitor Training

- **Wandb**: If `wandb_project` is set, metrics are logged to Weights & Biases
- **Logs**: HTML rollout transcripts are saved to the log directory
- **Metrics**: `metrics.jsonl` in the log directory contains per-step metrics

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `Qwen/Qwen3-30B-Instruct-2507` | Policy model to train |
| `grader_llm_name` | `Qwen/Qwen3-235B-A22B-Instruct-2507` | Judge model for rubric evaluation |
| `groups_per_batch` | 32 | Number of unique prompts per training batch |
| `train_group_size` | 8 | Number of rollouts per prompt (for GRPO) |
| `learning_rate` | 1e-5 | Learning rate for LoRA training |
| `max_tokens` | 2048 | Maximum tokens for policy model response |
| `lora_rank` | 32 | LoRA rank for parameter-efficient training |
| `judge_max_retries` | 3 | Max retries for failed judge calls |
| `judge_temperature` | 0.7 | Temperature for judge model sampling |

## Data Format

### Input (JSONL)

Each line in `train.jsonl` / `test.jsonl`:

```json
{
    "task_id": "uuid",
    "prompt": "User instruction to follow...",
    "rubrics": [
        "The response should correctly calculate...",
        "The response should bold and capitalize...",
        "The response should create exactly 5 groups..."
    ]
}
```

### Judge Evaluation

The judge evaluates all rubrics in a single call and returns:

```json
{
    "ratings": [
        {"rating": "Yes", "rationale": "The calculation is correct..."},
        {"rating": "No", "rationale": "Names are not bolded..."},
        {"rating": "Yes", "rationale": "5 groups were created..."}
    ]
}
```

Pass rate = 2/3 = 0.667 (used as the RL reward)

## Error Handling

The recipe handles judge failures gracefully:

1. **Judge API errors**: If the Tinker judge call fails after retries, a `JudgeError` is raised
2. **Skipped samples**: The fault-tolerant rollout catches `JudgeError` and skips that sample
3. **Training continues**: Valid samples in the batch are used for training; failed samples are excluded
4. **No fake rewards**: Failed judge calls do NOT use default scores like 0.0

This ensures that training data quality is maintained even when some judge calls fail.

## Performance Tips

1. **Increase `groups_per_batch`**: More prompts per batch = better GPU utilization
2. **Adjust `train_group_size`**: More rollouts per prompt = better advantage estimation
3. **Use `num_substeps > 1`**: Multiple optimizer updates per sampling iteration
4. **Monitor KL divergence**: Keep `kl_sample_train_v1` below 0.01 for stable training

## Metrics

Key metrics logged during training:

| Metric | Description |
|--------|-------------|
| `env/all/pass_rate` | Average rubric pass rate across all rollouts |
| `env/all/num_passed` | Average number of rubrics passed per response |
| `env/all/num_rubrics` | Average number of rubrics per prompt |
| `optim/kl_sample_train_v1` | KL divergence (should stay < 0.01) |
| `env/all/reward/total` | Average total reward (same as pass_rate) |

