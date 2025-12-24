# RLVR: RL with Verifiable Rubrics

This recipe implements reinforcement learning with verifiable rubrics for instruction-following tasks. It uses an LLM judge (GPT-5-mini by default) to evaluate model responses against a set of rubrics, computing a pass rate that serves as the reward signal for GRPO-style training.

## Overview

RLVR trains models to better follow complex instructions by:

1. **Sampling**: The model generates a response to an instruction
2. **Judging**: GPT-5-mini evaluates the response against all rubrics in a single API call
3. **Rewarding**: The rubric pass rate (0-1) becomes the reward signal
4. **Training**: GRPO-style updates improve the model based on which responses earned higher pass rates

## Dataset

The dataset (`data/single_turn_if_sample_1k.json`) contains 1000 instruction-following prompts, each with 15-40 rubrics that define what a good response should satisfy. For example:

- **Prompt**: "Analyze the following budget data and identify councils responsible for housing..."
- **Rubrics**:
  - "The response should list council types responsible for housing..."
  - "The response should identify the four exceptions to the standard rule..."
  - etc.

## Prerequisites

1. **Tinker API Key**: Set `TINKER_API_KEY` environment variable
2. **OpenAI API Key**: Set `OPENAI_API_KEY` environment variable (for GPT-5-mini judge)

```bash
export TINKER_API_KEY=your_tinker_key
export OPENAI_API_KEY=your_openai_key
```

## Usage

### Basic Training

```bash
python -m tinker_cookbook.recipes.rlvr.train \
    model_name="meta-llama/Llama-3.1-8B" \
    group_size=8 \
    groups_per_batch=32 \
    learning_rate=4e-5 \
    max_tokens=2048
```

### With Qwen Model

```bash
python -m tinker_cookbook.recipes.rlvr.train \
    model_name="Qwen/Qwen3-8B" \
    group_size=16 \
    groups_per_batch=32 \
    learning_rate=2e-5 \
    max_tokens=2048
```

### With Custom Dataset

```bash
python -m tinker_cookbook.recipes.rlvr.train \
    model_name="meta-llama/Llama-3.1-8B" \
    data_path="/path/to/your/dataset.json" \
    train_ratio=0.9
```

### With Weights & Biases Logging

```bash
python -m tinker_cookbook.recipes.rlvr.train \
    model_name="meta-llama/Llama-3.1-8B" \
    wandb_project="my-rlvr-experiments" \
    wandb_name="llama-8b-run1"
```

## CLI Arguments

### Model Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `model_name` | `meta-llama/Llama-3.1-8B` | Tinker-supported model to train |
| `lora_rank` | `32` | LoRA rank for fine-tuning |

### Dataset Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `data_path` | Built-in dataset | Path to custom RLVR dataset JSON |
| `train_ratio` | `0.9` | Fraction of data for training |
| `data_seed` | `42` | Random seed for train/test split |

### Training Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `group_size` | `8` | Number of samples per prompt for GRPO |
| `groups_per_batch` | `32` | Number of prompts per training batch |
| `learning_rate` | `4e-5` | Learning rate |
| `max_tokens` | `2048` | Maximum tokens for model response |
| `temperature` | `1.0` | Sampling temperature |
| `kl_penalty_coef` | `0.0` | KL penalty coefficient (0 = disabled) |
| `num_substeps` | `1` | Optimizer substeps per batch |

### Judge Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `judge_model` | `gpt-5-mini` | OpenAI model for rubric evaluation |
| `judge_max_retries` | `3` | Max retries on judge API failure |
| `judge_temperature` | `0.0` | Judge sampling temperature |

### Logging Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `log_path` | Auto-generated | Directory for logs and checkpoints |
| `wandb_project` | `None` | W&B project name |
| `wandb_name` | Auto-generated | W&B run name |
| `eval_every` | `10` | Evaluation frequency (steps) |
| `save_every` | `20` | Checkpoint save frequency (steps) |

## Expected Results

With the default configuration on Llama-3.1-8B:

- **Initial pass rate**: ~0.3-0.4 (baseline instruction following)
- **After training**: ~0.5-0.6+ (improved instruction following)

The exact improvements depend on:
- Model size and capability
- Number of training steps
- Hyperparameter tuning
- Difficulty of rubrics in the dataset

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  RLVR Dataset   │────>│  Model Sampling  │────>│  GPT-5-mini     │
│  (1000 prompts  │     │  (Tinker API)    │     │  Judge          │
│   + rubrics)    │     │                  │     │  (OpenAI API)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          v
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Updated Model  │<────│  GRPO Training   │<────│  Rubric Pass    │
│  Weights        │     │  (Tinker API)    │     │  Rate (0-1)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Custom Dataset Format

To use your own dataset, create a JSON file with an array of objects:

```json
[
  {
    "task_id": "unique-id-1",
    "prompt": "Your instruction here...",
    "rubric - 1. criterion": "First rubric criterion",
    "rubric - 2. criterion": "Second rubric criterion",
    ...
  },
  ...
]
```

Each object should have:
- `task_id`: Unique identifier (optional but recommended)
- `prompt`: The instruction/prompt for the model
- `rubric - N. criterion`: Rubric criteria (any number, N = 1, 2, 3, ...)

## Cost Considerations

This recipe uses OpenAI's GPT-5-mini as the judge. Cost depends on:

- **Prompts per batch**: `groups_per_batch × group_size`
- **Tokens per judge call**: ~500-2000 input + ~500-1000 output
- **Training steps**: Number of batches in the dataset

For rough estimation with 32 groups × 8 samples = 256 judge calls per batch:
- Each call: ~$0.0001-0.0005
- Per batch: ~$0.03-0.15
- Full training (31 batches): ~$1-5

## Troubleshooting

### OpenAI API Rate Limits

If you hit rate limits, reduce `group_size` or `groups_per_batch`, or add delays between batches.

### JSON Parse Errors in Dataset

The included dataset may have special characters. The loader automatically handles unescaped tabs. For custom datasets, ensure proper JSON escaping.

### Out of Memory

Reduce `max_tokens` or use a smaller model. The judge calls are independent of model memory.

