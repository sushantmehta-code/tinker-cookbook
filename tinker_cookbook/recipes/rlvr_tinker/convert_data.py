"""
Data Converter for RLVR Tinker

Converts the RLVR dataset from the original JSON format to a simpler JSONL format
suitable for the Tinker-based rubric evaluation.

Input format (rlvr/data/single_turn_if_sample_1k.json):
{
    "task_id": "uuid",
    "prompt": "...",
    "rubric - 1. criterion": "The response should...",
    "rubric - 2. criterion": "...",
    ...
}

Output format (JSONL):
{
    "task_id": "uuid",
    "prompt": "User instruction...",
    "rubrics": [
        "The response should correctly calculate...",
        "The response should bold and capitalize...",
        ...
    ]
}

Usage:
    python -m tinker_cookbook.recipes.rlvr_tinker.convert_data
"""

import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
SOURCE_DATA_PATH = Path(__file__).parent.parent / "rlvr" / "data" / "single_turn_if_sample_1k.json"
OUTPUT_DIR = Path(__file__).parent / "data"
TRAIN_OUTPUT_PATH = OUTPUT_DIR / "train.jsonl"
TEST_OUTPUT_PATH = OUTPUT_DIR / "test.jsonl"


def load_rlvr_data(path: Path) -> list[dict]:
    """Load RLVR data from JSON file, handling unescaped tabs."""
    logger.info(f"Loading RLVR data from {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        # Fix unescaped tab characters (common issue in this dataset)
        content = content.replace("\t", "\\t")
        data = json.loads(content)

    logger.info(f"Loaded {len(data)} samples")
    return data


def extract_rubrics(item: dict) -> list[str]:
    """Extract non-empty rubric criteria from a data item."""
    rubrics = []

    # Rubric keys follow pattern: "rubric - N. criterion"
    for key, value in item.items():
        if key.startswith("rubric") and "criterion" in key.lower() and value:
            # Only include non-empty rubrics
            value_str = str(value).strip()
            if value_str:
                rubrics.append(value_str)

    return rubrics


def convert_item(item: dict) -> dict | None:
    """Convert a single RLVR item to the simplified format."""
    prompt = item.get("prompt", "").strip()
    task_id = item.get("task_id", "")

    if not prompt:
        return None

    rubrics = extract_rubrics(item)

    if not rubrics:
        return None

    return {
        "task_id": task_id,
        "prompt": prompt,
        "rubrics": rubrics,
    }


def convert_dataset(
    source_path: Path,
    train_output_path: Path,
    test_output_path: Path,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[int, int]:
    """Convert the RLVR dataset to JSONL format with train/test split."""
    # Load source data
    raw_data = load_rlvr_data(source_path)

    # Convert items
    converted = []
    for item in raw_data:
        converted_item = convert_item(item)
        if converted_item:
            converted.append(converted_item)

    logger.info(f"Converted {len(converted)} valid samples (with prompt + rubrics)")

    # Log rubric statistics
    rubric_counts = [len(item["rubrics"]) for item in converted]
    avg_rubrics = sum(rubric_counts) / len(rubric_counts) if rubric_counts else 0
    max_rubrics = max(rubric_counts) if rubric_counts else 0
    min_rubrics = min(rubric_counts) if rubric_counts else 0
    logger.info(f"Rubrics per sample: min={min_rubrics}, max={max_rubrics}, avg={avg_rubrics:.1f}")

    # Shuffle and split
    rng = random.Random(seed)
    rng.shuffle(converted)

    split_idx = int(len(converted) * train_ratio)
    train_data = converted[:split_idx]
    test_data = converted[split_idx:]

    logger.info(f"Split: {len(train_data)} train, {len(test_data)} test")

    # Ensure output directory exists
    train_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write train data
    with open(train_output_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Wrote train data to {train_output_path}")

    # Write test data
    with open(test_output_path, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Wrote test data to {test_output_path}")

    return len(train_data), len(test_data)


def main():
    """Main entry point for data conversion."""
    logger.info("Starting RLVR data conversion")

    if not SOURCE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Source data not found at {SOURCE_DATA_PATH}\n"
            "Make sure the RLVR dataset is available in tinker_cookbook/recipes/rlvr/data/"
        )

    train_count, test_count = convert_dataset(
        source_path=SOURCE_DATA_PATH,
        train_output_path=TRAIN_OUTPUT_PATH,
        test_output_path=TEST_OUTPUT_PATH,
    )

    logger.info(f"Conversion complete: {train_count} train, {test_count} test samples")
    logger.info(f"Output files:")
    logger.info(f"  - {TRAIN_OUTPUT_PATH}")
    logger.info(f"  - {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

