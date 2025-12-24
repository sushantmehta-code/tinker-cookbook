"""
RLVR (RL with Verifiable Rubrics) Recipe

This recipe implements reinforcement learning with verifiable rubrics for
instruction-following tasks. It uses an LLM judge (GPT-5-mini) to evaluate
model responses against a set of rubrics, computing a pass rate that serves
as the reward signal for GRPO-style training.
"""

