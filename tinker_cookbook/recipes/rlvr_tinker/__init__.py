"""
RLVR with Tinker Sampling Client for Judge

This recipe implements RL with Verifiable Rubrics (RLVR) using Tinker's sampling
client for the judge model instead of external APIs like OpenAI.

Key features:
- Batched judge calls: 1 call per prompt evaluates all rubrics at once
- Binary Yes/No scoring per rubric with pass rate as reward
- Fault-tolerant rollouts that skip failed judge calls
- Uses TinkerMessageCompleter for fast inference on Tinker infrastructure
"""

