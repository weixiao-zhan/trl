# Segment Policy Optimization (SPO)

## Overview

Segment Policy Optimization (SPO) is a hybrid reinforcement learning algorithm that combines ideas from:
- **GRPO (Group Relative Policy Optimization)**: Uses sequence-level rewards
- **PPO (Proximal Policy Optimization)**: Uses a value network for dense per-token feedback
- **Entropy-based Segmentation**: Uses high-entropy tokens as natural segment boundaries

## Motivation

The key insight comes from the **80-20 rule paper**: entropy naturally identifies important decision points in a sequence. In language generation:
- **High-entropy tokens** indicate uncertain/important decisions (e.g., choosing between multiple plausible words)
- **Low-entropy tokens** indicate more deterministic continuations (e.g., completing a phrase)

SPO leverages this by:
1. Using entropy to segment sequences at critical decision points
2. Treating each segment as a unit with shared advantages
3. Using a value network (like PPO) to provide segment-level feedback

This approach bridges GRPO's sequence-level rewards and PPO's token-level value estimates, providing a middle ground that respects the natural structure of language.

## How It Works

### 1. Entropy-Based Segmentation

```python
# Example: High entropy tokens become segment boundaries
# Token:     "The"  "cat"  "is"   "very" "|"    "cute" "and"  "fluffy"
# Entropy:    0.2    0.3    0.1    2.8   |      0.4    0.2    0.5
#                                  ^-- Top 5% entropy (boundary)
# Segments:  [----Segment 1-----]  | [----Segment 2----------]
```

### 2. Value Network

A separate value network estimates the expected return for each segment. This provides richer feedback than sequence-level rewards alone.

### 3. Segment-Level Advantages

All tokens within the same segment share the same advantage value:
```python
Advantage(segment) = Reward(segment) + γ * Value(next_segment) - Value(segment)
```

### 4. Policy Update

Uses PPO-style clipped objective with segment advantages:
```python
Loss = -min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage) + β * KL + α * value_loss
```

## Usage

### Basic Example

```python
from datasets import load_dataset
from trl.experimental.spo import SPOConfig, SPOTrainer

# Load dataset
dataset = load_dataset("trl-lib/tldr", split="train")

# Define reward function
def reward_func(completions, **kwargs):
    return [float(len(set(completion))) for completion in completions]

# Configure SPO
config = SPOConfig(
    output_dir="spo-model",
    learning_rate=1e-5,
    entropy_percentile=0.05,  # 5% of tokens (highest entropy) are boundaries
    use_value_network=True,
    vf_coef=0.1,  # Value function loss coefficient
    gamma=1.0,    # Discount factor
    lam=0.95,     # GAE lambda
)

# Initialize trainer
trainer = SPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_func,
    args=config,
    train_dataset=dataset,
)

# Train
trainer.train()
```

### Advanced: Using a Pretrained Value Model

For best results, provide a pretrained value model (typically a reward model checkpoint):

```python
from transformers import AutoModelForSequenceClassification

# Load pretrained value model (reward model checkpoint)
value_model = AutoModelForSequenceClassification.from_pretrained(
    "path/to/reward/model",
    num_labels=1
)

trainer = SPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_func,
    args=config,
    train_dataset=dataset,
    value_model=value_model,  # Use pretrained value model
)
```

Alternatively, provide the path in the config:

```python
config = SPOConfig(
    output_dir="spo-model",
    value_model_path="path/to/reward/model",  # Load value model from path
    entropy_percentile=0.05,  # 5% of tokens are boundaries
)
```

### Without Value Network

If no value model is provided, SPO falls back to using sequence-level advantages (similar to GRPO but with entropy-based segmentation):

```python
config = SPOConfig(
    output_dir="spo-model",
    use_value_network=True,  # Will fall back if no model provided
    entropy_percentile=0.05,  # 5% of tokens are boundaries
)

# No value_model parameter - will use sequence-level advantages
trainer = SPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_func,
    args=config,
    train_dataset=dataset,
)
```

⚠️ **Note**: Without a value model, SPO still performs entropy-based segmentation but uses simpler advantage computation. For best results, train a reward model first and use it as the value model.

## Configuration

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entropy_percentile` | float | 0.05 | Fraction of tokens to use as boundaries. E.g., 0.05 means top 5% highest-entropy tokens are boundaries. Edge cases: 0.0 = GRPO-like (no segmentation), 1.0 = PPO-like (per-token). |
| `use_value_network` | bool | True | Whether to use a value network for computing advantages. If no value model is provided, falls back to sequence-level advantages. |
| `value_model_path` | str | None | Path to pretrained value model (typically a reward model checkpoint). If None and no value_model is provided, falls back to not using a value network. |
| `vf_coef` | float | 0.1 | Value function loss coefficient (only used when value network is available). |
| `gamma` | float | 1.0 | Discount factor for computing returns. |
| `lam` | float | 0.95 | Lambda parameter for GAE (Generalized Advantage Estimation). |
| `normalize_advantages` | bool | True | Whether to normalize advantages within each batch. |

All other parameters are inherited from `GRPOConfig`.

## Comparison with GRPO and PPO

| Aspect | GRPO | PPO | SPO |
|--------|------|-----|-----|
| **Reward Signal** | Sequence-level | Token-level | Segment-level |
| **Value Network** | ❌ | ✅ | ✅ |
| **Advantage Computation** | Group-relative | Per-token GAE | Per-segment GAE |
| **Segmentation** | N/A | N/A | Entropy-based |
| **Use Case** | Simple, fast | Dense feedback | Balanced approach |

## When to Use SPO

**Use SPO when:**
- You want richer feedback than sequence-level rewards (GRPO)
- But don't need full token-level granularity (PPO)
- Your tasks have natural decision points that entropy can capture
- You want to balance computational efficiency with feedback quality

**Use GRPO when:**
- You want the simplest, fastest approach
- Sequence-level rewards are sufficient
- You don't want to train a value network

**Use PPO when:**
- You need maximum feedback granularity
- You have dense per-token rewards
- Computational cost is less of a concern

## Logged Metrics

SPO logs the following metrics during training:

- `spo/policy_loss`: Policy gradient loss
- `spo/value_loss`: Value function loss (if using value network)
- `spo/mean_segments_per_sequence`: Average number of segments per sequence
- `spo/clip_fraction`: Fraction of clipped probability ratios
- Standard GRPO metrics (rewards, KL divergence, entropy, etc.)

## Implementation Details

### Segment Boundary Detection

```python
def _get_segment_boundaries(entropies, mask, percentile):
    # 1. Compute global entropy threshold
    threshold = quantile(entropies[mask], percentile)
    
    # 2. Mark high-entropy tokens as boundaries
    boundaries = entropies >= threshold
    
    # 3. Assign segment IDs via cumulative sum
    segment_ids = boundaries.cumsum(dim=1)
    
    return boundaries, segment_ids
```

### Segment Advantage Computation

```python
def _compute_segment_advantages(rewards, values, segment_ids):
    for each segment:
        # Sum rewards within segment
        seg_reward = rewards[segment].sum()
        
        # Get segment value (last token's value)
        seg_value = values[segment][-1]
        
        # Bootstrap from next segment
        next_value = values[next_segment][0]
        
        # Compute TD error
        advantage = seg_reward + γ * next_value - seg_value
        
        # All tokens in segment share this advantage
        advantages[segment] = advantage
    
    return advantages
```

## Experimental Status

⚠️ **Note**: SPO is an experimental feature. The API and implementation may change in future releases.

## Citation

If you use SPO in your research, please cite:

```bibtex
@article{spo2025,
    title        = {{Segment Policy Optimization: Bridging GRPO and PPO with Entropy-based Segmentation}},
    author       = {},
    year         = 2025,
    note         = {Experimental implementation in TRL}
}
```

## References

- [DeepSeekMath: GRPO Paper](https://arxiv.org/abs/2402.03300)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
- [80-20 Rule Paper on Entropy and RL]

## Contributing

SPO is in active development. Contributions, bug reports, and feedback are welcome!
