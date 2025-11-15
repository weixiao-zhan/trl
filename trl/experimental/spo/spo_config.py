# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Optional

from ...trainer.grpo_config import GRPOConfig


@dataclass
class SPOConfig(GRPOConfig):
    """
    Configuration class for Segment Policy Optimization (SPO) Trainer.
    
    SPO combines ideas from GRPO and PPO:
    - Uses sequence-level rewards like GRPO
    - Uses a value network like PPO to provide per-segment advantages
    - Segments sequences based on entropy, treating high-entropy tokens as boundaries
    - All tokens within a segment share the same advantage value
    
    Inherits from GRPOConfig and adds segment-specific parameters.
    
    Args:
        entropy_percentile (`float`, *optional*, defaults to `0.95`):
            Percentile threshold for determining segment boundaries. Tokens with entropy
            at or above this percentile will be marked as segment boundaries.
            
            Valid range: [0.0, 1.0]
            - 0.95: Top 5% highest-entropy tokens are boundaries (recommended default)
            - 0.90: Top 10% highest-entropy tokens are boundaries
            - Higher values: fewer boundaries (coarser segmentation)
            - Lower values: more boundaries (finer segmentation)
        value_model_path (`str`, *optional*):
            Path to a pretrained value model. If None, must be provided via the 
            `value_model` parameter in SPOTrainer. Typically a reward model checkpoint.
        vf_coef (`float`, *optional*, defaults to `0.1`):
            Value function loss coefficient (same as PPO).
        gamma (`float`, *optional*, defaults to `1.0`):
            Discount factor for computing returns.
        lam (`float`, *optional*, defaults to `0.95`):
            Lambda parameter for GAE (Generalized Advantage Estimation).
        normalize_advantages (`bool`, *optional*, defaults to `True`):
            Whether to normalize advantages within each segment.
        segment_value_aggregation (`str`, *optional*, defaults to `"last"`):
            How to aggregate per-token values to get segment-level values. Options:
            - "last": Use the last token's value in each segment
            - "mean": Average all token values within each segment
        use_grpo_advantages (`bool`, *optional*, defaults to `False`):
            Whether to use GRPO's group-normalized advantages as sequence rewards instead of 
            raw rewards from reward functions. When True, SPO will use the normalized advantages
            which already account for group-relative comparisons. When False (default), SPO 
            uses raw rewards and computes its own advantages via the value network.
    """
    
    entropy_percentile: float = 0.95
    value_model_path: Optional[str] = None
    vf_coef: float = 0.1
    gamma: float = 1.0
    lam: float = 0.95
    normalize_advantages: bool = True
    segment_value_aggregation: str = "last"  # "last" or "mean"
    use_grpo_advantages: bool = False
