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

import textwrap
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from accelerate import logging
from datasets import Dataset, IterableDataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)

from ...models import AutoModelForCausalLMWithValueHead, create_reference_model, prepare_fsdp
from ...trainer.grpo_trainer import GRPOTrainer
from ...trainer.utils import prepare_deepspeed
from .spo_config import SPOConfig


logger = logging.get_logger(__name__)


class SPOTrainer(GRPOTrainer):
    """
    Trainer for Segment Policy Optimization (SPO).

    SPO is a hybrid approach that combines:
    - Group Relative Policy Optimization (GRPO): Uses sequence-level rewards
    - Proximal Policy Optimization (PPO): Uses a value network for dense feedback
    - Entropy-based segmentation: Segments sequences at high-entropy tokens
    
    The key idea from the 80-20 rule paper is that entropy naturally identifies important
    decision points in a sequence. SPO uses these high-entropy tokens as segment boundaries,
    and all tokens within a segment share the same advantage value computed from a value network.

    Example:

    ```python
    from datasets import load_dataset
    from trl.experimental import SPOConfig, SPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        return [float(len(set(completion))) for completion in completions]

    args = SPOConfig(
        model="Qwen/Qwen2-0.5B-Instruct",
        entropy_percentile=0.95,  # Top 5% highest-entropy tokens are boundaries
        use_value_network=True,
    )

    trainer = SPOTrainer(
        model=args.model,
        reward_funcs=reward_func,
        args=args,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`str | PreTrainedModel`):
            Model to be trained (policy model).
        reward_funcs (`RewardFunc | list[RewardFunc]`):
            Reward functions for computing rewards.
        args ([`SPOConfig`], *optional*):
            Configuration for this trainer.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset for training.
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict`, *optional*):
            Dataset for evaluation.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class for data processing.
        reward_processing_classes ([`~transformers.PreTrainedTokenizerBase`] or `list`, *optional*):
            Processing classes for reward models.
        callbacks (`list` of [`~transformers.TrainerCallback`], *optional*):
            Callbacks for training.
        optimizers (`tuple`, *optional*, defaults to `(None, None)`):
            Optimizer and scheduler tuple.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration.
        rollout_func (`RolloutFunc`, *optional*):
            Custom rollout function.
        value_model (`str | AutoModelForSequenceClassification`, *optional*):
            Pretrained value model (typically a reward model checkpoint). Can be either:
            - A string: Model ID to load with AutoModelForSequenceClassification.from_pretrained()
            - An AutoModelForSequenceClassification instance: Already loaded value/reward model
            If None, falls back to sequence-level advantages (GRPO-like behavior).
    """

    _tag_names = ["trl", "spo"]
    _name = "SPO"
    _paper = {
        "title": "Segment Policy Optimization: Bridging GRPO and PPO with Entropy-based Segmentation",
        "id": "experimental",
        "citation": textwrap.dedent("""\
            @article{spo2025,
                title        = {{Segment Policy Optimization: Bridging GRPO and PPO with Entropy-based Segmentation}},
                author       = {Weixiao Zhan},
                year         = 2025,
                note         = {Experimental implementation}
            }
            """),
    }

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Any,
        args: Optional[SPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple = (None, None),
        peft_config: Optional[Any] = None,
        rollout_func: Optional[Callable] = None,
        value_model: Optional[Union[str, PreTrainedModel]] = None,
    ):
        # Initialize args if not provided
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = SPOConfig(f"{model_name}-SPO")

        # Store SPO-specific parameters
        self.entropy_percentile = args.entropy_percentile
        self.vf_coef = args.vf_coef
        self.gamma = args.gamma
        self.lam = args.lam
        self.normalize_advantages = args.normalize_advantages
        self.segment_value_aggregation = args.segment_value_aggregation
        self.use_grpo_advantages = args.use_grpo_advantages
        self.use_value_network = True  # SPO always uses a value network
 
        # Initialize parent GRPO trainer
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            rollout_func=rollout_func,
        )

        # Initialize value model - REQUIRED for SPO
        # Try value_model parameter first, then fall back to config
        if value_model is None and hasattr(args, 'value_model_path') and args.value_model_path is not None:
            value_model = args.value_model_path
        
        if value_model is not None:
            # Handle both string and model instances
            if isinstance(value_model, str):
                # Load from model ID, using same init kwargs as main model
                model_init_kwargs = args.model_init_kwargs or {}
                self.value_model = AutoModelForSequenceClassification.from_pretrained(
                    value_model, num_labels=1, **model_init_kwargs
                )
            else:
                # Use provided model instance
                self.value_model = value_model
        else:
            # No value model provided - this is an error
            raise ValueError(
                "SPO requires a value model. Please provide one via either:\n"
                "  - Pass `value_model` parameter (str or model) to SPOTrainer, or\n"
                "  - Set `value_model_path` in SPOConfig\n"
                "Typically, you should use a reward model checkpoint trained on your task."
            )

        # Prepare value model for training (note: not yet optimized, see TODO below)
        # TODO: Add value model parameters to optimizer for joint training
        if self.is_deepspeed_enabled:
            self.value_model = prepare_deepspeed(self.value_model, self.accelerator)
        elif self.is_fsdp_enabled:
            self.value_model = prepare_fsdp(self.value_model, self.accelerator)
        else:
            # Remove evaluation_mode=True to allow training, but we still use torch.no_grad()
            # in _compute_loss for now since optimizer doesn't include value model params yet
            self.value_model = self.accelerator.prepare_model(self.value_model, evaluation_mode=True)

        logger.info(f"Initialized SPO trainer with entropy_percentile={self.entropy_percentile}")

    def _get_segment_boundaries(
        self, entropies: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Identify segment boundaries based on entropy percentile.
        
        Args:
            entropies: Tensor of shape (batch_size, seq_len) with per-token entropy
            mask: Binary mask of shape (batch_size, seq_len), 1 for valid tokens
            
        Returns:
            segment_boundaries: Boolean tensor indicating segment boundary positions
            segment_ids: Integer tensor assigning each token to a segment
            
        The entropy_percentile parameter directly specifies the percentile threshold.
        Tokens with entropy >= percentile(entropy_percentile) are marked as boundaries.
        For example, entropy_percentile=0.95 marks tokens at or above the 95th percentile
        (top 5% highest-entropy tokens) as boundaries.
        """
        # Use percentile directly without transformation
        segment_boundaries = self.get_high_entropy_mask(entropies, mask, self.entropy_percentile)
        
        # Create segment IDs by cumulative sum of boundaries
        # Each boundary marks the start of a new segment
        segment_ids = segment_boundaries.long().cumsum(dim=1)
        
        # Ensure padding tokens don't create segments
        segment_ids = segment_ids * mask.long()
        
        return segment_boundaries, segment_ids

    def _compute_segment_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        segment_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute advantages at the segment level using vectorized operations.
        
        Args:
            rewards: Per-token rewards, shape (batch_size, seq_len)
            values: Value predictions, shape (batch_size, seq_len)
            segment_ids: Segment assignment for each token, shape (batch_size, seq_len)
            mask: Binary mask, shape (batch_size, seq_len)
            
        Returns:
            advantages: Per-segment advantages, shape (batch_size, seq_len)
                       All tokens in the same segment share the same advantage
        """
        batch_size, seq_len = rewards.shape
        device = rewards.device
        
        # Flatten for vectorized operations
        flat_rewards = rewards.reshape(-1)
        flat_values = values.reshape(-1)
        flat_segment_ids = segment_ids.reshape(-1)
        flat_mask = mask.reshape(-1).bool()
        
        # Only process valid tokens
        valid_rewards = flat_rewards[flat_mask]
        valid_values = flat_values[flat_mask]
        valid_segment_ids = flat_segment_ids[flat_mask]
        
        # Create unique global segment IDs (to handle multiple sequences)
        # Add batch offset to segment IDs to make them globally unique
        batch_indices = torch.arange(batch_size, device=device).repeat_interleave(seq_len)[flat_mask]
        global_segment_ids = valid_segment_ids + batch_indices * (segment_ids.max() + 1)
        
        # Filter out padding segments (segment_id == 0)
        non_padding_mask = valid_segment_ids > 0
        if non_padding_mask.sum() == 0:
            return torch.zeros_like(rewards)
        
        valid_rewards = valid_rewards[non_padding_mask]
        valid_values = valid_values[non_padding_mask]
        global_segment_ids = global_segment_ids[non_padding_mask]
        
        # Get unique segments and create mapping
        unique_segments, inverse_indices = torch.unique(global_segment_ids, return_inverse=True)
        num_segments = len(unique_segments)
        
        # Sum rewards per segment using scatter_add
        segment_rewards = torch.zeros(num_segments, device=device)
        segment_rewards.scatter_add_(0, inverse_indices, valid_rewards)
        
        # Create position indices for each token (needed for both modes)
        positions = torch.arange(len(valid_values), device=device)
        
        # Get segment values based on mode
        if self.segment_value_aggregation == "last":
            # For each segment, find the last token's value
            # For each segment, find the maximum position (last token)
            segment_last_positions = torch.zeros(num_segments, dtype=torch.long, device=device)
            segment_last_positions.scatter_reduce_(
                0, inverse_indices, positions, reduce="amax", include_self=False
            )
            # Get values at those positions
            segment_values = valid_values[segment_last_positions]
        elif self.segment_value_aggregation == "mean":
            # Average values per segment
            segment_values = torch.zeros(num_segments, device=device)
            segment_counts = torch.zeros(num_segments, device=device)
            segment_values.scatter_add_(0, inverse_indices, valid_values)
            segment_counts.scatter_add_(0, inverse_indices, torch.ones_like(valid_values))
            segment_values = segment_values / segment_counts.clamp(min=1.0)
        else:
            raise ValueError(f"Unknown segment_value_mode: {self.segment_value_aggregation}")
        
        # Bootstrap from next segment's first value
        # Create a shifted version where each segment gets the next segment's first value
        # For the last segment in each sequence, use 0
        next_segment_values = torch.zeros(num_segments, device=device)
        
        # For each segment, find the first token of the next segment
        # This requires identifying segment boundaries within each sequence
        for i in range(num_segments - 1):
            # Check if segment i+1 belongs to the same sequence as segment i
            # by comparing batch indices
            seg_i_mask = inverse_indices == i
            seg_i_plus_1_mask = inverse_indices == i + 1
            
            if seg_i_mask.any() and seg_i_plus_1_mask.any():
                # Get batch index for segment i and i+1
                seg_i_batch = global_segment_ids[seg_i_mask][0] // (segment_ids.max() + 1)
                seg_i_plus_1_batch = global_segment_ids[seg_i_plus_1_mask][0] // (segment_ids.max() + 1)
                
                # Only bootstrap if they're in the same batch
                if seg_i_batch == seg_i_plus_1_batch:
                    # Get first value of next segment
                    next_seg_positions = positions[seg_i_plus_1_mask]
                    first_next_pos = next_seg_positions.min()
                    next_segment_values[i] = valid_values[first_next_pos]
        
        # Compute TD error for each segment
        segment_advantages = segment_rewards + self.gamma * next_segment_values - segment_values
        
        # Broadcast segment advantages back to tokens
        token_advantages = segment_advantages[inverse_indices]
        
        # Scatter back to full tensor
        advantages = torch.zeros_like(rewards)
        flat_advantages = advantages.view(-1)
        
        # Rebuild the non-padding mask for the flattened view
        flat_mask_indices = torch.where(flat_mask)[0]
        non_padding_indices = flat_mask_indices[non_padding_mask]
        flat_advantages[non_padding_indices] = token_advantages
        
        advantages = flat_advantages.view(batch_size, seq_len)
        
        # Normalize advantages if requested - using GLOBAL statistics across all GPUs
        if self.normalize_advantages:
            valid_advantages = advantages[mask.bool()]
            if valid_advantages.numel() > 0:
                # Gather advantages from all GPUs for global normalization
                # Note: We can't use pad_across_processes with a sentinel value for floats
                # Instead, gather the sizes and manually pad
                local_size = torch.tensor([valid_advantages.size(0)], device=device)
                all_sizes = self.accelerator.gather(local_size)
                max_size = all_sizes.max().item()
                
                # Pad locally
                if valid_advantages.size(0) < max_size:
                    padding = torch.zeros(max_size - valid_advantages.size(0), device=device)
                    padded_advantages = torch.cat([valid_advantages, padding])
                else:
                    padded_advantages = valid_advantages
                
                gathered_advantages = self.accelerator.gather(padded_advantages)
                
                # Flatten and take only the valid elements
                total_valid = all_sizes.sum().item()
                gathered_advantages = gathered_advantages.view(-1)[:total_valid]
                
                if gathered_advantages.numel() > 1:
                    # Compute global mean and std
                    mean = gathered_advantages.mean()
                    std = gathered_advantages.std()
                    advantages = (advantages - mean) / (std + 1e-8)
                    advantages = advantages * mask  # Zero out padding
        
        return advantages

    def _compute_loss(self, model, inputs):
        """
        Compute SPO loss with segment-level advantages.
        
        This overrides the GRPO loss computation to use segment-based advantages
        from the value network instead of sequence-level advantages.
        """
        # Get completions and masks
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Compute per-token log probs and entropies
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        # Get segment boundaries based on entropy
        segment_boundaries, segment_ids = self._get_segment_boundaries(entropies, completion_mask)

        # Compute KL divergence if using reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
        else:
            per_token_kl = None

        # Get value predictions from value network
        with torch.no_grad():
            # Handle different value model types
            # Create position_ids to handle padding correctly
            position_ids = attention_mask.cumsum(1) - attention_mask.long()
            masked_input_ids = torch.masked_fill(input_ids, ~attention_mask.bool(), 0)
            
            # Unwrap the value model to access attributes safely
            unwrapped_value_model = self.accelerator.unwrap_model(self.value_model)
            
            # Check if this is AutoModelForCausalLMWithValueHead or AutoModelForSequenceClassification
            if hasattr(unwrapped_value_model, 'v_head'):
                # This is AutoModelForCausalLMWithValueHead - use its forward method directly
                # It returns (lm_logits, loss, value) tuple
                _, _, per_token_values = unwrapped_value_model(
                    input_ids=masked_input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                # Extract values for completion tokens only
                values = per_token_values[:, -logits_to_keep:]  # (batch_size, logits_to_keep)
            else:
                # This is AutoModelForSequenceClassification (reward model style)
                # Get hidden states from the value model backbone
                lm_backbone = getattr(unwrapped_value_model, unwrapped_value_model.base_model_prefix)
                output = lm_backbone(
                    input_ids=masked_input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    return_dict=True,
                    output_hidden_states=True,
                    use_cache=False,
                )
                
                # Get per-token values by applying score head to hidden states
                # Shape: (batch_size, seq_len, 1)
                per_token_values = unwrapped_value_model.score(output.hidden_states[-1])
                per_token_values = per_token_values.squeeze(-1)  # (batch_size, seq_len)
                
                # Extract values for completion tokens only
                values = per_token_values[:, -logits_to_keep:]  # (batch_size, logits_to_keep)

        # Choose between raw rewards or GRPO's normalized advantages
        if self.use_grpo_advantages:
            # Use GRPO's group-normalized advantages as sequence rewards
            sequence_rewards = inputs["advantages"]
        else:
            # Use raw rewards from reward functions (default)
            # SPO computes its own advantages using the value network
            sequence_rewards = inputs["rewards"]
        
        # Create per-token rewards (distribute sequence reward across tokens)
        batch_size = completion_ids.size(0)
        per_token_rewards = torch.zeros_like(completion_ids, dtype=torch.float32)
        for b in range(batch_size):
            valid_mask = completion_mask[b].bool()
            num_valid = valid_mask.sum()
            if num_valid > 0:
                # Distribute sequence reward/advantage across valid tokens
                per_token_rewards[b, valid_mask] = sequence_rewards[b] / num_valid

        # Compute segment-level advantages
        advantages = self._compute_segment_advantages(
            per_token_rewards, values, segment_ids, completion_mask
        )

        # Compute importance sampling ratio
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        
        log_ratio = per_token_logps - old_per_token_logps
        ratio = torch.exp(log_ratio)

        # PPO-style clipped loss with segment advantages
        # Both ratio and advantages are (batch_size, seq_len)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - self.epsilon_low, 1.0 + self.epsilon_high) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2)

        # Add KL penalty if using reference model
        if self.beta != 0.0 and per_token_kl is not None:
            policy_loss = policy_loss + self.beta * per_token_kl

        # Compute value loss
        # Compute returns for value loss
        returns = advantages + values
        value_loss = 0.5 * ((values - returns.detach()) ** 2)
        
        # Mask and reduce
        value_loss = (value_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

        # Mask and reduce policy loss
        policy_loss = (policy_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        
        # Combine losses
        loss = policy_loss + self.vf_coef * value_loss
        loss = loss / self.args.gradient_accumulation_steps

        # Log metrics
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["spo/policy_loss"].append(self.accelerator.gather(policy_loss).mean().item())
        if self.use_value_network:
            self._metrics[mode]["spo/value_loss"].append(self.accelerator.gather(value_loss).mean().item())
        
        # Log segment statistics
        num_segments_per_seq = segment_ids.max(dim=1)[0]
        mean_segments = self.accelerator.gather(num_segments_per_seq.float()).mean().item()
        self._metrics[mode]["spo/mean_segments_per_sequence"].append(mean_segments)
        
        # Log clipping statistics
        is_clipped = ((ratio < 1.0 - self.epsilon_low) | (ratio > 1.0 + self.epsilon_high)) & (advantages != 0)
        clip_fraction = (is_clipped.float() * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        self._metrics[mode]["spo/clip_fraction"].append(self.accelerator.gather(clip_fraction).mean().item())

        return loss
