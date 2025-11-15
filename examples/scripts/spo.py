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

"""
Example script for training with Segment Policy Optimization (SPO).

SPO is a hybrid approach that combines:
- GRPO's sequence-level rewards
- PPO's value network for dense feedback
- Entropy-based segmentation to identify natural decision boundaries

Run:
    python examples/scripts/spo.py \
        --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
        --output_dir spo-qwen2-0.5b \
        --entropy_percentile 0.95 \
        --use_value_network true \
        --num_train_epochs 1
"""

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from trl.experimental.spo import SPOConfig, SPOTrainer
from trl.rewards import accuracy_reward
from trl.models import AutoModelForCausalLMWithValueHead

def main():
    # Load dataset - explicitly cast to Dataset type
    dataset = load_dataset("yentinglin/aime_2025", split="train")
    assert isinstance(dataset, Dataset), f"Expected Dataset, got {type(dataset)}"
    
    # Preprocess dataset to have the required 'prompt' column
    # The SPO trainer expects a 'prompt' column for the input
    def preprocess_function(examples):
        # Use the 'problem' as the prompt for math problem solving
        examples["prompt"] = examples["problem"]
        return examples
    
    dataset = dataset.map(preprocess_function, batched=True)
    
    # Take enough samples for batch size (need at least 16 for generation_batch_size=16)
    # We'll use 32 samples to have enough for 2 training steps
    dataset = dataset.select(range(min(32, len(dataset))))
    
    # Model checkpoint to use for policy model
    model_checkpoint = "Qwen/Qwen3-0.6B"
    
    # Create value model with the same base as policy model but with a value head
    # NOTE: For testing purposes, we use the same base model for both policy and value.
    # In production, you would typically use a pre-trained reward model checkpoint.
    value_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_checkpoint,
        # Value head initialization parameters
        v_head_init_strategy="normal",  # Initialize value head with normal distribution
        v_head_initializer_range=0.2,   # Standard deviation for value head initialization
        summary_dropout_prob=0.1,       # Dropout for value head
    )
    
    # Configure SPO 
    # 
    # Batch size configuration explanation:
    # - num_generations: Number of completions to generate per prompt (rollouts per question)
    # - generation_batch_size: Number of unique prompts to process per batch
    # - per_device_train_batch_size = generation_batch_size * num_generations
    #
    # Testing with 16 rollouts per question from 16 questions:
    #   - num_generations = 16 (16 rollouts per question)
    #   - generation_batch_size = 16 (16 unique questions) - automatically calculated
    #   - per_device_train_batch_size = 16 * 16 = 256
    config = SPOConfig(
        output_dir="spo-qwen3-0.6b",
        learning_rate=1e-5,
        num_train_epochs=1,
        logging_steps=1,
        max_steps=2,  # Only run 2 steps for testing
        # SPO-specific parameters
        entropy_percentile=0.95,  # Tokens at/above 95th percentile (top 5% entropy) are boundaries
        vf_coef=0.1,              # Value function loss coefficient
        gamma=1.0,                # Discount factor
        lam=0.95,                 # GAE lambda
        normalize_advantages=True,
        segment_value_aggregation="last",  # Use last token's value in each segment
        use_grpo_advantages=False,  # Use raw rewards, not GRPO's normalized advantages
        # Generation parameters
        max_prompt_length=1024,
        max_completion_length=1024,
        temperature=0.7,
        # Training parameters: 16 questions with 16 rollouts each (effective batch size = 256)
        # To avoid CUDA OOM, we use gradient accumulation with smaller mini-batches:
        # - Mini-batch size: 32 (2 questions × 16 rollouts)
        # - Gradient accumulation: 8 steps
        # - Effective batch size: 32 × 8 = 256 (16 questions × 16 rollouts)
        per_device_train_batch_size=32,   # Mini-batch: 2 questions * 16 rollouts
        num_generations=16,               # 16 rollouts per question
        gradient_accumulation_steps=8,    # Accumulate gradients over 8 mini-batches
        bf16=True,
    )
    
    # Initialize trainer with both models using the same base checkpoint
    trainer = SPOTrainer(
        model=model_checkpoint,  # Policy model: regular causal LM
        reward_funcs=accuracy_reward,
        args=config,
        train_dataset=dataset,
        value_model=value_model,  # Value model: same base + value head
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    main()
