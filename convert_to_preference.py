# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from huggingface_hub import ModelCard
from transformers import HfArgumentParser
from datasets import Dataset
import json


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the dataset to the Hugging Face Hub.
        repo_id (`str`, *optional*, defaults to `"trl-lib/tldr-preference"`):
            Hugging Face repository ID to push the dataset to.
        dataset_num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of workers to use for dataset processing.
    """

    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the dataset to the Hugging Face Hub."},
    )
    repo_id: str = field(
        default="trl-lib/tldr-preference",
        metadata={"help": "Hugging Face repository ID to push the dataset to."},
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of workers to use for dataset processing."},
    )

def preprocess_function(examples):
    return {
        "prompt": examples["prompt"][0]["content"],  # Extract the content from the list of dictionaries
        "chosen": examples["chosen"][0]["content"],  # Extract the content from the list of dictionaries
        "rejected": examples["rejected"][0]["content"]  # Extract the content from the list of dictionaries
    }

def to_preference(example):   
    
        
    # prompt = f"'"role"': '"user"', '"content"':{example['prompt']} ?\n\n"      
        
    prompt = [{
        "role" : "user",
        "content": f"{example['prompt']} ?"        
    }]
    
    chosen_idx = 1
    rejected_idx = 1
    chosen = [{
        "role" : example["chosen"][chosen_idx]["role"],
        "content": example["chosen"][chosen_idx]["content"]
    }]
    rejected = [{
        "role" : example["rejected"][rejected_idx]["role"],
        "content": example["rejected"][chosen_idx]["content"]
    }]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

def to_flatpreference(example):   
    
        
    # prompt = f"'"role"': '"user"', '"content"':{example['prompt']} ?\n\n"      
        
    # prompt = [{
    #     "role" : "user",
    #     "content":         
    # }]
    
    chosen_idx = 1
    rejected_idx = 1
    # chosen = {
    #     "role" : example["chosen"][chosen_idx]["role"],
    #     "content": example["chosen"][chosen_idx]["content"]
    # }
    # rejected = {
    #     "role" : example["rejected"][rejected_idx]["role"],
    #     "content": example["rejected"][chosen_idx]["content"]
    # }
    return {"prompt": f"{example['prompt']} ?", "chosen": f"{example["chosen"][chosen_idx]["content"]}", "rejected": f"{example["rejected"][chosen_idx]["content"]}"}

model_card = ModelCard("""
---
tags: [trl]
---

## Summary

This dataset is modified version of HuggingFaceH4/ultrafeedback_binarized, specifically curated to train models using the [TRL library](https://github.com/huggingface/trl) for preference learning and Reinforcement Learning from Human Feedback (RLHF) tasks. Providing a rich source of paired text data for training models to understand and generate concise summaries.

## Data Structure

- **Format**: [Standard](https://huggingface.co/docs/trl/main/dataset_formats#standard)
- **Type**: [Preference](https://huggingface.co/docs/trl/main/dataset_formats#preference)

Columns:
- `"prompt"`: The unabridged Reddit post.
- `"chosen"`: The concise "TL;DR" summary appended by the author.
- `"rejected"`: An alternative summary or response that was not selected.

This structure enables models to learn the relationship between detailed content and its abbreviated form, enhancing their summarization capabilities.

## Generation script

The script used to generate this dataset can be found [here](https://github.com/huggingface/trl/blob/main/examples/datasets/tldr_preference.py).
""")

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs").select(range(25000))

    dataset = dataset.map(
        to_flatpreference,
        num_proc=script_args.dataset_num_proc,
        remove_columns=["prompt_id","messages","score_chosen","score_rejected"],
    )

    # # Convert the dictionary to a Hugging Face Dataset
    # dataset = Dataset.from_dict(dataset)

    # # Apply the preprocessing function
    # processed_dataset = dataset.map(preprocess_function)

    # print(dataset[0])

    if script_args.push_to_hub:
        dataset.push_to_hub(script_args.repo_id)
        model_card.push_to_hub(script_args.repo_id, repo_type="dataset")
