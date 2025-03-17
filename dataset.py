# %%
from transformers import AutoTokenizer
from trl import apply_chat_template

# %%
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

# %%
example = {
    "prompt": [{"role": "user", "content": "What color is the sky?"}],
    "completion": [{"role": "assistant", "content": "It is blue."}]
}

# %%
apply_chat_template(example, tokenizer)

# %%
dataset_dict = {
    "prompt": [[{"role": "user", "content": "What color is the sky?"}],
               [{"role": "user", "content": "Where is the sun?"}]],
    "completion": [[{"role": "assistant", "content": "It is blue."}],
                   [{"role": "assistant", "content": "In the sky."}]]
}

# %%
from datasets import Dataset
dataset = Dataset.from_dict(dataset_dict)
dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

# %%
dataset_dict

# %%
for d in dataset:
    print(d)

# %%
from huggingface_hub import login
login()

# %%
apply_chat_template(example, AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct"))

# %%
from datasets import load_dataset

ds = load_dataset("flyingfishinwater/ultrafeedback_clean",split="train_prefs")

# %%
ds[0]

# %%
# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch import torch


# %%
%pip install acclera

# %%
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# %%
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# %%
!python convert_to_preference.py --push_to_hub --repo_id Sivaganesh07/flat_preference

# %%
# train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train").select(range(25000))

# %%
train_dataset = load_dataset("Sivaganesh07/flat_preference", split="train_prefs")

# %%
train_dataset[0]

# %%
# keys_to_remove = ["score_chosen", "score_rejected","prompt_id","messages"]
# filtered_data = {k: v for k, v in data.items() if k not in keys_to_remove}

# print(filtered_data)

# %%
# dataset = Dataset.from_pandas(pd.DataFrame(data=filtered_data))

# %%
# def fix_labels(example):
#     example["score_chosen"] = float(example["score_chosen"])  # Convert string to float
#     example["score_rejected"] = float(example["score_rejected"])  # Convert string to float
#     return example

# %%
# train_dataset = train_dataset.map(fix_labels)

# %%
training_args = DPOConfig(output_dir="llama-3.2-1B-DPO",
                          logging_steps=10,
                          report_to='wandb',
                          padding_value=tokenizer.eos_token_id
                          )

# %%
import torch
import gc

gc.collect()
torch.cuda.empty_cache()

# %%
del  tokenizer

# %%
!nvidia-smi


# %%
import torch
torch.cuda.empty_cache()
torch.cuda.memory_allocated()

# %%
taskkill /PID 5728 /F #Kills GPU process

# %%
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)

# %%
%pip install wandb

# %%
import wandb


# %%
wandb.login()

# %%
trainer.train_dataset[0]

# %%
trainer.train()

# %% [markdown]
# Takes lot of time with just DPO Trainer . check if you have cude enabled in the env

# %%
!pip uninstall torch torchvision torchaudio

# %%
%pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# %%
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# %%
%pip install --force-reinstall --upgrade --no-cache-dir --no-deps unsloth unsloth_zoo

# %%
%pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# %% [markdown]
# !pip install pytorch-triton

# %%
%pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# %%
!git clone https://github.com/triton-lang/triton.git;
!cd triton/python;


# %%
bnb_config = BitsAndBytesConfig(
    # load_in_4bit=True,
    # llm_int8_threshold=6.0,
    # llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# %%
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    # load_in_4bit=True,
    quantization_config=bnb_config,
    # attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = False

# %%
%pip install peft

# %%
import gc
import os

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    
)
from trl import ORPOConfig, ORPOTrainer

# %%
# Load the adapter.
model = PeftModel.from_pretrained(
    model
)


