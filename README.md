# DPO Trained Model Working Directory

This directory contains the workflow and scripts for training a Direct Preference Optimization (DPO) model.

## Overview

The process consists of the following steps:

1.  **Dataset Preparation:** Converting conversation datasets into preference datasets.
2.  **DPO Training:** Training a DPO model using the prepared dataset.
3.  **Model Merging:** Merging the trained LoRA adapters with the base model.
4.  **Quantization:** Quantizing the merged model into GGUF and Q4_K_M formats.
5.  **Ollama Integration:** Loading the quantized model into Ollama.

## Steps

### 1. Dataset Preparation

* **Script:** `convert_to_preference.py`
* **Description:** This utility script converts conversation datasets into preference datasets suitable for DPO training.
* **Output:** The generated preference dataset is pushed to a Hugging Face repository.
* **Usage:** ex. !python convert_to_preference.py --push_to_hub --repo_id <hf repo>

### 2. DPO Training

* **Notebook:** `lora.ipynb`
* **Description:** This Jupyter Notebook uses the prepared preference dataset to train a DPO model using LoRA adapters.
* **Details:**
    * The DPO trainer utilizes two adapters: one for training and another as a reference model.
    * check lora notebook for ref

### 3. Model Merging

* **Description:** After training is complete, the trained LoRA adapters are merged with the base f16 model.
* **Details:** check lora notebook for ref

### 4. Quantization

* **Description:** The merged model is quantized to GGUF format and then further quantized to Q4_K_M format.
* **Details:**
    * This step reduces the model's size and improves performance for inference.
    * check lora notebook for ref

### 5. Ollama Integration

* **Model:** [https://ollama.com/Sivaganeshk07/dpo_trained_llama](https://ollama.com/Sivaganeshk07/dpo_trained_llama)
* **Description:** The Q4_K_M quantized model is loaded into Ollama, a tool for running large language models locally.
* **Usage:** Ollama run Sivaganeshk07/dpo_trained_llama

## Dependencies

* llama.cpp to build quantized models
* Ollama
* Required packages: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
* pip install transformers
* pip install peft
* pip install wandb
* Increase reserved memory os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
* Hugging face credentials

## Usage

Start with main.ipynb to create datasets and then view lora.ipynb

## Contributing

Verify how DPO Trainer is configured and the contribute more on how to fine tune further.

## License

Open Source
