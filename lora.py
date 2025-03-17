# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# %% [markdown]
# Generate Adapter for train and reference

# %%


model_path = "llama_adapter"  # Update this with the correct path

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# %%


lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Update based on model
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Train the model...

# Save adapter
model.save_pretrained("llama3_lora_adapter")

# %%
%pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# %%
%pip install transformers

# %%
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# %%


config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# %%
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", 
                                             quantization_config=config,
                                             low_cpu_mem_usage=True,
                                             torch_dtype=torch.float16,
                                            device_map="auto",)
model.config.use_cache = False

# %%
%pip install peft

# %%
# from peft import prepare_model_for_kbit_training

# model = prepare_model_for_kbit_training(model)

# %%
# from peft import LoraConfig,PeftModel

# config = LoraConfig(
#     r=32,
#     lora_alpha=8,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM"
# )

# %%
# from peft import get_peft_model

# model = get_peft_model(model, config)

# %%
!huggingface-cli login


# %%
!huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir "llama_adapter"


# %%
from peft import get_peft_model
from peft import LoraConfig,PeftModel

# %%

# Load the adapter.
model = PeftModel.from_pretrained(
    model,
    "llama3_lora_adapter",
    is_trainable=True,
    adapter_name="train1",
)

# %%
# Load the adapter a second time, with a different name, which will be our reference model.
model.load_adapter("llama3_lora_adapter", adapter_name="reference")

# %%
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

# %%
train_dataset = load_dataset("Sivaganesh07/flat_preference", split="train_prefs").select(range(1000))

# %%
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# %%
training_args = DPOConfig(output_dir="llama-3.2-1B-DPO",
                          logging_steps=10,
                          report_to='wandb',
                          model_adapter_name="train1",
                          ref_adapter_name="reference",
                          padding_value=tokenizer.eos_token_id,
                          per_device_train_batch_size=1,  # Reduce to 1
    per_device_eval_batch_size=1,  
    gradient_accumulation_steps=8,  # Adjust to maintain effective batch size
    fp16=True,  # Enable mixed precision
    dataloader_num_workers=1,  # Reduce workers if needed
                          )

# %%
%pip install wandb

# %%
import wandb
wandb.login()

# %%
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)

# %%
#Increase reserved memory
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
os.getenv("PYTORCH_CUDA_ALLOC_CONF")

# %%
trainer.train()

# %%
trainer.save_model("llama-3.2-1B-DPO")

# %% [markdown]
# Merge trained model with based model

# %%
# Note that we reload the model in fp16 
fp16_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)


# Merge adapter with base model
newmodel = PeftModel.from_pretrained(fp16_model, "llama-3.2-1B-DPO/train1")
newmodel_merged = newmodel.merge_and_unload()


# %%
# Move merged model to GPU (or CPU if no GPU available)
newmodel_merged = newmodel_merged.to("cuda" if torch.cuda.is_available() else "cpu")

# Save the fully merged model
newmodel_merged.save_pretrained("merged_llama_dpo")
tokenizer.save_pretrained("merged_llama_dpo")

# %% [markdown]
# To Inference

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "merged_llama_dpo"


# Load model with optimizations
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    # offload_folder="offload",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# %%
def generate_text(prompt, max_new_tokens=1000):
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.5,
            top_k=50,
            top_p=0.9,
            use_cache=True
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# %%
prompt = "E. Nelson, Physical Review 150, 1079 (1966)."
output = generate_text(prompt)
print(output)

# %% [markdown]
# To convert to gguf format

# %%
!python E:/Source/llama.cpp/convert_hf_to_gguf.py merged_llama_dpo

# %%
# run the quantize script
!cd E:/Source/llama.cpp/build/bin && llama-quantize E:/Source/Test1/merged_llama_dpo/Llama-3.2-1B-Instruct-F16.gguf E:/Source/Test1/merged_llama_dpo/ggml-model-Q4_K_M.gguf Q4_K_M

# %% [markdown]
# To start llama

# %%
!cd E:/Source/llama.cpp/build/bin && llama-server -m E:/Source/test_cpp/Llama-FineTuned/ggml-model-Q4_K_M.gguf -c 2048

# %% [markdown]
# To create model file

# %%
tuned_model_path = "E:/Source/Test1/merged_llama_dpo/ggml-model-Q4_K_M.gguf"
sys_message = "You are a personnel AI assistant who helps in answering question in clear and consice manner"

# %%
cmds = []

# %%
base_model = f"FROM {tuned_model_path}"

template = '''TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"
"""'''

params = '''PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|reserved_special_token|>"'''

system = f'''SYSTEM """{sys_message}"""'''

# %%
cmds.append(base_model)
cmds.append(template)
cmds.append(params)
cmds.append(system)

# %%
def generate_modelfile(cmds):
    content = ""
    for command in cmds:
        content += command + "\n"
    print(content)
    with open("Modelfile", "w") as file:
        file.write(content)

# %%
generate_modelfile(cmds)

# %%
import torch
torch.cuda.empty_cache()
torch.cuda.memory_allocated()  # Check allocated memory


# %%

!ollama create llama_DPO_Distilled -f E:/Source/Test1/Modelfile


# %%
!ollama cp llama_DPO_Distilled Sivaganeshk07/dpo_trained_llama
!ollama push Sivaganeshk07/dpo_trained_llama


