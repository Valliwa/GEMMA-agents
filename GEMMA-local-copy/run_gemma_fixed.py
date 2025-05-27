import sys
import os
sys.path.insert(0, './packages')

# Set ALL cache directories to current folder
os.environ['HF_HOME'] = './hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = './hf_cache'
os.environ['TORCH_HOME'] = './torch_cache'
os.environ['TRITON_CACHE_DIR'] = './triton_cache'
os.environ['XDG_CACHE_HOME'] = './cache'

# Create cache directories
import pathlib
pathlib.Path('./hf_cache').mkdir(exist_ok=True)
pathlib.Path('./torch_cache').mkdir(exist_ok=True)
pathlib.Path('./triton_cache').mkdir(exist_ok=True)
pathlib.Path('./cache').mkdir(exist_ok=True)

# Disable torch optimizations that cause cache issues
os.environ['TORCHDYNAMO_DISABLE'] = '1'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Loading Gemma model...")
print(f"CUDA available: {torch.cuda.is_available()}")

# Set your token here
token = "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"  # Replace with your actual token

model_name = "google/gemma-2-9b-it"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, cache_dir="./hf_cache")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=token,
    device_map="auto",
    torch_dtype=torch.float16,
    cache_dir="./hf_cache",
    # Disable optimizations that cause cache issues
    torch_compile=False,
    use_cache=True
)

print("Model loaded successfully!")

# Test with a prompt
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
