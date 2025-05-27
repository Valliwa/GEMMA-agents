import sys
import os
sys.path.insert(0, './packages')

# Set custom cache directory
os.environ['HF_HOME'] = './hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = './hf_cache'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Loading Gemma model...")
print(f"CUDA available: {torch.cuda.is_available()}")

# Set your token here (replace with your actual token)
token = "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"  # Replace this with your actual token

model_name = "google/gemma-2-9b-it"

# Load tokenizer and model with token
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=token,
    device_map="auto",
    torch_dtype=torch.float16,
    cache_dir="./hf_cache"
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
print("\n" + "="*50)
print("RESPONSE:")
print("="*50)
print(response)
