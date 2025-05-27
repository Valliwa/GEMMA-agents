import sys
import os
sys.path.insert(0, './packages')

# Disable the optimization causing the cache issue
os.environ['TORCHDYNAMO_DISABLE'] = '1'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# The model is already cached, so this should be fast
token = "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"  # Replace with your token
model_name = "google/gemma-2-9b-it"

print("Loading cached model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=token,
    device_map="auto",
    torch_dtype=torch.float16
)

print("Model ready! Enter your prompts (type 'quit' to exit):")

while True:
    prompt = input("\nYour prompt: Write a children story linking quantum computing and consciuosness ")
    if prompt.lower() == 'quit':
        break
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGemma: {response[len(prompt):].strip()}")
