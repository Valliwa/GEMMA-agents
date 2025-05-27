import sys
import os
sys.path.insert(0, './packages')

# Disable optimizations
os.environ['TORCHDYNAMO_DISABLE'] = '1'

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Configure quantization for memory efficiency
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

token = "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"  # Replace with your token
model_name = "google/gemma-2-9b-it"

print("Loading quantized model (this will use ~5GB instead of ~18GB)...")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=token,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=quantization_config
)

print("Quantized model ready! Enter your prompts (type 'quit' to exit):")

while True:
    prompt = input("\nYour prompt: ")
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
