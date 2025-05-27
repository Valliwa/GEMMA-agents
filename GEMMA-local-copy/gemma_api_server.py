import sys
import os
import pathlib
import torch

sys.path.insert(0, './packages')

# Set ALL cache directories to current folder BEFORE importing anything
os.environ['HF_HOME'] = './hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = './hf_cache'
os.environ['TORCH_HOME'] = './torch_cache'
os.environ['TRITON_CACHE_DIR'] = './triton_cache'
os.environ['XDG_CACHE_HOME'] = './cache'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'

# Disable all PyTorch optimizations that cause cache issues
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

# Create all cache directories
cache_dirs = ['./hf_cache', './torch_cache', './triton_cache', './cache']
for cache_dir in cache_dirs:
    pathlib.Path(cache_dir).mkdir(exist_ok=True)



# Add this to the TOP of your gemma_server.py, right after the cache directory setup:
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import json
import time
from typing import List, Dict, Any, Optional
import logging

# FORCE SINGLE GPU USAGE for better performance
os.environ['CUDA_VISIBLE_DEVICES'] = '5'  # Use GPU 7 since it's showing 96% utilization
print(f"🎯 FORCING GPU 5 usage for optimal performance")

# Set PyTorch to use only one GPU
torch.cuda.set_device(0)  # This will map to GPU 7 due to CUDA_VISIBLE_DEVICES

app = FastAPI(
    title="Gemma API Server", 
    description="Local Gemma 2 9B API Server - Claude-compatible endpoint",
    version="1.0.0"
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Global variables for model
tokenizer = None
model = None

# Pydantic models for API compatibility with Anthropic's API
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gemma-2-9b-it"
    messages: List[Message]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Usage

# Claude-style API request (for compatibility with existing agent code)
class ClaudeRequest(BaseModel):
    model: str = "gemma-2-9b-it"
    max_tokens: int = 1024
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0

@app.on_event("startup")
async def load_model():
    """Load the Gemma model on startup without quantization to avoid cache issues"""
    global tokenizer, model
    
    logger.info("Loading Gemma model (without quantization to avoid cache issues)...")
    
    # Replace with your actual Hugging Face token
    token = "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"  # TODO: Replace with your token
    model_name = "google/gemma-2-9b-it"
    
    try:
        # Load without quantization first to avoid cache permission issues
        # The model will still use float16 which saves significant memory
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=token,
            cache_dir="./hf_cache"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            device_map="auto",
            torch_dtype=torch.float16,  # This alone saves ~50% memory
            cache_dir="./hf_cache",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info("✅ Gemma model loaded successfully!")
        logger.info(f"🔧 Model device: {next(model.parameters()).device}")
        logger.info(f"💾 Memory usage: ~9GB (float16, no quantization)")
        logger.info("📝 If you need less memory, you can try quantization later")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        
        # Fallback: Try with explicit device placement
        try:
            logger.info("🔄 Trying fallback loading method...")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                token=token,
                cache_dir="./hf_cache"
            )
            
            # Try loading on CPU first, then move to GPU
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=token,
                torch_dtype=torch.float16,
                cache_dir="./hf_cache",
                trust_remote_code=True,
                device_map="cpu"  # Load on CPU first
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.to("cuda")
                logger.info("✅ Model moved to GPU")
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            logger.info("✅ Gemma model loaded successfully with fallback method!")
            
        except Exception as e2:
            logger.error(f"❌ Fallback loading also failed: {str(e2)}")
            raise e2

def format_messages_for_gemma(messages: List[Message]) -> str:
    """Convert chat messages to Gemma's expected format"""
    formatted_prompt = ""
    
    for message in messages:
        if message.role == "system":
            formatted_prompt += f"<start_of_turn>user\nSystem: {message.content}<end_of_turn>\n"
        elif message.role == "user" or message.role == "human":
            formatted_prompt += f"<start_of_turn>user\n{message.content}<end_of_turn>\n"
        elif message.role == "assistant":
            formatted_prompt += f"<start_of_turn>model\n{message.content}<end_of_turn>\n"
    
    # Add the model turn
    formatted_prompt += "<start_of_turn>model\n"
    return formatted_prompt

# Replace your generate_response function in gemma_server.py with this optimized version:

def generate_response(prompt: str, max_tokens: int = 1024, temperature: float = 0.7, top_p: float = 1.0) -> tuple:
    """Optimized generation for better performance"""
    try:
        # OPTIMIZATION 1: Truncate very long prompts
        original_length = len(prompt)
        if len(prompt) > 4000:
            prompt = "..." + prompt[-4000:]
            logger.info(f"⚡ Truncated prompt: {original_length} -> {len(prompt)} chars")
        
        # OPTIMIZATION 2: Shorter tokenization window
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000).to(model.device)
        input_length = inputs['input_ids'].shape[1]
        
        # OPTIMIZATION 3: Limit output tokens
        max_tokens = min(max_tokens, 300)
        
        logger.info(f"⚡ Generating: {input_length} input -> max {max_tokens} output tokens")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                use_cache=True,
                early_stopping=True
            )
        
        # Decode only the new tokens (response)
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        # Clean up the response
        response = response.strip()
        if response.endswith("<end_of_turn>"):
            response = response[:-13].strip()
        
        prompt_tokens = input_length
        completion_tokens = len(outputs[0]) - input_length
        
        logger.info(f"✅ Generated {completion_tokens} tokens")
        
        return response, prompt_tokens, completion_tokens
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# ALSO ADD: Memory cleanup after each request
@app.middleware("http")
async def cleanup_memory(request: Request, call_next):
    response = await call_next(request)
    
    # Clear GPU cache after each request
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return response       

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "model": "gemma-2-9b-it",
        "description": "Local Gemma API Server - Claude-compatible",
        "endpoints": ["/v1/chat/completions", "/v1/messages", "/health"],
        "memory_usage": "~9GB (float16, no quantization)"
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "gpu_available": torch.cuda.is_available(),
        "device": str(next(model.parameters()).device) if model else None,
        "cache_dirs_writable": all(os.access(d, os.W_OK) for d in ['./hf_cache', './torch_cache', './triton_cache', './cache'] if os.path.exists(d))
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Format messages for Gemma
        prompt = format_messages_for_gemma(request.messages)
        
        # Generate response
        response_text, prompt_tokens, completion_tokens = generate_response(
            prompt, 
            request.max_tokens or 1024,
            request.temperature or 0.7,
            request.top_p or 1.0
        )
        
        # Create OpenAI-compatible response
        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/messages")
async def claude_messages(request: ClaudeRequest):
    """Claude-compatible messages endpoint for the coding agent"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Format messages for Gemma
        prompt = format_messages_for_gemma(request.messages)
        
        # Generate response
        response_text, prompt_tokens, completion_tokens = generate_response(
            prompt,
            request.max_tokens,
            request.temperature or 0.7,
            request.top_p or 1.0
        )
        
        # Claude-style response format
        return {
            "id": f"msg_{int(time.time())}",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": response_text
                }
            ],
            "model": request.model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens
            }
        }
        
    except Exception as e:
        logger.error(f"Claude messages error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Simple test endpoint
@app.post("/test")
async def test_generation(prompt: str = "Hello, how are you?"):
    """Simple test endpoint"""
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        response, _, _ = generate_response(prompt, max_tokens=100)
        return {"prompt": prompt, "response": response}
    except Exception as e:
        return {"error": str(e)}

# Middleware for logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}s")
    return response

if __name__ == "__main__":
    print("🚀 Starting Gemma API Server...")
    print("📍 This will run on http://localhost:8000")
    print("🔧 Compatible with Claude API endpoints")
    print("💾 Using float16 model (~9GB VRAM, no quantization)")
    print("📝 All cache directories set to current folder")
    print("\n⚠️  IMPORTANT: Replace 'your_token_here' with your actual Hugging Face token!")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
