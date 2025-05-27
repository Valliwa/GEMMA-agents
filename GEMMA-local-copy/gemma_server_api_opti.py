# Add these optimizations to your gemma_server.py

def generate_response(prompt: str, max_tokens: int = 1024, temperature: float = 0.7, top_p: float = 1.0) -> tuple:
    """Optimized generation for long prompts"""
    try:
        # OPTIMIZATION 1: Truncate very long prompts
        if len(prompt) > 8000:  # ~2000 tokens
            logger.warning(f"⚠️ Truncating long prompt: {len(prompt)} chars -> 8000 chars")
            prompt = prompt[-8000:]  # Keep the end (most recent context)
        
        # OPTIMIZATION 2: Use shorter max_length for tokenization
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048,  # Reduced from 4096
            padding=False
        ).to(model.device)
        
        input_length = inputs['input_ids'].shape[1]
        
        # OPTIMIZATION 3: Limit output tokens for long inputs
        if input_length > 1500:
            max_tokens = min(max_tokens, 512)  # Shorter responses for long inputs
            logger.info(f"🔧 Long input detected ({input_length} tokens), limiting output to {max_tokens}")
        
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
                # OPTIMIZATION 4: Use faster generation settings
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
        
        logger.info(f"✅ Generated {completion_tokens} tokens from {prompt_tokens} input tokens")
        
        return response, prompt_tokens, completion_tokens
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
