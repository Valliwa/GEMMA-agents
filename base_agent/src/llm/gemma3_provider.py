# base_agent/src/llm/gemma3_provider.py
"""
Gemma 3 27B Local Server Provider for SICA
Connects to your local gemma_api_server.py
"""

import requests
import json
import time
import logging
from typing import Dict, Any, List, Optional, Generator
from ..config import GEMMA3_SYSTEM_PROMPTS

logger = logging.getLogger(__name__)

class Gemma3Provider:
    """Gemma 3 27B provider for SICA system"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:8000",
                 timeout: int = 300,
                 max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.model_name = "gemma-3-27b-it"
        
        # Gemma 3 specific settings
        self.max_context_length = 128000
        self.default_max_tokens = 4096
        self.supports_function_calling = True
        
        # Check server availability on initialization
        self._check_server_health()
    
    def _check_server_health(self):
        """Check if Gemma 3 server is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"✅ Gemma 3 server healthy: {health_data.get('status', 'unknown')}")
                
                if health_data.get('gpu_memory'):
                    gpu_info = health_data['gpu_memory']
                    logger.info(f"📊 GPU Memory: {gpu_info.get('allocated_gb', 'unknown')}GB allocated")
                
                return True
            else:
                logger.warning(f"⚠️ Gemma 3 server responded with status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Cannot connect to Gemma 3 server: {str(e)}")
            logger.error("Make sure your gemma_api_server.py is running on localhost:8000")
            return False
    
    def generate_response(self, 
                         messages: List[Dict[str, str]], 
                         max_tokens: Optional[int] = None,
                         temperature: float = 0.7,
                         tools: Optional[List[Dict]] = None,
                         **kwargs) -> str:
        """Generate response using Gemma 3 27B"""
        
        max_tokens = max_tokens or self.default_max_tokens
        
        # Prepare the request payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 1.0)
        }
        
        # Add tools if provided (for function calling)
        if tools:
            payload["tools"] = tools
        
        # Make the request with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"🧠 Generating with Gemma 3 (attempt {attempt + 1}/{self.max_retries})")
                
                response = self.session.post(
                    f"{self.base_url}/v1/messages",
                    json=payload,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Extract content from Claude-style response
                if "content" in result and result["content"]:
                    if isinstance(result["content"], list):
                        content = result["content"][0].get("text", "")
                    else:
                        content = result["content"]
                    
                    # Log usage stats
                    if "usage" in result:
                        usage = result["usage"]
                        logger.info(f"📊 Tokens used: {usage.get('input_tokens', 0)} input + {usage.get('output_tokens', 0)} output")
                    
                    return content.strip()
                else:
                    raise Exception("No content in response")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"⏰ Request timeout (attempt {attempt + 1})")
                if attempt == self.max_retries - 1:
                    raise Exception("Request timed out after all retries")
                time.sleep(2 ** attempt)  # exponential backoff
                
            except requests.exceptions.RequestException as e:
                logger.error(f"🌐 Request error: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise Exception(f"Request failed: {str(e)}")
                time.sleep(2 ** attempt)
                
            except Exception as e:
                logger.error(f"💥 Generation error: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise Exception(f"Generation failed: {str(e)}")
                time.sleep(1)
        
        raise Exception("All retry attempts failed")
    
    def generate_with_openai_format(self,
                                   messages: List[Dict[str, str]],
                                   **kwargs) -> Dict[str, Any]:
        """Generate using OpenAI-compatible format"""
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.default_max_tokens),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0),
            "stream": False
        }
        
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]
        
        try:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            raise Exception(f"OpenAI format generation failed: {str(e)}")
    
    def create_coding_agent_messages(self, 
                                   task: str, 
                                   context: str = "",
                                   tools: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """Create messages optimized for coding tasks"""
        
        # Use Gemma 3 specific system prompt
        system_prompt = GEMMA3_SYSTEM_PROMPTS["coding_agent"]
        
        if tools:
            tools_str = "\n".join([f"- {tool}" for tool in tools])
            system_prompt = system_prompt.format(tools=tools_str)
        else:
            system_prompt = system_prompt.format(tools="None specified")
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add context if provided
        if context:
            messages.append({
                "role": "user", 
                "content": f"Context:\n{context}\n\nTask: {task}"
            })
        else:
            messages.append({
                "role": "user",
                "content": task
            })
        
        return messages
    
    def evaluate_code(self, 
                     code: str, 
                     benchmark_description: str,
                     evaluation_criteria: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate code using Gemma 3's advanced reasoning"""
        
        criteria_text = ""
        if evaluation_criteria:
            criteria_text = f"Evaluation criteria:\n" + "\n".join([f"- {c}" for c in evaluation_criteria])
        
        messages = [
            {"role": "system", "content": GEMMA3_SYSTEM_PROMPTS["benchmark_evaluator"]},
            {"role": "user", "content": f"""Please evaluate this code implementation:

**Benchmark:** {benchmark_description}

**Code to evaluate:**
```python
{code}
```

{criteria_text}

Please provide:
1. Overall score (1-10)
2. Detailed analysis of strengths and weaknesses
3. Specific suggestions for improvement
4. Assessment of correctness, efficiency, and maintainability

Format your response as JSON with the following structure:
{{
    "overall_score": <number>,
    "correctness_score": <number>,
    "efficiency_score": <number>, 
    "readability_score": <number>,
    "strengths": ["<strength1>", "<strength2>"],
    "weaknesses": ["<weakness1>", "<weakness2>"],
    "suggestions": ["<suggestion1>", "<suggestion2>"],
    "detailed_analysis": "<detailed text analysis>"
}}"""}
        ]
        
        try:
            response = self.generate_response(messages, max_tokens=2048)
            
            # Try to parse JSON response
            if response.strip().startswith('{'):
                return json.loads(response)
            else:
                # Fallback: extract JSON from response
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx]
                    return json.loads(json_str)
                else:
                    # Return structured fallback
                    return {
                        "overall_score": 5,
                        "detailed_analysis": response,
                        "error": "Could not parse JSON response"
                    }
                    
        except Exception as e:
            logger.error(f"Code evaluation failed: {str(e)}")
            return {
                "overall_score": 0,
                "error": str(e),
                "detailed_analysis": "Evaluation failed due to technical error"
            }
    
    def plan_improvements(self, 
                         current_performance: Dict[str, Any],
                         benchmark_results: List[Dict[str, Any]],
                         codebase_context: str) -> Dict[str, Any]:
        """Generate improvement plan using Gemma 3's strategic planning"""
        
        # Prepare performance summary
        performance_summary = f"""
Current Performance Summary:
{json.dumps(current_performance, indent=2)}

Recent Benchmark Results:
{json.dumps(benchmark_results[-5:], indent=2)}  # Last 5 results

Codebase Context:
{codebase_context[:5000]}  # First 5K chars
"""
        
        messages = [
            {"role": "system", "content": GEMMA3_SYSTEM_PROMPTS["improvement_planner"]},
            {"role": "user", "content": f"""{performance_summary}

Based on this information, please create a detailed improvement plan:

1. **Priority Analysis**: What are the 3 highest-impact areas for improvement?
2. **Implementation Strategy**: Step-by-step plan for each improvement
3. **Expected Outcomes**: Quantify expected performance gains
4. **Risk Assessment**: Potential challenges and mitigation strategies
5. **Timeline**: Estimated effort and sequencing

Format as JSON:
{{
    "priority_improvements": [
        {{
            "area": "<improvement area>",
            "impact_score": <1-10>,
            "effort_score": <1-10>,
            "description": "<detailed description>",
            "implementation_steps": ["<step1>", "<step2>"], 
            "expected_outcome": "<quantified improvement>"
        }}
    ],
    "implementation_plan": {{
        "phase_1": ["<task1>", "<task2>"],
        "phase_2": ["<task3>", "<task4>"],
        "phase_3": ["<task5>", "<task6>"]
    }},
    "success_metrics": ["<metric1>", "<metric2>"],
    "risk_factors": ["<risk1>", "<risk2>"]
}}"""}
        ]
        
        try:
            response = self.generate_response(messages, max_tokens=3072)
            
            # Parse JSON response
            if response.strip().startswith('{'):
                return json.loads(response)
            else:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx]
                    return json.loads(json_str)
                else:
                    return {
                        "priority_improvements": [],
                        "error": "Could not parse improvement plan",
                        "raw_response": response
                    }
                    
        except Exception as e:
            logger.error(f"Improvement planning failed: {str(e)}")
            return {
                "priority_improvements": [],
                "error": str(e)
            }
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get current server performance statistics"""
        try:
            response = self.session.get(f"{self.base_url}/stats", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Stats endpoint returned {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def __str__(self):
        return f"Gemma3Provider(url={self.base_url}, model={self.model_name})"

# Convenience function for quick testing
def test_gemma3_provider():
    """Test the Gemma 3 provider"""
    provider = Gemma3Provider()
    
    test_messages = [
        {"role": "user", "content": "Write a Python function to calculate the Fibonacci sequence up to n terms. Include proper documentation and error handling."}
    ]
    
    try:
        response = provider.generate_response(test_messages, max_tokens=800)
        print("✅ Gemma 3 Provider Test Successful!")
        print(f"Response: {response[:200]}...")
        return True
    except Exception as e:
        print(f"❌ Gemma 3 Provider Test Failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_gemma3_provider()
