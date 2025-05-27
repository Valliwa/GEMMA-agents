#!/usr/bin/env python3
"""
SICA Web Interface
Provides a comprehensive web interface for monitoring and examining agent responses
"""

import os
import json
import time
import asyncio
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from flask import Flask, render_template, request, jsonify, Response, stream_template
from flask_socketio import SocketIO, emit
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sica_web_interface_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
GEMMA3_SERVER_URL = "http://localhost:8000"
RESULTS_DIR = Path("results")
CURRENT_EXPERIMENT = None
MONITORING_ACTIVE = False

class SICAWebInterface:
    """Main interface class for SICA web monitoring"""
    
    def __init__(self):
        self.gemma3_available = False
        self.current_stats = {}
        self.recent_responses = []
        self.experiment_data = {}
        self.monitoring_thread = None
        
    def check_gemma3_status(self) -> Dict[str, Any]:
        """Check Gemma 3 server status"""
        try:
            response = requests.get(f"{GEMMA3_SERVER_URL}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                self.gemma3_available = True
                return {
                    "available": True,
                    "status": health_data.get("status", "unknown"),
                    "model": health_data.get("model", "unknown"),
                    "gpu_memory": health_data.get("gpu_memory", {}),
                    "enhanced_features": health_data.get("enhanced_features", {})
                }
            else:
                self.gemma3_available = False
                return {"available": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            self.gemma3_available = False
            return {"available": False, "error": str(e)}
    
    def get_gemma3_stats(self) -> Dict[str, Any]:
        """Get detailed Gemma 3 performance stats"""
        try:
            response = requests.get(f"{GEMMA3_SERVER_URL}/stats", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def test_agent_response(self, prompt: str, max_tokens: int = 500) -> Dict[str, Any]:
        """Test agent response with custom prompt"""
        try:
            payload = {
                "model": "gemma-3-27b-it",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            start_time = time.time()
            response = requests.post(
                f"{GEMMA3_SERVER_URL}/v1/messages",
                json=payload,
                timeout=120
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result['content'][0]['text'] if isinstance(result['content'], list) else result['content']
                
                response_data = {
                    "success": True,
                    "content": content,
                    "response_time": response_time,
                    "usage": result.get("usage", {}),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Store in recent responses
                self.recent_responses.append({
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "response": content[:200] + "..." if len(content) > 200 else content,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "response_time": f"{response_time:.2f}s",
                    "tokens": result.get("usage", {})
                })
                
                # Keep only last 10 responses
                if len(self.recent_responses) > 10:
                    self.recent_responses.pop(0)
                
                return response_data
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_experiment_list(self) -> List[Dict[str, Any]]:
        """Get list of available experiments"""
        experiments = []
        if RESULTS_DIR.exists():
            for exp_dir in RESULTS_DIR.iterdir():
                if exp_dir.is_dir() and exp_dir.name.startswith("run_"):
                    metadata_file = exp_dir / "metadata.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file) as f:
                                metadata = json.load(f)
                            
                            # Count agents
                            agent_count = len([d for d in exp_dir.iterdir() if d.name.startswith("agent_")])
                            
                            experiments.append({
                                "id": exp_dir.name,
                                "path": str(exp_dir),
                                "agent_count": agent_count,
                                "current_iteration": metadata.get("agent_iteration", 0),
                                "start_time": metadata.get("experiment_start_timestamp", "unknown"),
                                "preferred_model": metadata.get("preferred_model", "unknown"),
                                "gemma3_available": metadata.get("gemma3_available", False)
                            })
                        except Exception as e:
                            print(f"Error reading metadata for {exp_dir}: {e}")
        
        return sorted(experiments, key=lambda x: x["id"], reverse=True)
    
    def get_experiment_details(self, exp_id: str) -> Dict[str, Any]:
        """Get detailed information about an experiment"""
        exp_dir = RESULTS_DIR / exp_id
        if not exp_dir.exists():
            return {"error": "Experiment not found"}
        
        try:
            # Load metadata
            metadata_file = exp_dir / "metadata.json"
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
            
            # Get agent iterations
            agents = []
            for agent_dir in sorted(exp_dir.iterdir()):
                if agent_dir.is_dir() and agent_dir.name.startswith("agent_"):
                    agent_info = self.get_agent_info(agent_dir)
                    agents.append(agent_info)
            
            return {
                "id": exp_id,
                "metadata": metadata,
                "agents": agents,
                "total_agents": len(agents)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_agent_info(self, agent_dir: Path) -> Dict[str, Any]:
        """Get information about a specific agent iteration"""
        try:
            agent_name = agent_dir.name
            
            # Get benchmark results
            benchmarks = {}
            benchmark_dir = agent_dir / "benchmarks"
            if benchmark_dir.exists():
                for bench_dir in benchmark_dir.iterdir():
                    if bench_dir.is_dir():
                        perf_file = bench_dir / "perf.json"
                        if perf_file.exists():
                            with open(perf_file) as f:
                                benchmarks[bench_dir.name] = json.load(f)
            
            # Get meta-improvement logs
            meta_logs = {}
            meta_dir = agent_dir / "meta_improvement_logs"
            if meta_dir.exists():
                summary_file = meta_dir / "summary.txt"
                if summary_file.exists():
                    meta_logs["summary"] = summary_file.read_text()
            
            return {
                "name": agent_name,
                "path": str(agent_dir),
                "benchmarks": benchmarks,
                "meta_improvement": meta_logs,
                "has_code": (agent_dir / "agent_code").exists()
            }
        except Exception as e:
            return {"name": agent_dir.name, "error": str(e)}
    
    def get_benchmark_traces(self, exp_id: str, agent_name: str, benchmark_name: str) -> List[Dict[str, Any]]:
        """Get benchmark execution traces"""
        traces_dir = RESULTS_DIR / exp_id / agent_name / "benchmarks" / benchmark_name / "traces"
        traces = []
        
        if traces_dir.exists():
            for trace_dir in traces_dir.iterdir():
                if trace_dir.is_dir():
                    trace_info = {"problem_id": trace_dir.name}
                    
                    # Read various files
                    files_to_read = ["trace.txt", "answer.txt", "summary.txt", "score.txt"]
                    for filename in files_to_read:
                        file_path = trace_dir / filename
                        if file_path.exists():
                            try:
                                content = file_path.read_text()
                                trace_info[filename.replace('.txt', '')] = content
                            except Exception as e:
                                trace_info[filename.replace('.txt', '')] = f"Error reading file: {e}"
                    
                    traces.append(trace_info)
        
        return traces

# Initialize the interface
sica_interface = SICAWebInterface()

# Routes
@app.route('/')
def index():
    """Main dashboard"""
    gemma3_status = sica_interface.check_gemma3_status()
    experiments = sica_interface.get_experiment_list()
    stats = sica_interface.get_gemma3_stats() if gemma3_status.get("available") else {}
    
    return render_template('dashboard.html',
                         gemma3_status=gemma3_status,
                         experiments=experiments,
                         stats=stats,
                         recent_responses=sica_interface.recent_responses)

@app.route('/test_agent')
def test_agent():
    """Agent testing interface"""
    return render_template('test_agent.html')

@app.route('/api/test_prompt', methods=['POST'])
def test_prompt():
    """Test agent with custom prompt"""
    data = request.get_json()
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 500)
    
    if not prompt:
        return jsonify({"success": False, "error": "No prompt provided"})
    
    result = sica_interface.test_agent_response(prompt, max_tokens)
    return jsonify(result)

@app.route('/experiment/<exp_id>')
def experiment_details(exp_id):
    """Detailed view of an experiment"""
    details = sica_interface.get_experiment_details(exp_id)
    if "error" in details:
        return render_template('error.html', error=details["error"])
    
    return render_template('experiment.html', experiment=details)

@app.route('/api/benchmark_traces/<exp_id>/<agent_name>/<benchmark_name>')
def get_benchmark_traces(exp_id, agent_name, benchmark_name):
    """Get benchmark execution traces"""
    traces = sica_interface.get_benchmark_traces(exp_id, agent_name, benchmark_name)
    return jsonify(traces)

@app.route('/api/status')
def api_status():
    """API endpoint for status updates"""
    return jsonify({
        "gemma3": sica_interface.check_gemma3_status(),
        "stats": sica_interface.get_gemma3_stats(),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/experiments')
def api_experiments():
    """API endpoint for experiments list"""
    return jsonify(sica_interface.get_experiment_list())

# WebSocket events for real-time updates
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status_update', {
        "gemma3": sica_interface.check_gemma3_status(),
        "timestamp": datetime.now().isoformat()
    })

@socketio.on('request_status')
def handle_status_request():
    emit('status_update', {
        "gemma3": sica_interface.check_gemma3_status(),
        "stats": sica_interface.get_gemma3_stats(),
        "recent_responses": sica_interface.recent_responses,
        "timestamp": datetime.now().isoformat()
    })

# Background monitoring
def background_monitoring():
    """Background thread for monitoring updates"""
    while MONITORING_ACTIVE:
        try:
            status_update = {
                "gemma3": sica_interface.check_gemma3_status(),
                "stats": sica_interface.get_gemma3_stats(),
                "timestamp": datetime.now().isoformat()
            }
            socketio.emit('status_update', status_update)
            time.sleep(10)  # Update every 10 seconds
        except Exception as e:
            print(f"Monitoring error: {e}")
            time.sleep(5)

if __name__ == '__main__':
    # Create templates directory and files
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    
    print("🚀 Starting SICA Web Interface...")
    print("📍 Interface will be available at: http://localhost:5000")
    print("🧠 Gemma 3 server expected at: http://localhost:8000")
    print("📊 Monitoring SICA experiments in: ./results/")
    
    # Start background monitoring
    MONITORING_ACTIVE = True
    monitoring_thread = threading.Thread(target=background_monitoring)
    monitoring_thread.daemon = True
    monitoring_thread.start()
    
    # Run the Flask app
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
