#!/usr/bin/env python3
"""
SICA Web Interface - Flask Application
Interactive web interface for monitoring and testing SICA agents
"""

import os
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import requests

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sica-web-interface-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
gemma3_status = {
    'available': False,
    'model': 'Unknown',
    'error': 'Not checked yet'
}

system_stats = {
    'gpu_memory': None,
    'last_updated': None
}

@app.route('/')
def dashboard():
    """Main dashboard page"""
    try:
        # Check Gemma 3 status
        check_gemma3_status()
        
        # Get SICA experiments
        experiments = get_sica_experiments()
        
        return render_template('dashboard.html', 
                             gemma3_status=gemma3_status,
                             stats=system_stats,
                             experiments=experiments)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/test_agent')
def test_agent():
    """Agent testing page"""
    return render_template('test_agent.html')

@app.route('/api/test_prompt', methods=['POST'])
def test_prompt():
    """API endpoint to test prompts with Gemma 3"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 500)
        
        if not prompt.strip():
            return jsonify({'success': False, 'error': 'Empty prompt'})
        
        # Record start time
        start_time = time.time()
        
        # Try to use Gemma 3 server
        if gemma3_status['available']:
            response = call_gemma3_api(prompt, max_tokens)
            if response:
                response_time = time.time() - start_time
                return jsonify({
                    'success': True,
                    'content': response['content'],
                    'response_time': response_time,
                    'usage': response.get('usage', {})
                })
        
        # Fallback response if Gemma 3 not available
        fallback_response = generate_fallback_response(prompt)
        response_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'content': fallback_response,
            'response_time': response_time,
            'usage': {'input_tokens': len(prompt.split()), 'output_tokens': len(fallback_response.split())}
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/experiment/<experiment_id>')
def experiment_details(experiment_id):
    """Show details for a specific experiment"""
    try:
        exp_data = get_experiment_data(experiment_id)
        if not exp_data:
            return render_template('error.html', error=f'Experiment {experiment_id} not found')
        
        return render_template('experiment.html', experiment=exp_data)
    except Exception as e:
        return render_template('error.html', error=str(e))

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    # Send initial status
    emit('status_update', {
        'gemma3': gemma3_status,
        'stats': system_stats
    })

@socketio.on('request_status')
def handle_status_request():
    """Handle status update requests"""
    check_gemma3_status()
    update_system_stats()
    emit('status_update', {
        'gemma3': gemma3_status,
        'stats': system_stats
    })

def check_gemma3_status():
    """Check if Gemma 3 server is available"""
    global gemma3_status
    
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            gemma3_status = {
                'available': True,
                'model': data.get('model', 'Gemma 3 27B'),
                'error': None
            }
        else:
            gemma3_status = {
                'available': False,
                'model': 'Unknown',
                'error': f'Server responded with status {response.status_code}'
            }
    except requests.exceptions.RequestException as e:
        gemma3_status = {
            'available': False,
            'model': 'Unknown',
            'error': 'Server not reachable'
        }

def call_gemma3_api(prompt, max_tokens=500):
    """Call Gemma 3 API server"""
    try:
        payload = {
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': 0.7
        }
        
        response = requests.post('http://localhost:8000/generate', 
                               json=payload, 
                               timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Gemma 3 API error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error calling Gemma 3 API: {e}")
        return None

def generate_fallback_response(prompt):
    """Generate a fallback response when Gemma 3 is not available"""
    
    # Simple pattern matching for common requests
    prompt_lower = prompt.lower()
    
    if 'python' in prompt_lower and 'function' in prompt_lower:
        if 'factorial' in prompt_lower:
            return '''def factorial(n):
    """Calculate factorial of n"""
    if n <= 0:
        return 1
    elif n == 1:
        return 1
    else:
        return n * factorial(n - 1)

# Example usage:
print(factorial(5))  # Output: 120'''
        
        elif 'fibonacci' in prompt_lower:
            return '''def fibonacci(n):
    """Generate fibonacci sequence up to n terms"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib_sequence = [0, 1]
    for i in range(2, n):
        fib_sequence.append(fib_sequence[i-1] + fib_sequence[i-2])
    
    return fib_sequence

# Example usage:
print(fibonacci(10))  # Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]'''
    
    # Generic fallback
    return f"""[SICA Web Interface - Fallback Response]

Gemma 3 server is not currently available. This is a simulated response.

Your prompt: "{prompt}"

To get full AI-powered responses:
1. Start your Gemma 3 server: python3 gemma_api_server.py
2. Wait for model loading to complete
3. Refresh this interface

The Gemma 3 server should be accessible at http://localhost:8000"""

def get_sica_experiments():
    """Get list of SICA experiments from results directory"""
    experiments = []
    results_dir = Path('./results')
    
    if results_dir.exists():
        for run_dir in results_dir.glob('run_*'):
            if run_dir.is_dir():
                try:
                    metadata_file = run_dir / 'metadata.json'
                    exp_info = {
                        'id': run_dir.name,
                        'path': str(run_dir),
                        'agent_count': 0,
                        'current_iteration': 0,
                        'gemma3_available': False
                    }
                    
                    if metadata_file.exists():
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                            exp_info.update({
                                'agent_count': len(list(run_dir.glob('agent_*'))),
                                'current_iteration': metadata.get('current_iteration', 0),
                                'gemma3_available': 'gemma3' in metadata.get('llm_models', [])
                            })
                    else:
                        # Count agent directories if no metadata
                        exp_info['agent_count'] = len(list(run_dir.glob('agent_*')))
                    
                    experiments.append(exp_info)
                    
                except Exception as e:
                    print(f"Error reading experiment {run_dir}: {e}")
                    continue
    
    return sorted(experiments, key=lambda x: x['id'], reverse=True)

def get_experiment_data(experiment_id):
    """Get detailed data for a specific experiment"""
    exp_dir = Path(f'./results/{experiment_id}')
    
    if not exp_dir.exists():
        return None
    
    exp_data = {
        'id': experiment_id,
        'path': str(exp_dir),
        'agents': [],
        'benchmarks': {},
        'metadata': {}
    }
    
    # Read metadata
    metadata_file = exp_dir / 'metadata.json'
    if metadata_file.exists():
        with open(metadata_file) as f:
            exp_data['metadata'] = json.load(f)
    
    # Read agent data
    for agent_dir in exp_dir.glob('agent_*'):
        if agent_dir.is_dir():
            agent_info = {
                'name': agent_dir.name,
                'path': str(agent_dir)
            }
            
            # Look for benchmark results
            benchmarks_dir = agent_dir / 'benchmarks'
            if benchmarks_dir.exists():
                for bench_dir in benchmarks_dir.iterdir():
                    if bench_dir.is_dir():
                        perf_file = bench_dir / 'perf.jsonl'
                        if perf_file.exists():
                            try:
                                with open(perf_file) as f:
                                    lines = f.readlines()
                                    if lines:
                                        perf_data = json.loads(lines[-1])  # Last line
                                        agent_info[f'{bench_dir.name}_score'] = perf_data.get('score', 0)
                            except:
                                pass
            
            exp_data['agents'].append(agent_info)
    
    return exp_data

def update_system_stats():
    """Update system statistics"""
    global system_stats
    
    # Try to get GPU memory info (if available)
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines and lines[0]:
                memory_used, memory_total, utilization = lines[0].split(', ')
                system_stats['gpu_memory'] = {
                    'allocated_gb': round(float(memory_used) / 1024, 1),
                    'total_gb': round(float(memory_total) / 1024, 1),
                    'utilization_percent': int(utilization)
                }
    except:
        # GPU monitoring not available
        pass
    
    system_stats['last_updated'] = datetime.now().isoformat()

def periodic_status_update():
    """Periodically update status and emit to clients"""
    while True:
        try:
            check_gemma3_status()
            update_system_stats()
            
            # Emit to all connected clients
            socketio.emit('status_update', {
                'gemma3': gemma3_status,
                'stats': system_stats
            })
            
        except Exception as e:
            print(f"Error in periodic update: {e}")
        
        time.sleep(30)  # Update every 30 seconds

if __name__ == '__main__':
    print("🌐 SICA Web Interface Starting...")
    print("=" * 50)
    
    # Check initial Gemma 3 status
    check_gemma3_status()
    update_system_stats()
    
    if gemma3_status['available']:
        print("✅ Gemma 3 server detected and ready")
    else:
        print("⚠️ Gemma 3 server not available - using fallback responses")
        print("   Start your Gemma 3 server for full functionality")
    
    print("")
    print("🚀 Starting web server...")
    print("📍 Interface available at: http://localhost:8080")
    print("🔧 Features:")
    print("   - Real-time agent monitoring")
    print("   - Interactive prompt testing")
    print("   - SICA experiment analysis")
    print("   - GPU performance tracking")
    print("")
    print("Press Ctrl+C to stop the server")
    print("")
    
    # Start periodic status updates in background thread
    status_thread = threading.Thread(target=periodic_status_update, daemon=True)
    status_thread.start()
    
    # Start the Flask-SocketIO server
    try:
        socketio.run(app, host='0.0.0.0', port=8080, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Server error: {e}")
