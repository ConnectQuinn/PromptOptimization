#!/usr/bin/env python3
"""
System Optimization Monitor

Tracks API limits and machine performance while experiments run.
Run this alongside your main experiment to monitor resource usage.

Usage:
    python src/system_optimization.py
    # or
    uv run src/system_optimization.py
"""

import time
import psutil
import requests
import json
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv
import signal
import sys

load_dotenv()

class SystemMonitor:
    def __init__(self, interval=10, log_file=None):
        self.interval = interval
        self.running = False
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.start_time = time.time()
        
        # Setup log file path in src/logs
        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        if log_file:
            self.log_file = os.path.join(logs_dir, log_file)
        else:
            # Create default log name matching experiment pattern
            experiment_name = os.getenv('EXPERIMENT_NAME', 'monitor')
            train_size = os.getenv('TRAIN_SIZE', '0')
            val_size = os.getenv('VAL_SIZE', '0') 
            test_size = os.getenv('TEST_SIZE', '0')
            timestamp = datetime.now().strftime("%H%M")
            default_name = f"{experiment_name}-{train_size}-{val_size}-{test_size}-{timestamp}.log"
            self.log_file = os.path.join(logs_dir, default_name)
        
    def get_system_stats(self):
        """Get current system performance metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        load_avg = psutil.getloadavg()
        
        # Find Python processes (likely your experiment)
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if 'main.py' in cmdline or 'experiment' in cmdline:
                        python_processes.append({
                            'pid': proc.info['pid'],
                            'cpu': proc.info['cpu_percent'],
                            'memory': proc.info['memory_percent'],
                            'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return {
            'timestamp': datetime.now().isoformat(),
            'uptime': time.time() - self.start_time,
            'cpu_percent': cpu_percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'load_avg_1m': load_avg[0],
            'load_avg_5m': load_avg[1],
            'load_avg_15m': load_avg[2],
            'cpu_cores': psutil.cpu_count(),
            'python_processes': python_processes
        }
    
    def check_openai_usage(self):
        """Check OpenAI API usage and limits"""
        if not self.api_key:
            return {'error': 'No API key found'}
        
        try:
            # Check usage endpoint (if available)
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Note: OpenAI doesn't have a public usage API endpoint
            # This is a placeholder for when they add one or for rate limit headers
            # from actual API calls
            
            # For now, we'll make a minimal test request to check if API is responsive
            test_response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json={
                    'model': 'gpt-4o-mini',
                    'messages': [{'role': 'user', 'content': 'test'}],
                    'max_tokens': 1
                },
                timeout=5
            )
            
            # Extract rate limit headers if present
            rate_limit_info = {}
            for header, value in test_response.headers.items():
                if 'ratelimit' in header.lower() or 'limit' in header.lower():
                    rate_limit_info[header] = value
            
            return {
                'status_code': test_response.status_code,
                'response_time_ms': test_response.elapsed.total_seconds() * 1000,
                'rate_limits': rate_limit_info,
                'api_responsive': test_response.status_code == 200
            }
            
        except requests.exceptions.Timeout:
            return {'error': 'API timeout', 'api_responsive': False}
        except requests.exceptions.RequestException as e:
            return {'error': str(e), 'api_responsive': False}
        except Exception as e:
            return {'error': f'Unexpected error: {str(e)}', 'api_responsive': False}
    
    def log_stats(self, stats):
        """Log statistics to file and console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Console output
        print(f"\n[{timestamp}] System Monitor Report")
        print("-" * 50)
        print(f"Uptime: {stats['system']['uptime']:.1f}s")
        print(f"CPU: {stats['system']['cpu_percent']:.1f}% (Cores: {stats['system']['cpu_cores']})")
        print(f"Load Avg: {stats['system']['load_avg_1m']:.2f}, {stats['system']['load_avg_5m']:.2f}, {stats['system']['load_avg_15m']:.2f}")
        print(f"Memory: {stats['system']['memory_used_gb']:.1f}GB ({stats['system']['memory_percent']:.1f}%)")
        
        if stats['system']['python_processes']:
            print(f"Experiment Processes:")
            for proc in stats['system']['python_processes']:
                print(f"  PID {proc['pid']}: CPU {proc['cpu']:.1f}%, Memory {proc['memory']:.1f}%")
        
        if 'error' not in stats['api']:
            print(f"API Status: {'✓ Responsive' if stats['api'].get('api_responsive') else '✗ Issues'}")
            print(f"API Response Time: {stats['api'].get('response_time_ms', 0):.0f}ms")
            if stats['api'].get('rate_limits'):
                print("Rate Limits:", stats['api']['rate_limits'])
        else:
            print(f"API Status: ✗ {stats['api']['error']}")
        
        # File logging
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(stats, indent=2) + '\n')
    
    def run(self):
        """Main monitoring loop"""
        self.running = True
        print(f"🔍 System Monitor Started (interval: {self.interval}s)")
        print(f"📊 Monitoring CPU, Memory, Load Average, and API responsiveness")
        print(f"📝 Logging to: {self.log_file}")
        print("Press Ctrl+C to stop\n")
        
        while self.running:
            try:
                system_stats = self.get_system_stats()
                api_stats = self.check_openai_usage()
                
                combined_stats = {
                    'system': system_stats,
                    'api': api_stats
                }
                
                self.log_stats(combined_stats)
                time.sleep(self.interval)
                
            except KeyboardInterrupt:
                print("\n\n⏹️  Stopping monitor...")
                self.running = False
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.interval)

def signal_handler(sig, frame):
    """Handle Ctrl+C signal"""
    print('\n\n⏹️  Monitor stopped by signal')
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(
        description="Monitor system resources and API limits during experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run monitor                              # Monitor with 10s intervals
  uv run monitor --interval 5                # Monitor every 5 seconds
  uv run monitor --log monitor.json          # Save detailed logs to src/logs/
  
Usage with experiments:
  # Terminal 1: Run your experiment
  uv run experiment
  
  # Terminal 2: Run system monitor
  uv run monitor --interval 5
        """
    )
    
    parser.add_argument(
        '--interval', '-i', 
        type=int, 
        default=10,
        help='Monitoring interval in seconds (default: 10)'
    )
    
    parser.add_argument(
        '--log', '-l',
        type=str,
        help='Log file to save detailed monitoring data (JSON format)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce console output (still logs to file if specified)'
    )
    
    args = parser.parse_args()
    
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create monitor
    monitor = SystemMonitor(
        interval=args.interval,
        log_file=args.log
    )
    
    if args.quiet:
        # Redirect stdout to reduce noise
        import io
        sys.stdout = io.StringIO()
    
    # Run monitor
    monitor.run()

if __name__ == "__main__":
    main()