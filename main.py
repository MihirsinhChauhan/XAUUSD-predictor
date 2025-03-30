import subprocess
import os
import sys
import time
import signal
import socket
import requests
from datetime import datetime

processes = []

def is_port_in_use(port):
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def wait_for_port(port, timeout=60):
    """Wait for a port to be open, indicating a service is running"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_in_use(port):
            return True
        time.sleep(1)
    return False

def start_api():
    """Start the FastAPI backend"""
    if is_port_in_use(8000):
        print(f"Port 8000 is already in use, another API instance might be running")
        return None
        
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    print(f"Started FastAPI backend (PID: {api_process.pid})")
    return api_process

def start_streamlit():
    """Start the Streamlit frontend"""
    if is_port_in_use(8501):
        print(f"Port 8501 is already in use, another Streamlit instance might be running")
        return None
        
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "streamlit_ui.py", 
         "--server.port", "8501",
         "--server.headless", "false",  # Disable headless mode to ensure browser opens
         "--server.enableCORS", "false",  # Disable CORS for easier debugging
         "--browser.serverAddress", "localhost",  # Ensure correct server address
         "--server.enableXsrfProtection", "false"],  # Disable XSRF protection for debugging
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # Line buffered
    )
    print(f"Started Streamlit frontend (PID: {streamlit_process.pid})")
    return streamlit_process

def wait_for_api(max_attempts=30):
    """Wait for the API to become available"""
    print("Waiting for API to start...")
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:8000/health", timeout=1)
            if response.status_code == 200:
                print("API is up and running!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"Waiting for API to start... Attempt {attempt+1}/{max_attempts}")
        time.sleep(1)
    
    print("Failed to connect to API after multiple attempts")
    return False

def cleanup(signum, frame):
    """Clean up processes on exit"""
    print("\nShutting down...")
    for process in processes:
        try:
            if process and process.poll() is None:  # If the process is still running
                process.terminate()
                print(f"Terminated process (PID: {process.pid})")
        except Exception as e:
            print(f"Error terminating process: {e}")
    
    # Forcefully kill any remaining processes after a grace period
    time.sleep(2)
    for process in processes:
        try:
            if process and process.poll() is None:
                process.kill()
                print(f"Forcefully killed process (PID: {process.pid})")
        except Exception as e:
            print(f"Error killing process: {e}")
    
    print("All processes terminated. Exiting.")
    sys.exit(0)

def print_urls():
    """Print the URLs to access the application"""
    print("\n" + "="*50)
    print("Application is running!")
    print("API URL: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Dashboard URL: http://localhost:8501")
    
    # Get local IP address for LAN access
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"Local network access: http://{local_ip}:8501")
    except Exception:
        pass
        
    print("=" * 50 + "\n")

def main():
    """Main function to start the application"""
    print(f"Starting XAUUSD Prediction Application at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    # Check if ports are already in use
    if is_port_in_use(8000):
        print("Warning: Port 8000 is already in use. FastAPI server might already be running.")
    
    if is_port_in_use(8501):
        print("Warning: Port 8501 is already in use. Streamlit server might already be running.")
    
    # Start the FastAPI backend
    api_process = start_api()
    if api_process:
        processes.append(api_process)
    else:
        print("Failed to start API server. Exiting.")
        return
    
    # Wait for the API to start up
    if not wait_for_api(max_attempts=30):
        print("API failed to start in time. Check logs for errors.")
        cleanup(None, None)
        return
    
    # Start the Streamlit frontend
    streamlit_process = start_streamlit()
    if streamlit_process:
        processes.append(streamlit_process)
    else:
        print("Failed to start Streamlit server. Exiting.")
        cleanup(None, None)
        return
    
    # Wait for Streamlit to start up
    if not wait_for_port(8501, timeout=30):
        print("Streamlit server failed to start in time. Check logs for errors.")
    else:
        print("Streamlit server is up and running!")
    
    # Print URLs
    print_urls()
    
    # Open browser automatically if requested
    if os.environ.get('OPEN_BROWSER', 'true').lower() == 'true':
        try:
            import webbrowser
            webbrowser.open('http://localhost:8501')
        except Exception as e:
            print(f"Failed to open browser: {e}")
    
    # Monitor process output
    try:
        while True:
            # Check if processes are still running
            if api_process.poll() is not None:
                print(f"FastAPI backend terminated unexpectedly with code {api_process.poll()}")
                cleanup(None, None)
                return
            
            if streamlit_process.poll() is not None:
                print(f"Streamlit frontend terminated unexpectedly with code {streamlit_process.poll()}")
                cleanup(None, None)
                return
            
            # Print any output from processes
            api_output = api_process.stdout.readline().strip()
            if api_output:
                print(f"[API] {api_output}")
                
            streamlit_output = streamlit_process.stdout.readline().strip()
            if streamlit_output:
                print(f"[Streamlit] {streamlit_output}")
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        cleanup(None, None)

if __name__ == "__main__":
    main()