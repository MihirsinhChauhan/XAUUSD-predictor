import subprocess
import os
import sys
import time
import signal

processes = []

def start_api():
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    print(f"Started FastAPI backend (PID: {api_process.pid})")
    return api_process

def start_streamlit():
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "streamlit_ui.py", "--server.port", "8501"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    print(f"Started Streamlit frontend (PID: {streamlit_process.pid})")
    return streamlit_process

def cleanup(signum, frame):
    print("\nShutting down...")
    for process in processes:
        try:
            if process.poll() is None:  # If the process is still running
                process.terminate()
                print(f"Terminated process (PID: {process.pid})")
        except Exception as e:
            print(f"Error terminating process: {e}")
    
    print("All processes terminated. Exiting.")
    sys.exit(0)

def main():
    print("Starting XAUUSD Prediction Application...")
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    # Start the FastAPI backend
    api_process = start_api()
    processes.append(api_process)
    
    # Wait for the API to start up
    print("Waiting for API to start...")
    time.sleep(3)
    
    # Start the Streamlit frontend
    streamlit_process = start_streamlit()
    processes.append(streamlit_process)
    
    # Print URLs
    print("\n" + "="*50)
    print("Application is running!")
    print("API URL: http://localhost:8000")
    print("Dashboard URL: http://localhost:8501")
    print("=" * 50 + "\n")
    
    # Monitor process output
    try:
        while True:
            # # Check if processes are still running
            # if api_process.poll() is not None:
            #     print("FastAPI backend terminated unexpectedly")
            #     cleanup(None, None)
            
            # if streamlit_process.poll() is not None:
            #     print("Streamlit frontend terminated unexpectedly")
            #     cleanup(None, None)
            
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