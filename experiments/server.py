""" Helper module for managing FastAPI server during experiments. """


import time
import threading
import uvicorn

# Global server thread
_server_thread: threading.Thread | None = None
_server_instance: uvicorn.Server | None = None

def start_fastapi_server():
    """Start the FastAPI server in a separate thread."""
    global _server_thread, _server_instance
    
    try:
        from sandbox.app import app
        
        config = uvicorn.Config(app, host="127.0.0.1", port=5000, log_level="info")
        _server_instance = uvicorn.Server(config)
        
        def run_server():
            _server_instance.run()
        
        _server_thread = threading.Thread(target=run_server, daemon=True)
        _server_thread.start()
        
        # Wait a bit for server to start
        time.sleep(3)
        return _server_thread
    except Exception as e:
        print(f"Failed to start FastAPI server: {e}")
        return None

def stop_fastapi_server():
    """Stop the FastAPI server thread."""
    global _server_thread, _server_instance
    
    if _server_instance:
        _server_instance.should_exit = True
        _server_instance = None
    
    if _server_thread and _server_thread.is_alive():
        _server_thread.join(timeout=5)
        _server_thread = None