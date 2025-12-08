#!/usr/bin/env python3
"""
Start all PAPR Memory Server services
- Web server (FastAPI)
- Memory processing Temporal worker
- Document processing Temporal worker
"""

import asyncio
import signal
import sys
from multiprocessing import Process
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def run_web_server():
    """Run the FastAPI web server"""
    import uvicorn
    from services.logger_singleton import LoggerSingleton
    
    logger = LoggerSingleton.get_logger(__name__)
    logger.info("Starting web server...")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5001,
        log_level="info",
        access_log=True,
    )


def run_memory_worker():
    """Run the memory processing Temporal worker"""
    import start_temporal_worker
    from services.logger_singleton import LoggerSingleton
    
    logger = LoggerSingleton.get_logger(__name__)
    logger.info("Starting memory processing Temporal worker...")
    
    asyncio.run(start_temporal_worker.main())


def run_document_worker():
    """Run the document processing Temporal worker"""
    import start_document_worker
    from services.logger_singleton import LoggerSingleton
    
    logger = LoggerSingleton.get_logger(__name__)
    logger.info("Starting document processing Temporal worker...")
    
    asyncio.run(start_document_worker.main())


def main():
    """Start all services as separate processes"""
    from services.logger_singleton import LoggerSingleton
    
    logger = LoggerSingleton.get_logger(__name__)
    logger.info("Starting PAPR Memory Server with all services...")
    
    # Create processes
    processes = []
    
    # Web server
    web_process = Process(target=run_web_server, name="WebServer")
    web_process.start()
    processes.append(web_process)
    logger.info(f"Web server started (PID: {web_process.pid})")
    
    # Memory worker
    memory_worker = Process(target=run_memory_worker, name="MemoryWorker")
    memory_worker.start()
    processes.append(memory_worker)
    logger.info(f"Memory worker started (PID: {memory_worker.pid})")
    
    # Document worker
    doc_worker = Process(target=run_document_worker, name="DocumentWorker")
    doc_worker.start()
    processes.append(doc_worker)
    logger.info(f"Document worker started (PID: {doc_worker.pid})")
    
    logger.info("All services started successfully!")
    
    # Handle shutdown gracefully
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down all services...")
        for process in processes:
            if process.is_alive():
                logger.info(f"Terminating {process.name} (PID: {process.pid})...")
                process.terminate()
        
        # Wait for all processes to finish
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                logger.warning(f"Force killing {process.name} (PID: {process.pid})...")
                process.kill()
        
        logger.info("All services stopped")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Wait for any process to exit
    try:
        import time
        while all(p.is_alive() for p in processes):
            time.sleep(1)
        
        # If any process died, log and exit
        for process in processes:
            if not process.is_alive():
                logger.error(f"{process.name} exited unexpectedly (exit code: {process.exitcode})")
        
        # Terminate remaining processes
        signal_handler(signal.SIGTERM, None)
        
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()

