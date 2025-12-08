import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from os import environ as env
from dotenv import find_dotenv, load_dotenv

# Load environment variables immediately
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

class LoggerSingleton:
    _instance = None
    _loggers = {}  # Dictionary to store named loggers
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._configure_base_logging()
            self._configure_third_party_loggers()
            self._initialized = True

    def _configure_base_logging(self):
        """Configure base logging settings"""
        # Get environment settings
        env_setting = env.get('LOGGING_ENV', 'development').lower()
        logging_to_file = env.get('LoggingtoFile', 'false').lower() == 'true'

        # Configure root logger
        root_logger = logging.getLogger()
        
        # Set level based on environment
        log_level = logging.WARNING if env_setting == 'production' else logging.INFO
        root_logger.setLevel(log_level)
        
        if not root_logger.handlers:  # Only configure if not already configured
            # Create console handler with explicit level
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)  # Set handler level to match logger
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

            # Add file handler if enabled
            if logging_to_file:
                try:
                    log_dir = 'logs'
                    os.makedirs(log_dir, exist_ok=True)
                    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    log_file = os.path.join(log_dir, f'app_{timestamp}.log')
                    
                    file_handler = RotatingFileHandler(
                        log_file,
                        maxBytes=30485760,  # 30MB
                        backupCount=5
                    )
                    file_handler.setLevel(log_level)  # Set handler level to match logger
                    file_handler.setFormatter(formatter)
                    root_logger.addHandler(file_handler)
                except Exception as e:
                    console_handler.setLevel(logging.WARNING)
                    root_logger.warning(f"Failed to initialize file logging: {e}")
        else:
            # Update existing handlers to use correct level
            for handler in root_logger.handlers:
                handler.setLevel(log_level)

    def _configure_third_party_loggers(self):
        """Configure logging levels for third-party libraries"""
        env_setting = env.get('LOGGING_ENV', 'development').lower()
        
        # List of third-party loggers to configure (including sub-modules)
        third_party_loggers = [
            'httpx',
            'httpcore',  # HTTP core library (used by httpx)
            'httpcore.http11',  # HTTP/1.1 implementation
            'httpcore.http2',  # HTTP/2 implementation
            'urllib3',
            'requests',
            'pinecone',
            'boto3',
            'botocore',
            'openai',
            'auth0',
            'werkzeug',
            'flask',
            'pymongo',  # MongoDB driver
            'pymongo.topology',  # MongoDB topology monitoring
            'pymongo.connection',  # MongoDB connection pool
            'pymongo.serverSelection',  # MongoDB server selection
            'pymongo.command',  # MongoDB command execution
            'motor',  # Async MongoDB driver
            'neo4j',  # Neo4j driver
            'neo4j.io',  # Neo4j I/O operations
            'neo4j.pool',  # Neo4j connection pool
            'neo4j.bolt',  # Neo4j Bolt protocol
            'asyncio',  # Asyncio library
            'temporalio',  # Temporal client
        ]

        # Set appropriate level based on environment
        # In development, use INFO to suppress DEBUG logs
        # In production, use WARNING to suppress INFO logs
        level = logging.WARNING if env_setting == 'production' else logging.INFO

        # Configure each third-party logger
        for logger_name in third_party_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
            # Prevent propagation to root logger to avoid duplicate logs
            logger.propagate = False

            # Remove any existing handlers that might have DEBUG level
            for handler in logger.handlers[:]:
                if handler.level < level:
                    logger.removeHandler(handler)

            # Ensure the logger has at least one handler with correct level
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setLevel(level)  # Set handler level explicitly
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            else:
                # Update existing handlers to use the correct level
                for handler in logger.handlers:
                    handler.setLevel(level)

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger instance with the singleton configuration"""
        if cls._instance is None:
            cls()

        if name not in cls._loggers:
            logger = logging.getLogger(name)
            
            # Set logger level based on environment
            env_setting = env.get('LOGGING_ENV', 'development').lower()
            log_level = logging.WARNING if env_setting == 'production' else logging.INFO
            logger.setLevel(log_level)
            
            # Allow propagation to root logger
            logger.propagate = True
            
            cls._loggers[name] = logger

        return cls._loggers[name] 