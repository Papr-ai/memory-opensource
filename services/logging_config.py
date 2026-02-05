import logging
from os import environ as env
from flask import Flask
from dotenv import find_dotenv, load_dotenv
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone

# Load environment variables immediately
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

def initialize_logging():
    """Initialize base logging configuration"""
    root_logger = logging.getLogger()
    
    # Get environment settings
    env_setting = env.get('LOGGING_ENV', 'development').lower()
    env_logging = env.get('LoggingtoFile', 'false').lower()
    
    # Set level based on environment (INFO in development, WARNING in production)
    log_level = logging.WARNING if env_setting == 'production' else logging.INFO
    root_logger.setLevel(log_level)
    
    if not root_logger.handlers:  # Only configure if not already configured
        # Create and configure handler with explicit level
        handler = logging.StreamHandler()
        handler.setLevel(log_level)  # Set handler level to match logger
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        
        # Add file handler if enabled
        if env_logging == 'true':
            log_dir = 'logs'
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            log_file = os.path.join(log_dir, f'app_{timestamp}.log')
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=30485760,  # 30MB
                backupCount=5
            )
            file_handler.setLevel(logging.INFO)  # Set handler level to match logger
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    else:
        # Update existing handlers to use correct level
        for handler in root_logger.handlers:
            handler.setLevel(log_level)

# Call initialization immediately
initialize_logging()

class LoggingConfig:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggingConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._configure_base_logging()

    def _configure_base_logging(self):
        """Initial configuration of logging system"""
        self.environment = env.get('LOGGING_ENV', 'development').lower()
        
        # Set appropriate log level (INFO in development, WARNING in production)
        log_level = logging.WARNING if self.environment == 'production' else logging.INFO
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        if not root_logger.handlers:  # Only add handler if none exists
            handler = logging.StreamHandler()
            handler.setLevel(log_level)  # Set handler level to match logger
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)
        else:
            # Update existing handlers to use correct level
            for handler in root_logger.handlers:
                handler.setLevel(log_level)

        # Configure memory_service logger
        memory_service_logger = logging.getLogger('memory_service')
        memory_service_logger.propagate = False  # Prevent duplicate logging
        memory_service_logger.setLevel(log_level)
        
        if not memory_service_logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)  # Set handler level to match logger
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            memory_service_logger.addHandler(handler)
        else:
            # Update existing handlers to use correct level
            for handler in memory_service_logger.handlers:
                handler.setLevel(log_level)

    def configure_app_logging(self, app: Flask = None):
        """Configure logging for the application and its dependencies"""
        # Set level based on environment (INFO in development, WARNING in production)
        level = logging.WARNING if self.environment == 'production' else logging.INFO
        
        # Configure common third-party loggers (suppress DEBUG in development)
        loggers_to_configure = [
            'flask',
            'werkzeug',
            'pinecone',
            'urllib3',
            'auth0',
            'boto3',
            'botocore',
            'openai',
            'requests',
            'httpx',
            'httpcore',
            'httpcore.http11',
            'httpcore.http2',
            'pymongo',
            'pymongo.topology',
            'pymongo.connection',
            'pymongo.serverSelection',
            'pymongo.command',
            'motor',
            'neo4j',
            'neo4j.io',
            'neo4j.pool',
            'neo4j.bolt',
            'asyncio',
            'temporalio',
        ]
        
        for logger_name in loggers_to_configure:
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
            logger.propagate = False  # Prevent duplicate logging
            
            # Remove any existing handlers that might have DEBUG level
            for handler in logger.handlers[:]:
                if handler.level < level:
                    logger.removeHandler(handler)
            
            # Ensure handler has correct level
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setLevel(level)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            else:
                # Update existing handlers to use correct level
                for handler in logger.handlers:
                    handler.setLevel(level)
        
        # Configure Flask-specific settings if app is provided
        if app:
            app.logger.setLevel(level)
            app.logger.propagate = False
            app.debug = False

def configure_logging(app: Flask = None):
    """
    Global function to configure logging settings.
    
    Args:
        app (Flask, optional): Flask application instance
    """
    # Get environment setting (fix: use env_setting to avoid shadowing imported env)
    env_setting = env.get('LOGGING_ENV', 'development').lower()
    
    # Set level based on environment (INFO in development, WARNING in production)
    log_level = logging.WARNING if env_setting == 'production' else logging.INFO
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(log_level)  # Set handler level to match logger
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        # Update existing handlers to use correct level
        for handler in root_logger.handlers:
            handler.setLevel(log_level)
            
    if app:
        app.logger.setLevel(log_level)
            
    # Configure common third-party loggers (suppress DEBUG in development)
    loggers_to_configure = [
        'flask',
        'werkzeug',
        'pinecone',
        'urllib3',
        'auth0',
        'boto3',
        'botocore',
        'openai',
        'requests',
        'httpx',
        'httpcore',
        'httpcore.http11',
        'httpcore.http2',
        'pymongo',
        'pymongo.topology',
        'pymongo.connection',
        'pymongo.serverSelection',
        'pymongo.command',
        'motor',
        'neo4j',
        'neo4j.io',
        'neo4j.pool',
        'neo4j.bolt',
        'asyncio',
        'temporalio',
        'memory_service'
    ]
    
    for logger_name in loggers_to_configure:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        logger.propagate = False
        
        # Remove any existing handlers that might have DEBUG level
        for handler in logger.handlers[:]:
            if handler.level < log_level:
                logger.removeHandler(handler)
        
        # Ensure handler has correct level
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        else:
            # Update existing handlers to use correct level
            for handler in logger.handlers:
                handler.setLevel(log_level)

def get_logger(name):
    """Get a logger instance with the correct configuration based on environment"""
    logger = logging.getLogger(name)
    
    # Only configure if handlers haven't been set up
    if not logger.handlers:
        env_setting = env.get('LOGGING_ENV', 'development').lower()
        env_logging = env.get('LoggingtoFile', 'false').lower()
        
        if env_logging == 'true':
            # Create logs directory if it doesn't exist
            log_dir = 'logs'
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            # Set up rotating file handler
            log_file = os.path.join(log_dir, f'app_{timestamp}.log')
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=30485760,  # 30MB
                backupCount=5
            )
        else:
            handler = logging.StreamHandler()
            
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Set levels based on environment
        if env_setting == 'production':
            logger.setLevel(logging.WARNING)
            handler.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)
            handler.setLevel(logging.INFO)
            
        logger.propagate = False  # Prevent duplicate logging
        
    return logger