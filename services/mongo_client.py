from __future__ import annotations

import os
import logging
from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database


_client: Optional[MongoClient] = None
_db: Optional[Database] = None


def get_mongo_db() -> Optional[Database]:
    """Return a singleton MongoDB Database if MONGO_URI is configured, else None.

    Environment variables:
    - MONGO_URI: standard Mongo connection string
    - MONGO_DB_NAME: optional explicit DB name; if omitted, uses URI's default database
    """
    global _client, _db

    # Check if cached database is still healthy
    if _db is not None and _client is not None:
        try:
            # Quick health check - ping the database
            _client.admin.command('ping')
            return _db
        except Exception as e:
            # Client is closed or unhealthy, check if we should reconnect
            error_str = str(e).lower()
            if "after close" in error_str or "closed" in error_str:
                # Connection was closed, try to recreate
                logging.warning(f"MongoDB client is closed: {e}. Recreating connection.")
                _client = None
                _db = None
            else:
                # Other error (network, timeout, etc.) - might be transient
                logging.warning(f"MongoDB client health check failed (may be transient): {e}. Will attempt to reconnect.")
                # Don't reset immediately for transient errors - let it try to reconnect on next use
                # But if the client is actually closed, we need to recreate
                try:
                    # Check if client is actually closed by trying to access it
                    _ = _client.server_info()
                except Exception:
                    # Client is definitely closed, recreate
                    _client = None
                    _db = None

    # If client exists but db doesn't (edge case), reuse client to get db
    if _client is not None:
        try:
            # Verify client is still alive
            _client.admin.command('ping')
            db_name = os.getenv("MONGO_DB_NAME")
            _db = _client.get_database(db_name) if db_name else _client.get_default_database()
            return _db
        except Exception as e:
            logging.warning(f"MongoDB client check failed: {e}. Recreating connection.")
            _client = None
            _db = None

    # Create new client only if it doesn't exist
    mongo_uri = os.getenv("MONGO_URI")

    # If MONGO_URI is not set, try to derive it from DATABASE_URI
    # This allows using the same MongoDB database as Parse Server for memory system
    if not mongo_uri:
        database_uri = os.getenv("DATABASE_URI")
        if database_uri and "mongodb" in database_uri:
            # Use the same database as Parse Server
            mongo_uri = database_uri
            logging.info(f"Derived MONGO_URI from DATABASE_URI (same database): {mongo_uri[:50]}...")
        else:
            return None

    # Create client with optimized pool and SSL/TLS settings
    # Note: Default maxPoolSize=100 is appropriate for most use cases.
    # If you need to customize pool settings, you can add parameters like:
    # maxPoolSize=20, minPoolSize=2, maxIdleTimeMS=60000, etc.
    # This client is shared across all MemoryGraph instances to prevent connection limit issues.
    
    # Configure SSL/TLS and connection settings for better stability
    # These settings help reduce SSL handshake errors in background tasks
    client_options = {
        'serverSelectionTimeoutMS': 10000,
        'connectTimeoutMS': 10000,
        'socketTimeoutMS': 30000,
        'retryWrites': True,
        'retryReads': True,
        # Reduce background task SSL errors by:
        # 1. Less frequent heartbeats (reduces connection attempts)
        # 2. Longer socket timeout (allows time for SSL handshake)
        'heartbeatFrequencyMS': 30000,  # 30 seconds (default is 10s)
    }
    
    # Configure PyMongo logger to reduce noise from background task SSL errors
    # These errors are non-critical - PyMongo automatically retries via AutoReconnect
    # Background task errors occur during connection pool maintenance and are expected
    pymongo_client_logger = logging.getLogger('pymongo.client')
    
    # Add a filter to downgrade background task SSL errors from ERROR to WARNING
    class BackgroundTaskErrorFilter(logging.Filter):
        """Filter to downgrade non-critical background task SSL errors.
        
        These errors occur during connection pool maintenance (removing stale sockets)
        and are automatically retried by PyMongo via AutoReconnect. They're not critical
        and don't indicate application-level connection failures.
        """
        def filter(self, record):
            # Check if this is a background task SSL error
            # Collect all text from the log record
            msg_parts = []
            
            # Get the main message
            if hasattr(record, 'getMessage'):
                try:
                    msg_parts.append(str(record.getMessage()))
                except:
                    pass
            elif hasattr(record, 'msg'):
                msg_parts.append(str(record.msg))
            
            # Get exception info if present
            if hasattr(record, 'exc_info') and record.exc_info:
                try:
                    if record.exc_info[1]:  # Exception instance
                        msg_parts.append(str(record.exc_info[1]))
                    if record.exc_info[2]:  # Traceback
                        import traceback
                        tb_text = ''.join(traceback.format_exception(*record.exc_info))
                        msg_parts.append(tb_text)
                except:
                    pass
            
            # Combine all text and check
            full_text = " ".join(msg_parts).lower()
            
            # Check for background task indicator - this is the key indicator
            # PyMongo logs: "MongoClient background task encountered an error"
            is_background_task = (
                'background task' in full_text or 
                'periodic task' in full_text or 
                '_process_periodic_tasks' in full_text or
                'update_pool' in full_text  # Part of background pool maintenance
            )
            
            # Check for SSL/TLS related errors in the traceback
            is_ssl_error = any(kw in full_text for kw in [
                'ssl', 'tls', 'handshake', 'autoreconnect', 'internal_error', 
                'tlsv1_alert', 'ssl handshake failed', 'sslerror', 'sslerror:',
                'tlsv1 alert internal error'  # The specific error from logs
            ])
            
            # If it's a background task with SSL error, downgrade to WARNING
            if is_background_task and is_ssl_error:
                # Downgrade from ERROR to WARNING
                record.levelno = logging.WARNING
                record.levelname = 'WARNING'
                # Modify message to indicate it's non-critical
                if hasattr(record, 'msg'):
                    original_msg = str(record.msg)
                    # Only add prefix if not already present
                    if '[non-critical' not in original_msg.lower():
                        record.msg = f"[Non-critical background task SSL error, auto-retrying] {original_msg}"
            return True
    
    # Add the filter to the PyMongo client logger and all its handlers
    # This ensures background task SSL errors are downgraded to WARNING
    filter_instance = BackgroundTaskErrorFilter()
    pymongo_client_logger.addFilter(filter_instance)
    
    # Also add filter to all handlers to ensure it's applied
    for handler in pymongo_client_logger.handlers:
        handler.addFilter(filter_instance)
    
    # Don't change the logger level - let the filter handle downgrading specific errors
    
    # Create client with error handling
    try:
        _client = MongoClient(
            mongo_uri,
            **client_options
        )
    except Exception as e:
        # Log critical connection errors
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to create MongoDB client: {e}")
        raise
    db_name = os.getenv("MONGO_DB_NAME")
    _db = _client.get_database(db_name) if db_name else _client.get_default_database()

    return _db


