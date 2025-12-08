import time
from neo4j import GraphDatabase, AsyncGraphDatabase, TRUST_SYSTEM_CA_SIGNED_CERTIFICATES, TRUST_ALL_CERTIFICATES
from neo4j.exceptions import ServiceUnavailable, TransientError
from services.logging_config import get_logger
import ssl
import certifi
import random
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field
from typing import List, Literal, TYPE_CHECKING, Dict, Any, Optional
from uuid import uuid4
import asyncio
import time
from contextlib import asynccontextmanager
import contextlib

from services.logging_config import get_logger
from services.logger_singleton import LoggerSingleton

# Create a logger instance for this module
logger = LoggerSingleton.get_logger(__name__)

class CircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=300):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout  # seconds
        self.last_failure_time = None
        self.is_open = False
        self._lock = asyncio.Lock()

    async def record_failure(self):
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now(timezone.utc)
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
                logger.warning("Circuit breaker opened due to multiple failures")

    async def can_try(self):
        async with self._lock:
            if not self.is_open:
                return True
            
            if self.last_failure_time and \
               (datetime.now(timezone.utc) - self.last_failure_time).total_seconds() > self.reset_timeout:
                self.reset()
                return True
            return False

    def reset(self):
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False
        logger.info("Circuit breaker reset")

class AsyncNeo4jConnection:
    _instance = None
    _initialized = False
    _driver = None
    _active_sessions = 0
    _active_transactions = 0

    def __new__(cls, uri=None, user=None, pwd=None, retries=5, delay=2):
        # Make singleton safe across event loops: if loop changed, create a new instance
        current_loop = None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop; fall back to get_event_loop for compatibility
            current_loop = asyncio.get_event_loop()

        instance = getattr(cls, "_instance", None)
        instance_loop = getattr(cls, "_instance_loop", None)
        if instance is None or instance_loop is not current_loop:
            instance = super(AsyncNeo4jConnection, cls).__new__(cls)
            cls._instance = instance
            cls._instance_loop = current_loop
            cls._initialized = False
        return instance

    def __init__(self, uri=None, user=None, pwd=None, retries=10, delay=5):
        if self._initialized:
            return
            
        # Validate URI for open-source edition only (cloud should always have it set)
        import os
        papr_edition = os.getenv("PAPR_EDITION", "").lower()
        if papr_edition == "opensource" and not uri:
            logger.warning("NEO4J_URL is None - Neo4j connection will be in fallback mode")
            # Set fallback mode immediately if URI is None in open-source
            self.fallback_mode = True
        else:
            self.fallback_mode = False
            
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__retries = retries
        self.__delay = delay
        self._initialized = True
        self.circuit_breaker = CircuitBreaker(failure_threshold=7, reset_timeout=600)

        # New attribute to track the last successful health check time.
        self._last_health_check = None

        # IMPORTANT: Use an instance-scoped asyncio.Lock bound to the current event loop
        # to avoid cross-event-loop issues in tests and ASGI lifespan restarts.
        # This intentionally shadows the class attribute if present.
        self._lock = asyncio.Lock()

        # Setup SSL context with certificate verification
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = True
        self.ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        # Try different certificate locations based on OS
        cert_locations = [
            '/etc/ssl/certs/ca-certificates.crt',  # Linux
            '/etc/ssl/cert.pem',  # macOS
            certifi.where()  # Use certifi's certificates as fallback
        ]
        
        cert_loaded = False
        for cert_path in cert_locations:
            try:
                self.ssl_context.load_verify_locations(cert_path)
                cert_loaded = True
                logger.info(f"Loaded SSL certificates from {cert_path}")
                break
            except Exception as e:
                logger.debug(f"Could not load certificates from {cert_path}: {e}")
        
        if not cert_loaded:
            logger.warning("Could not load SSL certificates, using default system certificates")
            self.ssl_context = ssl.create_default_context()

    async def connect_with_retries(self):
        """Establish connection with retries and circuit breaker"""
        logger.info("Starting connect_with_retries")
        if not await self.circuit_breaker.can_try():
            logger.warning("Circuit breaker is open, using fallback mode")
            self.fallback_mode = True
            return None

        # Validate URI before attempting connection (only for open-source edition)
        # Cloud edition should always have NEO4J_URL set, so we skip validation to maintain backward compatibility
        import os
        papr_edition = os.getenv("PAPR_EDITION", "").lower()
        if papr_edition == "opensource":
            if not self.__uri:
                error_msg = "NEO4J_URL is not set. Cannot connect to Neo4j. Using fallback mode."
                logger.warning(error_msg)
                self.fallback_mode = True
                return None  # Return None to allow graceful fallback instead of raising

        
        try:
            if self._driver is not None:
                logger.info("Driver exists, doing health check")
                try:
                    async with self._driver.session() as session:
                        await session.run("RETURN 1")
                        logger.info("Health check passed")
                    return self._driver
                except Exception as e:
                    logger.warning(f"Connection health check failed: {str(e)}")
                    await self._driver.close()
                    self._driver = None

            for attempt in range(self.__retries):
                try:
                    logger.info(f"Attempting Neo4j connection (attempt {attempt + 1}/{self.__retries})")
                    
                    connection_args = {
                        "max_connection_lifetime": 180,    # 3 minutes
                        "max_connection_pool_size": 50,
                        "connection_timeout": 20,          # More aggressive timeout
                        "max_transaction_retry_time": 20,  # More aggressive retry
                        "keep_alive": True,
                        "fetch_size": 500,                 # Reduced batch size
                        "connection_acquisition_timeout": 60  # Added timeout for connection acquisition

                    }

                    is_secure = self.__uri.startswith(('bolt+s://', 'neo4j+s://', 'bolt+ssc://', 'neo4j+ssc://'))
                    logger.info(f"Using secure connection: {is_secure}")

                    if not is_secure and self.__uri.startswith(('bolt://', 'neo4j://')):
                        connection_args.update({
                            "trust": TRUST_SYSTEM_CA_SIGNED_CERTIFICATES,
                            "encrypted": False
                        })

                    logger.info("Creating driver...")
                    if is_secure:
                        # Do not pass ssl_context when using a secure scheme
                        self._driver = AsyncGraphDatabase.driver(
                            self.__uri,
                            auth=(self.__user, self.__pwd),
                            **connection_args
                        )
                    else:
                        # For non-secure connections (bolt://), don't pass ssl_context
                        # This prevents connection failures in Docker/local environments
                        self._driver = AsyncGraphDatabase.driver(
                            self.__uri,
                            auth=(self.__user, self.__pwd),
                            **connection_args
                        )
                    
                    logger.info("Driver created, attempting test connection...")
                    async with self._driver.session() as session:
                        logger.info("Session created, running test query...")
                        await session.run("RETURN 1")
                        logger.info("Test query successful")
                        
                    self._last_health_check = datetime.now(timezone.utc)
                    logger.info("Successfully connected to Neo4j")
                    self.circuit_breaker.reset()
                    self.fallback_mode = False
                    return self._driver
                    
                except Exception as e:
                    logger.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                    if self._driver:
                        await self._driver.close()
                        self._driver = None
                    if attempt < self.__retries - 1:
                        max_delay = min(self.__delay * (2 ** attempt), 60)
                        jitter = random.uniform(0, 0.1 * max_delay)
                        delay = max_delay + jitter
                        logger.info(f"Retrying in {delay:.2f} seconds...")
                        await asyncio.sleep(delay)
            
            await self.circuit_breaker.record_failure()
            raise Exception("All connection attempts failed")
            
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            self.fallback_mode = True
            raise

    async def get_driver(self):
        """
        Get the Neo4j driver, reconnecting if necessary.
        """
        try:
            logger.info("Getting driver")
            current_time = datetime.now(timezone.utc)
            
            # Add connection staleness check
            if self._last_health_check and \
               (current_time - self._last_health_check).total_seconds() > 180:  # 3 minutes
                logger.info("Connection potentially stale, forcing reconnection")
                if self._driver:
                    await self._driver.close()
                self._driver = None

            if self._driver is None:
                logger.info("Driver is None, connecting with retries")
                await self.connect_with_retries()
            else:
                logger.info("Driver is not None, doing health check")
                try:
                    logger.info("Doing health check")
                    async with self._driver.session() as session:
                        async with asyncio.timeout(3):  # Reduced timeout
                            await session.run("RETURN 1")
                            self._last_health_check = current_time
                            logger.info("Health check passed")
                except Exception as e:
                    logger.warning(f"Driver health check failed: {str(e)}. Reconnecting...")
                    if self._driver:
                        await self._driver.close()
                    self._driver = None
                    await self.connect_with_retries()
            
            return self._driver
        except Exception as e:
            logger.error(f"Error getting driver: {str(e)}")
            raise

    @asynccontextmanager
    async def get_session(self):
        """
        Async context manager for Neo4j sessions with better cleanup
        """
        driver = None
        session = None
        logger.info("Getting session")
        
        try:
            driver = await self.get_driver()
            logger.info("Got driver")
            self._active_sessions += 1
            logger.warning(f"Opening new session. Active sessions: {self._active_sessions}")

            # Use async session instead of sync session
            async with driver.session() as session:
                logger.info("Got session")
                yield session
                
        except Exception as e:
            logger.error(f"Error in session: {e}")
            raise
        finally:
            self._active_sessions -= 1
            logger.warning(f"Closing session. Remaining active sessions: {self._active_sessions}")

    @asynccontextmanager
    async def get_transaction(self):
        """
        Async context manager for Neo4j transactions.
        Usage:
            async with connection.get_transaction() as tx:
                result = await tx.run("MATCH (n) RETURN n")
        """
        async with self.get_session() as session:
            self._active_transactions += 1
            logger.warning(f"Opening new transaction. Active transactions: {self._active_transactions}")

            async with session.begin_transaction() as tx:
                try:
                    yield tx
                except Exception as e:
                    logger.error(f"Error in transaction: {e}")
                    await tx.rollback()
                    raise
                finally:
                    self._active_transactions -= 1
                    logger.warning(f"Closing transaction. Remaining active transactions: {self._active_transactions}")

    async def get_connection_stats(self):
        """Get current connection statistics"""
        return {
            "active_sessions": self._active_sessions,
            "active_transactions": self._active_transactions
        }


    async def close(self):
        """Close the Neo4j connection"""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def warm_connection(self):
        """Warm up the connection pool without impacting app stability"""
        try:
            logger.info("Attempting to warm up Neo4j connection")
            driver = await self.get_driver()
            async with driver.session() as session:
                start = time.time()
                result = await session.run("RETURN 1")
                await result.consume()
                duration = time.time() - start
                logger.info(f"Connection warm-up successful. Duration: {duration:.2f}s")
                return True
        except Exception as e:
            logger.warning(f"Warm-up attempt failed (this is non-fatal): {str(e)}")
            return False

    async def start_background_warmup(self):
        """Start background warm-up task"""
        async def warmup_task():
            while True:
                try:
                    # Warm up every 3 minutes (less than Azure's 4-minute timeout)
                    await self.warm_connection()
                    await asyncio.sleep(180)
                except Exception as e:
                    logger.warning(f"Background warm-up task error (will retry): {str(e)}")
                    await asyncio.sleep(5)

        # Create task but don't wait for it
        asyncio.create_task(warmup_task())

    async def keep_warm(self):
        """Keep connection warm with periodic checks"""
        while True:
            try:
                await self.warm_connection()
                # Run every 2.5 minutes (150 seconds)
                # This ensures we refresh before both Neo4j (3 min) 
                # and Azure (4 min) timeouts
                await asyncio.sleep(150)
            except Exception as e:
                logger.warning(f"Keep-warm task error (will retry): {str(e)}")
                await asyncio.sleep(5)

 
class Neo4jConnection:

    _instance = None
    _initialized = False

    def __new__(cls, uri=None, user=None, pwd=None, retries=5, delay=2):
        if cls._instance is None:
            cls._instance = super(Neo4jConnection, cls).__new__(cls)
        return cls._instance

    def __init__(self, uri=None, user=None, pwd=None, retries=10, delay=5):
        if self._initialized:
            return
            
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        self.__retries = retries
        self.__delay = delay
        self._initialized = True
        
        # Initialize connection immediately
        self.connect_with_retries()

    def connect_with_retries(self):
        for attempt in range(self.__retries):
            try:
                self.__driver = GraphDatabase.driver(
                    self.__uri, 
                    auth=(self.__user, self.__pwd)
                )
                # Verify connectivity
                self.__driver.verify_connectivity()
                logger.info("Connected to Neo4j successfully")
                return self.__driver
            except Exception as e:
                self.__driver = None
                if attempt < self.__retries - 1:
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying in {self.__delay} seconds...")
                    time.sleep(self.__delay)
                else:
                    logger.error(f"Failed to connect to Neo4j after {self.__retries} attempts: {e}")
                    raise

    def get_driver(self):
        if self.__driver is None:
            return self.connect_with_retries()
        return self.__driver

    def session(self, **kwargs):
        driver = self.get_driver()
        if driver is None:
            raise RuntimeError("Failed to initialize Neo4j driver")
        return driver.session(**kwargs)
    
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
            self.__driver = None


