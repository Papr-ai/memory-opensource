# url_utils.py

from urllib.parse import urlparse
from services.logging_config import get_logger

from services.logger_singleton import LoggerSingleton

# Create a logger instance for this module
logger = LoggerSingleton.get_logger(__name__)


def handle_url(source_url):
    """
    Sanitize a URL by removing all query parameters and ensuring the URL ends with .pdf.

    Args:
        source_url (str): The original URL that needs to be sanitized.

    Returns:
        str: The sanitized URL ending with .pdf.
    """
    parsed_url = urlparse(source_url)
    
    # Reconstruct the base URL without query parameters
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"

    # Check if the base URL ends with .pdf
    if base_url.lower().endswith('.pdf'):
        sanitized_url = base_url
    else:
        logger.warning(f"The URL does not end with .pdf: {base_url}")
        sanitized_url = base_url  # This will return the URL even if it doesn't end with .pdf

    # Log the final sanitized URL
    logger.info(f"Sanitized URL: {sanitized_url}")

    return sanitized_url

def clean_url(url):
    """Clean a URL by removing comments and spaces."""
    if not url:
        return url
    
    # Split by '#' and take the first part (before any comment)
    url = url.split('#')[0]
    
    # Remove leading/trailing whitespace
    url = url.strip()
    
    logger.debug(f"Cleaned URL: {url}")
    return url


def get_parse_server_url():
    """
    Get Parse Server URL at runtime, applying localhost override for open-source local testing.
    This function reads from os.environ at runtime, so it respects test environment overrides.
    Only applies override when:
    - PAPR_EDITION=opensource
    - AND running locally (pytest or not in Docker container)
    This ensures Docker deployments using service names are not affected.
    """
    import os
    
    url = os.getenv("PARSE_SERVER_URL", "")
    if not url:
        return url
    
    # Only apply override for open-source edition
    papr_edition = os.getenv("PAPR_EDITION", "").lower()
    if papr_edition != "opensource":
        return clean_url(url)
    
    # Only apply override when running locally (tests or local dev), not in Docker
    # Check if we're in pytest (PYTEST_CURRENT_TEST is set by pytest)
    is_pytest = "PYTEST_CURRENT_TEST" in os.environ
    
    # Check if we're running in Docker container
    is_docker = False
    if os.path.exists("/.dockerenv"):
        is_docker = True
    elif os.path.exists("/proc/self/cgroup"):
        try:
            with open("/proc/self/cgroup", "r") as f:
                if "docker" in f.read():
                    is_docker = True
        except (IOError, OSError):
            pass  # If we can't read it, assume not Docker
    
    # Only override if we're running locally (pytest or not in Docker) AND URL contains Docker service name
    if (is_pytest or not is_docker) and "parse-server" in url:
        # Replace Docker service name with localhost
        url = url.replace("parse-server", "localhost")
        # Remove trailing /parse if present, since the code adds /parse/classes
        if url.endswith("/parse"):
            url = url[:-6]  # Remove trailing "/parse"
        elif url.endswith("/parse/"):
            url = url[:-7]  # Remove trailing "/parse/"
    
    return clean_url(url)


def get_qdrant_url():
    """
    Get Qdrant URL at runtime, applying localhost override for open-source local testing.
    This function reads from os.environ at runtime, so it respects test environment overrides.
    Only applies override when:
    - PAPR_EDITION=opensource
    - AND running locally (pytest or not in Docker container)
    This ensures Docker deployments using service names are not affected.
    """
    import os
    
    url = os.getenv("QDRANT_URL", "")
    if not url:
        return url
    
    # Only apply override for open-source edition
    papr_edition = os.getenv("PAPR_EDITION", "").lower()
    if papr_edition != "opensource":
        return clean_url(url)
    
    # Only apply override when running locally (tests or local dev), not in Docker
    # Check if we're in pytest (PYTEST_CURRENT_TEST is set by pytest)
    is_pytest = "PYTEST_CURRENT_TEST" in os.environ
    
    # Check if we're running in Docker container
    is_docker = False
    if os.path.exists("/.dockerenv"):
        is_docker = True
    elif os.path.exists("/proc/self/cgroup"):
        try:
            with open("/proc/self/cgroup", "r") as f:
                if "docker" in f.read():
                    is_docker = True
        except (IOError, OSError):
            pass  # If we can't read it, assume not Docker
    
    # Only override if we're running locally (pytest or not in Docker) AND URL contains Docker service name
    if (is_pytest or not is_docker) and "qdrant" in url and "localhost" not in url:
        # Replace Docker service name with localhost (but preserve port if different)
        # qdrant:6333 -> localhost:6333
        url = url.replace("qdrant", "localhost")
    
    return clean_url(url)