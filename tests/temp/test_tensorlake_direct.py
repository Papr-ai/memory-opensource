#!/usr/bin/env python3
"""
Test TensorLake provider directly with real API key
"""

import asyncio
import os
from dotenv import load_dotenv
from core.document_processing.providers.tensorlake import TensorLakeProvider

async def test_tensorlake_provider():
    """Test TensorLake provider directly"""

    # Load environment
    use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
    if use_dotenv:
        load_dotenv()

    # Configure provider
    config = {
        "api_key": os.getenv("TENSORLAKE_API_KEY"),
        "base_url": os.getenv("TENSORLAKE_BASE_URL", "https://api.tensorlake.ai"),
        "timeout": 60
    }

    print(f"Testing TensorLake with API key: {config['api_key'][:20]}...")
    print(f"Base URL: {config['base_url']}")

    provider = TensorLakeProvider(config)

    # Test configuration validation
    print("Validating configuration...")
    is_valid = await provider.validate_config()
    print(f"Config validation: {'PASSED' if is_valid else 'FAILED'}")

    if not is_valid:
        print("Configuration validation failed, cannot proceed with document processing")
        return False

    # Test health check
    print("Running health check...")
    is_healthy = await provider.health_check()
    print(f"Health check: {'PASSED' if is_healthy else 'FAILED'}")

    if not is_healthy:
        print("Health check failed")
        return False

    # Test document processing with a simple PDF
    print("Testing document processing...")
    pdf_content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 1 /Root 1 0 R >>\nstartxref\n50\n%%EOF"

    try:
        result = await provider.process_document(
            file_content=pdf_content,
            filename="test.pdf",
            upload_id="test-upload-123"
        )

        print(f"Processing completed successfully!")
        print(f"Pages processed: {result.total_pages}")
        print(f"Confidence: {result.confidence}")
        if result.pages:
            print(f"First page content (preview): {result.pages[0].content[:100]}...")

        return True

    except Exception as e:
        print(f"Processing failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_tensorlake_provider())
    print(f"\nTest {'PASSED' if success else 'FAILED'}")