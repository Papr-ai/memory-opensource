#!/usr/bin/env python3
"""
Test TensorLake provider with a real PDF file
"""

import asyncio
import os
from dotenv import load_dotenv
from core.document_processing.providers.tensorlake import TensorLakeProvider

def create_real_pdf():
    """Create a real PDF using reportlab"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        import io

        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)

        # Add some content
        p.drawString(100, 750, "Test Document for TensorLake")
        p.drawString(100, 720, "This is a real PDF document created with reportlab.")
        p.drawString(100, 690, "It contains actual text content that should be extractable.")
        p.drawString(100, 660, "Line 4: Additional content for testing purposes.")
        p.drawString(100, 630, "Line 5: More text to ensure the PDF has meaningful content.")

        p.showPage()
        p.save()

        return buffer.getvalue()
    except ImportError:
        print("reportlab not available, using simple text-based approach")
        # If reportlab is not available, create a very basic but valid PDF
        return b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
/Font <<
/F1 5 0 R
>>
>>
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Test Document) Tj
ET
endstream
endobj

5 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj

xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
0000000368 00000 n
trailer
<<
/Size 6
/Root 1 0 R
>>
startxref
465
%%EOF"""

async def test_tensorlake_with_real_pdf():
    """Test TensorLake provider with a real PDF"""

    # Load environment variables conditionally
    use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
    if use_dotenv:
        load_dotenv()

    # Configure provider
    config = {
        "api_key": os.getenv("TENSORLAKE_API_KEY"),
        "base_url": os.getenv("TENSORLAKE_BASE_URL", "https://api.tensorlake.ai"),
        "timeout": 120  # Increased timeout
    }

    print(f"Testing TensorLake with real PDF...")
    print(f"API key: {config['api_key'][:20]}...")

    provider = TensorLakeProvider(config)

    # Test configuration validation
    print("Validating configuration...")
    is_valid = await provider.validate_config()
    print(f"Config validation: {'PASSED' if is_valid else 'FAILED'}")

    if not is_valid:
        print("Configuration validation failed")
        return False

    # Create real PDF content
    print("Creating real PDF...")
    pdf_content = create_real_pdf()
    print(f"PDF size: {len(pdf_content)} bytes")

    # Test document processing
    print("Testing document processing...")

    try:
        result = await provider.process_document(
            file_content=pdf_content,
            filename="real_test.pdf",
            upload_id="test-upload-real-pdf"
        )

        print(f"✅ Processing completed successfully!")
        print(f"Pages processed: {result.total_pages}")
        print(f"Confidence: {result.confidence}")
        print(f"Processing time: {result.processing_time}")
        if result.pages:
            print(f"First page content: {result.pages[0].content[:200]}...")

        return True

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_tensorlake_with_real_pdf())
    print(f"\nTest {'PASSED' if success else 'FAILED'}")