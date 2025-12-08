"""
Integration tests for the new document processing system
"""

import pytest
import asyncio
import json
import uuid
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Import the components we're testing
from core.document_processing.providers.base import DocumentProvider, ProcessingResult, DocumentPage
from core.document_processing.providers.tensorlake import TensorLakeProvider
from core.document_processing.providers.reducto import ReductoProvider
from core.document_processing.providers.gemini import GeminiVisionProvider
from core.document_processing.provider_manager import DocumentProcessorFactory, TenantConfigManager
from core.document_processing.security import FileValidator
from core.document_processing.parse_integration import ParseDocumentIntegration


class TestDocumentProviders:
    """Test document processing providers"""

    @pytest.mark.asyncio
    async def test_tensorlake_provider_initialization(self):
        """Test TensorLake provider initialization"""
        config = {
            "api_key": "test_key",
            "base_url": "https://api.tensorlake.ai",
            "timeout": 300
        }

        provider = TensorLakeProvider(config)
        assert provider.api_key == "test_key"
        assert provider.base_url == "https://api.tensorlake.ai"
        assert provider.provider_name == "tensorlake"

    @pytest.mark.asyncio
    async def test_reducto_provider_initialization(self):
        """Test Reducto provider initialization"""
        config = {
            "api_key": "test_key",
            "base_url": "https://api.reducto.ai",
            "timeout": 300
        }

        provider = ReductoProvider(config)
        assert provider.api_key == "test_key"
        assert provider.base_url == "https://api.reducto.ai"
        assert provider.provider_name == "reducto"

    @pytest.mark.asyncio
    async def test_gemini_provider_initialization(self):
        """Test Gemini provider initialization"""
        config = {
            "api_key": "test_key",
            "model": "gemini-2.5-flash"
        }

        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                provider = GeminiVisionProvider(config)
                assert provider.api_key == "test_key"
                assert provider.model == "gemini-2.5-flash"
                assert provider.provider_name == "geminivision"

    @pytest.mark.asyncio
    async def test_provider_supported_formats(self):
        """Test provider supported formats"""
        tensorlake_config = {"api_key": "test"}
        reducto_config = {"api_key": "test"}
        gemini_config = {"api_key": "test"}

        tensorlake = TensorLakeProvider(tensorlake_config)
        reducto = ReductoProvider(reducto_config)

        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                gemini = GeminiVisionProvider(gemini_config)

        assert "pdf" in tensorlake.get_supported_formats()
        assert "png" in tensorlake.get_supported_formats()

        assert "pdf" in reducto.get_supported_formats()
        assert "docx" in reducto.get_supported_formats()

        assert "pdf" in gemini.get_supported_formats()
        assert "webp" in gemini.get_supported_formats()


class TestFileValidation:
    """Test file validation and security"""

    @pytest.mark.asyncio
    async def test_valid_pdf_file(self):
        """Test validation of valid PDF file"""
        # Simulate PDF content
        pdf_content = b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n'
        filename = "test.pdf"

        is_valid, error = await FileValidator.validate_file(pdf_content, filename)
        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_invalid_file_size(self):
        """Test validation fails for oversized files"""
        # Create content larger than 50MB
        large_content = b"x" * (51 * 1024 * 1024)
        filename = "large.pdf"

        is_valid, error = await FileValidator.validate_file(large_content, filename)
        assert is_valid is False
        assert "exceeds limit" in error

    @pytest.mark.asyncio
    async def test_unsupported_file_type(self):
        """Test validation fails for unsupported file types"""
        content = b"some content"
        filename = "test.exe"

        is_valid, error = await FileValidator.validate_file(content, filename)
        assert is_valid is False
        assert "not allowed" in error

    @pytest.mark.asyncio
    async def test_malicious_content_detection(self):
        """Test detection of potentially malicious content"""
        malicious_content = b'<script>alert("xss")</script>'
        filename = "test.html"

        is_valid, error = await FileValidator.validate_file(malicious_content, filename)
        assert is_valid is False
        assert "malicious content" in error

    def test_file_hash_calculation(self):
        """Test file hash calculation"""
        content = b"test content"
        hash1 = FileValidator.calculate_file_hash(content)
        hash2 = FileValidator.calculate_file_hash(content)

        assert hash1 == hash2  # Same content should produce same hash
        assert len(hash1) == 64  # SHA-256 produces 64-character hex string


class TestProviderManager:
    """Test provider management and factory"""

    @pytest.mark.asyncio
    async def test_tenant_config_manager(self):
        """Test tenant configuration management"""
        mock_memory_graph = Mock()

        config_manager = TenantConfigManager(mock_memory_graph)

        # Test default configuration
        config = await config_manager.get_tenant_config("test_org", "test_namespace")
        assert config.organization_id == "test_org"
        assert config.namespace == "test_namespace"
        assert len(config.providers) > 0

    @pytest.mark.asyncio
    async def test_document_processor_factory(self):
        """Test document processor factory"""
        mock_memory_graph = Mock()
        config_manager = TenantConfigManager(mock_memory_graph)
        factory = DocumentProcessorFactory(config_manager)

        # Mock a working provider
        with patch.object(factory, '_create_provider_instance') as mock_create:
            mock_provider = Mock(spec=DocumentProvider)
            mock_provider.health_check = AsyncMock(return_value=True)
            mock_create.return_value = mock_provider

            processor = await factory.create_processor("test_org", "test_namespace")
            assert processor is not None

    @pytest.mark.asyncio
    async def test_provider_fallback_logic(self):
        """Test provider fallback when primary fails"""
        mock_memory_graph = Mock()
        config_manager = TenantConfigManager(mock_memory_graph)
        factory = DocumentProcessorFactory(config_manager)

        call_count = 0

        async def mock_create_provider(config):
            nonlocal call_count
            call_count += 1

            mock_provider = Mock(spec=DocumentProvider)
            # First provider fails health check, second succeeds
            mock_provider.health_check = AsyncMock(return_value=call_count > 1)
            return mock_provider

        with patch.object(factory, '_create_provider_instance', side_effect=mock_create_provider):
            processor = await factory.create_processor("test_org", "test_namespace")
            assert processor is not None
            assert call_count >= 2  # Should have tried multiple providers


class TestParseIntegration:
    """Test Parse Server integration"""

    @pytest.mark.asyncio
    async def test_parse_document_integration(self):
        """Test Parse Server document record creation"""
        mock_memory_graph = Mock()
        integration = ParseDocumentIntegration(mock_memory_graph)

        # Test Post record creation
        post_id = await integration.create_post_record(
            content="Test document content",
            upload_id="test-upload-123",
            organization_id="org-123",
            namespace_id="namespace-456",
            user_id="user-789"
        )

        assert post_id is not None
        assert len(post_id) > 0

        # Test PostSocial record creation
        post_social_id = await integration.create_post_social_record(
            post_object_id=post_id,
            user_id="user-789",
            organization_id="org-123",
            namespace_id="namespace-456"
        )

        assert post_social_id is not None
        assert len(post_social_id) > 0

        # Test PageVersion record creation
        page_version_id = await integration.create_page_version_record(
            post_object_id=post_id,
            content="Processed document content",
            user_id="user-789",
            organization_id="org-123",
            namespace_id="namespace-456"
        )

        assert page_version_id is not None
        assert len(page_version_id) > 0

    @pytest.mark.asyncio
    async def test_full_document_record_creation(self):
        """Test complete document record set creation"""
        mock_memory_graph = Mock()
        integration = ParseDocumentIntegration(mock_memory_graph)

        result = await integration.create_full_document_record(
            content="Complete document content",
            upload_id="test-upload-456",
            user_id="user-789",
            organization_id="org-123",
            namespace_id="namespace-456",
            metadata={"source": "upload", "type": "pdf"},
            processing_metadata={"provider": "tensorlake", "confidence": 0.95}
        )

        assert "post" in result
        assert "postSocial" in result
        assert "pageVersion" in result
        assert all(len(obj_id) > 0 for obj_id in result.values())


class TestMultiTenantScoping:
    """Test multi-tenant functionality"""

    def test_organization_namespace_scoping(self):
        """Test proper organization/namespace scoping"""
        from services.multi_tenant_utils import extract_multi_tenant_context

        # Mock auth response
        mock_auth_response = Mock()
        mock_auth_response.organization_id = "org-123"
        mock_auth_response.namespace_id = "namespace-456"
        mock_auth_response.is_legacy_auth = False
        mock_auth_response.auth_type = "organization"

        context = extract_multi_tenant_context(mock_auth_response)

        assert context['organization_id'] == "org-123"
        assert context['namespace_id'] == "namespace-456"
        assert context['is_legacy_auth'] is False
        assert context['auth_type'] == "organization"

    def test_legacy_auth_context(self):
        """Test legacy authentication context"""
        from services.multi_tenant_utils import extract_multi_tenant_context

        # Mock legacy auth response
        mock_auth_response = Mock()
        mock_auth_response.organization_id = None
        mock_auth_response.namespace_id = None
        mock_auth_response.is_legacy_auth = True
        mock_auth_response.auth_type = "legacy"

        context = extract_multi_tenant_context(mock_auth_response)

        assert context['organization_id'] is None
        assert context['namespace_id'] is None
        assert context['is_legacy_auth'] is True
        assert context['auth_type'] == "legacy"


class TestErrorHandling:
    """Test error handling and edge cases"""

    @pytest.mark.asyncio
    async def test_provider_unavailable_error(self):
        """Test handling when all providers are unavailable"""
        mock_memory_graph = Mock()
        config_manager = TenantConfigManager(mock_memory_graph)
        factory = DocumentProcessorFactory(config_manager)

        async def failing_create_provider(config):
            mock_provider = Mock(spec=DocumentProvider)
            mock_provider.health_check = AsyncMock(return_value=False)
            return mock_provider

        with patch.object(factory, '_create_provider_instance', side_effect=failing_create_provider):
            with pytest.raises(ValueError, match="No healthy document processing provider available"):
                await factory.create_processor("test_org", "test_namespace")

    @pytest.mark.asyncio
    async def test_invalid_provider_config(self):
        """Test handling of invalid provider configuration"""
        config = {"api_key": ""}  # Invalid empty API key

        provider = TensorLakeProvider(config)
        is_valid = await provider.validate_config()
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_processing_failure_handling(self):
        """Test handling of processing failures"""
        config = {"api_key": "test_key"}
        provider = TensorLakeProvider(config)

        # Mock HTTP error
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = Exception("API Error")
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            with pytest.raises(Exception, match="TensorLake processing failed"):
                await provider.process_document(
                    b"test content",
                    "test.pdf",
                    "upload-123"
                )


@pytest.mark.asyncio
async def test_end_to_end_document_processing():
    """End-to-end test of document processing workflow"""

    # Create test data
    upload_id = str(uuid.uuid4())
    file_content = b"%PDF-1.4\nTest PDF content"
    filename = "test.pdf"
    organization_id = "test-org"
    namespace_id = "test-namespace"

    # Test file validation
    is_valid, error = await FileValidator.validate_file(file_content, filename)
    assert is_valid is True

    # Test provider factory
    mock_memory_graph = Mock()
    config_manager = TenantConfigManager(mock_memory_graph)
    factory = DocumentProcessorFactory(config_manager)

    # Mock successful processing
    with patch.object(factory, 'create_processor') as mock_create:
        mock_provider = Mock(spec=DocumentProvider)
        mock_provider.provider_name = "gemini"

        # Mock processing result
        mock_result = ProcessingResult(
            pages=[
                DocumentPage(
                    page_number=1,
                    content="Extracted text content",
                    confidence=0.95,
                    metadata={}
                )
            ],
            total_pages=1,
            processing_time=2.5,
            confidence=0.95,
            metadata={},
            provider_specific={}
        )

        mock_provider.process_document = AsyncMock(return_value=mock_result)
        mock_create.return_value = mock_provider

        # Test processing
        processor = await factory.create_processor(organization_id, namespace_id)
        result = await processor.process_document(file_content, filename, upload_id)

        assert result.total_pages == 1
        assert result.pages[0].content == "Extracted text content"
        assert result.confidence == 0.95

    # Test Parse integration
    integration = ParseDocumentIntegration(mock_memory_graph)
    parse_records = await integration.create_full_document_record(
        content="Extracted text content",
        upload_id=upload_id,
        user_id="test-user",
        organization_id=organization_id,
        namespace_id=namespace_id
    )

    assert "post" in parse_records
    assert "postSocial" in parse_records
    assert "pageVersion" in parse_records


if __name__ == "__main__":
    pytest.main([__file__, "-v"])