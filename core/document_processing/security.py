"""
Security utilities for document processing
"""

import hashlib
import os
import tempfile
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class FileValidator:
    """Validates uploaded files for security and compliance"""

    ALLOWED_MIME_TYPES = {
        'application/pdf': ['.pdf'],
        'image/png': ['.png'],
        'image/jpeg': ['.jpg', '.jpeg'],
        'image/webp': ['.webp'],
        'text/html': ['.html', '.htm'],
        'text/plain': ['.txt'],
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx']
    }

    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MALICIOUS_PATTERNS = [
        b'<script',
        b'javascript:',
        b'vbscript:',
        b'onload=',
        b'onerror=',
        b'data:text/html',
        b'eval(',
        b'document.write'
    ]

    @classmethod
    async def validate_file(
        cls,
        file_content: bytes,
        filename: str,
        organization_limits: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Comprehensive file validation"""

        logger.info(f"🔒 FileValidator.validate_file called for {filename}, size: {len(file_content)} bytes")
        
        # Size validation
        max_size = organization_limits.get('max_file_size', cls.MAX_FILE_SIZE) if organization_limits else cls.MAX_FILE_SIZE
        if len(file_content) > max_size:
            logger.warning(f"🔒 File {filename} too large: {len(file_content)} > {max_size}")
            return False, f"File size {len(file_content)} exceeds limit {max_size}"

        if len(file_content) < 10:
            logger.warning(f"🔒 File {filename} too small: {len(file_content)} bytes")
            return False, "File appears to be empty"

        # MIME type validation using python-magic if available
        mime_type = None
        try:
            import magic
            mime_type = magic.from_buffer(file_content, mime=True)
            logger.info(f"🔒 python-magic detected MIME type: {mime_type} for {filename}")
        except ImportError:
            # Fallback to simple extension checking
            logger.info(f"🔒 python-magic not available, using extension-based detection for {filename}")
            file_ext = os.path.splitext(filename.lower())[1]
            # Map extensions to MIME types
            ext_to_mime = {
                '.pdf': 'application/pdf',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.webp': 'image/webp',
                '.html': 'text/html',
                '.htm': 'text/html',
                '.txt': 'text/plain',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            }
            mime_type = ext_to_mime.get(file_ext)
            logger.info(f"🔒 Extension {file_ext} mapped to MIME type: {mime_type}")
        except Exception as e:
            logger.error(f"🔒 Error in python-magic: {e}")
            mime_type = None

        if not mime_type:
            logger.warning(f"🔒 Could not determine MIME type for {filename}")
            return False, "Could not determine file type"

        allowed_types = organization_limits.get('allowed_mime_types', cls.ALLOWED_MIME_TYPES) if organization_limits else cls.ALLOWED_MIME_TYPES
        if mime_type not in allowed_types:
            return False, f"File type {mime_type} not allowed"

        # Extension validation
        file_ext = os.path.splitext(filename.lower())[1]
        if file_ext not in allowed_types[mime_type]:
            return False, f"File extension {file_ext} doesn't match MIME type {mime_type}"

        # Content validation for text-based files
        if mime_type in ['text/html', 'text/plain']:
            for pattern in cls.MALICIOUS_PATTERNS:
                if pattern in file_content.lower():
                    return False, f"File contains potentially malicious content"

        # PDF-specific validation
        if mime_type == 'application/pdf':
            if not file_content.startswith(b'%PDF-'):
                return False, "Invalid PDF file format"

        # Additional security checks
        logger.info(f"🔒 Checking suspicious content for {filename}")
        if cls._contains_suspicious_content(file_content):
            logger.warning(f"🔒 File {filename} contains suspicious content patterns")
            return False, "File contains suspicious content patterns"

        logger.info(f"🔒 File {filename} passed all validation checks")
        return True, None

    @classmethod
    def calculate_file_hash(cls, file_content: bytes) -> str:
        """Calculate SHA-256 hash of file content"""
        return hashlib.sha256(file_content).hexdigest()

    @classmethod
    def _contains_suspicious_content(cls, file_content: bytes) -> bool:
        """Check for suspicious content patterns"""

        # Check for embedded executables (basic check)
        suspicious_headers = [
            b'MZ',  # DOS/Windows executable
            b'\x7fELF',  # Linux executable
            b'\xfe\xed\xfa',  # Mach-O executable
        ]

        for header in suspicious_headers:
            if file_content.startswith(header):
                return True

        # Check for suspicious JavaScript patterns in non-JS files
        js_patterns = [
            b'XMLHttpRequest',
            b'fetch(',
            b'window.location',
            b'document.cookie',
            b'localStorage'
        ]

        content_lower = file_content.lower()
        suspicious_count = sum(1 for pattern in js_patterns if pattern in content_lower)

        # If multiple suspicious JS patterns found, flag as suspicious
        return suspicious_count >= 3


class SecureFileStorage:
    """Secure file storage with encryption and access control"""

    def __init__(self, storage_config: Dict[str, Any]):
        self.config = storage_config
        self.storage_type = storage_config.get('type', 'local')

        if self.storage_type == 's3':
            try:
                import boto3
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=storage_config.get('access_key'),
                    aws_secret_access_key=storage_config.get('secret_key'),
                    region_name=storage_config.get('region', 'us-east-1')
                )
                self.bucket_name = storage_config.get('bucket_name')
            except ImportError:
                logger.error("boto3 not installed for S3 storage")
                self.s3_client = None
        elif self.storage_type == 'local':
            from pathlib import Path
            self.base_path = Path(storage_config.get('base_path', '/tmp/document_storage'))
            self.base_path.mkdir(parents=True, exist_ok=True)

    async def store_file(
        self,
        file_content: bytes,
        upload_id: str,
        organization_id: str,
        namespace: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store file securely with organization scoping"""

        # Generate secure file path
        file_path = self._generate_file_path(upload_id, organization_id, namespace)

        if self.storage_type == 's3' and self.s3_client:
            return await self._store_s3(file_content, file_path, metadata)
        else:
            return await self._store_local(file_content, file_path, metadata)

    async def retrieve_file(
        self,
        file_path: str,
        organization_id: str,
        namespace: Optional[str] = None
    ) -> Optional[bytes]:
        """Retrieve file with access control validation"""

        # Validate access permissions
        if not self._validate_access(file_path, organization_id, namespace):
            logger.warning(f"Access denied for {organization_id} to {file_path}")
            return None

        if self.storage_type == 's3' and self.s3_client:
            return await self._retrieve_s3(file_path)
        else:
            return await self._retrieve_local(file_path)

    async def delete_file(
        self,
        file_path: str,
        organization_id: str,
        namespace: Optional[str] = None
    ) -> bool:
        """Delete file with access control validation"""

        if not self._validate_access(file_path, organization_id, namespace):
            logger.warning(f"Delete access denied for {organization_id} to {file_path}")
            return False

        if self.storage_type == 's3' and self.s3_client:
            return await self._delete_s3(file_path)
        else:
            return await self._delete_local(file_path)

    def _generate_file_path(
        self,
        upload_id: str,
        organization_id: str,
        namespace: Optional[str]
    ) -> str:
        """Generate secure, hierarchical file path"""

        # Hash organization ID for additional security
        org_hash = hashlib.sha256(organization_id.encode()).hexdigest()[:16]
        namespace_part = f"/{namespace}" if namespace else ""

        return f"organizations/{org_hash}{namespace_part}/uploads/{upload_id}"

    def _validate_access(
        self,
        file_path: str,
        organization_id: str,
        namespace: Optional[str]
    ) -> bool:
        """Validate that organization has access to file path"""

        expected_path = self._generate_file_path("", organization_id, namespace).rsplit("/", 1)[0]
        return file_path.startswith(expected_path)

    async def _store_s3(
        self,
        file_content: bytes,
        file_path: str,
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Store file in S3 with encryption"""

        try:
            extra_args = {
                'ServerSideEncryption': 'AES256',
                'ContentType': 'application/octet-stream'
            }

            if metadata:
                # Convert metadata to string format for S3
                extra_args['Metadata'] = {
                    k: str(v) for k, v in metadata.items() if isinstance(k, str)
                }

            # Upload with encryption
            with tempfile.NamedTemporaryFile() as tmp_file:
                tmp_file.write(file_content)
                tmp_file.flush()

                self.s3_client.upload_file(
                    tmp_file.name,
                    self.bucket_name,
                    file_path,
                    ExtraArgs=extra_args
                )

            return f"s3://{self.bucket_name}/{file_path}"

        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            raise

    async def _retrieve_s3(self, file_path: str) -> Optional[bytes]:
        """Retrieve file from S3"""

        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_path)
            return response['Body'].read()
        except Exception as e:
            logger.error(f"S3 retrieval failed: {e}")
            return None

    async def _delete_s3(self, file_path: str) -> bool:
        """Delete file from S3"""

        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=file_path)
            return True
        except Exception as e:
            logger.error(f"S3 deletion failed: {e}")
            return False

    async def _store_local(
        self,
        file_content: bytes,
        file_path: str,
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Store file locally with proper permissions"""

        import json
        import aiofiles

        full_path = self.base_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file with restricted permissions
        async with aiofiles.open(full_path, 'wb') as f:
            await f.write(file_content)

        # Set secure permissions (owner read/write only)
        os.chmod(full_path, 0o600)

        # Store metadata separately if provided
        if metadata:
            metadata_path = full_path.with_suffix('.metadata.json')
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(json.dumps(metadata, default=str))
            os.chmod(metadata_path, 0o600)

        return str(full_path)

    async def _retrieve_local(self, file_path: str) -> Optional[bytes]:
        """Retrieve file from local storage"""

        import aiofiles

        full_path = self.base_path / file_path

        if not full_path.exists():
            return None

        try:
            async with aiofiles.open(full_path, 'rb') as f:
                return await f.read()
        except Exception as e:
            logger.error(f"Local file retrieval failed: {e}")
            return None

    async def _delete_local(self, file_path: str) -> bool:
        """Delete file from local storage"""

        full_path = self.base_path / file_path

        try:
            if full_path.exists():
                full_path.unlink()

            # Also delete metadata file if it exists
            metadata_path = full_path.with_suffix('.metadata.json')
            if metadata_path.exists():
                metadata_path.unlink()

            return True
        except Exception as e:
            logger.error(f"Local file deletion failed: {e}")
            return False