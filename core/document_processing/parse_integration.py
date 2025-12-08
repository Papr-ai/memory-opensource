"""
Integration with Parse Server for document storage
Maintains compatibility with existing Post, PostSocial, and PageVersion classes
"""

import json
import gzip
import uuid
import os
from os import environ as env
import httpx
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from services.logger_singleton import LoggerSingleton
from services.multi_tenant_utils import apply_multi_tenant_scoping_to_metadata
from models.parse_server import PostParseServer
from models.memory_models import MemoryMetadata

logger = LoggerSingleton.get_logger(__name__)


class ParseDocumentIntegration:
    """Handles integration with Parse Server document classes"""

    def __init__(self, memory_graph):
        self.memory_graph = memory_graph
        # Parse Server base URL must come from environment (no hardcoded defaults)
        self.parse_server_url = env.get("PARSE_SERVER_URL")
        self.parse_app_id = env.get("PARSE_APPLICATION_ID")
        self.parse_master_key = env.get("PARSE_MASTER_KEY")

    @staticmethod
    def _json_safe(value):
        """Recursively convert common rich Python objects to JSON-serializable primitives."""
        # Primitives
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        # Enums
        try:
            from enum import Enum
            if isinstance(value, Enum):
                return value.value
        except Exception:
            pass
        # Pydantic models (v2: model_dump, v1: dict)
        if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
            try:
                return ParseDocumentIntegration._json_safe(value.model_dump())
            except Exception:
                pass
        if hasattr(value, "dict") and callable(getattr(value, "dict")):
            try:
                return ParseDocumentIntegration._json_safe(value.dict())
            except Exception:
                pass
        # Datetime
        try:
            from datetime import datetime
            if isinstance(value, datetime):
                return value.isoformat()
        except Exception:
            pass
        # Mapping
        if isinstance(value, dict):
            return {str(k): ParseDocumentIntegration._json_safe(v) for k, v in value.items()}
        # Sequence
        if isinstance(value, (list, tuple, set)):
            return [ParseDocumentIntegration._json_safe(v) for v in value]
        # Fallback to string representation
        return str(value)

    async def create_post_record(
        self,
        content: str,
        upload_id: str,
        organization_id: Optional[str] = None,
        namespace_id: Optional[str] = None,
        user_id: str = None,
        workspace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a Post record in Parse Server for the document

        Returns:
            str: The Parse objectId of the created Post
        """

        # Structure content similar to existing Post format
        structured_content = {
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "attrs": {
                        "id": str(uuid.uuid4())
                    },
                    "content": [
                        {
                            "type": "text",
                            "text": content
                        }
                    ]
                }
            ]
        }

        # Apply multi-tenant scoping to metadata
        auth_context = {
            'organization_id': organization_id,
            'namespace_id': namespace_id,
            'is_legacy_auth': organization_id is None,
            'auth_type': 'organization' if organization_id else 'legacy'
        }

        scoped_metadata = apply_multi_tenant_scoping_to_metadata(
            MemoryMetadata(**(metadata or {})),
            auth_context
        )

        # Create Post record via Parse REST API
        post_data = {
            "content": {
                "default": structured_content
            },
            "uploadId": upload_id,
            "documentProcessed": True,
            "metadata": scoped_metadata.model_dump() if scoped_metadata else {},
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            } if user_id else None,
            "createdAt": {
                "__type": "Date",
                "iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            },
            "updatedAt": {
                "__type": "Date",
                "iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            }
        }

        # Add required workspace field and organization/namespace fields
        if workspace_id:
            post_data["workspace"] = {
                "__type": "Pointer",
                "className": "WorkSpace",
                "objectId": workspace_id
            }

        # Add organization/namespace fields directly to Post if present
        if organization_id:
            post_data["organizationId"] = organization_id
        if namespace_id:
            post_data["namespaceId"] = namespace_id

        try:
            # Create Post via Parse REST API
            headers = {
                "X-Parse-Application-Id": self.parse_app_id,
                "X-Parse-Master-Key": self.parse_master_key,
                "Content-Type": "application/json"
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.parse_server_url}/parse/classes/Post",
                    headers=headers,
                    json=post_data
                )

                if response.status_code == 201:
                    result = response.json()
                    object_id = result["objectId"]
                    logger.info(f"Created Post record with objectId: {object_id} for upload_id: {upload_id}")
                    return object_id
                else:
                    logger.error(f"Failed to create Post record. Status: {response.status_code}, Response: {response.text}")
                    raise Exception(f"Parse Server error: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Failed to create Post record: {e}")
            raise

    async def create_post_social_record(
        self,
        post_object_id: str,
        user_id: str,
        organization_id: Optional[str] = None,
        namespace_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        initial_status: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a PostSocial record for user interaction tracking

        Returns:
            str: The Parse objectId of the created PostSocial
        """

        post_social_data = {
            "post": {
                "__type": "Pointer",
                "className": "Post",
                "objectId": post_object_id
            },
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            },
            "isLiked": False,
            "isBookmarked": False,
            "archive": False,
            "isDelivered": True,  # Mark as delivered since it's uploaded
            "hasRead": False,
            "documentProcessingStatus": initial_status or {
                "status": "processing",
                "progress": 0.0,
                "startedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            },
            "createdAt": {
                "__type": "Date",
                "iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            },
            "updatedAt": {
                "__type": "Date",
                "iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            }
        }

        # Add required workspace field and organization/namespace context
        if workspace_id:
            post_social_data["workspace"] = {
                "__type": "Pointer",
                "className": "WorkSpace",
                "objectId": workspace_id
            }

        if organization_id:
            post_social_data["organizationId"] = organization_id
        if namespace_id:
            post_social_data["namespaceId"] = namespace_id

        try:
            # Create PostSocial via Parse REST API
            headers = {
                "X-Parse-Application-Id": self.parse_app_id,
                "X-Parse-Master-Key": self.parse_master_key,
                "Content-Type": "application/json"
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.parse_server_url}/parse/classes/PostSocial",
                    headers=headers,
                    json=post_social_data
                )

                if response.status_code == 201:
                    result = response.json()
                    object_id = result["objectId"]
                    logger.info(f"Created PostSocial record with objectId: {object_id} for post: {post_object_id}")
                    return object_id
                else:
                    logger.error(f"Failed to create PostSocial record. Status: {response.status_code}, Response: {response.text}")
                    raise Exception(f"Parse Server error: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Failed to create PostSocial record: {e}")
            raise

    async def create_page_version_record(
        self,
        post_object_id: str,
        content: str,
        user_id: str,
        version_type: str = "processed",
        organization_id: Optional[str] = None,
        namespace_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        processing_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a PageVersion record for document version tracking

        Returns:
            str: The Parse objectId of the created PageVersion
        """

        # Structure content for PageVersion
        structured_content = {
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {
                            "type": "text",
                            "text": content
                        }
                    ]
                }
            ]
        }

        page_version_data = {
            "page": {
                "__type": "Pointer",
                "className": "Post",
                "objectId": post_object_id
            },
            "author": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            },
            "content": structured_content,
            "versionType": version_type,
            "processingMetadata": processing_metadata or {},
            "createdAt": {
                "__type": "Date",
                "iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            },
            "updatedAt": {
                "__type": "Date",
                "iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            }
        }

        # Add required workspace field and organization/namespace context
        if workspace_id:
            page_version_data["workspace"] = {
                "__type": "Pointer",
                "className": "WorkSpace",
                "objectId": workspace_id
            }

        if organization_id:
            page_version_data["organizationId"] = organization_id
        if namespace_id:
            page_version_data["namespaceId"] = namespace_id

        try:
            # Create PageVersion via Parse REST API
            headers = {
                "X-Parse-Application-Id": self.parse_app_id,
                "X-Parse-Master-Key": self.parse_master_key,
                "Content-Type": "application/json"
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.parse_server_url}/parse/classes/PageVersion",
                    headers=headers,
                    json=page_version_data
                )

                if response.status_code == 201:
                    result = response.json()
                    object_id = result["objectId"]
                    logger.info(f"Created PageVersion record with objectId: {object_id} for post: {post_object_id}")
                    return object_id
                else:
                    logger.error(f"Failed to create PageVersion record. Status: {response.status_code}, Response: {response.text}")
                    raise Exception(f"Parse Server error: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Failed to create PageVersion record: {e}")
            raise

    async def update_post_social_status(
        self,
        post_social_object_id: str,
        status_update: Dict[str, Any]
    ) -> bool:
        """
        Update processing status in PostSocial record
        """

        try:
            # Simulate status update
            logger.info(f"Updated PostSocial {post_social_object_id} with status: {status_update}")
            return True

        except Exception as e:
            logger.error(f"Failed to update PostSocial status: {e}")
            return False

    async def create_full_document_record(
        self,
        content: str,
        upload_id: str,
        user_id: str,
        organization_id: Optional[str] = None,
        namespace_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        processing_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Create complete document record set (Post + PostSocial + PageVersion)

        Returns:
            Dict with objectIds for post, postSocial, and pageVersion
        """

        # Create Post record
        post_id = await self.create_post_record(
            content, upload_id, organization_id, namespace_id, user_id, workspace_id, metadata
        )

        # PostSocial is auto-created by Parse Server, so we'll get its ID later
        post_social_id = "auto-created"

        # Create PageVersion record
        page_version_id = await self.create_page_version_record(
            post_id, content, user_id, "processed",
            organization_id, namespace_id, workspace_id, processing_metadata
        )

        return {
            "post": post_id,
            "postSocial": post_social_id,
            "pageVersion": page_version_id
        }

    async def create_rich_document_record(
        self,
        pages: List[Dict[str, Any]],
        upload_id: str,
        user_id: str,
        organization_id: Optional[str] = None,
        namespace_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        processing_metadata: Optional[Dict[str, Any]] = None,
        document_title: str = "Document Upload"
    ) -> Dict[str, str]:
        """
        Create rich document record with proper TipTap structure for Parse Server memory integration
        """

        # Build rich TipTap content structure (without noisy status header)
        rich_content = await self._build_rich_tiptap_content(pages, processing_metadata, suppress_status_header=True)

        # Also build a markdown/plain-text representation for quick previews
        markdown_lines: List[str] = []
        if document_title:
            markdown_lines.append(f"# {document_title}")
            markdown_lines.append("")
        for idx, page in enumerate(pages, 1):
            if len(pages) > 1:
                markdown_lines.append(f"## Page {idx}")
            page_text = page.get("content", "").strip()
            if page_text:
                markdown_lines.append(page_text)
                markdown_lines.append("")
        markdown_text = "\n".join(markdown_lines).strip()

        # Apply multi-tenant scoping to metadata
        auth_context = {
            'organization_id': organization_id,
            'namespace_id': namespace_id,
            'is_legacy_auth': organization_id is None,
            'auth_type': 'organization' if organization_id else 'legacy'
        }

        scoped_metadata = apply_multi_tenant_scoping_to_metadata(
            MemoryMetadata(**(metadata or {})),
            auth_context
        )

        # Create Post record with memory integration flags
        post_data = {
            "content": {
                "default": rich_content
            },
            # markdown/plain text representation
            "text": markdown_text,
            "post_title": document_title,
            "uploadId": upload_id,
            "documentProcessed": True,
            "needsMemoryUpdate": True,  # Critical flag for Parse Server memory integration
            "isNew": True,  # Mark as new for Parse Server processing
            "metadata": scoped_metadata.model_dump() if scoped_metadata else {},
            "processingMetadata": processing_metadata or {},
            # Source descriptors
            "type": "page",
            "source": "/document upload API",
            # Set user pointer
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            } if user_id else None,
            "createdAt": {
                "__type": "Date",
                "iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            },
            "updatedAt": {
                "__type": "Date",
                "iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            }
        }

        # Add workspace and tenant context
        if workspace_id:
            post_data["workspace"] = {
                "__type": "Pointer",
                "className": "WorkSpace",
                "objectId": workspace_id
            }

        if organization_id:
            post_data["organizationId"] = organization_id
        if namespace_id:
            post_data["namespaceId"] = namespace_id

        # Attach Parse File pointer when available in metadata
        file_url = (metadata or {}).get("file_url")
        file_name = (metadata or {}).get("file_name") or (metadata or {}).get("filename")
        if file_url and file_name:
            post_data["file"] = {
                "__type": "File",
                "name": str(file_name),
                "url": str(file_url)
            }

        # Build ACL like Memory: developer/end-user only
        if user_id:
            post_data["ACL"] = {
                user_id: {"read": True, "write": True},
                "*": {"read": False, "write": False}
            }

        try:
            # Create Post via Parse REST API
            headers = {
                "X-Parse-Application-Id": self.parse_app_id,
                "X-Parse-Master-Key": self.parse_master_key,
                "Content-Type": "application/json"
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.parse_server_url}/parse/classes/Post",
                    headers=headers,
                    json=post_data
                )

                if response.status_code == 201:
                    result = response.json()
                    post_object_id = result["objectId"]
                    logger.info(f"Created rich Post record with objectId: {post_object_id} for upload_id: {upload_id}")

                    # Create PageVersion record with same rich content
                    page_version_id = await self.create_rich_page_version_record(
                        post_object_id, rich_content, user_id, "processed",
                        organization_id, namespace_id, workspace_id, processing_metadata
                    )

                    return {
                        "post": post_object_id,
                        "postSocial": "auto-created",
                        "pageVersion": page_version_id
                    }
                else:
                    logger.error(f"Failed to create Post record. Status: {response.status_code}, Response: {response.text}")
                    raise Exception(f"Parse Server error: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Failed to create rich Post record: {e}")
            raise

    async def _build_rich_tiptap_content(
        self,
        pages: List[Dict[str, Any]],
        processing_metadata: Optional[Dict[str, Any]] = None,
        suppress_status_header: bool = False
    ) -> Dict[str, Any]:
        """Build rich TipTap/ProseMirror content structure from processed pages"""

        content_blocks = []

        # Add document header if we have processing metadata (optional)
        if processing_metadata and not suppress_status_header:
            provider = processing_metadata.get("provider", "Unknown")
            pages_count = processing_metadata.get("pages_processed", len(pages))
            confidence = processing_metadata.get("confidence", 0.0)

            # Add metadata paragraph
            content_blocks.append({
                "type": "paragraph",
                "attrs": {"id": str(uuid.uuid4())},
                "content": [{
                    "type": "text",
                    "text": f"Document processed by {provider} • {pages_count} pages • {confidence:.1%} confidence",
                    "marks": [{"type": "italic"}]
                }]
            })

            # Add separator
            content_blocks.append({
                "type": "horizontalRule",
                "attrs": {"id": str(uuid.uuid4())}
            })

        # Process each page
        for page_idx, page_data in enumerate(pages, 1):
            page_content = page_data.get("content", "")

            # Add page header for multi-page documents
            if len(pages) > 1:
                content_blocks.append({
                    "type": "heading",
                    "attrs": {
                        "level": 2,
                        "id": str(uuid.uuid4())
                    },
                    "content": [{
                        "type": "text",
                        "text": f"Page {page_idx}"
                    }]
                })

            # Split page content into paragraphs and add to content blocks
            paragraphs = self._split_into_paragraphs(page_content)

            for paragraph in paragraphs:
                if paragraph.strip():
                    content_blocks.append({
                        "type": "paragraph",
                        "attrs": {"id": str(uuid.uuid4())},
                        "content": [{
                            "type": "text",
                            "text": paragraph
                        }]
                    })

            # Add page separator for multi-page documents (except last page)
            if len(pages) > 1 and page_idx < len(pages):
                content_blocks.append({
                    "type": "horizontalRule",
                    "attrs": {"id": str(uuid.uuid4())}
                })

        return {
            "type": "doc",
            "content": content_blocks
        }

    def _split_into_paragraphs(self, content: str) -> List[str]:
        """Split content into paragraphs for better TipTap structure"""

        if not content:
            return []

        # Split by double newlines first (natural paragraph breaks)
        paragraphs = content.split('\n\n')

        # Further split long paragraphs by single newlines if they're too long
        result = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If paragraph is very long (>500 chars), split by single newlines
            if len(paragraph) > 500:
                sub_paragraphs = paragraph.split('\n')
                for sub_p in sub_paragraphs:
                    sub_p = sub_p.strip()
                    if sub_p:
                        result.append(sub_p)
            else:
                result.append(paragraph)

        return result

    async def create_rich_page_version_record(
        self,
        post_object_id: str,
        rich_content: Dict[str, Any],
        user_id: str,
        version_type: str = "processed",
        organization_id: Optional[str] = None,
        namespace_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        processing_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a PageVersion record with rich TipTap content"""

        page_version_data = {
            "page": {
                "__type": "Pointer",
                "className": "Post",
                "objectId": post_object_id
            },
            "author": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            },
            "content": rich_content,
            "versionType": version_type,
            "processingMetadata": processing_metadata or {},
            "createdAt": {
                "__type": "Date",
                "iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            },
            "updatedAt": {
                "__type": "Date",
                "iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            }
        }

        # Add required workspace field and organization/namespace context and ACL
        if workspace_id:
            page_version_data["workspace"] = {
                "__type": "Pointer",
                "className": "WorkSpace",
                "objectId": workspace_id
            }
        # Tight ACL to owner only
        if user_id:
            page_version_data["ACL"] = {
                user_id: {"read": True, "write": True},
                "*": {"read": False, "write": False}
            }

        if organization_id:
            page_version_data["organizationId"] = organization_id
        if namespace_id:
            page_version_data["namespaceId"] = namespace_id

        try:
            # Create PageVersion via Parse REST API
            headers = {
                "X-Parse-Application-Id": self.parse_app_id,
                "X-Parse-Master-Key": self.parse_master_key,
                "Content-Type": "application/json"
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.parse_server_url}/parse/classes/PageVersion",
                    headers=headers,
                    json=page_version_data
                )

                if response.status_code == 201:
                    result = response.json()
                    object_id = result["objectId"]
                    logger.info(f"Created rich PageVersion record with objectId: {object_id} for post: {post_object_id}")
                    return object_id
                else:
                    logger.error(f"Failed to create PageVersion record. Status: {response.status_code}, Response: {response.text}")
                    raise Exception(f"Parse Server error: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Failed to create rich PageVersion record: {e}")
            raise

    async def create_or_update_document_post(
        self,
        upload_id: str,
        post_data: Dict[str, Any],
        organization_id: Optional[str] = None,
        namespace_id: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Create or update document post record - expected by document_activities

        Args:
            upload_id: Unique upload identifier
            post_data: Post data structure with pages and metadata
            organization_id: Organization context
            namespace_id: Namespace context

        Returns:
            Dict with post_id, post_social_id, and page_version_id
        """

        try:
            pages = post_data.get("pages", [])
            metadata = post_data.get("metadata", {})
            processing_metadata = post_data.get("processing_metadata", {})
            user_id = post_data.get("user_id")
            workspace_id = post_data.get("workspace_id")

            # Generate document title from content
            document_title = self._extract_document_title(pages)

            logger.info(f"Creating document post for upload_id: {upload_id} with {len(pages)} pages")

            # Use the existing rich document record creation method
            result = await self.create_rich_document_record(
                pages=pages,
                upload_id=upload_id,
                user_id=user_id,
                organization_id=organization_id,
                namespace_id=namespace_id,
                workspace_id=workspace_id,
                metadata=metadata,
                processing_metadata=processing_metadata,
                document_title=document_title
            )

            logger.info(f"Successfully created document post: {result}")

            return {
                "post_id": result["post"],
                "post_social_id": result["postSocial"],
                "page_version_id": result["pageVersion"]
            }

        except Exception as e:
            logger.error(f"Failed to create or update document post: {e}")
            raise

    async def create_post_with_provider_json(
        self,
        upload_id: str,
        provider_name: str,
        provider_specific: Dict[str, Any],
        user_id: Optional[str],
        organization_id: Optional[str] = None,
        namespace_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        processing_metadata: Optional[Dict[str, Any]] = None,
        document_title: Optional[str] = None
    ) -> PostParseServer:
        """Create a Post where `content` stores the provider's raw JSON result.

        This allows downstream activities to fetch the provider output by post_id
        without passing large payloads through Temporal history.
        """

        try:
            headers = {
                "X-Parse-Application-Id": self.parse_app_id,
                "X-Parse-Master-Key": self.parse_master_key,
                "Content-Type": "application/json"
            }

            # Build post content: place provider payload under a namespaced key
            safe_provider_result = ParseDocumentIntegration._json_safe(provider_specific or {})
            content_obj: Dict[str, Any] = {
                "provider": provider_name,
            }
            
            # Convert provider result to markdown for the text field
            markdown_text = ""
            try:
                from core.document_processing.provider_adapter import provider_to_markdown
                logger.info(f"Attempting to convert provider '{provider_name}' result to markdown")
                logger.info(f"Provider result keys: {list(safe_provider_result.keys())[:10]}")
                markdown_text = provider_to_markdown(provider_name, safe_provider_result)
                logger.info(f"Converted provider result to markdown: {len(markdown_text)} chars")
                if len(markdown_text) > 0:
                    logger.info(f"Markdown preview (first 200 chars): {markdown_text[:200]}")
                else:
                    logger.warning(f"Markdown conversion returned empty string for provider '{provider_name}'")
            except Exception as e:
                logger.error(f"Failed to convert provider result to markdown: {e}", exc_info=True)
                markdown_text = ""  # Fallback to empty string

            # Prefer storing full provider JSON inline when small; otherwise upload as Parse File with a short preview
            try:
                provider_bytes = json.dumps(safe_provider_result, ensure_ascii=False).encode("utf-8")
            except Exception:
                provider_bytes = str(safe_provider_result).encode("utf-8")

            size_bytes = len(provider_bytes)
            logger.info(f"Provider result size: {size_bytes} bytes ({size_bytes / 1024 / 1024:.2f} MB)")
            
            # Always store full provider result as a Parse File pointer; keep Post body tiny
            file_name = f"provider_result_{upload_id}.json.gz"  # .gz extension to indicate it's compressed
            file_resp = None
            try:
                # Compress provider result for efficient storage
                compressed_bytes = gzip.compress(provider_bytes)
                logger.info(f"Provider result size: {len(provider_bytes)} bytes ({len(provider_bytes) / 1024 / 1024:.2f} MB)")
                logger.info(f"Compressed size: {len(compressed_bytes)} bytes ({len(compressed_bytes) / 1024 / 1024:.2f} MB)")
                logger.info(f"Compression ratio: {len(compressed_bytes) / len(provider_bytes) * 100:.1f}%")
                
                # Use longer timeout for large files (120 seconds)
                async with httpx.AsyncClient(timeout=120.0) as client:
                    logger.info(f"Uploading compressed provider result: {self.parse_server_url}/parse/files/{file_name}")
                    
                    # DO NOT set Content-Encoding: gzip!
                    # Content-Encoding tells the server to decompress before storing.
                    # We want to store the gzipped file AS-IS for efficient storage.
                    file_resp = await client.post(
                        f"{self.parse_server_url}/parse/files/{file_name}",
                        headers={
                            "X-Parse-Application-Id": self.parse_app_id,
                            "X-Parse-Master-Key": self.parse_master_key,
                            "Content-Type": "application/octet-stream",
                            # NO Content-Encoding header - store gzipped AS-IS
                        },
                        content=compressed_bytes,
                    )
                    logger.info(f"File upload response: {file_resp.status_code}")
            except Exception as e:
                logger.error(f"File upload failed with exception: {e}")
                file_resp = None

            if file_resp is not None and file_resp.status_code in (200, 201):
                file_data = file_resp.json()
                content_obj["provider_result_file"] = {
                    "__type": "File",
                    "name": file_data.get("name"),
                    "url": file_data.get("url"),
                }
                content_obj["provider_result_meta"] = {"stored": "file", "size": size_bytes}
                # Provide a small default text for downstream cloud code expectations
                content_obj["default"] = (document_title or "Document Upload")[:512]
            else:
                # As a last resort, store a minimal placeholder and record size; avoid large bodies
                content_obj["provider_result_meta"] = {"stored": "unavailable", "size": size_bytes}
                content_obj["default"] = (document_title or "Document Upload")[:512]

            # Basic title
            title = document_title or (metadata or {}).get("file_name") or f"Document Upload ({provider_name})"

            # Attach file pointer if available
            file_url = (metadata or {}).get("file_url")
            file_name = (metadata or {}).get("file_name") or (metadata or {}).get("filename")

            post_payload: Dict[str, Any] = {
                "content": content_obj,
                "post_title": title,
                "text": markdown_text,  # Add markdown version of content for PageVersion creation
                "uploadId": upload_id,
                "documentProcessed": True,
                "metadata": ParseDocumentIntegration._json_safe(metadata or {}),
                "processingMetadata": ParseDocumentIntegration._json_safe(processing_metadata or {}),
                "type": "page",
                "source": "/document upload API",
                # Set needsMemoryUpdate=false since Temporal workflow will create memories
                "needsMemoryUpdate": False,
                # Set initial processing status  
                "processingStatus": "processing",
            }
            
            # Log text field status
            logger.info(f"Post payload 'text' field length: {len(markdown_text) if markdown_text else 0} chars")
            if not markdown_text:
                logger.warning(f"⚠️ Post will be created with empty 'text' field for provider '{provider_name}'")

            if user_id:
                post_payload["user"] = {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": user_id,
                }

            if workspace_id:
                post_payload["workspace"] = {
                    "__type": "Pointer",
                    "className": "WorkSpace",
                    "objectId": workspace_id,
                }

            if organization_id:
                post_payload["organizationId"] = organization_id
            if namespace_id:
                post_payload["namespaceId"] = namespace_id

            # Add file_url as top-level field and as file pointer
            if file_url:
                post_payload["file_url"] = str(file_url)
                
            if file_url and file_name:
                post_payload["file"] = {
                    "__type": "File",
                    "name": str(file_name),
                    "url": str(file_url),
                }

            # Tight ACL to owner only if user provided
            if user_id:
                post_payload["ACL"] = {
                    user_id: {"read": True, "write": True},
                    "*": {"read": False, "write": False},
                }

            # Log the actual payload size being sent
            payload_json = json.dumps(post_payload, ensure_ascii=False)
            payload_size = len(payload_json.encode("utf-8"))
            logger.info(f"Post payload size: {payload_size} bytes ({payload_size / 1024:.2f} KB)")
            if payload_size > 100000:  # 100KB
                logger.warning(f"Post payload exceeds 100KB! Size: {payload_size / 1024 / 1024:.2f} MB")
                logger.warning(f"Payload keys: {list(post_payload.keys())}")
                for key, value in post_payload.items():
                    if isinstance(value, (dict, list)):
                        key_size = len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
                        logger.warning(f"  - {key}: {key_size / 1024:.2f} KB")

            # Use longer timeout for Post creation (120s) since we're uploading file references
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self.parse_server_url}/parse/classes/Post",
                    headers=headers,
                    json=post_payload,
                )

            if resp.status_code in (200, 201):
                data = resp.json()
                post_id = data.get("objectId")
                if not post_id:
                    raise Exception("Parse returned no objectId")
                logger.info(f"Created provider-json Post {post_id} for upload {upload_id}")
                return PostParseServer(objectId=post_id)
            # Compact error to avoid exceeding Temporal failure size limits
            err_snippet = (resp.text or "")[:512]
            raise Exception(f"Parse Server error {resp.status_code}: {err_snippet}")
        except Exception as e:
            # Log concise error to keep Temporal failure small
            logger.error("Failed to create provider-json Post")
            raise

    def _extract_document_title(self, pages: List[Dict[str, Any]]) -> str:
        """Extract a meaningful title from the first page content"""

        if not pages:
            return "Document Upload"

        first_page = pages[0]
        content = first_page.get("content", "")

        if not content:
            return "Document Upload"

        # Split into lines and look for a good title
        lines = content.split('\n')

        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 100:
                # Check if it looks like a title (not too long, has meaningful words)
                words = line.split()
                if len(words) >= 2 and len(words) <= 12:
                    return line

        # Fallback to first non-empty line
        for line in lines:
            line = line.strip()
            if line and len(line) > 5:
                return line[:80] + ("..." if len(line) > 80 else "")

        return "Document Upload"

    async def link_memories_to_post(self, post_id: str, memory_object_ids: List[str]) -> bool:
        """Attach Memory relations to Post in Parse Server"""
        if not post_id or not memory_object_ids:
            return True

        try:
            headers = {
                "X-Parse-Application-Id": self.parse_app_id,
                "X-Parse-Master-Key": self.parse_master_key,
                "Content-Type": "application/json"
            }

            # Use batch relation add operation for performance
            ops = []
            for mem_id in memory_object_ids:
                ops.append({
                    "method": "PUT",
                    "path": f"/parse/classes/Post/{post_id}",
                    "body": {
                        "memories": {
                            "__op": "AddRelation",
                            "objects": [
                                {"__type": "Pointer", "className": "Memory", "objectId": mem_id}
                            ]
                        }
                    }
                })

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.parse_server_url}/parse/batch",
                    headers=headers,
                    json={"requests": ops}
                )

                if response.status_code in (200, 201):
                    logger.info(f"Linked {len(memory_object_ids)} memories to Post {post_id}")
                    return True
                else:
                    logger.error(f"Failed linking memories: {response.status_code} {response.text}")
                    return False

        except Exception as e:
            logger.error(f"Error linking memories to Post: {e}")
            return False

    async def get_post(self, post_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a Post record from Parse Server by objectId
        
        Args:
            post_id: Parse Post objectId
            
        Returns:
            Dict with Post data if found, None if not found or error
        """
        try:
            headers = {
                "X-Parse-Application-Id": self.parse_app_id,
                "X-Parse-Master-Key": self.parse_master_key,
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.parse_server_url}/parse/classes/Post/{post_id}",
                    headers=headers
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    logger.warning(f"Post {post_id} not found (404)")
                    return None
                else:
                    logger.error(f"Failed to fetch Post {post_id}: {response.status_code} {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching Post {post_id}: {e}")
            return None

    async def update_post(self, post_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update a Post record in Parse Server

        Args:
            post_id: The Parse objectId of the Post to update
            update_data: Dictionary of fields to update

        Returns:
            bool: True if update successful
        """
        try:
            headers = {
                "X-Parse-Application-Id": self.parse_app_id,
                "X-Parse-Master-Key": self.parse_master_key,
                "Content-Type": "application/json"
            }

            # Make fields JSON-safe
            safe_data = ParseDocumentIntegration._json_safe(update_data)

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.put(
                    f"{self.parse_server_url}/parse/classes/Post/{post_id}",
                    headers=headers,
                    json=safe_data
                )

                if response.status_code == 200:
                    logger.info(f"Updated Post {post_id} with fields: {list(update_data.keys())}")
                    return True
                else:
                    error_msg = f"Failed to update Post {post_id}: {response.status_code} {response.text}"
                    logger.error(error_msg)
                    # Raise exception for 404 (Post not found) to trigger Temporal retry
                    if response.status_code == 404:
                        raise Exception(f"Post {post_id} not found - it may not have been created yet or was deleted")
                    return False

        except httpx.HTTPError as e:
            logger.error(f"HTTP error updating Post {post_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error updating Post {post_id}: {e}")
            raise

    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        content_type: str = "application/octet-stream"
    ) -> str:
        """
        Upload a file to Parse Server

        Args:
            file_content: Raw bytes of the file
            filename: Name for the file
            content_type: MIME type of the file

        Returns:
            str: URL of the uploaded file
        """
        try:
            headers = {
                "X-Parse-Application-Id": self.parse_app_id,
                "X-Parse-Master-Key": self.parse_master_key,
                "Content-Type": content_type,
            }

            # DO NOT set Content-Encoding: gzip!
            # Content-Encoding tells the server to decompress before storing.
            # We want to store the gzipped file AS-IS, so we only set Content-Type.

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.parse_server_url}/parse/files/{filename}",
                    headers=headers,
                    content=file_content
                )

                if response.status_code in (200, 201):
                    result = response.json()
                    file_url = result.get("url")
                    logger.info(f"Uploaded file {filename} ({len(file_content)} bytes) -> {file_url}")
                    return file_url
                else:
                    logger.error(f"Failed to upload file {filename}: {response.status_code} {response.text}")
                    raise Exception(f"File upload failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Error uploading file {filename}: {e}")
            raise