from typing import Optional
from models.parse_server import DocumentUploadStatusType, DocumentUploadStatusResponse
from services.memory_management import update_document_upload_status
import logging

from services.logger_singleton import LoggerSingleton

# Create a logger instance for this module
logger = LoggerSingleton.get_logger(__name__)

async def update_processing_status(
    upload_id: str,
    filename: Optional[str] = None,
    current_page: Optional[int] = None,
    total_pages: Optional[int] = None,
    status: DocumentUploadStatusType = DocumentUploadStatusType.PROCESSING,
    error: Optional[str] = None,
    objectId: Optional[str] = None,
    post_objectId: Optional[str] = None,
    file_url: Optional[str] = None
) -> Optional[DocumentUploadStatusResponse]:
    """Update document processing status in Parse Server"""
    logger.info(f"Updating status for upload_id: {upload_id}")
    
    if not objectId:
        logger.error("No Memory objectId provided for status update")
        return None

    # Calculate progress as a float between 0 and 1
    if current_page is not None and total_pages is not None and total_pages > 0:
        progress = current_page / total_pages
    else:
        progress = 0.0

    try:
        response = await update_document_upload_status(
            objectId=objectId,
            filename=filename,
            status=status,
            progress=progress,
            current_page=current_page,
            total_pages=total_pages,
            error=error,
            post_objectId=post_objectId,
            upload_id=upload_id,
            file_url=file_url
        )
        logger.info(f"Successfully updated document status for memory {objectId}" + 
                   (f" and post {post_objectId}" if post_objectId else ""))
        return response
    except Exception as e:
        logger.error(f"Error updating document status: {e}", exc_info=True)
        return None 