import asyncio
import os
from os import environ as env
import httpx
import pytest
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env
_ENV = find_dotenv()
if _ENV:
    load_dotenv(_ENV)


def test_create_post_with_provider_json_real_parse():
    from core.document_processing.parse_integration import ParseDocumentIntegration
    from models.parse_server import PostParseServer

    parse_url = env.get("PARSE_SERVER_URL")
    app_id = env.get("PARSE_APPLICATION_ID")
    master_key = env.get("PARSE_MASTER_KEY")
    workspace_id = (
        env.get("WORKSPACE_ID")
        or env.get("TEST_WORKSPACE_ID")
        or env.get("PARSE_WORKSPACE_ID")
        or env.get("DEFAULT_WORKSPACE_ID")
    )
    # Optional user/tenant context
    user_id = env.get("USER_ID") or env.get("TEST_USER_ID") or env.get("PARSE_USER_ID")
    organization_id = env.get("ORGANIZATION_ID") or env.get("ORG_ID") or env.get("TEST_ORGANIZATION_ID")
    namespace_id = env.get("NAMESPACE_ID") or env.get("NS_ID") or env.get("TEST_NAMESPACE_ID")
    if not (parse_url and app_id and master_key and workspace_id):
        pytest.skip("Real Parse test skipped: set PARSE_SERVER_URL, PARSE_APPLICATION_ID, PARSE_MASTER_KEY, and WORKSPACE_ID in .env")

    async def run():
        integration = ParseDocumentIntegration(None)
        # Use actual PDF file from repo for a realistic metadata/file pointer
        pdf_path = "/Users/shawkatkabbara/Documents/GitHub/memory/tests/2502.02533v1.pdf"
        file_meta = {"file_name": "2502.02533v1.pdf"}
        if os.path.exists(pdf_path):
            # We only attach a file pointer in metadata; actual upload not needed for this test
            file_meta.update({"filename": "2502.02533v1.pdf"})

        result = await integration.create_or_update_document_post(
            upload_id="up_real_1",
            post_data={
                "pages": [{"content": "Sample page content for Post creation."}],
                "metadata": file_meta,
                "processing_metadata": {"provider": "reducto"},
                "user_id": user_id,
                "workspace_id": workspace_id,
            },
            organization_id=organization_id,
            namespace_id=namespace_id,
        )

        # result is a dict from create_or_update_document_post
        assert isinstance(result, dict)
        assert result.get("post_id")

        # Cleanup: delete the created Post
        headers = {
            "X-Parse-Application-Id": app_id,
            "X-Parse-Master-Key": master_key,
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.delete(f"{parse_url}/parse/classes/Post/{result['post_id']}", headers=headers)
            # Some cloud code installs error on delete with master key (e.g., "req is not defined").
            # Treat cleanup as best-effort so the creation contract remains the assertion surface.
            if resp.status_code not in (200, 204):
                # Attempt soft cleanup (archive) to avoid leaving noise
                try:
                    await client.put(
                        f"{parse_url}/parse/classes/Post/{result['post_id']}",
                        headers=headers,
                        json={"archive": True}
                    )
                except Exception:
                    pass

    asyncio.run(run())


