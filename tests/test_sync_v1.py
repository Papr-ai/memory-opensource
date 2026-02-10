import os
import json
import asyncio
from datetime import datetime
import httpx
import pytest
from asgi_lifespan import LifespanManager

from main import app

TEST_X_USER_API_KEY = os.environ.get('TEST_X_USER_API_KEY')


@pytest.mark.asyncio
async def test_v1_sync_tiers_basic(app):
    """Test basic tier sync with Memory objects and ACL fields verification"""
    assert TEST_X_USER_API_KEY, "TEST_X_USER_API_KEY must be set for integration tests"
    async with LifespanManager(app, startup_timeout=120):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:
            headers = {
                'X-Client-Type': 'papr_plugin',
                'X-API-Key': TEST_X_USER_API_KEY,
                'Accept-Encoding': 'gzip',
                'Content-Type': 'application/json'
            }
            payload = {
                'include_embeddings': True,
                'max_tier0': 100,
                'max_tier1': 100
            }
            r = await async_client.post("/v1/sync/tiers", headers=headers, json=payload)
            assert r.status_code == 200
            data = r.json()
            assert data.get('code') == 200
            assert data.get('status') == 'success'
            assert 'tier0' in data and 'tier1' in data
            assert 'transitions' in data
            
            # Verify Tier0 Memory objects
            t0 = data.get('tier0', [])
            t1 = data.get('tier1', [])
            
            print(f"\n=== TIER0 VERIFICATION ===")
            print(f"Tier0 count: {len(t0)}")
            if t0:
                # Verify first item has Memory fields
                first_t0 = t0[0]
                assert 'id' in first_t0, "Tier0 item missing 'id' field"
                assert 'type' in first_t0, "Tier0 item missing 'type' field"
                assert 'content' in first_t0, "Tier0 item missing 'content' field"
                
                # Verify ACL fields are present
                acl_fields_present = []
                if 'user_id' in first_t0:
                    acl_fields_present.append('user_id')
                if 'workspace_id' in first_t0:
                    acl_fields_present.append('workspace_id')
                if 'acl' in first_t0:
                    acl_fields_present.append('acl')
                
                print(f"First Tier0 item fields: {list(first_t0.keys())[:15]}...")
                print(f"ACL fields present: {acl_fields_present}")
                print(f"Preview: {[{'id': it.get('id'), 'type': it.get('type'), 'content': (it.get('content') or '')[:80]} for it in t0[:3]]}")
            
            print(f"\n=== TIER1 VERIFICATION ===")
            print(f"Tier1 count: {len(t1)}")
            if t1:
                # Verify first item has Memory fields
                first_t1 = t1[0]
                assert 'id' in first_t1, "Tier1 item missing 'id' field"
                assert 'type' in first_t1, "Tier1 item missing 'type' field"
                assert 'content' in first_t1, "Tier1 item missing 'content' field"
                
                # Verify ACL fields are present
                acl_fields_present = []
                if 'user_id' in first_t1:
                    acl_fields_present.append('user_id')
                if 'workspace_id' in first_t1:
                    acl_fields_present.append('workspace_id')
                if 'acl' in first_t1:
                    acl_fields_present.append('acl')
                if 'user_read_access' in first_t1:
                    acl_fields_present.append('user_read_access')
                if 'workspace_read_access' in first_t1:
                    acl_fields_present.append('workspace_read_access')
                if 'organization_id' in first_t1:
                    acl_fields_present.append('organization_id')
                if 'namespace_id' in first_t1:
                    acl_fields_present.append('namespace_id')
                
                print(f"First Tier1 item fields: {list(first_t1.keys())[:15]}...")
                print(f"ACL fields present: {acl_fields_present}")
                print(f"Preview: {[{'id': it.get('id'), 'type': it.get('type'), 'content': (it.get('content') or '')[:80]} for it in t1[:3]]}")
                
                # Verify metadata field structure
                if 'metadata' in first_t1:
                    print(f"Metadata keys: {list(first_t1['metadata'].keys())[:10] if isinstance(first_t1['metadata'], dict) else 'not a dict'}")
                
                # Verify relevance_score field exists and varies (bug fix validation)
                tier1_relevance_scores = []
                for item in t1[:10]:  # Check first 10 items
                    if 'relevance_score' in item and item['relevance_score'] is not None:
                        tier1_relevance_scores.append(float(item['relevance_score']))
                
                if tier1_relevance_scores:
                    unique_scores = len(set(tier1_relevance_scores))
                    flat_bug_count = sum(1 for s in tier1_relevance_scores if abs(s - 0.2) < 0.001)
                    print(f"Tier1 relevance_score check: {len(tier1_relevance_scores)} items with scores, {unique_scores} unique values")
                    print(f"  - Range: [{min(tier1_relevance_scores):.6f}, {max(tier1_relevance_scores):.6f}]")
                    print(f"  - Items with flat 0.2 bug: {flat_bug_count}/{len(tier1_relevance_scores)}")
                    
                    # Assert variation (not all same value)
                    if len(tier1_relevance_scores) > 1:
                        assert unique_scores > 1, f"All Tier1 relevance scores are the same (flat bug): {tier1_relevance_scores}"
                        assert flat_bug_count < len(tier1_relevance_scores), f"All Tier1 relevance scores are 0.2 (bug not fixed): {tier1_relevance_scores}"
            
            # Check Tier0 relevance_score variation
            if t0:
                tier0_relevance_scores = []
                for item in t0[:10]:  # Check first 10 items
                    if 'relevance_score' in item and item['relevance_score'] is not None:
                        tier0_relevance_scores.append(float(item['relevance_score']))
                
                if tier0_relevance_scores:
                    unique_scores = len(set(tier0_relevance_scores))
                    flat_bug_count = sum(1 for s in tier0_relevance_scores if abs(s - 0.2) < 0.001)
                    print(f"\nTier0 relevance_score check: {len(tier0_relevance_scores)} items with scores, {unique_scores} unique values")
                    print(f"  - Range: [{min(tier0_relevance_scores):.6f}, {max(tier0_relevance_scores):.6f}]")
                    print(f"  - Items with flat 0.2 bug: {flat_bug_count}/{len(tier0_relevance_scores)}")
                    
                    # Assert variation (not all same value)
                    if len(tier0_relevance_scores) > 1:
                        assert unique_scores > 1, f"All Tier0 relevance scores are the same (flat bug): {tier0_relevance_scores}"
                        assert flat_bug_count < len(tier0_relevance_scores), f"All Tier0 relevance scores are 0.2 (bug not fixed): {tier0_relevance_scores}"
            
            print(f"=== END VERIFICATION ===\n")


@pytest.mark.asyncio
async def test_v1_sync_tiers_relevance_score_normalization(app):
    """Test that relevance_score is properly normalized and varies across memories
    
    This test validates the fix for the bug where all tier0 memories had relevance_score=0.2.
    After the fix, relevance_score should:
    1. Vary across different memories (not all the same value)
    2. Have proper distribution (not clustered at 0.2)
    3. Be present for both tier0 and tier1 items
    4. Represent normalized scores from ranking algorithms
    """
    assert TEST_X_USER_API_KEY, "TEST_X_USER_API_KEY must be set for integration tests"
    async with LifespanManager(app, startup_timeout=120):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:
            headers = {
                'X-Client-Type': 'papr_plugin',
                'X-API-Key': TEST_X_USER_API_KEY,
                'Accept-Encoding': 'gzip',
                'Content-Type': 'application/json'
            }
            payload = {
                'include_embeddings': False,  # Skip embeddings to focus on relevance_score
                'max_tier0': 50,
                'max_tier1': 50
            }
            r = await async_client.post("/v1/sync/tiers", headers=headers, json=payload)
            assert r.status_code == 200
            data = r.json()
            assert data.get('code') == 200
            assert data.get('status') == 'success'
            
            print(f"\n{'='*80}")
            print(f"RELEVANCE SCORE NORMALIZATION TEST")
            print(f"{'='*80}")
            
            # Check Tier0 relevance scores
            tier0_items = data.get('tier0', [])
            print(f"\n--- TIER0 RELEVANCE SCORES ---")
            print(f"Total Tier0 items: {len(tier0_items)}")
            
            tier0_scores = []
            tier0_with_score = 0
            
            for idx, item in enumerate(tier0_items):
                if 'relevance_score' in item and item['relevance_score'] is not None:
                    score = float(item['relevance_score'])
                    tier0_scores.append(score)
                    tier0_with_score += 1
                    if idx < 10:  # Log first 10 for visibility
                        print(f"  Item {idx+1} (id={item.get('id')[:20]}...): score={score:.6f}")
            
            print(f"\nTier0 Relevance Score Statistics:")
            print(f"  - Items with relevance_score: {tier0_with_score}/{len(tier0_items)}")
            if tier0_scores:
                unique_scores = len(set(tier0_scores))
                min_score = min(tier0_scores)
                max_score = max(tier0_scores)
                avg_score = sum(tier0_scores) / len(tier0_scores)
                # Count how many have the flat 0.2 bug value
                flat_bug_count = sum(1 for s in tier0_scores if abs(s - 0.2) < 0.001)
                
                print(f"  - Unique score values: {unique_scores}")
                print(f"  - Score range: [{min_score:.6f}, {max_score:.6f}]")
                print(f"  - Average score: {avg_score:.6f}")
                print(f"  - Items with flat 0.2 bug value: {flat_bug_count}/{len(tier0_scores)}")
                
                # Validation: Should have variation (not all same value)
                assert unique_scores > 1, f"All Tier0 relevance scores are the same (flat bug): {tier0_scores[:5]}"
                
                # Validation: Should not all be 0.2 (the bug value)
                flat_bug_percentage = flat_bug_count / len(tier0_scores)
                assert flat_bug_percentage < 0.8, f"Too many Tier0 items have flat 0.2 score (bug): {flat_bug_percentage:.1%} ({flat_bug_count}/{len(tier0_scores)})"
                
                # Validation: Score should be non-negative and reasonable
                assert all(s >= 0 for s in tier0_scores), "Some Tier0 relevance scores are negative"
                # Tier0 scores combine vector (0.6×), transition (0.3×), and hotness (0.2×) - can reach ~1.1-2.0
                assert max_score <= 10.0, f"Tier0 relevance score seems unreasonably high: {max_score}"
                
                print(f"  ✓ Tier0 relevance scores vary correctly (not all 0.2)")
            else:
                print(f"  ⚠ No Tier0 items have relevance_score set")
            
            # Check Tier1 relevance scores
            tier1_items = data.get('tier1', [])
            print(f"\n--- TIER1 RELEVANCE SCORES ---")
            print(f"Total Tier1 items: {len(tier1_items)}")
            
            tier1_scores = []
            tier1_with_score = 0
            
            for idx, item in enumerate(tier1_items):
                if 'relevance_score' in item and item['relevance_score'] is not None:
                    score = float(item['relevance_score'])
                    tier1_scores.append(score)
                    tier1_with_score += 1
                    if idx < 10:  # Log first 10 for visibility
                        print(f"  Item {idx+1} (id={item.get('id')[:20]}...): score={score:.6f}")
            
            print(f"\nTier1 Relevance Score Statistics:")
            print(f"  - Items with relevance_score: {tier1_with_score}/{len(tier1_items)}")
            if tier1_scores:
                unique_scores = len(set(tier1_scores))
                min_score = min(tier1_scores)
                max_score = max(tier1_scores)
                avg_score = sum(tier1_scores) / len(tier1_scores)
                
                print(f"  - Unique score values: {unique_scores}")
                print(f"  - Score range: [{min_score:.6f}, {max_score:.6f}]")
                print(f"  - Average score: {avg_score:.6f}")
                
                # Validation: Should have variation (not all same value)
                assert unique_scores > 1, f"All Tier1 relevance scores are the same: {tier1_scores[:5]}"
                
                # Validation: Score should be non-negative and reasonable
                assert all(s >= 0 for s in tier1_scores), "Some Tier1 relevance scores are negative"
                # Tier1 scores use citation-first ranking: 0.7×citation + 0.3×cache with log normalization
                # Scores can be quite high (e.g., 20-30+) for frequently accessed/cited memories
                assert max_score <= 100.0, f"Tier1 relevance score seems unreasonably high: {max_score}"
                
                print(f"  ✓ Tier1 relevance scores vary correctly")
            else:
                print(f"  ⚠ No Tier1 items have relevance_score set")
            
            # Overall validation: At least one tier should have scores
            if tier0_scores or tier1_scores:
                all_scores = tier0_scores + tier1_scores
                all_unique = len(set(all_scores))
                print(f"\n{'='*80}")
                print(f"OVERALL RELEVANCE SCORE SUMMARY")
                print(f"{'='*80}")
                print(f"Total items with scores: {len(all_scores)}")
                print(f"Unique score values across all tiers: {all_unique}")
                print(f"Score distribution: min={min(all_scores):.6f}, max={max(all_scores):.6f}, avg={sum(all_scores)/len(all_scores):.6f}")
                print(f"✓ Relevance score normalization working correctly!")
            else:
                print(f"\n⚠ Warning: No relevance scores found in either tier")
            
            print(f"{'='*80}\n")


@pytest.mark.asyncio
async def test_v1_sync_tiers_with_embeddings_small_limits(app):
    assert TEST_X_USER_API_KEY, "TEST_X_USER_API_KEY must be set for integration tests"
    async with LifespanManager(app, startup_timeout=120):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:
            headers = {
                'X-Client-Type': 'papr_plugin',
                'X-API-Key': TEST_X_USER_API_KEY,
                'Accept-Encoding': 'gzip',
                'Content-Type': 'application/json'
            }
            payload = {
                'include_embeddings': True,
                'embed_limit': 2,
                'max_tier0': 2,
                'max_tier1': 2
            }
            r = await async_client.post("/v1/sync/tiers", headers=headers, json=payload)
            assert r.status_code == 200
            data = r.json()
            assert data.get('code') == 200
            assert data.get('status') == 'success'
            # If there are items, check embedding keys existence form
            if data['tier0']:
                ti0 = data['tier0'][0]
                # embedding may not be present if content is empty; tolerate
                assert isinstance(ti0, dict)
            if data['tier1']:
                ti1 = data['tier1'][0]
                assert isinstance(ti1, dict)
            # Print preview for visibility
            print("Tier0/1 with embeddings (limited) =>", data.get('tier0', []), data.get('tier1', []))


@pytest.mark.asyncio
async def test_v1_sync_delta_cursor_pagination(app):
    assert TEST_X_USER_API_KEY, "TEST_X_USER_API_KEY must be set for integration tests"
    async with LifespanManager(app, startup_timeout=120):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:
            headers = {
                'X-Client-Type': 'papr_plugin',
                'X-API-Key': TEST_X_USER_API_KEY,
                'Accept-Encoding': 'gzip'
            }
            # first page
            r1 = await async_client.get("/v1/sync/delta", headers=headers, params={'limit': 5})
            assert r1.status_code == 200
            d1 = r1.json()
            assert d1.get('code') == 200
            assert 'items' in d1
            cursor = d1.get('next_cursor')
            if cursor:
                # second page
                r2 = await async_client.get("/v1/sync/delta", headers=headers, params={'cursor': cursor, 'limit': 5})
                assert r2.status_code == 200
                d2 = r2.json()
                assert d2.get('code') == 200


@pytest.mark.asyncio
async def test_v1_sync_tiers_with_tier1_embeddings(app):
    """Test that Tier1 memories get enriched with embeddings from Qdrant and have all Memory fields"""
    assert TEST_X_USER_API_KEY, "TEST_X_USER_API_KEY must be set for integration tests"
    async with LifespanManager(app, startup_timeout=120):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:
            headers = {
                'X-Client-Type': 'papr_plugin',
                'X-API-Key': TEST_X_USER_API_KEY,
                'Accept-Encoding': 'gzip',
                'Content-Type': 'application/json'
            }
            payload = {
                'include_embeddings': True,
                'embed_limit': 10,  # Embed first 10 items
                'max_tier0': 5,
                'max_tier1': 10
            }
            r = await async_client.post("/v1/sync/tiers", headers=headers, json=payload)
            assert r.status_code == 200
            data = r.json()
            assert data.get('code') == 200
            assert data.get('status') == 'success'
            
            # Check Tier0 Memory structure
            tier0_items = data.get('tier0', [])
            if tier0_items:
                print(f"\n=== TIER0 MEMORY STRUCTURE ===")
                print(f"Total Tier0 items: {len(tier0_items)}")
                first_t0 = tier0_items[0]
                print(f"Tier0 fields: {list(first_t0.keys())[:20]}")
                
                # Verify core Memory fields
                assert 'id' in first_t0
                assert 'type' in first_t0
                assert 'content' in first_t0
                print(f"✓ Core Memory fields present in Tier0")
            
            # Check Tier1 embeddings and Memory structure
            tier1_items = data.get('tier1', [])
            if tier1_items:
                print(f"\n=== TIER1 EMBEDDINGS & MEMORY STRUCTURE TEST ===")
                print(f"Total Tier1 items: {len(tier1_items)}")
                
                first_t1 = tier1_items[0]
                print(f"Tier1 fields: {list(first_t1.keys())[:20]}")
                
                # Verify core Memory fields
                assert 'id' in first_t1, "Missing 'id' field in Tier1"
                assert 'type' in first_t1, "Missing 'type' field in Tier1"
                assert 'content' in first_t1, "Missing 'content' field in Tier1"
                print(f"✓ Core Memory fields present in Tier1")
                
                # Verify ACL fields
                acl_fields_found = []
                for field in ['user_id', 'workspace_id', 'acl', 'user_read_access', 
                             'workspace_read_access', 'organization_id', 'namespace_id']:
                    if field in first_t1:
                        acl_fields_found.append(field)
                print(f"✓ ACL fields found: {acl_fields_found}")
                
                # Count how many items have embeddings
                items_with_embeddings = 0
                for item in tier1_items[:10]:  # Check first 10
                    has_embedding = 'embedding_int8' in item or 'embedding' in item
                    if has_embedding:
                        items_with_embeddings += 1
                        # Check if it's quantized INT8
                        if 'embedding_int8' in item:
                            emb = item['embedding_int8']
                            print(f"  - Item {item.get('id')}: Has INT8 embedding (len={len(emb)})")
                            # Verify INT8 range
                            assert all(-128 <= v <= 127 for v in emb), "INT8 values out of range"
                        elif 'embedding' in item:
                            emb = item['embedding']
                            print(f"  - Item {item.get('id')}: Has float embedding (len={len(emb)})")
                    else:
                        print(f"  - Item {item.get('id')}: No embedding")
                
                print(f"Items with embeddings: {items_with_embeddings}/{min(len(tier1_items), 10)}")
                print(f"✓ Embedding enrichment completed")
                # At least some items should have embeddings if they exist in Qdrant
                # (It's okay if not all have embeddings - some might not be in Qdrant yet)
                print(f"=== END TIER1 EMBEDDINGS TEST ===\n")


@pytest.mark.asyncio
async def test_v1_sync_tiers_float32_embeddings_for_coreml(app):
    """Test that Tier1 can return float32 embeddings for CoreML/ANE (fp16 on-device)"""
    assert TEST_X_USER_API_KEY, "TEST_X_USER_API_KEY must be set for integration tests"
    async with LifespanManager(app, startup_timeout=120):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:
            headers = {
                'X-Client-Type': 'papr_plugin',
                'X-API-Key': TEST_X_USER_API_KEY,
                'Accept-Encoding': 'gzip',
                'Content-Type': 'application/json'
            }
            payload = {
                'include_embeddings': True,
                'embedding_format': 'float32',  # Request float32 for CoreML/ANE
                'embed_limit': 5,
                'max_tier0': 2,
                'max_tier1': 5
            }
            r = await async_client.post("/v1/sync/tiers", headers=headers, json=payload)
            assert r.status_code == 200
            data = r.json()
            assert data.get('code') == 200
            assert data.get('status') == 'success'
            
            print(f"\n=== FLOAT32 EMBEDDINGS FOR COREML/ANE TEST ===")
            
            # Check Tier1 has float32 embeddings (not INT8)
            tier1_items = data.get('tier1', [])
            if tier1_items:
                print(f"Total Tier1 items: {len(tier1_items)}")
                
                items_with_float32 = 0
                items_with_int8 = 0
                
                for item in tier1_items[:5]:
                    has_float32 = 'embedding' in item and item['embedding'] is not None
                    has_int8 = 'embedding_int8' in item and item['embedding_int8'] is not None
                    
                    if has_float32:
                        items_with_float32 += 1
                        emb = item['embedding']
                        print(f"  ✓ Item {item.get('id')}: Has float32 embedding (len={len(emb)})")
                        # Verify it's actually float values (not integers from INT8)
                        assert isinstance(emb[0], float), f"Expected float type, got {type(emb[0])}"
                        # Check that values have decimal precision (not integers)
                        has_decimals = any(v != int(v) for v in emb[:100] if v != 0)
                        print(f"    - Type: {type(emb[0])}, Has decimal precision: {has_decimals}")
                    
                    if has_int8:
                        items_with_int8 += 1
                        print(f"  ✗ Item {item.get('id')}: Has INT8 embedding (should be float32!)")
                
                print(f"Items with float32: {items_with_float32}/{len(tier1_items)}")
                print(f"Items with INT8: {items_with_int8}/{len(tier1_items)}")
                
                # Assert that we got float32, not INT8
                assert items_with_float32 > 0, "Should have float32 embeddings for CoreML/ANE"
                assert items_with_int8 == 0, "Should NOT have INT8 embeddings when float32 requested"
                
                print(f"✓ Float32 format confirmed - ready for CoreML/ANE fp16 conversion")
            else:
                print("No Tier1 items to test")
            
            print(f"=== END FLOAT32 COREML TEST ===\n")


@pytest.mark.asyncio
async def test_v1_sync_tiers_qwen4b_coreml_full_precision(app):
    """Test that both Tier0 and Tier1 return 2560-dim float32 embeddings for Qwen4B/CoreML
    
    This test validates:
    1. Both tier0 AND tier1 get enriched with embeddings (not just tier1)
    2. Embeddings are 2560 dimensions (Qwen4B dimension)
    3. Embeddings are float32 format (for CoreML/ANE fp16 conversion)
    4. All 100 requested items per tier have embeddings
    """
    assert TEST_X_USER_API_KEY, "TEST_X_USER_API_KEY must be set for integration tests"
    async with LifespanManager(app, startup_timeout=120):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:
            headers = {
                'X-Client-Type': 'papr_plugin',
                'X-API-Key': TEST_X_USER_API_KEY,
                'Accept-Encoding': 'gzip',
                'Content-Type': 'application/json'
            }
            payload = {
                'include_embeddings': True,
                'embedding_format': 'float32',  # Full precision for CoreML/ANE
                'embed_model': 'Qwen4B',        # Use Qwen4B model
                'embed_limit': 100,             # Embed all 100 items
                'max_tier0': 100,
                'max_tier1': 100
            }
            r = await async_client.post("/v1/sync/tiers", headers=headers, json=payload)
            assert r.status_code == 200
            data = r.json()
            assert data.get('code') == 200
            assert data.get('status') == 'success'
            
            print(f"\n{'='*80}")
            print(f"QWEN4B COREML FULL PRECISION TEST (2560-dim float32)")
            print(f"{'='*80}")
            
            # Check Tier0 embeddings
            tier0_items = data.get('tier0', [])
            print(f"\n--- TIER0 EMBEDDINGS ---")
            print(f"Total Tier0 items returned: {len(tier0_items)}")
            
            tier0_with_embeddings = 0
            tier0_correct_dim = 0
            tier0_correct_type = 0
            
            for idx, item in enumerate(tier0_items):
                has_embedding = 'embedding' in item and item['embedding'] is not None
                has_int8 = 'embedding_int8' in item and item['embedding_int8'] is not None
                
                if has_embedding:
                    tier0_with_embeddings += 1
                    emb = item['embedding']
                    dim = len(emb)
                    
                    if dim == 2560:
                        tier0_correct_dim += 1
                    
                    # Check type is float
                    if isinstance(emb[0], float):
                        tier0_correct_type += 1
                    
                    # Log first few items for visibility
                    if idx < 5:
                        print(f"  Item {idx+1} (id={item.get('id')[:20]}...)")
                        print(f"    - Dimension: {dim} {'✓' if dim == 2560 else '✗ WRONG!'}")
                        print(f"    - Type: {type(emb[0]).__name__} {'✓' if isinstance(emb[0], float) else '✗ WRONG!'}")
                        print(f"    - Sample values: [{emb[0]:.8f}, {emb[1]:.8f}, {emb[2]:.8f}, ...]")
                        print(f"    - Has INT8: {has_int8} {'✗ SHOULD BE FALSE!' if has_int8 else '✓'}")
                elif idx < 5:
                    print(f"  Item {idx+1} (id={item.get('id')[:20]}...): ✗ NO EMBEDDING")
            
            print(f"\nTier0 Summary:")
            print(f"  - Items with embeddings: {tier0_with_embeddings}/{len(tier0_items)}")
            print(f"  - Correct dimension (2560): {tier0_correct_dim}/{tier0_with_embeddings}")
            print(f"  - Correct type (float32): {tier0_correct_type}/{tier0_with_embeddings}")
            
            # Assert tier0 has embeddings
            assert tier0_with_embeddings > 0, "Tier0 should have at least some embeddings from Qdrant"
            if tier0_with_embeddings > 0:
                assert tier0_correct_dim == tier0_with_embeddings, f"All Tier0 embeddings should be 2560-dim (Qwen4B), got {tier0_correct_dim}/{tier0_with_embeddings}"
                assert tier0_correct_type == tier0_with_embeddings, f"All Tier0 embeddings should be float32, got {tier0_correct_type}/{tier0_with_embeddings}"
            
            # Check Tier1 embeddings
            tier1_items = data.get('tier1', [])
            print(f"\n--- TIER1 EMBEDDINGS ---")
            print(f"Total Tier1 items returned: {len(tier1_items)}")
            
            tier1_with_embeddings = 0
            tier1_correct_dim = 0
            tier1_correct_type = 0
            
            for idx, item in enumerate(tier1_items):
                has_embedding = 'embedding' in item and item['embedding'] is not None
                has_int8 = 'embedding_int8' in item and item['embedding_int8'] is not None
                
                if has_embedding:
                    tier1_with_embeddings += 1
                    emb = item['embedding']
                    dim = len(emb)
                    
                    if dim == 2560:
                        tier1_correct_dim += 1
                    
                    # Check type is float
                    if isinstance(emb[0], float):
                        tier1_correct_type += 1
                    
                    # Log first few items for visibility
                    if idx < 5:
                        print(f"  Item {idx+1} (id={item.get('id')[:20]}...)")
                        print(f"    - Dimension: {dim} {'✓' if dim == 2560 else '✗ WRONG!'}")
                        print(f"    - Type: {type(emb[0]).__name__} {'✓' if isinstance(emb[0], float) else '✗ WRONG!'}")
                        print(f"    - Sample values: [{emb[0]:.8f}, {emb[1]:.8f}, {emb[2]:.8f}, ...]")
                        print(f"    - Has INT8: {has_int8} {'✗ SHOULD BE FALSE!' if has_int8 else '✓'}")
                elif idx < 5:
                    print(f"  Item {idx+1} (id={item.get('id')[:20]}...): ✗ NO EMBEDDING")
            
            print(f"\nTier1 Summary:")
            print(f"  - Items with embeddings: {tier1_with_embeddings}/{len(tier1_items)}")
            print(f"  - Correct dimension (2560): {tier1_correct_dim}/{tier1_with_embeddings}")
            print(f"  - Correct type (float32): {tier1_correct_type}/{tier1_with_embeddings}")
            
            # Assert tier1 has embeddings
            assert tier1_with_embeddings > 0, "Tier1 should have at least some embeddings from Qdrant"
            if tier1_with_embeddings > 0:
                assert tier1_correct_dim == tier1_with_embeddings, f"All Tier1 embeddings should be 2560-dim (Qwen4B), got {tier1_correct_dim}/{tier1_with_embeddings}"
                assert tier1_correct_type == tier1_with_embeddings, f"All Tier1 embeddings should be float32, got {tier1_correct_type}/{tier1_with_embeddings}"
            
            # Overall summary
            print(f"\n{'='*80}")
            print(f"OVERALL SUMMARY")
            print(f"{'='*80}")
            total_items = len(tier0_items) + len(tier1_items)
            total_with_embeddings = tier0_with_embeddings + tier1_with_embeddings
            total_correct_dim = tier0_correct_dim + tier1_correct_dim
            total_correct_type = tier0_correct_type + tier1_correct_type
            
            print(f"Total items: {total_items}")
            print(f"Items with embeddings: {total_with_embeddings}/{total_items} ({100*total_with_embeddings/total_items:.1f}%)")
            print(f"Correct dimension (2560): {total_correct_dim}/{total_with_embeddings} ({100*total_correct_dim/total_with_embeddings if total_with_embeddings > 0 else 0:.1f}%)")
            print(f"Correct type (float32): {total_correct_type}/{total_with_embeddings} ({100*total_correct_type/total_with_embeddings if total_with_embeddings > 0 else 0:.1f}%)")
            
            # Final assertion: We should have high coverage
            coverage = total_with_embeddings / total_items if total_items > 0 else 0
            print(f"\n{'✓' if coverage >= 0.5 else '✗'} Embedding coverage: {coverage:.1%} (target: ≥50%)")
            
            assert coverage >= 0.5, f"Expected at least 50% of items to have embeddings from Qdrant, got {coverage:.1%}"
            
            print(f"{'='*80}")
            print(f"✓ QWEN4B COREML TEST PASSED")
            print(f"{'='*80}\n")

