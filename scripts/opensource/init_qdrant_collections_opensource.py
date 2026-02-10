#!/usr/bin/env python3
"""
Initialize Qdrant Collections for Open Source Edition

This script creates the required Qdrant collections on startup:
- Main collection: Variable dimensions based on embedding model
  - Local (Qwen3-Embedding-0.6B): 1024 dimensions
  - Cloud (Qwen3-Embedding-4B): 2560 dimensions
- Property collection: 384 dimensions for property vectors

Usage:
    python scripts/init_qdrant_collections_opensource.py
"""

import os
import sys
import asyncio
import argparse
from typing import Optional

try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Distance, VectorParams, ScalarQuantization, ScalarQuantizationConfig, ScalarType, HnswConfigDiff
except ImportError:
    print("❌ Error: qdrant-client not installed. Install with: pip install qdrant-client")
    sys.exit(1)


async def create_collection(
    client: AsyncQdrantClient,
    collection_name: str,
    vector_size: int,
    description: str
) -> bool:
    """Create a Qdrant collection with optimized settings."""
    try:
        # Check if collection already exists
        collections = await client.get_collections()
        existing_collections = [col.name for col in collections.collections]
        
        if collection_name in existing_collections:
            print(f"✅ Collection '{collection_name}' already exists")
            return True
        
        # Create collection with optimized settings
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
                # Enable quantization for faster search
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True
                    )
                ),
                # Optimize for speed
                on_disk=False,  # Keep vectors in RAM for faster access
                hnsw_config=HnswConfigDiff(
                    m=16,  # Number of connections per layer
                    ef_construct=100,  # Size of the dynamic candidate list
                    full_scan_threshold=10000,  # Threshold for full scan
                    max_indexing_threads=4  # Number of threads for indexing
                )
            )
        )
        
        print(f"✅ Created collection '{collection_name}' ({description})")
        return True
        
    except Exception as e:
        print(f"❌ Error creating collection '{collection_name}': {e}")
        return False


async def init_qdrant_collections(
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    main_collection: Optional[str] = None,
    property_collection: Optional[str] = None,
    create_both_property_collections: bool = True
) -> bool:
    """Initialize Qdrant collections."""
    
    # Get configuration from environment or arguments
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY", "")
    
    # Determine embedding dimensions based on local vs cloud embeddings
    use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
    if use_local_embeddings:
        # Local Qwen3-Embedding-0.6B produces 1024 dimensions
        embedding_dimensions = int(os.getenv("LOCAL_EMBEDDING_DIMENSIONS", "1024"))
        embedding_model_name = "Qwen0.6B (local)"
    else:
        # Cloud Qwen3-Embedding-4B produces 2560 dimensions
        embedding_dimensions = 2560
        embedding_model_name = "Qwen4B (cloud)"
    
    # Main collection (for memory embeddings) - auto-select based on dimensions
    if main_collection is None:
        if embedding_dimensions == 1024:
            main_collection = os.getenv("QDRANT_COLLECTION_QWEN0pt6B", "Qwen0pt6B")
        else:
            main_collection = os.getenv("QDRANT_COLLECTION_QWEN4B", "Qwen4B")
    
    # Property collection (for property vectors)
    property_collection = property_collection or os.getenv("QDRANT_PROPERTY_COLLECTION", "neo4j_properties_dev")
    
    print("=" * 60)
    print("🔧 Initializing Qdrant Collections")
    print("=" * 60)
    print(f"Qdrant URL: {qdrant_url}")
    print(f"Embedding Model: {embedding_model_name}")
    print(f"Main Collection: {main_collection} ({embedding_dimensions} dimensions)")
    print(f"Property Collection: {property_collection} (384 dimensions)")
    if create_both_property_collections:
        print(f"Also creating: neo4j_properties (384 dimensions)")
    print("=" * 60)
    print()
    
    # Initialize Qdrant client
    try:
        client = AsyncQdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key if qdrant_api_key else None,
            timeout=30.0
        )
        
        # Test connection
        await client.get_collections()
        print("✅ Connected to Qdrant")
        print()
        
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        return False
    
    # Create main collection with appropriate dimensions
    success_main = await create_collection(
        client=client,
        collection_name=main_collection,
        vector_size=embedding_dimensions,
        description=f"Memory embeddings ({embedding_model_name})"
    )
    
    # Create property collection (384 dimensions) - dev version
    success_property_dev = await create_collection(
        client=client,
        collection_name=property_collection,
        vector_size=384,
        description="Property vectors (sentence-bert) - dev"
    )
    
    # Also create production property collection if requested
    success_property_prod = True
    if create_both_property_collections and property_collection != "neo4j_properties":
        success_property_prod = await create_collection(
            client=client,
            collection_name="neo4j_properties",
            vector_size=384,
            description="Property vectors (sentence-bert) - production"
        )
    
    print()
    if success_main and success_property_dev and success_property_prod:
        print("=" * 60)
        print("✅ All Qdrant collections initialized successfully!")
        print("=" * 60)
        return True
    else:
        print("=" * 60)
        print("⚠️  Some collections failed to initialize")
        print("=" * 60)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Initialize Qdrant collections for open-source edition"
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Qdrant URL (default: from QDRANT_URL env var or http://localhost:6333)"
    )
    parser.add_argument(
        "--qdrant-api-key",
        default=None,
        help="Qdrant API key (default: from QDRANT_API_KEY env var)"
    )
    parser.add_argument(
        "--main-collection",
        default=None,
        help="Main collection name (default: from QDRANT_COLLECTION_QWEN4B env var or 'Qwen4B')"
    )
    parser.add_argument(
        "--property-collection",
        default=None,
        help="Property collection name (default: from QDRANT_PROPERTY_COLLECTION env var or 'neo4j_properties_dev')"
    )
    parser.add_argument(
        "--create-both-property-collections",
        action="store_true",
        default=True,
        help="Create both neo4j_properties and neo4j_properties_dev collections (default: True)"
    )
    parser.add_argument(
        "--no-create-both-property-collections",
        dest="create_both_property_collections",
        action="store_false",
        help="Only create the specified property collection"
    )
    
    args = parser.parse_args()
    
    try:
        success = asyncio.run(init_qdrant_collections(
            qdrant_url=args.qdrant_url,
            qdrant_api_key=args.qdrant_api_key,
            main_collection=args.main_collection,
            property_collection=args.property_collection,
            create_both_property_collections=args.create_both_property_collections
        ))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

