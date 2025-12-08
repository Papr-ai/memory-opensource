#!/usr/bin/env python3
"""
Direct Neo4j query to see what nodes actually exist
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

NEO4J_URL = os.getenv('NEO4J_URL')
NEO4J_SECRET = os.getenv('NEO4J_SECRET')
WORKSPACE_ID = os.getenv('TEST_WORKSPACE_ID', 'pohYfXWoOK')

print(f"Connecting to Neo4j: {NEO4J_URL}")
print(f"Workspace ID: {WORKSPACE_ID}")
print("=" * 80)

driver = GraphDatabase.driver(NEO4J_URL, auth=("neo4j", NEO4J_SECRET))

# Test user IDs from the test
test_users = [
    "security_test_user_001",  # schema_id approach
    "security_test_user_002",  # agentic approach
    "security_test_user_003",  # graph_override approach
]

with driver.session() as session:
    print("\n1. CHECKING FOR TEST USER NODES:")
    print("=" * 80)

    for user_id in test_users:
        result = session.run("""
            MATCH (n)
            WHERE n.user_id = $user_id
              AND n.workspace_id = $workspace_id
            RETURN count(n) as count,
                   collect(DISTINCT labels(n)[0]) as node_types
        """, user_id=user_id, workspace_id=WORKSPACE_ID)

        record = result.single()
        if record and record["count"] > 0:
            print(f"✅ {user_id}: {record['count']} nodes")
            print(f"   Node types: {record['node_types']}")
        else:
            print(f"❌ {user_id}: 0 nodes found")

    print("\n2. CHECKING ALL NODES WITH WORKSPACE_ID:")
    print("=" * 80)

    result = session.run("""
        MATCH (n)
        WHERE n.workspace_id = $workspace_id
        RETURN labels(n) as labels,
               n.user_id as user_id,
               n.id as node_id,
               n.name as name,
               n.content as content
        ORDER BY n.createdAt DESC
        LIMIT 20
    """, workspace_id=WORKSPACE_ID)

    count = 0
    for record in result:
        count += 1
        print(f"\nNode {count}:")
        print(f"  Labels: {record['labels']}")
        print(f"  User ID: {record['user_id']}")
        print(f"  Node ID: {record['node_id']}")
        print(f"  Name: {record['name']}")
        print(f"  Content: {record['content'][:100] if record['content'] else None}...")

    if count == 0:
        print("❌ No nodes found with this workspace_id")

    print("\n3. CHECKING ALL NODES (NO FILTERS):")
    print("=" * 80)

    result = session.run("""
        MATCH (n)
        RETURN labels(n) as labels,
               n.user_id as user_id,
               n.workspace_id as workspace_id,
               n.id as node_id
        ORDER BY n.createdAt DESC
        LIMIT 10
    """)

    count = 0
    for record in result:
        count += 1
        print(f"\nNode {count}:")
        print(f"  Labels: {record['labels']}")
        print(f"  User ID: {record['user_id']}")
        print(f"  Workspace ID: {record['workspace_id']}")
        print(f"  Node ID: {record['node_id']}")

    if count == 0:
        print("❌ No nodes found at all in Neo4j!")

    print("\n4. CHECKING FOR MEMORY NODES:")
    print("=" * 80)

    result = session.run("""
        MATCH (n:Memory)
        RETURN n.id as memory_id,
               n.user_id as user_id,
               n.workspace_id as workspace_id,
               n.externalUserId as external_user_id
        ORDER BY n.createdAt DESC
        LIMIT 10
    """)

    count = 0
    for record in result:
        count += 1
        print(f"\nMemory {count}:")
        print(f"  Memory ID: {record['memory_id']}")
        print(f"  User ID: {record['user_id']}")
        print(f"  Workspace ID: {record['workspace_id']}")
        print(f"  External User ID: {record['external_user_id']}")

    print(f"\n\nTotal Memory nodes: {count}")

driver.close()
print("\n" + "=" * 80)
print("Done!")
