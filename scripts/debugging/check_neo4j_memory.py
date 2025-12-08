#!/usr/bin/env python3
"""
Quick script to check Neo4j for Memory nodes
"""
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables conditionally
use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
if use_dotenv:
    load_dotenv()

def check_memory_nodes():
    # Get Neo4j connection details
    neo4j_url = os.getenv('NEO4J_URL')
    neo4j_secret = os.getenv('NEO4J_SECRET')
    
    if not neo4j_url or not neo4j_secret:
        print("❌ NEO4J_URL or NEO4J_SECRET not found in environment")
        return
    
    print(f"🔗 Connecting to Neo4j: {neo4j_url}")
    
    try:
        # Create driver (bolt+s:// scheme already includes encryption)
        driver = GraphDatabase.driver(neo4j_url, auth=("neo4j", neo4j_secret))
        
        with driver.session() as session:
            print("\n📊 CHECKING NEO4J DATABASE")
            print("=" * 50)
            
            # 1. Check all node labels
            print("\n1️⃣ ALL NODE LABELS:")
            result = session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")
            labels = [record["label"] for record in result]
            for label in labels:
                print(f"   • {label}")
            
            # 2. Check Memory nodes count
            print(f"\n2️⃣ MEMORY NODES COUNT:")
            result = session.run("MATCH (m:Memory) RETURN count(m) as count")
            memory_count = result.single()["count"]
            print(f"   • Total Memory nodes: {memory_count}")
            
            # 3. Check recent Memory nodes
            print(f"\n3️⃣ RECENT MEMORY NODES (last 24 hours):")
            result = session.run("""
                MATCH (m:Memory) 
                WHERE m.createdAt >= datetime() - duration('P1D')
                RETURN m.content, m.id, m.createdAt
                ORDER BY m.createdAt DESC
                LIMIT 10
            """)
            
            for record in result:
                content_preview = record["content"][:100] + "..." if len(record["content"]) > 100 else record["content"]
                print(f"   • ID: {record['id']}")
                print(f"     Content: {content_preview}")
                print(f"     Created: {record['createdAt']}")
                print()
            
            # 4. Check Developer nodes
            print(f"\n4️⃣ DEVELOPER NODES:")
            result = session.run("MATCH (d:Developer) RETURN d.name, d.email, d.id ORDER BY d.createdAt DESC LIMIT 10")
            for record in result:
                print(f"   • {record['d.name']} ({record['d.email']}) - ID: {record['d.id']}")
            
            # 5. Check CodeSnippet nodes
            print(f"\n5️⃣ CODESNIPPET NODES:")
            result = session.run("MATCH (c:CodeSnippet) RETURN c.title, c.language, c.id ORDER BY c.createdAt DESC LIMIT 10")
            for record in result:
                print(f"   • {record['c.title']} ({record['c.language']}) - ID: {record['c.id']}")
            
            # 6. Check Memory-Developer connections
            print(f"\n6️⃣ MEMORY-DEVELOPER CONNECTIONS:")
            result = session.run("""
                MATCH (m:Memory)-[r]-(d:Developer) 
                RETURN m.content, d.name, type(r) as relationship_type
                LIMIT 5
            """)
            
            connections = list(result)
            if connections:
                for record in connections:
                    content_preview = record["m.content"][:80] + "..." if len(record["m.content"]) > 80 else record["m.content"]
                    print(f"   • Memory: {content_preview}")
                    print(f"     Connected to Developer: {record['d.name']}")
                    print(f"     Relationship: {record['relationship_type']}")
                    print()
            else:
                print("   • No Memory-Developer connections found")
            
            # 7. Check for Sarah Kim or Mike Chen specifically
            print(f"\n7️⃣ SEARCHING FOR SARAH KIM & MIKE CHEN:")
            result = session.run("""
                MATCH (n)
                WHERE n.content CONTAINS "Sarah Kim" OR n.content CONTAINS "Mike Chen" 
                   OR n.name = "Sarah Kim" OR n.name = "Mike Chen"
                RETURN labels(n) as node_labels, n.name, n.content, n.id
                LIMIT 10
            """)
            
            for record in result:
                labels_str = ", ".join(record["node_labels"])
                content = record["n.content"] or "N/A"
                content_preview = content[:100] + "..." if len(content) > 100 else content
                print(f"   • Labels: [{labels_str}]")
                print(f"     Name: {record['n.name'] or 'N/A'}")
                print(f"     Content: {content_preview}")
                print(f"     ID: {record['n.id']}")
                print()
        
        driver.close()
        print("✅ Neo4j check completed successfully!")
        
    except Exception as e:
        print(f"❌ Error connecting to Neo4j: {e}")

if __name__ == "__main__":
    check_memory_nodes()
