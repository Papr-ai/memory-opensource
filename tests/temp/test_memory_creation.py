#!/usr/bin/env python3
"""
Test script to verify memory creation from messages
"""
import asyncio
import httpx
import time

BASE_URL = "http://localhost:8000"
API_KEY = "f80c5a2940f21882420b41690522cb2c"
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

async def test_memory_creation():
    """Test that memories are actually created from messages"""
    session_id = f"memory_creation_test_{int(time.time())}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        print(f"🧪 Testing memory creation with session: {session_id}")
        
        # Send 15 meaningful messages to trigger processing
        messages = [
            "I prefer using Python for data analysis because it has excellent libraries like pandas and numpy.",
            "My goal is to become a machine learning engineer within the next 2 years.",
            "I learned that React hooks make state management much easier than class components.",
            "The task I'm working on involves building a recommendation system for e-commerce.",
            "I have experience with PostgreSQL and MongoDB for database management.",
            "My preference is to work remotely 3 days a week and in-office 2 days.",
            "I need to improve my skills in deep learning, specifically with PyTorch.",
            "The context of our project is building a fintech application for small businesses.",
            "I discovered that microservices architecture works better for our scaling needs.",
            "My learning goal is to master Kubernetes deployment and orchestration.",
            "I prefer using TypeScript over JavaScript for better type safety.",
            "The fact is that our current system handles 10,000 requests per minute.",
            "I have a task to optimize our database queries to reduce response time by 50%.",
            "My goal is to implement automated testing that covers 90% of our codebase.",
            "I learned that caching with Redis improved our API performance by 300%."
        ]
        
        for i, message in enumerate(messages, 1):
            message_data = {
                "content": message,
                "role": "user",
                "sessionId": session_id,
                "process_messages": True
            }
            
            print(f"📝 Sending message {i}/15...")
            response = await client.post(f"{BASE_URL}/v1/messages", headers=HEADERS, json=message_data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Message {i} stored: {result['objectId']}")
            else:
                print(f"   ❌ Message {i} failed: {response.status_code} - {response.text}")
                
            # Small delay between messages
            await asyncio.sleep(0.3)
        
        print("\n⏳ Waiting for background processing to complete...")
        await asyncio.sleep(10)  # Give time for processing
        
        print("\n🔍 Checking if memories were created...")
        
        # Check memory endpoint to see if memories were created
        memory_response = await client.get(
            f"{BASE_URL}/v1/memory/search",
            headers=HEADERS,
            params={
                "query": "Python data analysis",
                "limit": 10
            }
        )
        
        if memory_response.status_code == 200:
            memory_data = memory_response.json()
            memory_count = len(memory_data.get("memories", []))
            print(f"📊 Found {memory_count} memories in search results")
            
            if memory_count > 0:
                print("✅ Memory creation is working!")
                for i, memory in enumerate(memory_data["memories"][:3], 1):
                    print(f"   Memory {i}: {memory.get('content', '')[:100]}...")
            else:
                print("❌ No memories found - memory creation may not be working")
        else:
            print(f"❌ Failed to search memories: {memory_response.status_code}")

if __name__ == "__main__":
    asyncio.run(test_memory_creation())
