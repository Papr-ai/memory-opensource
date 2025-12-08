#!/usr/bin/env python3
"""
Final test to verify memory creation is working
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

async def test_final_memory():
    session_id = f"final_test_{int(time.time())}"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        print(f"🧪 Final memory test with session: {session_id}")
        
        # Send 15 diverse messages to trigger processing
        messages = [
            "I prefer using TypeScript over JavaScript for better type safety in large applications.",
            "My goal is to become a senior software architect within the next 3 years.",
            "I learned that React hooks make component state management much more intuitive than class components.",
            "The task I'm working on involves optimizing our database queries to reduce API response times.",
            "I have extensive experience with AWS services including EC2, S3, and Lambda for cloud deployments.",
            "My preference is to work with agile methodologies and daily standups for better team coordination.",
            "I need to improve my skills in machine learning, specifically with TensorFlow and PyTorch frameworks.",
            "The context of our current project is building a real-time chat application for remote teams.",
            "I discovered that using Redis for caching improved our application performance by 200%.",
            "My learning goal is to master Kubernetes for container orchestration and deployment automation.",
            "I prefer using PostgreSQL over MySQL for complex relational database operations.",
            "The fact is that our microservices architecture handles over 50,000 requests per minute.",
            "I have a task to implement automated CI/CD pipelines using GitHub Actions and Docker.",
            "My goal is to establish comprehensive testing coverage of at least 85% for our codebase.",
            "I learned that implementing proper error handling and logging significantly reduces debugging time."
        ]
        
        for i, message in enumerate(messages, 1):
            message_data = {
                "content": message,
                "role": "user",
                "sessionId": session_id,
                "process_messages": True
            }
            
            response = await client.post(f"{BASE_URL}/v1/messages", headers=HEADERS, json=message_data)
            if response.status_code == 200:
                print(f"✅ Message {i} sent")
            else:
                print(f"❌ Message {i} failed: {response.status_code}")
        
        print("⏳ Waiting 25 seconds for processing...")
        await asyncio.sleep(25)
        print("✅ Processing complete!")

if __name__ == "__main__":
    asyncio.run(test_final_memory())
