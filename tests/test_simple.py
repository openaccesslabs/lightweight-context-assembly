#!/usr/bin/env python3
"""
Simple test - just one message with memory extraction
"""
import asyncio
import aiohttp
import sqlite3

async def simple_test():
    """Send one message and check if memory gets extracted"""
    
    print("=" * 60)
    print("Simple Memory Extraction Test")
    print("=" * 60)
    
    # Clear old memories via API (don't delete database file!)
    async with aiohttp.ClientSession() as session:
        # Get all memories
        async with session.get("http://localhost:8081/memories/alice") as resp:
            if resp.status == 200:
                data = await resp.json()
                if data['total'] > 0:
                    print(f"Found {data['total']} old memories, clearing...")
                    #  Delete each one
                    for mem in data['memories']:
                        await session.delete(
                            f"http://localhost:8081/memories/alice/{mem['type']}/{mem['key']}"
                        )
                print("✓ Ready to test")
    
        # Send one message with clear facts
        payload = {
            "user_id": "alice",
            "thread_id": "test-1",
            "messages": [
                {
                    "role": "user",
                    "content": "Hi! I'm a Python developer working at Google. I prefer FastAPI for building APIs."
                }
            ],
            "temperature": 0.7,
            "max_tokens": 200
        }
        
        print("\n[1] Sending message with clear facts...")
        async with session.post(
            "http://localhost:8081/v1/chat/completions",
            json=payload
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"✗ Error: {text}")
                return
            
            data = await resp.json()
            print(f"✓ Got response: {data['reply'][:100]}...")
            print(f"  Context: {data['context_usage']:.1%}")
        
        # Wait for async extraction to complete
        print("\n[2] Waiting 8 seconds for memory extraction...")
        await asyncio.sleep(8)
        
        # Check database
        print("\n[3] Checking database...")
        try:
            conn = sqlite3.connect('memory.db')
            cursor = conn.execute("SELECT memory_type, key, value FROM memories WHERE user_id='alice'")
            memories = cursor.fetchall()
            conn.close()
            
            if memories:
                print(f"✓ Found {len(memories)} memories:")
                for mem_type, key, value in memories:
                    print(f"  [{mem_type}] {key} = {value}")
            else:
                print("✗ No memories found in database")
                print("\nCheck the memory proxy logs for extraction errors!")
        except Exception as e:
            print(f"✗ Database error: {e}")
        
        # Try via API
        print("\n[4] Checking via API...")
        async with session.get("http://localhost:8081/memories/alice") as resp:
            data = await resp.json()
            print(f"✓ API reports {data['total']} memories")
            if data['memories']:
                for mem in data['memories']:
                    print(f"  [{mem['type']}] {mem['key']} = {mem['value']}")

if __name__ == "__main__":
    print("\nMake sure:")
    print("1. llama-server is running on port 8080")
    print("2. memory proxy is running on port 8081")
    print()
    asyncio.run(simple_test())
