#!/usr/bin/env python3
"""
Example test conversation to demonstrate memory proxy functionality
"""

import asyncio
import aiohttp
import json

PROXY_URL = "http://localhost:8081"

async def send_message(user_id: str, thread_id: str, message: str):
    """Send a message to the proxy"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "user_id": user_id,
            "thread_id": thread_id,
            "messages": [{"role": "user", "content": message}],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        async with session.post(
            f"{PROXY_URL}/v1/chat/completions",
            json=payload
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"Error: {text}")
                return None
            
            data = await resp.json()
            return data

async def get_memories(user_id: str):
    """Get all stored memories for a user"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{PROXY_URL}/memories/{user_id}") as resp:
            return await resp.json()

async def test_conversation():
    """Test a multi-turn conversation with memory"""
    user_id = "test-user"
    thread_id = "test-thread-1"
    
    print("=" * 60)
    print("Memory Proxy Test Conversation")
    print("=" * 60)
    
    # Turn 1: Introduce preferences
    print("\n[Turn 1] Stating preferences...")
    response = await send_message(
        user_id, 
        thread_id,
        "Hi! I'm a Python developer and I prefer using FastAPI for web services."
    )
    if response:
        print(f"Assistant: {response['reply']}")
        print(f"Context usage: {response['context_usage']:.1%}")
        print(f"Memories used: {len(response['used_memories'])}")
    
    await asyncio.sleep(5)  # Give extraction time to run (embeddings are slow)
    
    # Turn 2: Ask related question
    print("\n[Turn 2] Testing memory recall...")
    response = await send_message(
        user_id,
        thread_id,
        "What web framework should I use for my next API project?"
    )
    if response:
        print(f"Assistant: {response['reply']}")
        print(f"Memories used: {response['used_memories']}")
    
    await asyncio.sleep(5)  # Wait for extraction
    
    # Turn 3: Add project context
    print("\n[Turn 3] Adding project context...")
    response = await send_message(
        user_id,
        thread_id,
        "I'm building a Matterport integration with a 4-week deadline."
    )
    if response:
        print(f"Assistant: {response['reply'][:200]}...")
        print(f"Context usage: {response['context_usage']:.1%}")
    
    await asyncio.sleep(5)  # Wait for extraction
    
    # Turn 4: Recall project context
    print("\n[Turn 4] Testing project memory...")
    response = await send_message(
        user_id,
        thread_id,
        "What am I currently working on?"
    )
    if response:
        print(f"Assistant: {response['reply']}")
        print(f"Memories used: {len(response['used_memories'])}")
    
    # Check stored memories
    print("\n" + "=" * 60)
    print("Stored Memories:")
    print("=" * 60)
    memories = await get_memories(user_id)
    for mem in memories['memories']:
        print(f"  [{mem['type']}] {mem['key']}: {mem['value']} (used {mem['access_count']}x)")
    print(f"\nTotal memories: {memories['total']}")

async def test_long_conversation():
    """Test context compression with long conversation"""
    user_id = "test-user-2"
    thread_id = "long-thread"
    
    print("\n" + "=" * 60)
    print("Testing Context Compression")
    print("=" * 60)
    
    # Send many messages to trigger compression
    for i in range(15):
        message = f"This is test message number {i+1}. " + ("Some filler text. " * 20)
        response = await send_message(user_id, thread_id, message)
        if response:
            print(f"Turn {i+1}: Context {response['context_usage']:.1%}, Compressed: {response['compressed_this_turn']}")
            if response['compressed_this_turn']:
                print(f"  ⚡ Compression triggered!")
                print(f"  Summary excerpt: {response['summary_excerpt']}")
        await asyncio.sleep(3)  # Wait for extraction between messages

async def main():
    """Run all tests"""
    print("\nMake sure the memory proxy is running on localhost:8081")
    print("And llama-server is running on localhost:8080\n")
    
    try:
        await test_conversation()
        await test_long_conversation()
        print("\n✅ All tests completed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
