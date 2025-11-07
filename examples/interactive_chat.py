#!/usr/bin/env python3
"""
Interactive chat client for memory proxy
Usage: python3 examples/interactive_chat.py
"""

import asyncio
import aiohttp
import sys

PROXY_URL = "http://localhost:8081"

async def chat(user_id: str, thread_id: str):
    """Interactive chat session"""
    
    print("=" * 60)
    print("Memory Proxy Interactive Chat")
    print("=" * 60)
    print(f"User: {user_id}")
    print(f"Thread: {thread_id}")
    print("Type 'exit' to quit, 'memories' to see stored facts")
    print("=" * 60)
    print()
    
    async with aiohttp.ClientSession() as session:
        while True:
            # Get user input
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break
            
            if not user_input:
                continue
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'memories':
                # Show stored memories
                async with session.get(f"{PROXY_URL}/memories/{user_id}") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data['total'] > 0:
                            print(f"\n📝 Stored memories ({data['total']}):")
                            for mem in data['memories']:
                                print(f"  [{mem['type']}] {mem['key']}: {mem['value']}")
                        else:
                            print("\n(No memories stored yet)")
                    print()
                continue
            
            # Send message to proxy
            payload = {
                "user_id": user_id,
                "thread_id": thread_id,
                "messages": [
                    {"role": "user", "content": user_input}
                ]
            }
            
            try:
                async with session.post(
                    f"{PROXY_URL}/v1/chat/completions",
                    json=payload
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        print(f"Error: {text}\n")
                        continue
                    
                    data = await resp.json()
                    
                    # Print response
                    print(f"Assistant: {data['reply']}")
                    
                    # Show context info
                    if data.get('used_memories'):
                        print(f"  💡 Recalled: {len(data['used_memories'])} memories")
                    
                    print(f"  📊 Context: {data['context_usage']:.1%}", end="")
                    if data.get('compressed_this_turn'):
                        print(" ⚡ (compressed!)", end="")
                    print("\n")
                    
            except aiohttp.ClientError as e:
                print(f"Connection error: {e}")
                print("Make sure memory proxy is running on port 8081\n")
            except Exception as e:
                print(f"Error: {e}\n")

def main():
    """Main entry point"""
    
    # Get user ID and thread ID
    if len(sys.argv) > 1:
        user_id = sys.argv[1]
    else:
        user_id = input("Enter your name (or press Enter for 'alice'): ").strip() or "alice"
    
    if len(sys.argv) > 2:
        thread_id = sys.argv[2]
    else:
        thread_id = input("Enter thread ID (or press Enter for 'default'): ").strip() or "default"
    
    print()
    
    try:
        asyncio.run(chat(user_id, thread_id))
    except KeyboardInterrupt:
        print("\n\nGoodbye!")

if __name__ == "__main__":
    main()
