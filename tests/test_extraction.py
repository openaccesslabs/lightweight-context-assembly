#!/usr/bin/env python3
"""
Test memory extraction manually
"""
import asyncio
import aiohttp
import json

async def test_extraction():
    """Test the extraction prompt"""
    
    async with aiohttp.ClientSession() as session:
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": """Extract NEW facts about the user from this conversation. Return ONLY a JSON array, no markdown.

Format: [{"type": "preference|fact|project|context", "key": "short-id", "value": "the fact"}]

Types:
- preference: User's stated preferences (language, tools, style)
- fact: Biographical or situational facts (job, location, projects)
- project: Active work context (deadlines, goals, blockers)
- context: Temporary state (current task, recent decision)

Rules:
- Only NEW information not previously stated
- Be specific, not generic
- One fact per object
- Empty array if no new facts
- Return ONLY the JSON array, no markdown formatting"""
                },
                {
                    "role": "user",
                    "content": """USER: Hi! I'm a Python developer and I prefer using FastAPI for web services.

ASSISTANT: Excellent! That's fantastic to hear."""
                }
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
        
        print("Sending extraction request to llama-server...")
        async with session.post(
            "http://localhost:8080/v1/chat/completions",
            json=payload
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"Error: {text}")
                return
            
            data = await resp.json()
            result = data['choices'][0]['message']['content']
            
            print("=" * 60)
            print("Raw response:")
            print("=" * 60)
            print(result)
            print()
            
            # Try to parse it
            result_stripped = result.strip()
            if result_stripped.startswith('```'):
                print("Detected markdown code block, stripping...")
                lines = result_stripped.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                result_stripped = '\n'.join(lines)
            
            print("=" * 60)
            print("After stripping:")
            print("=" * 60)
            print(result_stripped)
            print()
            
            try:
                facts = json.loads(result_stripped)
                print("=" * 60)
                print("Parsed facts:")
                print("=" * 60)
                for fact in facts:
                    print(f"  {fact['type']}: {fact['key']} = {fact['value']}")
                print(f"\nTotal: {len(facts)} facts")
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")

if __name__ == "__main__":
    asyncio.run(test_extraction())
