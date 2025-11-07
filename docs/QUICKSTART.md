# Quick Start Guide

## Prerequisites

1. **llama-server running with embeddings enabled**
   
   ```bash
   llama-server -hf ggml-org/gemma-3-1b-it-GGUF \
     --embedding \
     --pooling mean \
     --port 8080 \
     --ctx-size 8192
   ```
   
   **Note:** llama-server serves its Web UI on port 8080. The memory proxy proxies it on port 8081 with memory features.

2. **Python 3.9+**

## Setup (2 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Start the proxy
python3 run.py
```

The proxy will:
- Create `memory.db` SQLite database
- Start on port 8081
- Serve the llama.cpp Web UI
- Connect to llama-server on port 8080

## Verify Installation

```bash
# Check health
curl http://localhost:8081/health

# Expected: {"status":"ok","llama_server":"http://localhost:8080"}
```

## Using the Web UI

**Open your browser to: http://localhost:8081/**

You'll see the familiar llama.cpp chat interface with memory features:
- 🧠 Conversations persist across sessions
- 📦 Automatic compression at 90% context
- 💾 All facts stored in SQLite

**Try it:**
1. Say: "I prefer Python for backend development"
2. Chat about something else
3. Ask: "What programming language do I prefer?"
4. The AI will remember! 🎉

## First Conversation

```bash
# Tell the assistant your preference
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "thread_id": "chat-1",
    "messages": [
      {"role": "user", "content": "I love Python and FastAPI"}
    ]
  }'

# Wait 2 seconds for memory extraction

# Ask a related question
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "thread_id": "chat-1",
    "messages": [
      {"role": "user", "content": "What should I use for my web API?"}
    ]
  }'

# The assistant will recall your Python/FastAPI preference!
```

## View Stored Memories

```bash
curl http://localhost:8081/memories/alice
```

You'll see extracted memories with their types, keys, and access counts.

## Run Test Script

```bash
cd examples
python test_conversation.py
```

This runs a full test including:
- Multi-turn conversation
- Memory storage and retrieval
- Context compression triggering

## Key Concepts

### user_id
Identifies the user. All memories are scoped to this ID.

### thread_id  
Identifies the conversation thread. Each thread has separate context and can be compressed independently.

### Compression
When context usage hits 90%, the system automatically:
1. Compresses old messages into a summary
2. Keeps recent 10 messages uncompressed
3. All messages stay in database

### Memory Extraction
After each response, the system automatically:
1. Analyzes the conversation
2. Extracts new facts/preferences
3. Stores them with embeddings
4. Retrieves them semantically in future conversations

## Configuration

Edit `config.json` to customize:

```json
{
  "compression_threshold": 0.9,     // Compress at 90% context
  "keep_recent_messages": 10,       // Keep 10 messages after compression
  "memory_similarity_threshold": 0.7, // Min similarity for retrieval
  "memory_top_k": 5                 // Max memories to inject per request
}
```

## Troubleshooting

### "Connection refused" error
- Ensure llama-server is running on port 8080
- Check `config.json` has correct `llama_server_url`

### No memories extracted
- Wait 2-3 seconds after messages (extraction is async)
- Check logs for JSON parsing errors
- Set `extraction_enabled: true` in config

### Context compression not triggering
- Send longer messages (100+ tokens each)
- Lower `compression_threshold` in config for testing
- Check logs for context usage percentage

### Database locked errors
- Close any SQLite browser tools
- Check file permissions on `memory.db`

## Next Steps

1. **Integrate with your app**: Use the OpenAI-compatible API
2. **Customize extraction**: Edit the extraction prompt in `memory_proxy.py`
3. **Add authentication**: Wrap endpoints with auth middleware
4. **Monitor performance**: Check logs for context usage and compression events

## Example Python Client

```python
import aiohttp

async def chat(user_id, thread_id, message):
    async with aiohttp.ClientSession() as session:
        payload = {
            "user_id": user_id,
            "thread_id": thread_id,
            "messages": [{"role": "user", "content": message}]
        }
        async with session.post(
            "http://localhost:8081/v1/chat/completions",
            json=payload
        ) as resp:
            return await resp.json()

# Usage
response = await chat("zay", "project-1", "I prefer Python")
print(response['reply'])
print(f"Context: {response['context_usage']:.1%}")
```

## Success Criteria

✅ Assistant remembers facts across conversations  
✅ Long conversations work without context overflow  
✅ Memories retrieve semantically (not keyword matching)  
✅ System works offline (no external APIs)  
✅ <300ms overhead per request

---

**Need help?** Check the logs - the proxy outputs detailed info about compression, memory retrieval, and context usage.
