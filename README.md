# Memory Proxy for llama.cpp

Lightweight semantic memory system that adds persistent memory and context-aware compression to llama.cpp through a proxy layer. Enables unlimited conversation length without modifying llama.cpp.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🧠 **Persistent Memory**: Facts, preferences, and context survive restarts
- 🔍 **Semantic Retrieval**: Automatic retrieval of relevant memories using embeddings
- 📦 **Context-Aware Compression**: Dynamically compresses at 90% context usage
- 🔌 **Zero llama.cpp Modifications**: Works as middleware proxy
- 💾 **Local-First**: SQLite + NumPy, no external services
- 🎯 **Accurate Token Counting**: Uses tiktoken for precise measurements
- 🔄 **OpenAI-Compatible API**: Easy integration with existing clients

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start llama-server with Embeddings

```bash
llama-server -hf ggml-org/gemma-3-1b-it-GGUF \
  --embedding \
  --pooling mean \
  --port 8080 \
  --ctx-size 8192
```

**Critical flags:**
- `--embedding`: Enable embeddings endpoint
- `--pooling mean`: Required for embeddings to work
- `--ctx-size 8192`: Larger context window (recommended)

**Note:** llama-server will serve its Web UI on port 8080, and the memory proxy will proxy it on port 8081 with memory features.

### 3. Start Memory Proxy

```bash
python3 run.py
```

Or use the helper script:
```bash
./scripts/START.sh
```

### 4. Use the Web UI

**Open your browser to: http://localhost:8081/**

You'll see the familiar llama.cpp chat interface, but now with:
- 🧠 Persistent memory across sessions
- 📦 Automatic context compression at 90%
- 💾 All conversations stored in SQLite

**The UI looks identical but remembers everything!**

### 5. Or Use the API

```bash
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "thread_id": "chat-1",
    "messages": [
      {"role": "user", "content": "I prefer Python and FastAPI for web development"}
    ]
  }'
```

### 6. Or Use the Interactive Terminal

```bash
python3 examples/interactive_chat.py
```

The assistant will remember preferences and context across all interfaces!

## Architecture

```
┌─────────┐     ┌──────────────────┐     ┌──────────────┐
│ Client  │────▶│  Memory Proxy    │────▶│ llama-server │
│         │◀────│  (FastAPI)       │◀────│              │
└─────────┘     └──────────────────┘     └──────────────┘
                        │
                        ▼
                 ┌─────────────┐
                 │   SQLite    │
                 │   + NumPy   │
                 └─────────────┘
```

**How it works:**
1. Client sends chat request to proxy
2. Proxy retrieves relevant memories (semantic search)
3. Proxy checks context usage and compresses if needed
4. Proxy forwards enriched request to llama-server
5. Proxy extracts new memories asynchronously
6. Response returned to client

## Project Structure

```
lightweight-context-assembly/
├── src/
│   └── memory_proxy.py          # Main proxy implementation
├── tests/
│   ├── test_offline.py          # Offline validation tests
│   ├── test_extraction.py       # Extraction testing
│   └── test_simple.py           # Simple integration test
├── examples/
│   └── test_conversation.py     # Full conversation example
├── docs/
│   ├── QUICKSTART.md            # Quick start guide
│   ├── IMPLEMENTATION.md        # Technical details
│   ├── CHANGES.md               # Changelog
│   └── VALIDATION.md            # Validation report
├── scripts/
│   └── START.sh                 # Startup helper script
├── config.json                  # Configuration (git-ignored)
├── config.example.json          # Example configuration
├── schema.sql                   # Database schema
├── requirements.txt             # Python dependencies
├── run.py                       # Main entry point
├── README.md                    # This file
├── LICENSE                      # MIT License
└── .gitignore                   # Git ignore rules
```

## Configuration

Copy `config.example.json` to `config.json` and customize:

```json
{
  "llama_server_url": "http://localhost:8080",
  "proxy_port": 8081,
  "memory_top_k": 5,
  "memory_similarity_threshold": 0.7,
  "compression_threshold": 0.9,
  "keep_recent_messages": 10,
  "max_tokens": 2000,
  "summary_max_tokens": 500,
  "tiktoken_encoding": "cl100k_base",
  "extraction_enabled": true,
  "database_path": "memory.db"
}
```

### Key Settings

- **compression_threshold** (0.9): Compress at 90% context usage
- **keep_recent_messages** (10): Messages to keep after compression
- **memory_similarity_threshold** (0.7): Min cosine similarity for retrieval
- **memory_top_k** (5): Max memories to inject per request

## API Endpoints

### POST `/v1/chat/completions`

Main chat endpoint with memory and compression.

**Request:**
```json
{
  "user_id": "alice",
  "thread_id": "project-1",
  "messages": [{"role": "user", "content": "..."}],
  "temperature": 0.7,
  "max_tokens": 2000
}
```

**Response:**
```json
{
  "reply": "...",
  "used_memories": [
    {"type": "preference", "key": "language", "value": "Python"}
  ],
  "summary_excerpt": "...",
  "context_usage": 0.45,
  "compressed_this_turn": false
}
```

### GET `/memories/{user_id}`

Retrieve all stored memories.

### DELETE `/memories/{user_id}/{memory_type}/{key}`

Delete a specific memory.

### GET `/health`

Health check endpoint.

## How Context Compression Works

1. **Every request** checks context usage (% of available window)
2. When usage exceeds 90%:
   - Compresses old messages into summary (<500 tokens)
   - Keeps recent N messages (default: 10)
   - **All messages stay in database** (never deleted)
3. Future requests use: `summary + recent messages`
4. Enables effectively unlimited conversation length

**Example:**
```
Messages 1-50: Context grows to 91%
Compression:   Messages 1-40 → summary (500 tokens)
               Messages 41-50 kept as-is
New context:   Summary + messages 41-50 = ~15% usage
```

## Memory Types

The system automatically extracts four types:

- **preference**: User preferences (e.g., "prefers Python")
- **fact**: Biographical facts (e.g., "works at Google")
- **project**: Active work (e.g., "building API with 4-week deadline")
- **context**: Temporary state (e.g., "debugging auth issue")

## Testing

### Run Offline Tests

```bash
python3 tests/test_offline.py
```

Tests database, token counting, and core logic without llama-server.

### Run Simple Integration Test

```bash
python3 tests/test_simple.py
```

Tests memory extraction with one message.

### Run Full Conversation Test

```bash
python3 examples/test_conversation.py
```

Tests multi-turn conversation with memory recall and compression.

## Performance

- Memory retrieval: <100ms for 10k memories
- Token counting: ~5ms (tiktoken)
- Context calculation: ~20ms
- **Total overhead: ~25ms per request**
- Compression (when triggered): ~3s (one-time, synchronous)

## Troubleshooting

### "Pooling type 'none' is not OAI compatible"

**Solution:** Start llama-server with `--pooling mean`:
```bash
llama-server --embedding --pooling mean ...
```

### No memories extracted

**Check:**
1. Wait 5-8 seconds after messages (extraction is async)
2. Look for extraction errors in proxy logs
3. Verify embeddings endpoint works:
   ```bash
   curl -X POST http://localhost:8080/v1/embeddings \
     -d '{"input":"test"}'
   ```

### Context overflow errors

**Solution:** Increase llama-server context:
```bash
llama-server --ctx-size 8192 ...
```

### Database locked errors

The proxy uses WAL mode which handles concurrent access. If issues persist, check file permissions on `memory.db`.

## Limitations

- Single-user focus (no authentication)
- SQLite suitable for <50k memories (switch to FAISS for more)
- Text-only (no multi-modal support)
- Requires llama-server with embedding support

## Roadmap

- [ ] Memory decay (reduce old/unused memories)
- [ ] FAISS backend for >100k memories
- [ ] Batch embedding API for lower latency
- [ ] Memory graph relationships
- [ ] Web UI for memory management
- [ ] Multi-agent memory scopes

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built for [llama.cpp](https://github.com/ggerganov/llama.cpp)
- Uses [tiktoken](https://github.com/openai/tiktoken) for accurate token counting
- Inspired by OpenAI's GPT memory features

## Support

For issues and questions:
- Open an issue on GitHub
- Check [docs/QUICKSTART.md](docs/QUICKSTART.md) for detailed setup
- See [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for technical details

---

**Built with ❤️ for the open source community**
