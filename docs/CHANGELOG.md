# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **Web UI Proxy Support**: Memory proxy now serves llama.cpp's embedded Web UI with transparent memory features
- **Streaming Support**: SSE (Server-Sent Events) for real-time chat responses
- **Accurate Token Counting**: Using tiktoken library (cl100k_base encoding) for precise context calculation
- **Optional user_id/thread_id**: Web UI defaults to `webui_user` and `webui_default`
- **OpenAI-Compatible API**: Response format compatible with llama.cpp Web UI
- **Interactive Chat Client**: Terminal-based chat example with memory display

### Features
- Persistent semantic memory with SQLite
- Context-aware compression at 90% threshold
- Semantic memory retrieval with cosine similarity
- Async memory extraction from conversations
- Automatic fact categorization (preference, fact, project, context)
- Browser-like headers for proper llama-server content negotiation

### Configuration
- Uses `tiktoken` library with `cl100k_base` encoding
- Configurable compression threshold, memory retrieval, token limits
- Example config provided in `config.example.json`

### Architecture
- FastAPI-based proxy middleware
- SQLite with WAL mode for concurrency
- NumPy for embedding similarity (suitable for <50k memories)
- aiohttp for async HTTP requests

## [0.1.0] - 2025-11-07

### Initial Release
- Complete memory proxy implementation per PRD
- Core memory system with embeddings
- Context compression with rolling summaries
- Message history persistence
- Memory extraction with LLM
