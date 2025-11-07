#!/usr/bin/env python3
"""
Memory Proxy for llama.cpp
Implements persistent semantic memory with context-aware compression
"""

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
import tiktoken
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
with open('config.json') as f:
    CONFIG = json.load(f)

# Validate configuration
def validate_config():
    if not 0.0 < CONFIG['compression_threshold'] <= 1.0:
        raise ValueError("compression_threshold must be between 0 and 1")
    if CONFIG['compression_threshold'] < 0.7:
        logger.warning("compression_threshold < 0.7 may cause frequent compressions")
    if CONFIG['compression_threshold'] > 0.95:
        logger.warning("compression_threshold > 0.95 risks context overflow")
    if CONFIG['keep_recent_messages'] < 5:
        logger.warning("keep_recent_messages < 5 may lose important context")
    logger.info(f"Compression threshold: {CONFIG['compression_threshold']:.0%}")
    logger.info(f"Keep recent: {CONFIG['keep_recent_messages']} messages")

validate_config()

# Initialize tiktoken encoder
try:
    TOKENIZER = tiktoken.get_encoding(CONFIG.get('tiktoken_encoding', 'cl100k_base'))
    logger.info(f"Initialized tiktoken with encoding: {CONFIG.get('tiktoken_encoding', 'cl100k_base')}")
except Exception as e:
    logger.warning(f"Failed to initialize tiktoken: {e}. Falling back to simple estimation.")
    TOKENIZER = None

# Request/Response Models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_id: str = "webui_user"  # Default for Web UI
    thread_id: str = "webui_default"
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 2000
    stream: bool = False  # Support streaming

class ChatResponse(BaseModel):
    # OpenAI-compatible fields (required for Web UI)
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "llama-proxy"
    choices: List[Dict] = []
    
    # Memory proxy extensions (optional, ignored by Web UI)
    used_memories: List[Dict[str, str]] = []
    summary_excerpt: Optional[str] = None
    context_usage: float = 0.0
    compressed_this_turn: bool = False

@dataclass
class ContextUsage:
    total_tokens: int
    max_tokens: int
    percentage: float
    messages_included: int

# Database Class
class MemoryDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database with schema"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            with open('schema.sql') as f:
                conn.executescript(f.read())
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
        finally:
            conn.close()
    
    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def store_memory(self, user_id: str, memory_type: str, key: str, 
                     value: str, embedding: np.ndarray):
        """Store or update memory with embedding"""
        embedding_bytes = embedding.tobytes()
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO memories (user_id, memory_type, key, value, embedding, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, memory_type, key)
                DO UPDATE SET 
                    value=excluded.value, 
                    embedding=excluded.embedding,
                    updated_at=CURRENT_TIMESTAMP
            """, (user_id, memory_type, key, value, embedding_bytes))
            conn.commit()
    
    def get_all_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve all memories for a user"""
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT memory_type, key, value, embedding, access_count
                FROM memories WHERE user_id = ?
            """, (user_id,)).fetchall()
        
        return [
            {
                'type': row['memory_type'],
                'key': row['key'],
                'value': row['value'],
                'embedding': np.frombuffer(row['embedding'], dtype=np.float32),
                'access_count': row['access_count']
            }
            for row in rows
        ]
    
    def increment_access_count(self, user_id: str, memory_type: str, key: str):
        """Track memory usage"""
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE memories 
                SET access_count = access_count + 1
                WHERE user_id = ? AND memory_type = ? AND key = ?
            """, (user_id, memory_type, key))
            conn.commit()
    
    def store_message(self, thread_id: str, user_id: str, role: str, content: str):
        """Store message (never deleted)"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO messages (thread_id, user_id, role, content)
                VALUES (?, ?, ?, ?)
            """, (thread_id, user_id, role, content))
            conn.commit()
    
    def get_thread_messages(self, thread_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """Retrieve ALL messages from thread"""
        query = """
            SELECT id, role, content, created_at 
            FROM messages 
            WHERE thread_id = ?
            ORDER BY created_at ASC
        """
        if limit:
            query += f" LIMIT {limit}"
        
        with self.get_connection() as conn:
            rows = conn.execute(query, (thread_id,)).fetchall()
        
        return [
            {
                'id': row['id'],
                'role': row['role'],
                'content': row['content'],
                'created_at': row['created_at']
            }
            for row in rows
        ]
    
    def get_messages_after_id(self, thread_id: str, message_id: int) -> List[Dict[str, Any]]:
        """Get messages after compression pointer (what LLM sees)"""
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT id, role, content, created_at 
                FROM messages 
                WHERE thread_id = ? AND id > ?
                ORDER BY created_at ASC
            """, (thread_id, message_id)).fetchall()
        
        return [
            {
                'id': row['id'],
                'role': row['role'],
                'content': row['content'],
                'created_at': row['created_at']
            }
            for row in rows
        ]
    
    def store_summary(self, thread_id: str, user_id: str, summary: str,
                      last_compressed_message_id: int, message_count: int):
        """Store thread summary with compression pointer"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO thread_summaries 
                (thread_id, user_id, summary, last_compressed_message_id, message_count, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(thread_id)
                DO UPDATE SET 
                    summary=excluded.summary, 
                    last_compressed_message_id=excluded.last_compressed_message_id,
                    message_count=excluded.message_count,
                    updated_at=CURRENT_TIMESTAMP
            """, (thread_id, user_id, summary, last_compressed_message_id, message_count))
            conn.commit()
        logger.info(f"Stored summary for thread {thread_id} (covers up to message {last_compressed_message_id})")
    
    def get_summary(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve thread summary with compression pointer"""
        with self.get_connection() as conn:
            row = conn.execute("""
                SELECT summary, last_compressed_message_id, message_count, updated_at
                FROM thread_summaries WHERE thread_id = ?
            """, (thread_id,)).fetchone()
        
        if row:
            return {
                'summary': row['summary'],
                'last_compressed_message_id': row['last_compressed_message_id'],
                'message_count': row['message_count'],
                'updated_at': row['updated_at']
            }
        return None
    
    def delete_memory(self, user_id: str, memory_type: str, key: str):
        """Delete specific memory"""
        with self.get_connection() as conn:
            conn.execute("""
                DELETE FROM memories 
                WHERE user_id = ? AND memory_type = ? AND key = ?
            """, (user_id, memory_type, key))
            conn.commit()

# Initialize database
db = MemoryDatabase(CONFIG['database_path'])

# LlamaClient Class
class LlamaClient:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or CONFIG['llama_server_url']
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def chat_completion(self, messages: List[Dict], 
                              temperature: float = 0.7,
                              max_tokens: int = 2000) -> str:
        """Call llama-server chat completion"""
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        async with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise HTTPException(status_code=resp.status, detail=f"llama-server error: {text}")
            
            data = await resp.json()
            return data['choices'][0]['message']['content']
    
    async def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from llama-server"""
        payload = {"input": text}
        
        async with self.session.post(
            f"{self.base_url}/v1/embeddings",
            json=payload
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise HTTPException(status_code=resp.status, detail=f"llama-server error: {text}")
            
            data = await resp.json()
            embedding = data['data'][0]['embedding']
            return np.array(embedding, dtype=np.float32)
    
    async def get_model_context_size(self) -> int:
        """Query model's context window size"""
        async with self.session.get(f"{self.base_url}/v1/models") as resp:
            if resp.status != 200:
                logger.warning("Failed to get model info, defaulting to 8192 context")
                return 8192
            data = await resp.json()
            context_length = data['data'][0].get('context_length', 8192)
            logger.info(f"Model context window: {context_length} tokens")
            return context_length

# Helper Functions
def count_tokens(text: str) -> int:
    """Count tokens using tiktoken"""
    if TOKENIZER is not None:
        try:
            return len(TOKENIZER.encode(text))
        except Exception as e:
            logger.warning(f"Tiktoken encoding failed: {e}, using fallback")
    
    # Fallback: 4 chars ≈ 1 token
    return len(text) // 4

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def format_messages(messages: List[Dict]) -> str:
    """Format messages for summarization"""
    return "\n\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in messages
    ])

def build_system_prompt(memories: List[Dict], summary: Optional[str] = None) -> str:
    """Build system prompt with memories and summary"""
    base_prompt = "You are a helpful AI assistant.\n\n"
    
    if memories:
        base_prompt += "=== KNOWN FACTS ABOUT USER ===\n"
        for mem in memories:
            base_prompt += f"- {mem['type'].capitalize()}: {mem['key']} → {mem['value']}\n"
        base_prompt += "\n"
    
    if summary:
        base_prompt += "=== RECENT CONVERSATION CONTEXT ===\n"
        base_prompt += summary + "\n\n"
    
    base_prompt += "Use these facts to personalize responses. Don't mention you have \"stored memories\" - respond naturally as if you remember."
    
    return base_prompt

async def retrieve_relevant_memories(user_id: str, query: str, 
                                     llm: LlamaClient) -> List[Dict[str, Any]]:
    """Retrieve top-k relevant memories using semantic search"""
    memories = db.get_all_memories(user_id)
    
    if not memories:
        return []
    
    # Get query embedding
    query_embedding = await llm.get_embedding(query)
    
    # Calculate similarities
    similarities = []
    for mem in memories:
        sim = cosine_similarity(query_embedding, mem['embedding'])
        if sim >= CONFIG['memory_similarity_threshold']:
            similarities.append((sim, mem))
    
    # Sort by similarity and take top-k
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_memories = [mem for _, mem in similarities[:CONFIG['memory_top_k']]]
    
    # Increment access counts
    for mem in top_memories:
        db.increment_access_count(user_id, mem['type'], mem['key'])
    
    return top_memories

async def calculate_context_usage(thread_id: str, user_id: str, 
                                  llm: LlamaClient) -> ContextUsage:
    """Calculate current context window usage"""
    max_context = await llm.get_model_context_size()
    summary = db.get_summary(thread_id)
    
    # Get messages that would be sent to LLM
    if summary:
        messages = db.get_messages_after_id(thread_id, summary['last_compressed_message_id'])
    else:
        messages = db.get_thread_messages(thread_id)
    
    # Get user memories
    memories = db.get_all_memories(user_id)
    
    # Count tokens
    token_count = 0
    
    # System prompt + memories
    system_prompt = build_system_prompt(memories, summary['summary'] if summary else None)
    token_count += count_tokens(system_prompt)
    
    # Messages
    for msg in messages:
        token_count += count_tokens(msg['content']) + 4
    
    # Calculate usable context
    reserved_tokens = CONFIG['max_tokens']
    usable_context = max_context - reserved_tokens
    percentage = token_count / usable_context
    
    return ContextUsage(
        total_tokens=token_count,
        max_tokens=usable_context,
        percentage=percentage,
        messages_included=len(messages)
    )

async def compress_thread(thread_id: str, user_id: str, llm: LlamaClient):
    """Compress thread history (messages stay in DB)"""
    all_messages = db.get_thread_messages(thread_id)
    keep_raw_count = CONFIG['keep_recent_messages']
    
    if len(all_messages) <= keep_raw_count:
        logger.warning(f"Thread {thread_id} has only {len(all_messages)} messages - nothing to compress")
        return
    
    messages_to_compress = all_messages[:-keep_raw_count]
    previous_summary = db.get_summary(thread_id)
    
    # Build compression prompt
    if previous_summary:
        context = f"""Previous summary:
{previous_summary['summary']}

New messages to integrate:
{format_messages(messages_to_compress[previous_summary['message_count']:])}

Create updated summary that integrates new information with previous context."""
    else:
        context = f"""Conversation to summarize:
{format_messages(messages_to_compress)}

Create concise summary of this conversation."""
    
    summary_prompt = [
        {
            "role": "system",
            "content": """Compress this conversation into <500 tokens.

Focus on:
- Facts established
- Decisions made
- Current state/blockers
- Key technical details

Be specific. Prioritize actionable information over pleasantries."""
        },
        {
            "role": "user",
            "content": context
        }
    ]
    
    summary = await llm.chat_completion(
        summary_prompt,
        temperature=0.3,
        max_tokens=CONFIG['summary_max_tokens']
    )
    
    # Store summary with pointer
    last_compressed_id = messages_to_compress[-1]['id']
    db.store_summary(thread_id, user_id, summary, last_compressed_id, len(messages_to_compress))
    
    # Log compression stats
    tokens_before = sum(count_tokens(m['content']) for m in messages_to_compress)
    tokens_after = count_tokens(summary)
    compression_ratio = tokens_after / tokens_before if tokens_before > 0 else 0
    
    logger.info(
        f"Compressed thread {thread_id}: "
        f"{len(messages_to_compress)} msgs ({tokens_before} tokens) → "
        f"summary ({tokens_after} tokens) [{compression_ratio:.1%}]. "
        f"Pointer at message {last_compressed_id}"
    )

async def extract_memories_wrapper(user_id: str, conversation: List[Dict]):
    """Wrapper that creates its own LlamaClient for async extraction"""
    async with LlamaClient() as llm:
        await extract_memories(user_id, conversation, llm)

async def extract_memories(user_id: str, conversation: List[Dict], llm: LlamaClient):
    """Extract and store new memories (async, non-blocking)"""
    if not CONFIG['extraction_enabled']:
        return
    
    extraction_prompt = [
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
            "content": format_messages(conversation[-4:])
        }
    ]
    
    try:
        result = await llm.chat_completion(extraction_prompt, temperature=0.3, max_tokens=500)
        
        logger.debug(f"Raw extraction result: {result[:200]}")
        
        # Strip markdown code blocks if present
        result = result.strip()
        if result.startswith('```'):
            # Remove ```json or ``` from start and ``` from end
            lines = result.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            result = '\n'.join(lines)
        
        logger.debug(f"After stripping: {result[:200]}")
        
        facts = json.loads(result)
        logger.info(f"Parsed {len(facts)} facts from extraction")
        
        if not isinstance(facts, list):
            logger.warning(f"Expected list, got {type(facts)}")
            return
        
        for i, fact in enumerate(facts):
            logger.debug(f"Fact {i}: {fact} (type: {type(fact)})")
            
            # Handle if fact is a string instead of dict
            if isinstance(fact, str):
                logger.warning(f"Skipping string fact (expected dict): {fact[:100]}")
                continue
            
            if not isinstance(fact, dict):
                logger.warning(f"Skipping non-dict fact: {type(fact)}")
                continue
            
            if not all(k in fact for k in ['type', 'key', 'value']):
                logger.warning(f"Skipping invalid fact (missing fields): {fact}")
                continue
            
            # Normalize type (some models return variations)
            fact_type = fact['type'].lower()
            if fact_type not in ['preference', 'fact', 'project', 'context']:
                logger.warning(f"Unknown type '{fact_type}', defaulting to 'fact'")
                fact_type = 'fact'
            
            # Get embedding for the fact
            embedding = await llm.get_embedding(fact['value'])
            db.store_memory(user_id, fact_type, fact['key'], fact['value'], embedding)
            logger.info(f"✓ Stored memory: {fact_type}/{fact['key']} = {fact['value'][:50]}")
    
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse extraction: {e}\nResponse: {result[:200]}")
    except Exception as e:
        logger.error(f"Memory extraction failed: {e}")

# FastAPI App
app = FastAPI(title="Memory Proxy for llama.cpp")

@app.on_event("startup")
async def startup():
    logger.info(f"Memory Proxy starting on port {CONFIG['proxy_port']}")
    logger.info(f"llama-server: {CONFIG['llama_server_url']}")

@app.get("/health")
async def health():
    return {"status": "ok", "llama_server": CONFIG['llama_server_url']}

@app.get("/memories/{user_id}")
async def get_memories(user_id: str):
    """Get all memories for a user"""
    memories = db.get_all_memories(user_id)
    return {
        "memories": [
            {
                "type": m['type'],
                "key": m['key'],
                "value": m['value'],
                "access_count": m['access_count']
            }
            for m in memories
        ],
        "total": len(memories)
    }

@app.delete("/memories/{user_id}/{memory_type}/{key}")
async def delete_memory(user_id: str, memory_type: str, key: str):
    """Delete a specific memory"""
    db.delete_memory(user_id, memory_type, key)
    return {"status": "deleted"}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Main chat endpoint with context-aware compression and streaming support"""
    
    # Store user message
    user_message = request.messages[-1].content
    db.store_message(request.thread_id, request.user_id, "user", user_message)
    
    async with LlamaClient() as llm:
        # Check context usage
        context_usage = await calculate_context_usage(request.thread_id, request.user_id, llm)
        logger.debug(f"Thread {request.thread_id}: {context_usage.percentage:.1%} context")
        
        compressed_this_turn = False
        
        # SYNCHRONOUS compression if threshold exceeded
        if context_usage.percentage >= CONFIG['compression_threshold']:
            logger.warning(f"Thread {request.thread_id} at {context_usage.percentage:.1%} - compressing")
            await compress_thread(request.thread_id, request.user_id, llm)
            
            # Recalculate after compression
            context_usage = await calculate_context_usage(request.thread_id, request.user_id, llm)
            logger.info(f"Compressed to {context_usage.percentage:.1%}")
            compressed_this_turn = True
        
        # Retrieve relevant memories
        relevant_memories = await retrieve_relevant_memories(request.user_id, user_message, llm)
        
        # Build context for LLM
        summary = db.get_summary(request.thread_id)
        system_prompt = build_system_prompt(relevant_memories, summary['summary'] if summary else None)
        
        # Get messages to send
        if summary:
            messages_to_send = db.get_messages_after_id(
                request.thread_id,
                summary['last_compressed_message_id']
            )
        else:
            messages_to_send = db.get_thread_messages(request.thread_id)
        
        # Build LLM messages
        llm_messages = [{"role": "system", "content": system_prompt}]
        llm_messages.extend([
            {"role": msg['role'], "content": msg['content']}
            for msg in messages_to_send
        ])
        
        # If streaming requested, just proxy to llama-server with streaming
        if request.stream:
            async def stream_response():
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{CONFIG['llama_server_url']}/v1/chat/completions",
                        json={
                            "messages": llm_messages,
                            "temperature": request.temperature,
                            "max_tokens": request.max_tokens,
                            "stream": True
                        }
                    ) as resp:
                        full_content = ""
                        async for line in resp.content:
                            chunk = line.decode('utf-8')
                            yield chunk
                            
                            # Collect content for storage
                            if chunk.startswith('data: ') and not chunk.startswith('data: [DONE]'):
                                try:
                                    data = json.loads(chunk[6:])
                                    if 'choices' in data and len(data['choices']) > 0:
                                        delta = data['choices'][0].get('delta', {})
                                        if 'content' in delta:
                                            full_content += delta['content']
                                except:
                                    pass
                        
                        # Store complete response after streaming
                        if full_content:
                            db.store_message(request.thread_id, request.user_id, "assistant", full_content)
                            
                            # Extract memories asynchronously
                            conversation = [{"role": msg['role'], "content": msg['content']} for msg in messages_to_send]
                            conversation.append({"role": "assistant", "content": full_content})
                            asyncio.create_task(extract_memories_wrapper(request.user_id, conversation))
            
            return StreamingResponse(stream_response(), media_type="text/event-stream")
        
        # Non-streaming response
        reply = await llm.chat_completion(
            llm_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Store assistant response
        db.store_message(request.thread_id, request.user_id, "assistant", reply)
        
        # Extract memories asynchronously (non-blocking)
        conversation = [{"role": msg['role'], "content": msg['content']} for msg in messages_to_send]
        conversation.append({"role": "assistant", "content": reply})
        asyncio.create_task(extract_memories_wrapper(request.user_id, conversation))
        
        return ChatResponse(
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": reply
                },
                "finish_reason": "stop"
            }],
            used_memories=[
                {"type": m['type'], "key": m['key'], "value": m['value']}
                for m in relevant_memories
            ],
            summary_excerpt=summary['summary'][:100] + "..." if summary else None,
            context_usage=context_usage.percentage,
            compressed_this_turn=compressed_this_turn
        )

@app.get("/", response_class=HTMLResponse)
async def serve_web_ui():
    """Serve llama-server Web UI through proxy"""
    async with aiohttp.ClientSession() as session:
        try:
            # Send browser-like headers so llama-server returns proper HTML
            async with session.get(
                f"{CONFIG['llama_server_url']}/",
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9"
                }
            ) as resp:
                if resp.status != 200:
                    return HTMLResponse(
                        content="""
                        <html>
                        <head><title>Memory Proxy - Error</title></head>
                        <body>
                            <h1>Error: llama-server not available</h1>
                            <p>Make sure llama-server is running on port 8080</p>
                            <pre>llama-server -hf model.gguf --embedding --pooling mean --port 8080</pre>
                        </body>
                        </html>
                        """,
                        status_code=503
                    )
                # aiohttp automatically decompresses gzip/deflate
                content = await resp.text()
                return HTMLResponse(content=content)
        except aiohttp.ClientError as e:
            logger.error(f"Failed to proxy Web UI: {e}")
            return HTMLResponse(
                content=f"""
                <html>
                <head><title>Memory Proxy - Connection Error</title></head>
                <body>
                    <h1>Connection Error</h1>
                    <p>Cannot connect to llama-server: {e}</p>
                </body>
                </html>
                """,
                status_code=502
            )

@app.get("/{full_path:path}")
async def proxy_static_files(full_path: str):
    """
    Proxy static files (JS, CSS, images) from llama-server.
    This is a catch-all route and must be defined LAST.
    """
    # Don't proxy our own API routes - they're already defined
    if full_path.startswith(('v1/chat/completions', 'memories/', 'health')):
        raise HTTPException(status_code=404, detail="Route handled elsewhere")
    
    async with aiohttp.ClientSession() as session:
        try:
            # Send browser-like headers for proper content negotiation
            async with session.get(
                f"{CONFIG['llama_server_url']}/{full_path}",
                allow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "*/*"
                }
            ) as resp:
                if resp.status == 404:
                    raise HTTPException(status_code=404, detail="Not found")
                
                # aiohttp automatically decompresses gzip/deflate
                content = await resp.read()
                
                # Pass through relevant headers
                # Exclude content-encoding since aiohttp decompressed
                headers = {
                    k: v for k, v in resp.headers.items()
                    if k.lower() in ['content-type', 'cache-control', 'etag', 'last-modified']
                    and k.lower() != 'content-encoding'
                }
                
                return Response(
                    content=content,
                    status_code=resp.status,
                    headers=headers
                )
        except aiohttp.ClientError as e:
            logger.error(f"Error proxying {full_path}: {e}")
            raise HTTPException(status_code=502, detail=f"llama-server unavailable: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=CONFIG['proxy_port'])
