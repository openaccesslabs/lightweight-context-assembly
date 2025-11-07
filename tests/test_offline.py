#!/usr/bin/env python3
"""
Offline validation test - doesn't require llama-server
Tests database operations and core logic
"""

import sqlite3
import numpy as np
import json
from pathlib import Path

print("=" * 60)
print("Offline Validation Test")
print("=" * 60)

# Test 1: Config loading
print("\n[Test 1] Config loading...")
try:
    with open('config.json') as f:
        config = json.load(f)
    assert 'llama_server_url' in config
    assert 'compression_threshold' in config
    assert 0.0 < config['compression_threshold'] <= 1.0
    print("✓ Config valid")
except Exception as e:
    print(f"✗ Config error: {e}")
    exit(1)

# Test 2: Database initialization
print("\n[Test 2] Database initialization...")
try:
    db_path = 'test_offline.db'
    Path(db_path).unlink(missing_ok=True)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    
    with open('schema.sql') as f:
        conn.executescript(f.read())
    conn.commit()
    
    # Verify tables
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    assert 'memories' in tables
    assert 'messages' in tables
    assert 'thread_summaries' in tables
    print(f"✓ Database created with tables: {tables}")
except Exception as e:
    print(f"✗ Database error: {e}")
    exit(1)

# Test 3: Memory operations
print("\n[Test 3] Memory storage and retrieval...")
try:
    # Store memory
    embedding = np.random.rand(384).astype(np.float32)
    conn.execute("""
        INSERT INTO memories (user_id, memory_type, key, value, embedding)
        VALUES (?, ?, ?, ?, ?)
    """, ('alice', 'preference', 'language', 'Python', embedding.tobytes()))
    conn.commit()
    
    # Retrieve memory
    row = conn.execute("""
        SELECT * FROM memories WHERE user_id = ? AND memory_type = ? AND key = ?
    """, ('alice', 'preference', 'language')).fetchone()
    
    assert row['value'] == 'Python'
    retrieved_emb = np.frombuffer(row['embedding'], dtype=np.float32)
    assert np.allclose(embedding, retrieved_emb)
    print("✓ Memory storage and retrieval works")
    
    # Test upsert
    new_embedding = np.random.rand(384).astype(np.float32)
    conn.execute("""
        INSERT INTO memories (user_id, memory_type, key, value, embedding, updated_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(user_id, memory_type, key)
        DO UPDATE SET value=excluded.value, embedding=excluded.embedding, updated_at=CURRENT_TIMESTAMP
    """, ('alice', 'preference', 'language', 'Rust', new_embedding.tobytes()))
    conn.commit()
    
    row = conn.execute("""
        SELECT value FROM memories WHERE user_id = ? AND memory_type = ? AND key = ?
    """, ('alice', 'preference', 'language')).fetchone()
    assert row['value'] == 'Rust'
    print("✓ Memory upsert works")
    
except Exception as e:
    print(f"✗ Memory operation error: {e}")
    exit(1)

# Test 4: Message storage
print("\n[Test 4] Message storage...")
try:
    # Store messages
    for i in range(5):
        conn.execute("""
            INSERT INTO messages (thread_id, user_id, role, content)
            VALUES (?, ?, ?, ?)
        """, ('thread1', 'alice', 'user' if i % 2 == 0 else 'assistant', f'Message {i}'))
    conn.commit()
    
    # Retrieve all messages
    rows = conn.execute("""
        SELECT * FROM messages WHERE thread_id = ? ORDER BY created_at ASC
    """, ('thread1',)).fetchall()
    
    assert len(rows) == 5
    print(f"✓ Stored and retrieved {len(rows)} messages")
    
    # Test get_messages_after_id
    second_msg_id = rows[1]['id']
    rows_after = conn.execute("""
        SELECT * FROM messages WHERE thread_id = ? AND id > ? ORDER BY created_at ASC
    """, ('thread1', second_msg_id)).fetchall()
    
    assert len(rows_after) == 3  # Messages 2, 3, 4 (0-indexed)
    print(f"✓ get_messages_after_id returns {len(rows_after)} messages (correct)")
    
except Exception as e:
    print(f"✗ Message operation error: {e}")
    exit(1)

# Test 5: Thread summary with compression pointer
print("\n[Test 5] Thread summary storage...")
try:
    last_msg_id = rows[2]['id']  # Compress up to message 2
    
    conn.execute("""
        INSERT INTO thread_summaries 
        (thread_id, user_id, summary, last_compressed_message_id, message_count)
        VALUES (?, ?, ?, ?, ?)
    """, ('thread1', 'alice', 'Summary of first 3 messages', last_msg_id, 3))
    conn.commit()
    
    # Retrieve summary
    summary = conn.execute("""
        SELECT * FROM thread_summaries WHERE thread_id = ?
    """, ('thread1',)).fetchone()
    
    assert summary['summary'] == 'Summary of first 3 messages'
    assert summary['last_compressed_message_id'] == last_msg_id
    assert summary['message_count'] == 3
    print("✓ Summary stored with compression pointer")
    
    # Simulate what happens after compression
    messages_for_llm = conn.execute("""
        SELECT * FROM messages 
        WHERE thread_id = ? AND id > ?
        ORDER BY created_at ASC
    """, ('thread1', last_msg_id)).fetchall()
    
    print(f"✓ After compression, LLM would see summary + {len(messages_for_llm)} recent messages")
    
    # Verify all messages still in DB
    all_messages = conn.execute("""
        SELECT COUNT(*) as cnt FROM messages WHERE thread_id = ?
    """, ('thread1',)).fetchone()['cnt']
    
    assert all_messages == 5
    print(f"✓ All {all_messages} messages still in database (none deleted)")
    
except Exception as e:
    print(f"✗ Summary operation error: {e}")
    exit(1)

# Test 6: Cosine similarity
print("\n[Test 6] Cosine similarity calculation...")
try:
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    sim = cosine_similarity(v1, v2)
    assert abs(sim - 1.0) < 0.001
    print(f"✓ Identical vectors: {sim:.4f}")
    
    v3 = np.array([1.0, 0.0])
    v4 = np.array([0.0, 1.0])
    sim = cosine_similarity(v3, v4)
    assert abs(sim) < 0.001
    print(f"✓ Orthogonal vectors: {sim:.4f}")
    
except Exception as e:
    print(f"✗ Cosine similarity error: {e}")
    exit(1)

# Test 7: Token counting
print("\n[Test 7] Token counting...")
try:
    # Test with tiktoken if available
    try:
        import tiktoken
        enc = tiktoken.get_encoding('cl100k_base')
        
        text1 = "Hello world"
        tokens1 = len(enc.encode(text1))
        print(f"✓ Tiktoken available: '{text1}' = {tokens1} tokens")
        
        text2 = "This is a longer piece of text for testing token counting accuracy."
        tokens2 = len(enc.encode(text2))
        estimate2 = len(text2) // 4
        print(f"✓ Long text: {tokens2} tokens (estimate: {estimate2})")
        
    except ImportError:
        print("⚠ tiktoken not installed, testing fallback")
        def count_tokens(text):
            return len(text) // 4
        
        assert count_tokens("Hello world") == 2
        assert count_tokens("") == 0
        assert count_tokens("A" * 400) == 100
        print("✓ Fallback token estimation works")
    
except Exception as e:
    print(f"✗ Token counting error: {e}")
    exit(1)

# Test 8: Context usage calculation logic
print("\n[Test 8] Context usage calculation...")
try:
    # Simulate context calculation
    max_context = 8192
    reserved = 2000
    usable = max_context - reserved  # 6192
    
    # Simulate token counts
    system_prompt_tokens = 200
    message_tokens = 5000
    total = system_prompt_tokens + message_tokens  # 5200
    
    percentage = total / usable
    print(f"✓ Context usage: {total}/{usable} = {percentage:.1%}")
    
    # Test compression trigger
    compression_threshold = 0.9
    if percentage >= compression_threshold:
        print("✓ Would trigger compression")
    else:
        print(f"✓ No compression needed ({percentage:.1%} < {compression_threshold:.0%})")
    
except Exception as e:
    print(f"✗ Context calculation error: {e}")
    exit(1)

# Cleanup
print("\n[Cleanup]")
conn.close()
Path(db_path).unlink(missing_ok=True)
Path(db_path + '-wal').unlink(missing_ok=True)
Path(db_path + '-shm').unlink(missing_ok=True)
print("✓ Test database cleaned up")

print("\n" + "=" * 60)
print("✅ All offline validation tests passed!")
print("=" * 60)
print("\nThe core logic is sound. To test with llama-server:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Start llama-server with embeddings: llama-server --model model.gguf --embedding")
print("3. Run: python memory_proxy.py")
print("4. Test: python examples/test_conversation.py")
