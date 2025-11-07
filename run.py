#!/usr/bin/env python3
"""
Memory Proxy Launcher
Run this to start the memory proxy server
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run
from memory_proxy import app
import uvicorn

if __name__ == "__main__":
    import json
    
    # Load config
    with open('config.json') as f:
        config = json.load(f)
    
    print(f"Starting Memory Proxy on port {config['proxy_port']}...")
    print(f"Connecting to llama-server at {config['llama_server_url']}")
    
    uvicorn.run(app, host="0.0.0.0", port=config['proxy_port'])
