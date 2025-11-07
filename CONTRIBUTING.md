# Contributing to Memory Proxy

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/lightweight-context-assembly.git
   cd lightweight-context-assembly
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Copy example config**
   ```bash
   cp config.example.json config.json
   ```

4. **Run tests**
   ```bash
   python3 tests/test_offline.py
   ```

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for functions and classes
- Keep functions focused and small
- Add comments only where necessary

## Testing

Before submitting a PR:

1. **Run offline tests**
   ```bash
   python3 tests/test_offline.py
   ```

2. **Test with llama-server** (if applicable)
   ```bash
   # Start llama-server
   llama-server -hf ggml-org/gemma-3-1b-it-GGUF --embedding --pooling mean --port 8080
   
   # Start proxy
   python3 run.py
   
   # Run integration tests
   python3 tests/test_simple.py
   python3 examples/test_conversation.py
   ```

3. **Check code compiles**
   ```bash
   python3 -m py_compile src/memory_proxy.py
   ```

## Project Structure

```
src/          # Main source code
tests/        # Test files
examples/     # Example scripts
docs/         # Documentation
scripts/      # Helper scripts
```

## Submitting Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear commit messages
   - Keep commits focused and atomic
   - Test your changes

3. **Update documentation**
   - Update README.md if adding features
   - Add examples if applicable
   - Update CHANGES.md

4. **Submit a pull request**
   - Describe what you changed and why
   - Reference any related issues
   - Ensure all tests pass

## Commit Message Format

```
type: brief description

Longer description if needed

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

## Areas for Contribution

- **Memory decay**: Reduce weight of old/unused memories
- **FAISS backend**: Support for >100k memories
- **Batch embeddings**: Reduce latency
- **Memory visualization**: Web UI for memory management
- **Multi-agent support**: Isolate memories per agent
- **Performance optimizations**: Speed improvements
- **Documentation**: Improve guides and examples
- **Tests**: Expand test coverage

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Questions about the codebase
- Documentation clarifications

## Code of Conduct

Be respectful and constructive in all interactions. This is an inclusive project open to contributors of all skill levels.
