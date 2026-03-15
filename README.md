# localgrep

**Local Semantic Code Search**

> Search your codebase by meaning, not just keywords. Powered by local embeddings (Ollama).

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

---

## Why localgrep?

Traditional search tools like `grep` and `ripgrep` require you to know the **exact keywords**. But when exploring unfamiliar code, you often think in concepts, not identifiers.

```bash
# With grep, you have to guess the right keyword:
grep -r "auth"           # maybe it's "authenticate"?
grep -r "middleware"     # or "interceptor"? "handler"?
grep -r "session"        # could be any of these...

# With localgrep, just describe what you're looking for:
localgrep search "authentication middleware"
# => src/auth/middleware.py (score: 0.87)
```

**Key advantages:**

- **Semantic understanding** -- finds code by meaning, not pattern matching
- **100% local** -- no cloud, no API keys, no data leaves your machine
- **Free forever** -- uses Ollama with open-source embedding models
- **Claude Code integration** -- works as an MCP server for AI coding agents

---

## Quick Start

```bash
# 1. Install
pip install localgrep

# 2. Pull the embedding model
ollama pull nomic-embed-text

# 3. Index your project and search
localgrep index .
localgrep search "database connection pooling"
```

That's it. Three commands to semantic search your entire codebase.

---

## Claude Code Integration

### One-line setup

```bash
localgrep install-claude
```

This automatically configures Claude Code's MCP settings and adds a search strategy guide to `CLAUDE.md`.

### Manual setup

Add to `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "localgrep": {
      "command": "localgrep",
      "args": ["serve"],
      "env": {
        "OLLAMA_HOST": "http://localhost:11434"
      }
    }
  }
}
```

Add to `~/.claude/CLAUDE.md`:

```markdown
## Code Search Strategy

1. When you know the exact keyword/symbol -> use Grep / Glob
2. When searching by concept or functionality -> use semantic_search
3. Score >= 0.7: high confidence, Score 0.3-0.7: reference, Score < 0.3: ignore
4. Always index before first search: run the reindex tool if index_status shows no index
```

---

## Features

- **Semantic search** -- natural language queries against your codebase
- **MCP server** -- seamless Claude Code integration via `localgrep serve`
- **Web dashboard** -- visual analytics at `localgrep dashboard`
- **Incremental indexing** -- only re-indexes changed files
- **Token usage analytics** -- track and compare semantic vs grep search efficiency
- **.gitignore aware** -- respects your ignore rules automatically
- **Multiple project support** -- manage indexes for different codebases

---

## Commands

| Command | Description |
|---------|-------------|
| `localgrep index [PATH]` | Index a project directory |
| `localgrep index --full [PATH]` | Force full re-indexing |
| `localgrep search "query"` | Semantic search |
| `localgrep search "query" -k 10` | Return top 10 results |
| `localgrep search "query" -t 0.5` | Minimum similarity threshold 0.5 |
| `localgrep search "query" -g "*.py"` | Filter by file pattern |
| `localgrep search "query" --json` | JSON output |
| `localgrep status` | Show index status |
| `localgrep config` | Show current configuration |
| `localgrep serve` | Start MCP server (stdio) |
| `localgrep dashboard` | Start web dashboard (http://localhost:8585) |
| `localgrep install-claude` | Configure Claude Code integration |
| `localgrep watch [PATH]` | Watch for file changes (coming soon) |

---

## Dashboard

The built-in web dashboard provides visual analytics for your search usage.

```bash
localgrep dashboard
# Open http://localhost:8585
```

<!-- TODO: Add screenshot -->
<!-- ![Dashboard Screenshot](docs/images/dashboard.png) -->

---

## Configuration

Each project can have its own configuration at `.localgrep/config.json`:

```json
{
  "ollama": {
    "host": "http://localhost:11434",
    "model": "nomic-embed-text"
  },
  "indexing": {
    "ignore": [
      "node_modules", ".git", "dist", "build",
      "__pycache__", ".venv", "*.lock"
    ],
    "max_file_size_kb": 512,
    "extensions": null
  },
  "chunking": {
    "max_lines": 100,
    "overlap_lines": 10,
    "min_lines": 3
  },
  "search": {
    "default_top_k": 5,
    "default_threshold": 0.3
  }
}
```

---

## How it Works

```
                        localgrep architecture

  ┌──────────┐       ┌───────────┐       ┌────────────┐
  │  CLI /   │──────>│  Indexer   │──────>│  Ollama    │
  │  MCP     │       │           │       │  Embedder  │
  └──────────┘       └─────┬─────┘       └────────────┘
                           │
                    ┌──────▼──────┐
                    │  SQLite +   │
                    │  sqlite-vec │
                    └──────┬──────┘
                           │
  ┌──────────┐       ┌─────▼─────┐
  │  Search  │<──────│  Vector   │
  │  Results │       │  Store    │
  └──────────┘       └───────────┘
```

**Pipeline:**

1. **File Crawling** -- walks the project tree, respects `.gitignore`
2. **Chunking** -- splits files into meaningful code chunks (function/class boundaries or sliding window)
3. **Embedding** -- generates vector embeddings via Ollama (`nomic-embed-text`)
4. **Vector Store** -- stores embeddings in SQLite with `sqlite-vec` extension
5. **Search** -- encodes your query, finds nearest vectors by cosine similarity

---

## Requirements

- **Python 3.11+**
- **[Ollama](https://ollama.ai)** -- must be installed and running locally

```bash
# Install Ollama (macOS)
brew install ollama

# Or download from https://ollama.ai

# Start Ollama and pull the model
ollama serve    # if not already running
ollama pull nomic-embed-text
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
