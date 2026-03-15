"""Semantic search vs grep benchmark on localgrep's own codebase."""

import asyncio
import subprocess
import time
import json
from pathlib import Path

from localgrep.embedder import OllamaEmbedder
from localgrep.store import VectorStore

PROJECT = Path("/Users/hyundoopark/workspace/localgrep")
DB = PROJECT / ".localgrep" / "index.db"

# 10 realistic queries a developer might ask
QUERIES = [
    ("vector similarity search implementation", "vector|similarity|cosine|distance"),
    ("how does file crawling work with gitignore", "gitignore|crawl|ignore|walk"),
    ("embedding generation from text", "embed|embedding|ollama|vector"),
    ("chunking strategy for source code", "chunk|split|window|overlap"),
    ("MCP server tool definitions", "mcp|tool|server|semantic_search"),
    ("database schema and migrations", "schema|CREATE TABLE|migrate|sqlite"),
    ("CLI command line interface", "typer|command|app.command|cli"),
    ("configuration file loading", "config|load_config|save_config|json"),
    ("incremental indexing changed files", "incremental|changed|upsert|mtime|hash"),
    ("search result scoring and ranking", "score|threshold|top_k|ranking|search"),
]


async def run_semantic(store, embedder, query, top_k=5):
    """Run semantic search and measure tokens."""
    start = time.monotonic()
    q_emb = await embedder.embed(query)
    results = store.search(q_emb, top_k=top_k, threshold=0.0)
    elapsed = (time.monotonic() - start) * 1000

    total_chars = sum(len(r.snippet) for r in results)
    return {
        "results": len(results),
        "total_chars": total_chars,
        "est_tokens": total_chars // 4,
        "time_ms": round(elapsed),
        "top_score": round(results[0].score, 3) if results else 0,
        "files": [f"{r.file}:{r.start_line}-{r.end_line}" for r in results],
    }


def run_grep(grep_pattern, top_k=5):
    """Run ripgrep and measure tokens."""
    start = time.monotonic()
    try:
        result = subprocess.run(
            ["rg", "-i", "--no-heading", "-C", "2", grep_pattern,
             str(PROJECT / "src"), str(PROJECT / "tests")],
            capture_output=True, text=True, timeout=10,
        )
        output = result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        output = ""
    elapsed = (time.monotonic() - start) * 1000

    lines = output.strip().split("\n") if output.strip() else []
    total_chars = len(output)

    return {
        "results": len([l for l in lines if l and not l.startswith("--")]),
        "total_chars": total_chars,
        "est_tokens": total_chars // 4,
        "time_ms": round(elapsed),
        "lines": len(lines),
    }


async def main():
    store = VectorStore(DB)
    async with OllamaEmbedder() as embedder:
        print("=" * 80)
        print("BENCHMARK: Semantic Search vs Grep (localgrep codebase, 21 files)")
        print("=" * 80)
        print()

        sem_total_tokens = 0
        grep_total_tokens = 0
        sem_total_time = 0
        grep_total_time = 0

        rows = []
        for i, (sem_query, grep_pattern) in enumerate(QUERIES, 1):
            sem = await run_semantic(store, embedder, sem_query)
            grp = run_grep(grep_pattern)

            sem_total_tokens += sem["est_tokens"]
            grep_total_tokens += grp["est_tokens"]
            sem_total_time += sem["time_ms"]
            grep_total_time += grp["time_ms"]

            rows.append({
                "query": sem_query,
                "grep_pattern": grep_pattern,
                "semantic": sem,
                "grep": grp,
            })

            print(f"Query {i}: {sem_query}")
            print(f"  Semantic: {sem['results']} results, ~{sem['est_tokens']} tokens, {sem['time_ms']}ms, top={sem['top_score']}")
            print(f"  Grep:     {grp['results']} results, ~{grp['est_tokens']} tokens, {grp['time_ms']}ms")
            if grp["est_tokens"] > 0:
                savings = (1 - sem["est_tokens"] / grp["est_tokens"]) * 100
                print(f"  Token savings: {savings:.0f}%")
            print()

        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"  Queries:          {len(QUERIES)}")
        print(f"  Semantic total:   ~{sem_total_tokens} tokens, {sem_total_time}ms")
        print(f"  Grep total:       ~{grep_total_tokens} tokens, {grep_total_time}ms")
        if grep_total_tokens > 0:
            overall_savings = (1 - sem_total_tokens / grep_total_tokens) * 100
            print(f"  Token savings:    {overall_savings:.1f}%")
        print(f"  Avg semantic:     ~{sem_total_tokens // len(QUERIES)} tokens/query")
        print(f"  Avg grep:         ~{grep_total_tokens // len(QUERIES)} tokens/query")

        # Output JSON for README
        summary = {
            "queries": len(QUERIES),
            "project_files": 21,
            "semantic_total_tokens": sem_total_tokens,
            "grep_total_tokens": grep_total_tokens,
            "token_savings_pct": round((1 - sem_total_tokens / grep_total_tokens) * 100, 1) if grep_total_tokens > 0 else 0,
            "semantic_avg_tokens": sem_total_tokens // len(QUERIES),
            "grep_avg_tokens": grep_total_tokens // len(QUERIES),
            "semantic_avg_ms": sem_total_time // len(QUERIES),
            "grep_avg_ms": grep_total_time // len(QUERIES),
        }
        with open(PROJECT / "benchmark_results.json", "w") as f:
            json.dump({"summary": summary, "details": rows}, f, indent=2, default=str)
        print(f"\nResults saved to benchmark_results.json")

    store.close()


if __name__ == "__main__":
    asyncio.run(main())
