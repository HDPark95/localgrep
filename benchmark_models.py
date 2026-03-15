"""임베딩 모델별 코드 검색 품질 비교 벤치마크.

localgrep 코드베이스에서 쿼리→기대 파일 매칭 정확도를 측정한다.
"""

import asyncio
import hashlib
import time
from pathlib import Path

from localgrep.chunker import SlidingWindowChunker
from localgrep.crawler import FileCrawler
from localgrep.embedder import OllamaEmbedder
from localgrep.store import VectorStore, Chunk as StoreChunk

PROJECT = Path("/Users/hyundoopark/workspace/localgrep")

# 모델별 차원 수
MODELS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "snowflake-arctic-embed": 1024,
    "rjmalagon/gte-qwen2-1.5b-instruct-embed-f16": 1536,
}

# 쿼리 → 기대하는 파일 (정답)
# 정답 파일이 top-K 안에 들어오면 hit
TEST_CASES = [
    ("vector cosine similarity search implementation", ["store.py"]),
    ("file directory traversal with gitignore filtering", ["crawler.py"]),
    ("generate embeddings from text using ollama", ["embedder.py"]),
    ("split source code into chunks sliding window", ["chunker.py"]),
    ("MCP server tool definition for claude code", ["server.py"]),
    ("sqlite database schema create table migration", ["store.py"]),
    ("typer CLI command line argument parsing", ["cli.py"]),
    ("load save configuration json dataclass", ["config.py"]),
    ("incremental indexing detect changed files hash", ["crawler.py", "store.py"]),
    ("async http client request error handling", ["embedder.py"]),
    ("web dashboard fastapi uvicorn html template", ["dashboard.py"]),
    ("search result ranking score threshold filtering", ["store.py"]),
    ("batch embedding multiple texts at once", ["embedder.py"]),
    ("project file statistics count chunks", ["store.py"]),
    ("binary file detection text file filtering", ["crawler.py"]),
]


async def index_with_model(model_name: str, dim: int) -> Path:
    """특정 모델로 인덱싱하고 DB 경로를 반환."""
    db_path = PROJECT / ".localgrep" / f"bench_{model_name.replace('/', '_')}.db"
    if db_path.exists():
        db_path.unlink()

    store = VectorStore(db_path)
    # 차원 수가 다르면 vec 테이블 재생성
    if dim != 768:
        store._conn.execute("DROP TABLE IF EXISTS chunks_vec")
        store._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
            f"chunk_id INTEGER PRIMARY KEY, embedding float[{dim}] distance_metric=cosine)"
        )
        store._conn.commit()

    embedder = OllamaEmbedder(model=model_name)
    # 차원 수 오버라이드
    embedder.DIMENSION = dim
    chunker = SlidingWindowChunker()
    crawler = FileCrawler(PROJECT, extra_ignore_patterns=["*.db", "benchmark*", ".localgrep"])

    files = crawler.crawl()
    # src/ 파일만 인덱싱 (테스트 정답이 src/ 기준)
    files = [f for f in files if f.relative_path.startswith("src/")]

    async with embedder:
        for fi in files:
            content = fi.path.read_text(encoding="utf-8", errors="replace")
            h = hashlib.sha256(content.encode()).hexdigest()
            file_id = store.upsert_file(fi.relative_path, fi.mtime, h)

            chunks = chunker.chunk(fi.relative_path, content)
            if not chunks:
                continue

            texts = [c.embeddable_text for c in chunks]
            embeddings = await embedder.embed_batch(texts)

            store_chunks = [
                StoreChunk(c.start_line, c.end_line, c.content, emb)
                for c, emb in zip(chunks, embeddings)
            ]
            store.add_chunks(file_id, store_chunks)

    store.close()
    return db_path


async def evaluate_model(model_name: str, dim: int, db_path: Path) -> dict:
    """모델의 검색 품질을 평가."""
    store = VectorStore(db_path)
    if dim != 768:
        store._conn.execute("DROP TABLE IF EXISTS chunks_vec")
        store._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
            f"chunk_id INTEGER PRIMARY KEY, embedding float[{dim}] distance_metric=cosine)"
        )
        # 기존 chunks에서 재삽입
        import struct
        rows = store._conn.execute("SELECT id, embedding FROM chunks").fetchall()
        for chunk_id, emb_blob in rows:
            store._conn.execute(
                "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, emb_blob),
            )
        store._conn.commit()

    embedder = OllamaEmbedder(model=model_name)
    embedder.DIMENSION = dim

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    total_score = 0.0
    total_time = 0.0

    async with embedder:
        for query, expected_files in TEST_CASES:
            start = time.monotonic()
            q_emb = await embedder.embed(query)
            results = store.search(q_emb, top_k=5, threshold=0.0)
            elapsed = time.monotonic() - start
            total_time += elapsed

            result_files = [Path(r.file).name for r in results]
            top_score = results[0].score if results else 0

            # Hit 판정: 기대 파일 중 하나라도 결과에 있으면 hit
            hit_1 = any(ef in result_files[:1] for ef in expected_files)
            hit_3 = any(ef in result_files[:3] for ef in expected_files)
            hit_5 = any(ef in result_files[:5] for ef in expected_files)

            hits_at_1 += int(hit_1)
            hits_at_3 += int(hit_3)
            hits_at_5 += int(hit_5)
            total_score += top_score

    store.close()

    n = len(TEST_CASES)
    return {
        "model": model_name,
        "dimension": dim,
        "hit@1": f"{hits_at_1}/{n} ({hits_at_1/n*100:.0f}%)",
        "hit@3": f"{hits_at_3}/{n} ({hits_at_3/n*100:.0f}%)",
        "hit@5": f"{hits_at_5}/{n} ({hits_at_5/n*100:.0f}%)",
        "avg_top_score": round(total_score / n, 3),
        "avg_latency_ms": round(total_time / n * 1000),
        "hit1_raw": hits_at_1,
        "hit3_raw": hits_at_3,
        "hit5_raw": hits_at_5,
    }


async def main():
    print("=" * 90)
    print("EMBEDDING MODEL BENCHMARK: Code Search Quality")
    print(f"Project: localgrep src/ | Test cases: {len(TEST_CASES)} queries")
    print("=" * 90)
    print()

    results = []
    for model_name, dim in MODELS.items():
        print(f"[{model_name}] (dim={dim})")
        print(f"  Indexing...", end=" ", flush=True)
        db_path = await index_with_model(model_name, dim)
        print("done")

        print(f"  Evaluating...", end=" ", flush=True)
        # vec 테이블 다시 생성 필요 — 깔끔하게 재인덱싱
        result = await evaluate_model(model_name, dim, db_path)
        print("done")

        print(f"  Hit@1: {result['hit@1']}  Hit@3: {result['hit@3']}  Hit@5: {result['hit@5']}")
        print(f"  Avg top score: {result['avg_top_score']}  Avg latency: {result['avg_latency_ms']}ms")
        print()
        results.append(result)

        # 임시 DB 정리
        db_path.unlink(missing_ok=True)

    # 결과 테이블
    print("=" * 90)
    print(f"{'Model':<50} {'Hit@1':>8} {'Hit@3':>8} {'Hit@5':>8} {'AvgScore':>9} {'Latency':>8}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: -x["hit5_raw"]):
        print(f"{r['model']:<50} {r['hit@1']:>8} {r['hit@3']:>8} {r['hit@5']:>8} {r['avg_top_score']:>9} {r['avg_latency_ms']:>6}ms")
    print("=" * 90)


if __name__ == "__main__":
    asyncio.run(main())
