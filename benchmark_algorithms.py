"""검색 알고리즘 비교 벤치마크.

1. Pure Vector (현재) - 코사인 유사도만
2. Hybrid (BM25 + Vector) - 키워드 + 시맨틱 합산
3. Vector + Reranking - 1차 검색 후 쿼리-청크 교차점수로 리랭킹
"""

import asyncio
import hashlib
import math
import re
import time
from collections import Counter
from pathlib import Path

from localgrep.chunker import SlidingWindowChunker
from localgrep.crawler import FileCrawler
from localgrep.embedder import OllamaEmbedder
from localgrep.store import VectorStore, Chunk as StoreChunk, SearchResult

PROJECT = Path("/Users/hyundoopark/workspace/localgrep")

# mxbai-embed-large가 최고 성능이므로 이걸로 테스트
MODEL = "mxbai-embed-large"
DIM = 1024

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


# ─── BM25 구현 (간단 버전) ────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """간단한 토큰화: 소문자 + 단어 분리."""
    return re.findall(r'[a-z_][a-z0-9_]*', text.lower())


class SimpleBM25:
    """간단한 BM25 구현."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[tuple[int, list[str]]] = []  # (chunk_id, tokens)
        self.df: Counter = Counter()
        self.avgdl: float = 0
        self.N: int = 0

    def add(self, chunk_id: int, text: str) -> None:
        tokens = tokenize(text)
        self.docs.append((chunk_id, tokens))
        self.df.update(set(tokens))

    def build(self) -> None:
        self.N = len(self.docs)
        self.avgdl = sum(len(t) for _, t in self.docs) / self.N if self.N else 1

    def search(self, query: str, top_k: int = 20) -> dict[int, float]:
        """BM25 점수를 반환. {chunk_id: score}"""
        q_tokens = tokenize(query)
        scores: dict[int, float] = {}

        for chunk_id, doc_tokens in self.docs:
            tf_map = Counter(doc_tokens)
            dl = len(doc_tokens)
            score = 0.0
            for qt in q_tokens:
                if qt not in tf_map:
                    continue
                tf = tf_map[qt]
                df = self.df.get(qt, 0)
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                score += idf * tf_norm
            if score > 0:
                scores[chunk_id] = score

        # 정규화 (0-1)
        if scores:
            max_s = max(scores.values())
            if max_s > 0:
                scores = {k: v / max_s for k, v in scores.items()}

        return dict(sorted(scores.items(), key=lambda x: -x[1])[:top_k])


# ─── 리랭킹 (쿼리-문서 토큰 오버랩) ──────────────────────────────

def rerank_score(query: str, snippet: str) -> float:
    """쿼리와 스니펫의 토큰 오버랩 기반 리랭킹 점수."""
    q_tokens = set(tokenize(query))
    d_tokens = set(tokenize(snippet))
    if not q_tokens:
        return 0.0
    overlap = q_tokens & d_tokens
    # Jaccard-like score
    return len(overlap) / len(q_tokens)


# ─── 인덱싱 ──────────────────────────────────────────────────────

async def build_index() -> tuple[Path, SimpleBM25, dict[int, dict]]:
    """인덱싱 + BM25 빌드."""
    db_path = PROJECT / ".localgrep" / "bench_algo.db"
    if db_path.exists():
        db_path.unlink()

    store = VectorStore(db_path)
    store._conn.execute("DROP TABLE IF EXISTS chunks_vec")
    store._conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
        f"chunk_id INTEGER PRIMARY KEY, embedding float[{DIM}] distance_metric=cosine)"
    )
    store._conn.commit()

    embedder = OllamaEmbedder(model=MODEL)
    embedder.DIMENSION = DIM
    chunker = SlidingWindowChunker()
    crawler = FileCrawler(PROJECT, extra_ignore_patterns=["*.db", "benchmark*", ".localgrep"])
    files = [f for f in crawler.crawl() if f.relative_path.startswith("src/")]

    bm25 = SimpleBM25()
    chunk_meta: dict[int, dict] = {}  # chunk_id -> {file, start_line, end_line, content}

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

            for c, emb in zip(chunks, embeddings):
                sc = StoreChunk(c.start_line, c.end_line, c.content, emb)
                store.add_chunks(file_id, [sc])

                # 방금 추가된 chunk_id 가져오기
                cur = store._conn.cursor()
                cur.execute("SELECT MAX(id) FROM chunks WHERE file_id = ?", (file_id,))
                chunk_id = cur.fetchone()[0]

                bm25.add(chunk_id, c.embeddable_text)
                chunk_meta[chunk_id] = {
                    "file": fi.relative_path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "content": c.content,
                }

    bm25.build()
    store.close()
    return db_path, bm25, chunk_meta


# ─── 알고리즘별 검색 ─────────────────────────────────────────────

async def search_pure_vector(store, embedder, query, top_k=5):
    """알고리즘 1: Pure Vector (현재)."""
    q_emb = await embedder.embed(query)
    return store.search(q_emb, top_k=top_k, threshold=0.0)


async def search_hybrid(store, embedder, bm25, chunk_meta, query, top_k=5, alpha=0.7):
    """알고리즘 2: Hybrid (Vector * alpha + BM25 * (1-alpha))."""
    q_emb = await embedder.embed(query)
    vec_results = store.search(q_emb, top_k=20, threshold=0.0)
    bm25_scores = bm25.search(query, top_k=20)

    # 벡터 결과를 chunk_id로 매핑
    combined: dict[int, float] = {}

    # 벡터 점수
    cur = store._conn.cursor()
    for r in vec_results:
        cur.execute(
            "SELECT id FROM chunks WHERE file_id = (SELECT id FROM files WHERE path = ?) "
            "AND start_line = ? AND end_line = ?",
            (r.file, r.start_line, r.end_line),
        )
        row = cur.fetchone()
        if row:
            combined[row[0]] = alpha * r.score

    # BM25 점수 합산
    for chunk_id, bm25_score in bm25_scores.items():
        combined[chunk_id] = combined.get(chunk_id, 0) + (1 - alpha) * bm25_score

    # 상위 K개
    sorted_ids = sorted(combined.items(), key=lambda x: -x[1])[:top_k]
    results = []
    for chunk_id, score in sorted_ids:
        meta = chunk_meta.get(chunk_id)
        if meta:
            results.append(SearchResult(
                file=meta["file"],
                start_line=meta["start_line"],
                end_line=meta["end_line"],
                score=round(score, 4),
                snippet=meta["content"],
            ))
    return results


async def search_vector_rerank(store, embedder, query, top_k=5):
    """알고리즘 3: Vector + Reranking."""
    q_emb = await embedder.embed(query)
    # 1차: 넓게 가져오기
    candidates = store.search(q_emb, top_k=20, threshold=0.0)

    # 2차: 리랭킹
    reranked = []
    for r in candidates:
        rr_score = rerank_score(query, r.snippet)
        combined = 0.6 * r.score + 0.4 * rr_score
        reranked.append(SearchResult(
            file=r.file,
            start_line=r.start_line,
            end_line=r.end_line,
            score=round(combined, 4),
            snippet=r.snippet,
        ))

    reranked.sort(key=lambda x: -x.score)
    return reranked[:top_k]


# ─── 평가 ────────────────────────────────────────────────────────

def evaluate(results: list[SearchResult], expected_files: list[str]) -> tuple[bool, bool, bool]:
    """Hit@1, Hit@3, Hit@5 반환."""
    result_files = [Path(r.file).name for r in results]
    hit_1 = any(ef in result_files[:1] for ef in expected_files)
    hit_3 = any(ef in result_files[:3] for ef in expected_files)
    hit_5 = any(ef in result_files[:5] for ef in expected_files)
    return hit_1, hit_3, hit_5


async def main():
    print("=" * 90)
    print(f"ALGORITHM BENCHMARK (model: {MODEL}, dim: {DIM})")
    print(f"Test cases: {len(TEST_CASES)} queries against localgrep src/")
    print("=" * 90)
    print()

    print("Building index + BM25...", end=" ", flush=True)
    db_path, bm25, chunk_meta = await build_index()
    print("done\n")

    algorithms = {
        "Pure Vector (cosine)": "vector",
        "Hybrid (0.7*vec + 0.3*BM25)": "hybrid",
        "Vector + Rerank (token overlap)": "rerank",
    }

    all_results = {}
    store = VectorStore(db_path)
    embedder = OllamaEmbedder(model=MODEL)
    embedder.DIMENSION = DIM

    async with embedder:
        for algo_name, algo_type in algorithms.items():
            h1 = h3 = h5 = 0
            total_time = 0.0

            for query, expected in TEST_CASES:
                start = time.monotonic()
                if algo_type == "vector":
                    results = await search_pure_vector(store, embedder, query)
                elif algo_type == "hybrid":
                    results = await search_hybrid(store, embedder, bm25, chunk_meta, query)
                elif algo_type == "rerank":
                    results = await search_vector_rerank(store, embedder, query)
                elapsed = time.monotonic() - start
                total_time += elapsed

                hit_1, hit_3, hit_5 = evaluate(results, expected)
                h1 += int(hit_1)
                h3 += int(hit_3)
                h5 += int(hit_5)

            n = len(TEST_CASES)
            all_results[algo_name] = {
                "hit1": h1, "hit3": h3, "hit5": h5,
                "avg_ms": round(total_time / n * 1000),
            }
            print(f"{algo_name}")
            print(f"  Hit@1: {h1}/{n} ({h1/n*100:.0f}%)  Hit@3: {h3}/{n} ({h3/n*100:.0f}%)  Hit@5: {h5}/{n} ({h5/n*100:.0f}%)")
            print(f"  Avg latency: {all_results[algo_name]['avg_ms']}ms")
            print()

    store.close()
    db_path.unlink(missing_ok=True)

    # 결과 테이블
    print("=" * 90)
    n = len(TEST_CASES)
    print(f"{'Algorithm':<40} {'Hit@1':>10} {'Hit@3':>10} {'Hit@5':>10} {'Latency':>10}")
    print("-" * 90)
    for name, r in all_results.items():
        print(f"{name:<40} {r['hit1']}/{n} ({r['hit1']/n*100:.0f}%){'':<2} {r['hit3']}/{n} ({r['hit3']/n*100:.0f}%){'':<2} {r['hit5']}/{n} ({r['hit5']/n*100:.0f}%){'':<2} {r['avg_ms']}ms")
    print("=" * 90)


if __name__ == "__main__":
    asyncio.run(main())
