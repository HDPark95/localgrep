"""임베딩 전략 비교 벤치마크.

1. Baseline - 현재 방식 (파일경로:줄번호 + 원본 코드)
2. Context-enriched - import문 + 클래스/함수 시그니처 추가
3. HyDE - 쿼리를 가상 코드로 변환 후 임베딩
4. Hierarchical - 파일 요약 먼저 매칭 → 해당 파일 청크만 검색
5. AST chunking - 함수/클래스 단위 분할
"""

import asyncio
import hashlib
import ast
import re
import time
from pathlib import Path
from dataclasses import dataclass

from localgrep.chunker import SlidingWindowChunker, Chunk
from localgrep.crawler import FileCrawler
from localgrep.embedder import OllamaEmbedder
from localgrep.store import VectorStore, Chunk as StoreChunk, SearchResult

PROJECT = Path("/Users/hyundoopark/workspace/localgrep")
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


# ═══════════════════════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════════════════════

def _make_store(name: str) -> tuple[VectorStore, Path]:
    db_path = PROJECT / ".localgrep" / f"bench_{name}.db"
    if db_path.exists():
        db_path.unlink()
    store = VectorStore(db_path)
    store._conn.execute("DROP TABLE IF EXISTS chunks_vec")
    store._conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
        f"chunk_id INTEGER PRIMARY KEY, embedding float[{DIM}] distance_metric=cosine)"
    )
    store._conn.commit()
    return store, db_path


def _get_src_files() -> list:
    crawler = FileCrawler(PROJECT, extra_ignore_patterns=["*.db", "benchmark*", ".localgrep"])
    return [f for f in crawler.crawl() if f.relative_path.startswith("src/")]


def _evaluate(results: list[SearchResult], expected: list[str]) -> tuple[bool, bool, bool]:
    rf = [Path(r.file).name for r in results]
    return (
        any(e in rf[:1] for e in expected),
        any(e in rf[:3] for e in expected),
        any(e in rf[:5] for e in expected),
    )


# ═══════════════════════════════════════════════════════════════════
# 전략 1: Baseline (현재 방식)
# ═══════════════════════════════════════════════════════════════════

async def strategy_baseline(embedder: OllamaEmbedder):
    """현재 방식: 파일경로:줄번호 + 원본 코드."""
    store, db_path = _make_store("baseline")
    chunker = SlidingWindowChunker()
    files = _get_src_files()

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
            store.add_chunks(file_id, [StoreChunk(c.start_line, c.end_line, c.content, emb)])

    # 평가
    h1 = h3 = h5 = 0
    for query, expected in TEST_CASES:
        q_emb = await embedder.embed(query)
        results = store.search(q_emb, top_k=5, threshold=0.0)
        a, b, c = _evaluate(results, expected)
        h1 += int(a); h3 += int(b); h5 += int(c)

    store.close()
    db_path.unlink(missing_ok=True)
    return h1, h3, h5


# ═══════════════════════════════════════════════════════════════════
# 전략 2: Context-enriched 임베딩
# ═══════════════════════════════════════════════════════════════════

def _extract_python_context(content: str) -> str:
    """파이썬 파일에서 모듈 docstring, import, 클래스/함수 시그니처 추출."""
    lines = content.split("\n")
    context_parts = []

    # 모듈 docstring (첫 번째 문자열)
    try:
        tree = ast.parse(content)
        docstring = ast.get_docstring(tree)
        if docstring:
            context_parts.append(f"Module: {docstring.split(chr(10))[0]}")
    except SyntaxError:
        pass

    # import문
    imports = [l.strip() for l in lines if l.strip().startswith(("import ", "from "))]
    if imports:
        context_parts.append("Imports: " + ", ".join(imports[:10]))

    # 클래스/함수 시그니처
    signatures = []
    for l in lines:
        stripped = l.strip()
        if stripped.startswith("class ") or stripped.startswith("def ") or stripped.startswith("async def "):
            signatures.append(stripped.split(":")[0])
    if signatures:
        context_parts.append("Definitions: " + "; ".join(signatures[:15]))

    return "\n".join(context_parts)


async def strategy_context_enriched(embedder: OllamaEmbedder):
    """청크에 파일 컨텍스트(import, 시그니처, docstring) 추가."""
    store, db_path = _make_store("context")
    chunker = SlidingWindowChunker()
    files = _get_src_files()

    for fi in files:
        content = fi.path.read_text(encoding="utf-8", errors="replace")
        h = hashlib.sha256(content.encode()).hexdigest()
        file_id = store.upsert_file(fi.relative_path, fi.mtime, h)

        file_context = _extract_python_context(content)
        chunks = chunker.chunk(fi.relative_path, content)
        if not chunks:
            continue

        # 각 청크에 파일 컨텍스트를 접두사로 추가
        texts = []
        for c in chunks:
            enriched = f"{c.header}\n{file_context}\n---\n{c.content}"
            texts.append(enriched[:6000])

        embeddings = await embedder.embed_batch(texts)
        for c, emb in zip(chunks, embeddings):
            store.add_chunks(file_id, [StoreChunk(c.start_line, c.end_line, c.content, emb)])

    h1 = h3 = h5 = 0
    for query, expected in TEST_CASES:
        q_emb = await embedder.embed(query)
        results = store.search(q_emb, top_k=5, threshold=0.0)
        a, b, c = _evaluate(results, expected)
        h1 += int(a); h3 += int(b); h5 += int(c)

    store.close()
    db_path.unlink(missing_ok=True)
    return h1, h3, h5


# ═══════════════════════════════════════════════════════════════════
# 전략 3: HyDE (Hypothetical Document Embeddings)
# ═══════════════════════════════════════════════════════════════════

# HyDE: 쿼리 → "이런 코드가 있을 것" 가상 코드 생성 → 그걸 임베딩
# LLM 없이 구현: 쿼리 키워드로 가상 Python 코드 스니펫 생성

def _generate_hypothetical_code(query: str) -> str:
    """쿼리에서 가상 코드 스니펫을 생성 (LLM 없는 휴리스틱 버전)."""
    # 쿼리 키워드를 Python 코드스럽게 변환
    words = re.findall(r'[a-z]+', query.lower())
    func_name = "_".join(words[:4])
    params = ", ".join(words[4:7]) if len(words) > 4 else "data"

    hypothetical = f'''def {func_name}({params}):
    """{query}"""
    # Implementation for {query}
    {" ".join(words)}
    return result
'''
    return hypothetical


async def strategy_hyde(embedder: OllamaEmbedder):
    """HyDE: 쿼리를 가상 코드로 변환 후 임베딩."""
    # 인덱싱은 baseline과 동일
    store, db_path = _make_store("hyde")
    chunker = SlidingWindowChunker()
    files = _get_src_files()

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
            store.add_chunks(file_id, [StoreChunk(c.start_line, c.end_line, c.content, emb)])

    h1 = h3 = h5 = 0
    for query, expected in TEST_CASES:
        # HyDE: 가상 코드 생성 → 쿼리+가상코드 합쳐서 임베딩
        hypo = _generate_hypothetical_code(query)
        hyde_query = f"{query}\n\n{hypo}"
        q_emb = await embedder.embed(hyde_query)
        results = store.search(q_emb, top_k=5, threshold=0.0)
        a, b, c = _evaluate(results, expected)
        h1 += int(a); h3 += int(b); h5 += int(c)

    store.close()
    db_path.unlink(missing_ok=True)
    return h1, h3, h5


# ═══════════════════════════════════════════════════════════════════
# 전략 4: Hierarchical (파일 요약 → 청크)
# ═══════════════════════════════════════════════════════════════════

async def strategy_hierarchical(embedder: OllamaEmbedder):
    """2단계: 파일 요약 매칭 → 상위 파일의 청크만 검색."""
    # 1. 파일 수준 인덱스
    store_files, db_files_path = _make_store("hier_files")
    # 2. 청크 수준 인덱스
    store_chunks, db_chunks_path = _make_store("hier_chunks")

    chunker = SlidingWindowChunker()
    files = _get_src_files()

    file_summaries: dict[str, str] = {}

    for fi in files:
        content = fi.path.read_text(encoding="utf-8", errors="replace")
        h = hashlib.sha256(content.encode()).hexdigest()

        # 파일 요약 생성
        ctx = _extract_python_context(content)
        summary = f"{fi.relative_path}\n{ctx}"
        file_summaries[fi.relative_path] = summary

        # 파일 수준 임베딩
        file_id_f = store_files.upsert_file(fi.relative_path, fi.mtime, h)
        file_emb = await embedder.embed(summary[:6000])
        store_files.add_chunks(file_id_f, [
            StoreChunk(1, 1, summary, file_emb)
        ])

        # 청크 수준 임베딩
        file_id_c = store_chunks.upsert_file(fi.relative_path, fi.mtime, h)
        chunks = chunker.chunk(fi.relative_path, content)
        if not chunks:
            continue
        texts = [c.embeddable_text for c in chunks]
        embeddings = await embedder.embed_batch(texts)
        for c, emb in zip(chunks, embeddings):
            store_chunks.add_chunks(file_id_c, [StoreChunk(c.start_line, c.end_line, c.content, emb)])

    h1 = h3 = h5 = 0
    for query, expected in TEST_CASES:
        q_emb = await embedder.embed(query)

        # 1단계: 파일 수준 매칭 (상위 3개 파일)
        file_results = store_files.search(q_emb, top_k=3, threshold=0.0)
        top_files = [r.file for r in file_results]

        # 2단계: 해당 파일의 청크만 검색
        all_chunk_results = store_chunks.search(q_emb, top_k=20, threshold=0.0)
        filtered = [r for r in all_chunk_results if r.file in top_files][:5]

        # 필터링 후 결과가 부족하면 전체에서 보충
        if len(filtered) < 5:
            remaining = [r for r in all_chunk_results if r.file not in top_files]
            filtered.extend(remaining[:5 - len(filtered)])

        a, b, c = _evaluate(filtered, expected)
        h1 += int(a); h3 += int(b); h5 += int(c)

    store_files.close()
    store_chunks.close()
    db_files_path.unlink(missing_ok=True)
    db_chunks_path.unlink(missing_ok=True)
    return h1, h3, h5


# ═══════════════════════════════════════════════════════════════════
# 전략 5: AST 기반 청킹
# ═══════════════════════════════════════════════════════════════════

def _ast_chunk_python(file_path: str, content: str) -> list[Chunk]:
    """Python 파일을 함수/클래스 단위로 청킹."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return SlidingWindowChunker().chunk(file_path, content)

    lines = content.split("\n")
    chunks: list[Chunk] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = node.end_lineno or start
            # 최소 3줄
            if end - start + 1 < 3:
                continue
            chunk_lines = lines[start - 1:end]
            chunks.append(Chunk(
                file_path=file_path,
                start_line=start,
                end_line=end,
                content="\n".join(chunk_lines),
            ))

    if not chunks:
        return SlidingWindowChunker().chunk(file_path, content)

    # 함수/클래스 사이 간극도 포함 (모듈 레벨 코드)
    # 중복 제거를 위해 줄 번호로 정렬
    chunks.sort(key=lambda c: c.start_line)

    return chunks


async def strategy_ast_chunking(embedder: OllamaEmbedder):
    """AST 기반: 함수/클래스 단위 청킹."""
    store, db_path = _make_store("ast")
    files = _get_src_files()

    for fi in files:
        content = fi.path.read_text(encoding="utf-8", errors="replace")
        h = hashlib.sha256(content.encode()).hexdigest()
        file_id = store.upsert_file(fi.relative_path, fi.mtime, h)

        if fi.relative_path.endswith(".py"):
            chunks = _ast_chunk_python(fi.relative_path, content)
        else:
            chunks = SlidingWindowChunker().chunk(fi.relative_path, content)

        if not chunks:
            continue

        texts = [c.embeddable_text for c in chunks]
        embeddings = await embedder.embed_batch(texts)
        for c, emb in zip(chunks, embeddings):
            store.add_chunks(file_id, [StoreChunk(c.start_line, c.end_line, c.content, emb)])

    h1 = h3 = h5 = 0
    for query, expected in TEST_CASES:
        q_emb = await embedder.embed(query)
        results = store.search(q_emb, top_k=5, threshold=0.0)
        a, b, c = _evaluate(results, expected)
        h1 += int(a); h3 += int(b); h5 += int(c)

    store.close()
    db_path.unlink(missing_ok=True)
    return h1, h3, h5


# ═══════════════════════════════════════════════════════════════════
# 전략 6: Context-enriched + AST (최적 조합)
# ═══════════════════════════════════════════════════════════════════

async def strategy_combined(embedder: OllamaEmbedder):
    """AST 청킹 + 컨텍스트 강화 임베딩 조합."""
    store, db_path = _make_store("combined")
    files = _get_src_files()

    for fi in files:
        content = fi.path.read_text(encoding="utf-8", errors="replace")
        h = hashlib.sha256(content.encode()).hexdigest()
        file_id = store.upsert_file(fi.relative_path, fi.mtime, h)

        file_context = _extract_python_context(content)

        if fi.relative_path.endswith(".py"):
            chunks = _ast_chunk_python(fi.relative_path, content)
        else:
            chunks = SlidingWindowChunker().chunk(fi.relative_path, content)

        if not chunks:
            continue

        texts = []
        for c in chunks:
            enriched = f"{c.header}\n{file_context}\n---\n{c.content}"
            texts.append(enriched[:6000])

        embeddings = await embedder.embed_batch(texts)
        for c, emb in zip(chunks, embeddings):
            store.add_chunks(file_id, [StoreChunk(c.start_line, c.end_line, c.content, emb)])

    h1 = h3 = h5 = 0
    for query, expected in TEST_CASES:
        q_emb = await embedder.embed(query)
        results = store.search(q_emb, top_k=5, threshold=0.0)
        a, b, c = _evaluate(results, expected)
        h1 += int(a); h3 += int(b); h5 += int(c)

    store.close()
    db_path.unlink(missing_ok=True)
    return h1, h3, h5


# ═══════════════════════════════════════════════════════════════════

async def main():
    print("=" * 90)
    print(f"EMBEDDING STRATEGY BENCHMARK (model: {MODEL})")
    print(f"Test cases: {len(TEST_CASES)} queries | Project: localgrep src/")
    print("=" * 90)
    print()

    embedder = OllamaEmbedder(model=MODEL)
    embedder.DIMENSION = DIM

    strategies = [
        ("1. Baseline (sliding window + raw)", strategy_baseline),
        ("2. Context-enriched (+ imports/sigs)", strategy_context_enriched),
        ("3. HyDE (hypothetical code query)", strategy_hyde),
        ("4. Hierarchical (file→chunk 2-stage)", strategy_hierarchical),
        ("5. AST chunking (function/class)", strategy_ast_chunking),
        ("6. AST + Context (combined best)", strategy_combined),
    ]

    results = []
    async with embedder:
        for name, fn in strategies:
            print(f"{name}...", end=" ", flush=True)
            start = time.monotonic()
            h1, h3, h5 = await fn(embedder)
            elapsed = time.monotonic() - start
            print(f"done ({elapsed:.1f}s)")

            n = len(TEST_CASES)
            results.append({
                "name": name,
                "h1": h1, "h3": h3, "h5": h5,
                "h1p": f"{h1/n*100:.0f}%",
                "h3p": f"{h3/n*100:.0f}%",
                "h5p": f"{h5/n*100:.0f}%",
            })

    print()
    print("=" * 90)
    n = len(TEST_CASES)
    print(f"{'Strategy':<45} {'Hit@1':>8} {'Hit@3':>8} {'Hit@5':>8}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: (-x["h5"], -x["h3"], -x["h1"])):
        print(f"{r['name']:<45} {r['h1']}/{n} ({r['h1p']}) {r['h3']}/{n} ({r['h3p']}) {r['h5']}/{n} ({r['h5p']})")
    print("=" * 90)


if __name__ == "__main__":
    asyncio.run(main())
