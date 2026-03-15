"""MCP 서버 - Claude Code와 연동하기 위한 JSON-RPC 서버."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from localgrep.chunker import SlidingWindowChunker
from localgrep.config import load_config
from localgrep.crawler import FileCrawler
from localgrep.embedder import OllamaEmbedder, OllamaEmbedderError
from localgrep.analytics import log_search, _estimate_tokens
from localgrep.store import VectorStore
from localgrep.store import Chunk as StoreChunk

mcp = FastMCP(
    "localgrep",
    instructions="로컬 코드베이스를 시맨틱 검색합니다. 키워드를 모를 때, 개념이나 기능으로 코드를 찾을 때 사용하세요.",
)


def _resolve_root(path: str | None) -> Path:
    return Path(path).resolve() if path else Path.cwd().resolve()


def _db_path(root: Path) -> Path:
    return root / ".localgrep" / "index.db"


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


@mcp.tool()
async def semantic_search(
    query: str,
    path: str | None = None,
    top_k: int = 5,
    threshold: float = 0.3,
    file_pattern: str | None = None,
) -> dict:
    """자연어 쿼리로 코드베이스를 시맨틱 검색합니다. 키워드를 모를 때, 개념이나 기능으로 코드를 찾을 때 사용하세요.

    Args:
        query: 검색할 자연어 쿼리 (예: '사용자 인증 처리', 'DB 커넥션 풀 설정')
        path: 검색 범위를 제한할 디렉토리 경로 (선택)
        top_k: 반환할 최대 결과 수 (기본: 5)
        threshold: 최소 유사도 점수 0.0-1.0 (기본: 0.3)
        file_pattern: 파일 필터 glob 패턴 (예: '*.py', '*.ts')
    """
    root = _resolve_root(path)
    db = _db_path(root)

    if not db.exists():
        return {"error": "인덱스가 없습니다. 먼저 reindex를 실행하세요."}

    config = load_config(root)
    store = VectorStore(db)
    embedder = OllamaEmbedder(
        host=config.ollama.host,
        model=config.ollama.model,
    )

    try:
        start = time.monotonic()
        query_embedding = await embedder.embed(query)
        results = store.search(
            query_embedding,
            top_k=top_k,
            threshold=threshold,
            file_pattern=file_pattern,
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        stats = store.get_stats()

        # 토큰 추정 및 로깅
        total_tokens = _estimate_tokens(query) + sum(
            _estimate_tokens(r.snippet) for r in results
        )
        scores = [r.score for r in results]
        log_search(
            project_path=str(root),
            query=query,
            search_type="semantic",
            results_count=len(results),
            total_tokens=total_tokens,
            search_time_ms=round(elapsed_ms, 1),
            top_score=max(scores) if scores else None,
            avg_score=round(sum(scores) / len(scores), 4) if scores else None,
        )

        return {
            "results": [
                {
                    "file": r.file,
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                    "score": r.score,
                    "snippet": r.snippet,
                }
                for r in results
            ],
            "query": query,
            "indexed_files": stats["indexed_files"],
            "search_time_ms": round(elapsed_ms),
        }
    except OllamaEmbedderError as e:
        return {"error": str(e)}
    finally:
        store.close()
        await embedder.close()


@mcp.tool()
async def index_status(path: str | None = None) -> dict:
    """현재 인덱스 상태를 확인합니다. 인덱싱된 파일 수, 마지막 업데이트 시간 등.

    Args:
        path: 확인할 프로젝트 경로 (기본: 현재 디렉토리)
    """
    root = _resolve_root(path)
    db = _db_path(root)

    if not db.exists():
        return {
            "indexed": False,
            "project_root": str(root),
            "message": "인덱스가 없습니다. reindex를 실행하세요.",
        }

    store = VectorStore(db)
    try:
        stats = store.get_stats()
        return {
            "indexed": True,
            "project_root": str(root),
            "indexed_files": stats["indexed_files"],
            "total_chunks": stats["total_chunks"],
            "last_updated": stats["last_updated"],
        }
    finally:
        store.close()


@mcp.tool()
async def reindex(path: str | None = None, full: bool = False) -> dict:
    """인덱스를 갱신합니다. 변경된 파일만 재인덱싱합니다.

    Args:
        path: 재인덱싱할 프로젝트 경로
        full: 전체 재인덱싱 여부 (기본: false, 변경분만)
    """
    root = _resolve_root(path)
    config = load_config(root)

    crawler = FileCrawler(
        root=root,
        max_file_size_kb=config.indexing.max_file_size_kb,
        extra_ignore_patterns=config.indexing.ignore,
    )

    store = VectorStore(_db_path(root))
    embedder = OllamaEmbedder(
        host=config.ollama.host,
        model=config.ollama.model,
    )
    chunker = SlidingWindowChunker(
        max_lines=config.chunking.max_lines,
        overlap_lines=config.chunking.overlap_lines,
        min_lines=config.chunking.min_lines,
    )

    try:
        if full:
            files = crawler.crawl()
        else:
            stats = store.get_stats()
            if stats["indexed_files"] == 0:
                files = crawler.crawl()
            else:
                known: dict[str, float] = {}
                cur = store._conn.cursor()
                cur.execute("SELECT path, mtime FROM files")
                for row in cur.fetchall():
                    known[row[0]] = row[1]
                files, deleted = crawler.get_changed_files(known)
                for d in deleted:
                    store.delete_file(d)

        if not files:
            return {"status": "no_changes", "message": "변경된 파일이 없습니다."}

        start = time.monotonic()
        indexed_count = 0

        for file_info in files:
            try:
                content = file_info.path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            h = _file_hash(file_info.path)
            file_id = store.upsert_file(file_info.relative_path, file_info.mtime, h)

            chunks = chunker.chunk(file_info.relative_path, content)
            if not chunks:
                continue

            texts = [c.embeddable_text for c in chunks]
            embeddings = await embedder.embed_batch(texts)

            store_chunks = [
                StoreChunk(
                    start_line=c.start_line,
                    end_line=c.end_line,
                    content=c.content,
                    embedding=emb,
                )
                for c, emb in zip(chunks, embeddings)
            ]
            store.add_chunks(file_id, store_chunks)
            indexed_count += 1

        elapsed = time.monotonic() - start
        stats = store.get_stats()

        return {
            "status": "completed",
            "indexed_files": stats["indexed_files"],
            "total_chunks": stats["total_chunks"],
            "files_processed": indexed_count,
            "elapsed_seconds": round(elapsed, 1),
        }
    except OllamaEmbedderError as e:
        return {"status": "error", "error": str(e)}
    finally:
        store.close()
        await embedder.close()


@mcp.tool()
async def log_grep_usage(
    query: str,
    path: str | None = None,
    results_count: int = 0,
    content_size: int = 0,
    search_time_ms: float = 0,
) -> dict:
    """grep 검색 사용량을 기록합니다. Claude가 Grep 도구를 사용한 뒤 호출하세요.

    Args:
        query: grep 검색 패턴
        path: 검색한 프로젝트 경로 (기본: 현재 디렉토리)
        results_count: 검색 결과 수
        content_size: 결과 텍스트의 총 글자 수
        search_time_ms: 검색 소요 시간(ms)
    """
    root = _resolve_root(path)
    total_tokens = _estimate_tokens(query) + (content_size // 4)
    log_search(
        project_path=str(root),
        query=query,
        search_type="grep",
        results_count=results_count,
        total_tokens=total_tokens,
        search_time_ms=search_time_ms,
    )
    return {"status": "logged", "total_tokens": total_tokens}


def run_server() -> None:
    """MCP 서버를 stdio 모드로 시작한다."""
    mcp.run(transport="stdio")
