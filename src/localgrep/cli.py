"""CLI 인터페이스 - typer 기반 커맨드라인 도구."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from localgrep.chunker import SlidingWindowChunker
from localgrep.config import load_config
from localgrep.crawler import FileCrawler
from localgrep.embedder import OllamaEmbedder, OllamaEmbedderError
from localgrep.analytics import log_search, _estimate_tokens
from localgrep.store import VectorStore
from localgrep.store import Chunk as StoreChunk

app = typer.Typer(
    name="localgrep",
    help="로컬 임베딩 모델(Ollama)을 사용한 시맨틱 코드 검색 CLI.",
)
console = Console()


def _resolve_root(path: Optional[Path]) -> Path:
    """프로젝트 루트 경로를 결정한다."""
    return (path or Path.cwd()).resolve()


def _db_path(root: Path) -> Path:
    return root / ".localgrep" / "index.db"


def _file_hash(path: Path) -> str:
    """파일 내용의 SHA-256 해시를 반환한다."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


@app.command()
def index(
    path: Optional[Path] = typer.Argument(None, help="인덱싱할 프로젝트 경로 (기본: 현재 디렉토리)"),
    full: bool = typer.Option(False, "--full", help="전체 재인덱싱 여부"),
) -> None:
    """프로젝트 파일을 크롤링하고 임베딩을 생성하여 인덱싱한다."""
    root = _resolve_root(path)
    config = load_config(root)

    console.print(f"[bold]인덱싱 시작:[/bold] {root}")

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
                if deleted:
                    console.print(f"  삭제된 파일: {len(deleted)}개")

        if not files:
            console.print("[green]변경된 파일이 없습니다.[/green]")
            return

        console.print(f"  대상 파일: {len(files)}개")

        start = time.monotonic()
        asyncio.run(_index_files(files, store, embedder, chunker))
        elapsed = time.monotonic() - start

        stats = store.get_stats()
        console.print(
            f"[green]인덱싱 완료![/green] "
            f"파일: {stats['indexed_files']}개, "
            f"청크: {stats['total_chunks']}개, "
            f"소요: {elapsed:.1f}초"
        )

        # 대시보드 프로젝트 자동 등록
        from localgrep.dashboard import register_project
        register_project(str(root))
    except OllamaEmbedderError as e:
        console.print(f"[red]에러:[/red] {e}")
        raise typer.Exit(1)
    finally:
        store.close()
        asyncio.run(embedder.close())


async def _index_files(
    files: list,
    store: VectorStore,
    embedder: OllamaEmbedder,
    chunker: SlidingWindowChunker,
) -> None:
    """파일 목록을 인덱싱한다."""
    for i, file_info in enumerate(files):
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

        if (i + 1) % 10 == 0:
            console.print(f"  진행: {i + 1}/{len(files)}")


@app.command()
def search(
    query: str = typer.Argument(..., help="검색할 자연어 쿼리"),
    top_k: int = typer.Option(5, "-k", "--top-k", help="반환할 최대 결과 수"),
    threshold: float = typer.Option(0.3, "-t", "--threshold", help="최소 유사도 점수 (0.0-1.0)"),
    file_pattern: Optional[str] = typer.Option(None, "-g", "--glob", help="파일 필터 glob 패턴"),
    json_output: bool = typer.Option(False, "--json", help="JSON 형식으로 출력"),
    path: Optional[Path] = typer.Option(None, "-p", "--path", help="프로젝트 경로"),
) -> None:
    """자연어 쿼리로 코드베이스를 시맨틱 검색한다."""
    root = _resolve_root(path)
    db = _db_path(root)

    if not db.exists():
        console.print("[red]인덱스가 없습니다. 먼저 'localgrep index'를 실행하세요.[/red]")
        raise typer.Exit(1)

    config = load_config(root)
    store = VectorStore(db)
    embedder = OllamaEmbedder(
        host=config.ollama.host,
        model=config.ollama.model,
    )

    try:
        start = time.monotonic()
        query_embedding = asyncio.run(embedder.embed(query))
        results = store.search(
            query_embedding,
            top_k=top_k,
            threshold=threshold,
            file_pattern=file_pattern,
        )
        elapsed_ms = (time.monotonic() - start) * 1000

        stats = store.get_stats()

        # 검색 로깅
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

        if json_output:
            output = {
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
            console.print_json(json.dumps(output, ensure_ascii=False))
        else:
            if not results:
                console.print("[yellow]검색 결과가 없습니다.[/yellow]")
                return

            console.print(
                f'[bold]검색:[/bold] "{query}" '
                f"({len(results)}건, {elapsed_ms:.0f}ms)\n"
            )
            for i, r in enumerate(results, 1):
                score_color = "green" if r.score >= 0.7 else "yellow" if r.score >= 0.5 else "dim"
                console.print(
                    f"[bold]{i}.[/bold] [{score_color}]{r.score:.2f}[/{score_color}] "
                    f"[cyan]{r.file}[/cyan]:{r.start_line}-{r.end_line}"
                )
                snippet_lines = r.snippet.splitlines()[:5]
                for line in snippet_lines:
                    console.print(f"    {line}")
                if len(r.snippet.splitlines()) > 5:
                    console.print("    ...")
                console.print()

    except OllamaEmbedderError as e:
        console.print(f"[red]에러:[/red] {e}")
        raise typer.Exit(1)
    finally:
        store.close()
        asyncio.run(embedder.close())


@app.command()
def watch(
    path: Optional[Path] = typer.Argument(None, help="감시할 프로젝트 경로 (기본: 현재 디렉토리)"),
) -> None:
    """파일 변경을 감지하여 자동으로 재인덱싱한다."""
    console.print("[yellow]watch는 Phase 3에서 구현 예정입니다.[/yellow]")
    raise typer.Exit(1)


@app.command()
def status(
    path: Optional[Path] = typer.Argument(None, help="확인할 프로젝트 경로 (기본: 현재 디렉토리)"),
) -> None:
    """현재 인덱스 상태를 출력한다."""
    root = _resolve_root(path)
    db = _db_path(root)

    if not db.exists():
        console.print("[yellow]인덱스가 없습니다. 'localgrep index'로 인덱싱하세요.[/yellow]")
        return

    store = VectorStore(db)
    try:
        stats = store.get_stats()
        table = Table(title="인덱스 상태")
        table.add_column("항목", style="bold")
        table.add_column("값")

        table.add_row("프로젝트 경로", str(root))
        table.add_row("DB 경로", str(db))
        table.add_row("인덱싱된 파일", str(stats["indexed_files"]))
        table.add_row("총 청크 수", str(stats["total_chunks"]))

        if stats["last_updated"]:
            dt = datetime.fromtimestamp(stats["last_updated"], tz=timezone.utc)
            table.add_row("마지막 업데이트", dt.strftime("%Y-%m-%d %H:%M:%S UTC"))
        else:
            table.add_row("마지막 업데이트", "-")

        console.print(table)
    finally:
        store.close()


@app.command("config")
def show_config() -> None:
    """프로젝트 설정을 보여준다."""
    from dataclasses import asdict

    root = Path.cwd().resolve()
    cfg = load_config(root)
    console.print_json(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))


@app.command()
def serve() -> None:
    """MCP 서버를 stdio 모드로 시작한다."""
    from localgrep.server import run_server

    run_server()


@app.command()
def dashboard(
    host: str = typer.Option("127.0.0.1", "--host", help="바인딩 호스트"),
    port: int = typer.Option(8585, "--port", help="포트 번호"),
) -> None:
    """웹 대시보드를 시작한다 (http://localhost:8585)."""
    from localgrep.dashboard import run_dashboard

    run_dashboard(host=host, port=port)


@app.command("install-claude")
def install_claude() -> None:
    """Claude Code MCP 설정과 CLAUDE.md 검색 가이드를 자동 구성한다."""
    claude_dir = Path.home() / ".claude"

    if not claude_dir.exists():
        claude_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[dim]디렉토리 생성: {claude_dir}[/dim]")

    # ── 1. mcp.json 설정 ──────────────────────────────────────────────
    mcp_json_path = claude_dir / "mcp.json"

    if mcp_json_path.exists():
        try:
            mcp_data = json.loads(mcp_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            mcp_data = {}
    else:
        mcp_data = {}

    if "mcpServers" not in mcp_data:
        mcp_data["mcpServers"] = {}

    mcp_data["mcpServers"]["localgrep"] = {
        "command": "localgrep",
        "args": ["serve"],
        "env": {
            "OLLAMA_HOST": "http://localhost:11434",
        },
    }

    mcp_json_path.write_text(
        json.dumps(mcp_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    console.print(f"[green]MCP 설정 완료:[/green] {mcp_json_path}")

    # ── 2. CLAUDE.md 가이드 추가 ──────────────────────────────────────
    claude_md_path = claude_dir / "CLAUDE.md"

    search_guide = """\

## Code Search Strategy (localgrep)

1. When you know the exact keyword/symbol -> use Grep / Glob
2. When searching by concept or functionality -> use semantic_search
3. Score >= 0.7: high confidence, Score 0.3-0.7: reference, Score < 0.3: ignore
4. Always index before first search: run the reindex tool if index_status shows no index
"""

    marker = "## Code Search Strategy (localgrep)"

    if claude_md_path.exists():
        existing = claude_md_path.read_text(encoding="utf-8")
        if marker in existing:
            console.print("[yellow]CLAUDE.md 가이드 이미 존재 — 스킵[/yellow]")
        else:
            claude_md_path.write_text(
                existing.rstrip() + "\n" + search_guide,
                encoding="utf-8",
            )
            console.print(f"[green]CLAUDE.md 가이드 추가:[/green] {claude_md_path}")
    else:
        claude_md_path.write_text(search_guide.lstrip(), encoding="utf-8")
        console.print(f"[green]CLAUDE.md 생성:[/green] {claude_md_path}")

    # ── 3. 완료 메시지 ────────────────────────────────────────────────
    console.print()
    console.print("[bold green]Claude Code 연동 설정 완료![/bold green]")
    console.print("Claude Code를 재시작하면 semantic_search 도구를 사용할 수 있습니다.")


if __name__ == "__main__":
    app()
