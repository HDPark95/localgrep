"""웹 대시보드 - FastAPI 기반 로컬 웹 UI."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from localgrep.analytics import get_daily_stats, get_recent_searches, get_summary, get_token_comparison
from localgrep.config import load_config
from localgrep.store import VectorStore

# ---------------------------------------------------------------------------
# Projects registry (~/.localgrep/projects.json)
# ---------------------------------------------------------------------------

PROJECTS_FILE = Path.home() / ".localgrep" / "projects.json"


def _load_projects() -> list[str]:
    """등록된 프로젝트 경로 목록을 로드한다."""
    if not PROJECTS_FILE.exists():
        return []
    try:
        data = json.loads(PROJECTS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_projects(projects: list[str]) -> None:
    """프로젝트 경로 목록을 저장한다."""
    PROJECTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROJECTS_FILE.write_text(
        json.dumps(projects, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def register_project(project_path: str) -> None:
    """프로젝트를 등록한다 (중복 무시)."""
    projects = _load_projects()
    resolved = str(Path(project_path).resolve())
    if resolved not in projects:
        projects.append(resolved)
        _save_projects(projects)


def unregister_project(project_path: str) -> None:
    """프로젝트를 등록 해제한다."""
    projects = _load_projects()
    resolved = str(Path(project_path).resolve())
    if resolved in projects:
        projects.remove(resolved)
        _save_projects(projects)


def _project_id(path: str) -> str:
    """프로젝트 경로를 짧은 ID로 변환한다."""
    return hashlib.md5(path.encode()).hexdigest()[:12]


def _db_path(root: Path) -> Path:
    return root / ".localgrep" / "index.db"


def _get_project_info(project_path: str) -> dict:
    """프로젝트의 인덱스 상태 정보를 가져온다."""
    root = Path(project_path)
    db = _db_path(root)
    pid = _project_id(project_path)
    info = {
        "id": pid,
        "path": project_path,
        "name": root.name,
        "indexed_files": 0,
        "total_chunks": 0,
        "last_updated": None,
        "last_updated_fmt": "-",
        "db_size": 0,
        "db_size_fmt": "-",
        "status": "none",  # none / stale / fresh
    }

    if not db.exists():
        return info

    try:
        db_size = db.stat().st_size
        info["db_size"] = db_size
        if db_size < 1024:
            info["db_size_fmt"] = f"{db_size} B"
        elif db_size < 1024 * 1024:
            info["db_size_fmt"] = f"{db_size / 1024:.1f} KB"
        else:
            info["db_size_fmt"] = f"{db_size / (1024 * 1024):.1f} MB"

        store = VectorStore(db)
        try:
            stats = store.get_stats()
            info["indexed_files"] = stats["indexed_files"]
            info["total_chunks"] = stats["total_chunks"]
            if stats["last_updated"]:
                info["last_updated"] = stats["last_updated"]
                dt = datetime.fromtimestamp(stats["last_updated"], tz=timezone.utc)
                info["last_updated_fmt"] = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                # 24시간 이내면 fresh, 아니면 stale
                age = time.time() - stats["last_updated"]
                info["status"] = "fresh" if age < 86400 else "stale"
            else:
                info["status"] = "none"
        finally:
            store.close()
    except Exception:
        info["status"] = "none"

    return info


def _get_project_files(project_path: str) -> list[dict]:
    """프로젝트의 인덱싱된 파일 목록을 가져온다."""
    root = Path(project_path)
    db = _db_path(root)
    if not db.exists():
        return []

    store = VectorStore(db)
    try:
        cur = store._conn.cursor()
        cur.execute(
            """
            SELECT f.path, f.mtime, COUNT(c.id) as chunk_count
            FROM files f
            LEFT JOIN chunks c ON c.file_id = f.id
            GROUP BY f.id
            ORDER BY f.path
            """
        )
        rows = cur.fetchall()
        result = []
        for path, mtime, chunk_count in rows:
            dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
            result.append({
                "path": path,
                "mtime": mtime,
                "mtime_fmt": dt.strftime("%Y-%m-%d %H:%M"),
                "chunk_count": chunk_count,
            })
        return result
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str
    project_path: str
    top_k: int = 5
    threshold: float = 0.3


class ReindexRequest(BaseModel):
    project_path: str
    full: bool = False


class AddProjectRequest(BaseModel):
    project_path: str


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

dashboard_app = FastAPI(title="localgrep Dashboard")


@dashboard_app.get("/api/projects")
async def api_projects():
    """등록된 프로젝트 목록 + 인덱스 상태."""
    projects = _load_projects()
    return [_get_project_info(p) for p in projects]


@dashboard_app.get("/api/projects/{project_id}/files")
async def api_project_files(project_id: str):
    """프로젝트의 인덱싱된 파일 목록."""
    projects = _load_projects()
    for p in projects:
        if _project_id(p) == project_id:
            return _get_project_files(p)
    raise HTTPException(status_code=404, detail="Project not found")


@dashboard_app.get("/api/projects/{project_id}/stats")
async def api_project_stats(project_id: str):
    """프로젝트 상세 통계."""
    projects = _load_projects()
    for p in projects:
        if _project_id(p) == project_id:
            info = _get_project_info(p)
            files = _get_project_files(p)
            info["files"] = files
            return info
    raise HTTPException(status_code=404, detail="Project not found")


@dashboard_app.post("/api/search")
async def api_search(req: SearchRequest):
    """시맨틱 검색."""
    root = Path(req.project_path)
    db = _db_path(root)
    if not db.exists():
        raise HTTPException(status_code=404, detail="Index not found. Run 'localgrep index' first.")

    config = load_config(root)

    from localgrep.embedder import OllamaEmbedder, OllamaEmbedderError

    store = VectorStore(db)
    embedder = OllamaEmbedder(host=config.ollama.host, model=config.ollama.model)
    try:
        start = time.monotonic()
        query_embedding = await embedder.embed(req.query)
        results = store.search(
            query_embedding,
            top_k=req.top_k,
            threshold=req.threshold,
        )
        elapsed_ms = (time.monotonic() - start) * 1000

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
            "query": req.query,
            "search_time_ms": round(elapsed_ms),
        }
    except OllamaEmbedderError as e:
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        store.close()
        await embedder.close()


@dashboard_app.post("/api/reindex")
async def api_reindex(req: ReindexRequest):
    """재인덱싱 트리거."""
    root = Path(req.project_path).resolve()
    if not root.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {root}")

    config = load_config(root)

    from localgrep.chunker import SlidingWindowChunker
    from localgrep.crawler import FileCrawler
    from localgrep.embedder import OllamaEmbedder, OllamaEmbedderError
    from localgrep.store import Chunk as StoreChunk

    crawler = FileCrawler(
        root=root,
        max_file_size_kb=config.indexing.max_file_size_kb,
        extra_ignore_patterns=config.indexing.ignore,
    )
    store = VectorStore(_db_path(root))
    embedder = OllamaEmbedder(host=config.ollama.host, model=config.ollama.model)
    chunker = SlidingWindowChunker(
        max_lines=config.chunking.max_lines,
        overlap_lines=config.chunking.overlap_lines,
        min_lines=config.chunking.min_lines,
    )

    try:
        if req.full:
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
            register_project(str(root))
            return {"status": "ok", "message": "No changed files", "files_indexed": 0}

        for file_info in files:
            try:
                content = file_info.path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            import hashlib as _hl
            h = _hl.sha256()
            with open(file_info.path, "rb") as f:
                for block in iter(lambda: f.read(65536), b""):
                    h.update(block)
            file_hash = h.hexdigest()

            file_id = store.upsert_file(file_info.relative_path, file_info.mtime, file_hash)
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

        register_project(str(root))
        final_stats = store.get_stats()
        return {
            "status": "ok",
            "message": f"Indexed {len(files)} files",
            "files_indexed": len(files),
            "total_files": final_stats["indexed_files"],
            "total_chunks": final_stats["total_chunks"],
        }
    except OllamaEmbedderError as e:
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        store.close()
        await embedder.close()


@dashboard_app.post("/api/projects/add")
async def api_add_project(req: AddProjectRequest):
    """프로젝트 추가."""
    resolved = str(Path(req.project_path).resolve())
    if not Path(resolved).is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {resolved}")
    register_project(resolved)
    return {"status": "ok", "path": resolved}


@dashboard_app.post("/api/projects/remove")
async def api_remove_project(req: AddProjectRequest):
    """프로젝트 삭제."""
    resolved = str(Path(req.project_path).resolve())
    unregister_project(resolved)
    return {"status": "ok", "path": resolved}


# ---------------------------------------------------------------------------
# Analytics API
# ---------------------------------------------------------------------------


@dashboard_app.get("/api/analytics/summary")
async def api_analytics_summary():
    """검색 분석 전체 요약."""
    return get_summary()


@dashboard_app.get("/api/analytics/daily")
async def api_analytics_daily(days: int = 7):
    """일별 검색 통계."""
    return get_daily_stats(days=days)


@dashboard_app.get("/api/analytics/recent")
async def api_analytics_recent(limit: int = 50):
    """최근 검색 기록."""
    return get_recent_searches(limit=limit)


@dashboard_app.get("/api/analytics/comparison")
async def api_analytics_comparison():
    """semantic vs grep 토큰 비교."""
    return get_token_comparison()


# ---------------------------------------------------------------------------
# HTML Templates (inline)
# ---------------------------------------------------------------------------

_CSS = """
:root {
    --bg: #0d1117;
    --bg-card: #161b22;
    --bg-hover: #1c2129;
    --border: #30363d;
    --text: #e6edf3;
    --text-dim: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --yellow: #d29922;
    --red: #f85149;
    --gray: #8b949e;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    min-height: 100vh;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

.container { max-width: 1200px; margin: 0 auto; padding: 0 24px; }

/* Nav */
nav {
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
    padding: 12px 0;
    position: sticky; top: 0; z-index: 100;
}
nav .container {
    display: flex; align-items: center; gap: 32px;
}
nav .logo {
    font-size: 20px; font-weight: 700; color: var(--text);
    display: flex; align-items: center; gap: 8px;
}
nav .logo span { color: var(--accent); }
nav .nav-links { display: flex; gap: 20px; }
nav .nav-links a {
    color: var(--text-dim); font-size: 14px; padding: 4px 8px;
    border-radius: 6px; transition: all 0.2s;
}
nav .nav-links a:hover, nav .nav-links a.active {
    color: var(--text); background: var(--bg-hover); text-decoration: none;
}

/* Header */
.page-header {
    padding: 32px 0 24px;
    display: flex; justify-content: space-between; align-items: center;
}
.page-header h1 { font-size: 24px; font-weight: 600; }

/* Cards grid */
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; padding-bottom: 40px; }
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    transition: border-color 0.2s;
    cursor: pointer;
}
.card:hover { border-color: var(--accent); }
.card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; }
.card-title { font-size: 16px; font-weight: 600; word-break: break-all; }
.card-path { font-size: 12px; color: var(--text-dim); margin-top: 2px; word-break: break-all; }
.status-dot {
    width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; margin-top: 4px;
}
.status-fresh { background: var(--green); }
.status-stale { background: var(--yellow); }
.status-none { background: var(--red); }
.card-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }
.stat { font-size: 13px; }
.stat-label { color: var(--text-dim); }
.stat-value { font-weight: 600; }

/* Buttons */
.btn {
    padding: 8px 16px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg-card);
    color: var(--text);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s;
    display: inline-flex; align-items: center; gap: 6px;
}
.btn:hover { background: var(--bg-hover); border-color: var(--accent); }
.btn-primary { background: var(--accent); color: #000; border-color: var(--accent); font-weight: 600; }
.btn-primary:hover { opacity: 0.9; }
.btn-sm { padding: 4px 10px; font-size: 12px; }
.btn-danger { border-color: var(--red); color: var(--red); }
.btn-danger:hover { background: var(--red); color: #fff; }

/* Table */
table { width: 100%; border-collapse: collapse; }
th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); font-size: 14px; }
th { color: var(--text-dim); font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
tr:hover td { background: var(--bg-hover); }

/* Bar chart */
.bar-chart { margin: 16px 0; }
.bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; font-size: 13px; }
.bar-label { width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-dim); text-align: right; }
.bar-fill { height: 20px; background: var(--accent); border-radius: 4px; min-width: 2px; transition: width 0.3s; }
.bar-value { color: var(--text-dim); min-width: 30px; }

/* Search */
.search-box {
    display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap;
}
.search-input {
    flex: 1; min-width: 250px;
    padding: 10px 16px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg-card);
    color: var(--text);
    font-size: 15px;
    outline: none;
    transition: border-color 0.2s;
}
.search-input:focus { border-color: var(--accent); }
select {
    padding: 10px 16px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg-card);
    color: var(--text);
    font-size: 14px;
    outline: none;
    min-width: 200px;
}

/* Results */
.result-item {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}
.result-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.result-file { font-weight: 600; color: var(--accent); font-size: 14px; }
.result-lines { color: var(--text-dim); font-size: 13px; }
.score-badge {
    padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600;
}
.score-high { background: rgba(63, 185, 80, 0.15); color: var(--green); }
.score-mid { background: rgba(210, 153, 34, 0.15); color: var(--yellow); }
.score-low { background: rgba(139, 148, 158, 0.15); color: var(--gray); }
.result-snippet {
    background: var(--bg);
    border-radius: 6px;
    padding: 12px;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 13px;
    line-height: 1.5;
    overflow-x: auto;
    white-space: pre-wrap;
    color: var(--text-dim);
    max-height: 200px;
    overflow-y: auto;
}

/* Modal */
.modal-overlay {
    display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.6); z-index: 200;
    align-items: center; justify-content: center;
}
.modal-overlay.active { display: flex; }
.modal {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    width: 480px; max-width: 90%;
}
.modal h3 { margin-bottom: 16px; }
.modal input[type="text"] {
    width: 100%;
    padding: 10px 16px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    margin-bottom: 16px;
    outline: none;
}
.modal-actions { display: flex; gap: 8px; justify-content: flex-end; }

/* Loading spinner */
.spinner {
    display: inline-block; width: 16px; height: 16px;
    border: 2px solid var(--border); border-top-color: var(--accent);
    border-radius: 50%; animation: spin 0.6s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Toast */
.toast {
    position: fixed; bottom: 24px; right: 24px;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px 20px; font-size: 14px;
    z-index: 300; transform: translateY(100px); opacity: 0;
    transition: all 0.3s;
}
.toast.show { transform: translateY(0); opacity: 1; }

/* Responsive */
@media (max-width: 640px) {
    .cards { grid-template-columns: 1fr; }
    .search-box { flex-direction: column; }
    nav .container { flex-wrap: wrap; gap: 12px; }
}
"""

_JS = """
// --- State ---
let projects = [];
let currentProject = null;

// --- API helpers ---
async function api(url, opts = {}) {
    const res = await fetch(url, {
        headers: { 'Content-Type': 'application/json' },
        ...opts
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'API error');
    }
    return res.json();
}

function toast(msg, duration = 3000) {
    const el = document.getElementById('toast');
    el.textContent = msg;
    el.classList.add('show');
    setTimeout(() => el.classList.remove('show'), duration);
}

// --- Projects page ---
async function loadProjects() {
    try {
        projects = await api('/api/projects');
        renderProjects();
    } catch (e) {
        toast('Failed to load projects: ' + e.message);
    }
}

function renderProjects() {
    const grid = document.getElementById('projects-grid');
    if (!grid) return;
    if (projects.length === 0) {
        grid.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; padding: 60px 0; color: var(--text-dim);">
                <p style="font-size: 18px; margin-bottom: 8px;">No projects registered</p>
                <p style="font-size: 14px;">Add a project or run <code>localgrep index</code> in a project directory.</p>
            </div>
        `;
        return;
    }
    grid.innerHTML = projects.map(p => `
        <div class="card" onclick="navigateTo('project', '${p.id}')">
            <div class="card-header">
                <div>
                    <div class="card-title">${escHtml(p.name)}</div>
                    <div class="card-path">${escHtml(p.path)}</div>
                </div>
                <div class="status-dot status-${p.status}" title="${p.status}"></div>
            </div>
            <div class="card-stats">
                <div class="stat"><span class="stat-label">Files:</span> <span class="stat-value">${p.indexed_files}</span></div>
                <div class="stat"><span class="stat-label">Chunks:</span> <span class="stat-value">${p.total_chunks}</span></div>
                <div class="stat"><span class="stat-label">Updated:</span> <span class="stat-value">${escHtml(p.last_updated_fmt)}</span></div>
                <div class="stat"><span class="stat-label">DB Size:</span> <span class="stat-value">${escHtml(p.db_size_fmt)}</span></div>
            </div>
            <div style="margin-top: 12px; display: flex; gap: 8px;">
                <button class="btn btn-sm btn-primary" onclick="event.stopPropagation(); reindex('${escHtml(p.path)}', false)">Re-index</button>
                <button class="btn btn-sm btn-danger" onclick="event.stopPropagation(); removeProject('${escHtml(p.path)}')">Remove</button>
            </div>
        </div>
    `).join('');
}

async function reindex(path, full) {
    toast('Re-indexing started...');
    try {
        const result = await api('/api/reindex', {
            method: 'POST',
            body: JSON.stringify({ project_path: path, full: full })
        });
        toast(result.message);
        await loadProjects();
    } catch (e) {
        toast('Re-index failed: ' + e.message, 5000);
    }
}

async function removeProject(path) {
    if (!confirm('Remove project from dashboard?')) return;
    try {
        await api('/api/projects/remove', {
            method: 'POST',
            body: JSON.stringify({ project_path: path })
        });
        toast('Project removed');
        await loadProjects();
    } catch (e) {
        toast('Failed: ' + e.message);
    }
}

function showAddModal() {
    document.getElementById('add-modal').classList.add('active');
    document.getElementById('add-path-input').value = '';
    document.getElementById('add-path-input').focus();
}

function hideAddModal() {
    document.getElementById('add-modal').classList.remove('active');
}

async function addProject() {
    const path = document.getElementById('add-path-input').value.trim();
    if (!path) return;
    try {
        await api('/api/projects/add', {
            method: 'POST',
            body: JSON.stringify({ project_path: path })
        });
        hideAddModal();
        toast('Project added');
        await loadProjects();
    } catch (e) {
        toast('Failed: ' + e.message);
    }
}

// --- Project detail page ---
async function loadProjectDetail(projectId) {
    try {
        const stats = await api(`/api/projects/${projectId}/stats`);
        currentProject = stats;
        renderProjectDetail(stats);
    } catch (e) {
        toast('Failed to load project: ' + e.message);
    }
}

function renderProjectDetail(p) {
    const main = document.getElementById('main-content');
    const files = p.files || [];
    const maxChunks = Math.max(...files.map(f => f.chunk_count), 1);
    const topFiles = files.slice().sort((a, b) => b.chunk_count - a.chunk_count).slice(0, 20);

    main.innerHTML = `
        <div class="page-header">
            <div>
                <a href="#" onclick="navigateTo('home'); return false;" style="font-size: 13px; color: var(--text-dim);">&#8592; Back to projects</a>
                <h1 style="margin-top: 4px;">${escHtml(p.name)}</h1>
                <div style="color: var(--text-dim); font-size: 13px; margin-top: 2px;">${escHtml(p.path)}</div>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
                <div class="status-dot status-${p.status}"></div>
                <button class="btn btn-primary" onclick="reindex('${escHtml(p.path)}', false)">Re-index</button>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px;">
            <div class="card" style="cursor:default;">
                <div class="stat-label">Files</div>
                <div style="font-size: 28px; font-weight: 700;">${p.indexed_files}</div>
            </div>
            <div class="card" style="cursor:default;">
                <div class="stat-label">Chunks</div>
                <div style="font-size: 28px; font-weight: 700;">${p.total_chunks}</div>
            </div>
            <div class="card" style="cursor:default;">
                <div class="stat-label">DB Size</div>
                <div style="font-size: 28px; font-weight: 700;">${escHtml(p.db_size_fmt)}</div>
            </div>
            <div class="card" style="cursor:default;">
                <div class="stat-label">Updated</div>
                <div style="font-size: 14px; font-weight: 600; margin-top: 8px;">${escHtml(p.last_updated_fmt)}</div>
            </div>
        </div>

        <h2 style="font-size: 18px; margin-bottom: 16px;">Chunk Distribution (Top 20)</h2>
        <div class="bar-chart">
            ${topFiles.map(f => `
                <div class="bar-row">
                    <div class="bar-label" title="${escHtml(f.path)}">${escHtml(f.path.split('/').pop())}</div>
                    <div class="bar-fill" style="width: ${Math.max((f.chunk_count / maxChunks) * 400, 2)}px;"></div>
                    <div class="bar-value">${f.chunk_count}</div>
                </div>
            `).join('')}
        </div>

        <h2 style="font-size: 18px; margin: 24px 0 16px;">Search in project</h2>
        <div class="search-box">
            <input class="search-input" id="detail-search-input" type="text" placeholder="Enter a semantic query..."
                onkeydown="if(event.key==='Enter') searchInProject()">
            <button class="btn btn-primary" onclick="searchInProject()">Search</button>
        </div>
        <div id="detail-search-results"></div>

        <h2 style="font-size: 18px; margin: 24px 0 16px;">Indexed Files (${files.length})</h2>
        <table>
            <thead><tr><th>Path</th><th>Chunks</th><th>Modified</th></tr></thead>
            <tbody>
                ${files.map(f => `
                    <tr>
                        <td style="font-family: monospace; font-size: 13px;">${escHtml(f.path)}</td>
                        <td>${f.chunk_count}</td>
                        <td style="color: var(--text-dim);">${escHtml(f.mtime_fmt)}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

async function searchInProject() {
    const input = document.getElementById('detail-search-input');
    const query = input.value.trim();
    if (!query || !currentProject) return;
    const container = document.getElementById('detail-search-results');
    container.innerHTML = '<div class="spinner"></div>';
    try {
        const data = await api('/api/search', {
            method: 'POST',
            body: JSON.stringify({
                query: query,
                project_path: currentProject.path,
                top_k: 10,
                threshold: 0.3
            })
        });
        renderSearchResults(container, data);
    } catch (e) {
        container.innerHTML = `<div style="color: var(--red);">Error: ${escHtml(e.message)}</div>`;
    }
}

// --- Search page ---
async function initSearchPage() {
    if (projects.length === 0) await loadProjects();
    const sel = document.getElementById('search-project-select');
    if (sel) {
        sel.innerHTML = '<option value="">Select a project...</option>' +
            projects.map(p => `<option value="${escHtml(p.path)}">${escHtml(p.name)} (${escHtml(p.path)})</option>`).join('');
    }
}

async function globalSearch() {
    const query = document.getElementById('global-search-input').value.trim();
    const projectPath = document.getElementById('search-project-select').value;
    if (!query || !projectPath) {
        toast('Enter a query and select a project');
        return;
    }
    const container = document.getElementById('global-search-results');
    container.innerHTML = '<div class="spinner"></div>';
    try {
        const data = await api('/api/search', {
            method: 'POST',
            body: JSON.stringify({
                query: query,
                project_path: projectPath,
                top_k: 10,
                threshold: 0.3
            })
        });
        renderSearchResults(container, data);
    } catch (e) {
        container.innerHTML = `<div style="color: var(--red);">Error: ${escHtml(e.message)}</div>`;
    }
}

function renderSearchResults(container, data) {
    if (data.results.length === 0) {
        container.innerHTML = '<div style="color: var(--text-dim); padding: 20px 0;">No results found.</div>';
        return;
    }
    const info = `<div style="color: var(--text-dim); font-size: 13px; margin-bottom: 12px;">${data.results.length} results in ${data.search_time_ms}ms</div>`;
    container.innerHTML = info + data.results.map(r => {
        let scoreClass = 'score-low';
        if (r.score >= 0.7) scoreClass = 'score-high';
        else if (r.score >= 0.5) scoreClass = 'score-mid';
        return `
            <div class="result-item">
                <div class="result-header">
                    <div>
                        <span class="result-file">${escHtml(r.file)}</span>
                        <span class="result-lines">:${r.start_line}-${r.end_line}</span>
                    </div>
                    <span class="score-badge ${scoreClass}">${r.score.toFixed(4)}</span>
                </div>
                <div class="result-snippet">${escHtml(r.snippet)}</div>
            </div>
        `;
    }).join('');
}

// --- Navigation ---
function navigateTo(page, param) {
    const main = document.getElementById('main-content');
    document.querySelectorAll('.nav-links a').forEach(a => a.classList.remove('active'));

    if (page === 'home') {
        document.querySelector('[data-page="home"]').classList.add('active');
        main.innerHTML = `
            <div class="page-header">
                <h1>Projects</h1>
                <button class="btn btn-primary" onclick="showAddModal()">+ Add Project</button>
            </div>
            <div class="cards" id="projects-grid"></div>
        `;
        loadProjects();
    } else if (page === 'project') {
        document.querySelector('[data-page="home"]').classList.add('active');
        loadProjectDetail(param);
    } else if (page === 'search') {
        document.querySelector('[data-page="search"]').classList.add('active');
        main.innerHTML = `
            <div class="page-header"><h1>Semantic Search</h1></div>
            <div class="search-box">
                <input class="search-input" id="global-search-input" type="text" placeholder="Enter a semantic query..."
                    onkeydown="if(event.key==='Enter') globalSearch()">
                <select id="search-project-select"></select>
                <button class="btn btn-primary" onclick="globalSearch()">Search</button>
            </div>
            <div id="global-search-results"></div>
        `;
        initSearchPage();
    } else if (page === 'analytics') {
        document.querySelector('[data-page="analytics"]').classList.add('active');
        main.innerHTML = `
            <div class="page-header"><h1>Search Analytics</h1></div>
            <div id="analytics-summary" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px;"></div>
            <h2 style="font-size: 18px; margin-bottom: 16px;">Daily Comparison (Last 7 Days)</h2>
            <div id="analytics-daily" style="margin-bottom: 32px;"></div>
            <h2 style="font-size: 18px; margin-bottom: 16px;">Semantic vs Grep</h2>
            <div id="analytics-comparison" style="margin-bottom: 32px;"></div>
            <h2 style="font-size: 18px; margin-bottom: 16px;">Recent Searches</h2>
            <div id="analytics-recent"></div>
        `;
        loadAnalytics();
    }
}

// --- Analytics page ---
async function loadAnalytics() {
    try {
        const [summary, daily, recent, comparison] = await Promise.all([
            api('/api/analytics/summary'),
            api('/api/analytics/daily'),
            api('/api/analytics/recent'),
            api('/api/analytics/comparison'),
        ]);
        renderAnalyticsSummary(summary);
        renderAnalyticsDaily(daily);
        renderAnalyticsComparison(comparison);
        renderAnalyticsRecent(recent);
    } catch (e) {
        toast('Failed to load analytics: ' + e.message);
    }
}

function renderAnalyticsSummary(s) {
    const el = document.getElementById('analytics-summary');
    if (!el) return;
    el.innerHTML = `
        <div class="card" style="cursor:default;">
            <div class="stat-label">Total Searches</div>
            <div style="font-size: 28px; font-weight: 700;">${s.total_searches}</div>
            <div style="font-size: 12px; color: var(--text-dim); margin-top: 4px;">
                <span style="color: var(--accent);">semantic: ${s.semantic_count}</span> /
                <span style="color: var(--gray);">grep: ${s.grep_count}</span>
            </div>
        </div>
        <div class="card" style="cursor:default;">
            <div class="stat-label">Total Tokens</div>
            <div style="font-size: 28px; font-weight: 700;">${s.total_tokens.toLocaleString()}</div>
        </div>
        <div class="card" style="cursor:default;">
            <div class="stat-label">Avg Tokens / Search</div>
            <div style="font-size: 28px; font-weight: 700;">${s.avg_tokens}</div>
        </div>
        <div class="card" style="cursor:default;">
            <div class="stat-label">Token Savings</div>
            <div style="font-size: 28px; font-weight: 700; color: ${s.savings_pct > 0 ? 'var(--green)' : s.savings_pct < 0 ? 'var(--red)' : 'var(--text)'};">
                ${s.savings_pct > 0 ? '+' : ''}${s.savings_pct}%
            </div>
            <div style="font-size: 12px; color: var(--text-dim); margin-top: 4px;">semantic vs grep</div>
        </div>
    `;
}

function renderAnalyticsDaily(days) {
    const el = document.getElementById('analytics-daily');
    if (!el) return;
    if (!days.length) {
        el.innerHTML = '<div style="color: var(--text-dim); padding: 20px 0;">No data yet.</div>';
        return;
    }
    const maxCount = Math.max(...days.map(d => Math.max(d.semantic, d.grep)), 1);
    el.innerHTML = `
        <div class="bar-chart">
            ${days.map(d => `
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; font-size: 13px;">
                    <div style="width: 90px; color: var(--text-dim); text-align: right; flex-shrink: 0;">${d.date}</div>
                    <div style="flex: 1; display: flex; flex-direction: column; gap: 2px;">
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <div style="height: 16px; background: var(--accent); border-radius: 3px; min-width: 2px; width: ${Math.max((d.semantic / maxCount) * 100, 0.5)}%;"></div>
                            <span style="color: var(--accent); font-size: 12px;">${d.semantic} semantic</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <div style="height: 16px; background: var(--gray); border-radius: 3px; min-width: 2px; width: ${Math.max((d.grep / maxCount) * 100, 0.5)}%;"></div>
                            <span style="color: var(--gray); font-size: 12px;">${d.grep} grep</span>
                        </div>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

function renderAnalyticsComparison(comp) {
    const el = document.getElementById('analytics-comparison');
    if (!el) return;
    const sem = comp.semantic || { count: 0, total_tokens: 0, avg_tokens: 0, avg_time_ms: 0 };
    const grp = comp.grep || { count: 0, total_tokens: 0, avg_tokens: 0, avg_time_ms: 0 };
    el.innerHTML = `
        <table>
            <thead><tr><th>Metric</th><th style="color: var(--accent);">Semantic</th><th style="color: var(--gray);">Grep</th></tr></thead>
            <tbody>
                <tr><td>Total Searches</td><td>${sem.count}</td><td>${grp.count}</td></tr>
                <tr><td>Total Tokens</td><td>${sem.total_tokens.toLocaleString()}</td><td>${grp.total_tokens.toLocaleString()}</td></tr>
                <tr><td>Avg Tokens</td><td>${sem.avg_tokens}</td><td>${grp.avg_tokens}</td></tr>
                <tr><td>Avg Time (ms)</td><td>${sem.avg_time_ms}</td><td>${grp.avg_time_ms}</td></tr>
            </tbody>
        </table>
    `;
}

function renderAnalyticsRecent(searches) {
    const el = document.getElementById('analytics-recent');
    if (!el) return;
    if (!searches.length) {
        el.innerHTML = '<div style="color: var(--text-dim); padding: 20px 0;">No searches logged yet.</div>';
        return;
    }
    el.innerHTML = `
        <table>
            <thead><tr><th>Time</th><th>Type</th><th>Query</th><th>Results</th><th>Tokens</th><th>Time (ms)</th></tr></thead>
            <tbody>
                ${searches.map(s => {
                    const dt = new Date(s.timestamp * 1000);
                    const timeStr = dt.toLocaleString();
                    const badgeColor = s.search_type === 'semantic' ? 'var(--accent)' : 'var(--gray)';
                    const badgeBg = s.search_type === 'semantic' ? 'rgba(88,166,255,0.15)' : 'rgba(139,148,158,0.15)';
                    return `
                        <tr>
                            <td style="white-space: nowrap; font-size: 12px; color: var(--text-dim);">${escHtml(timeStr)}</td>
                            <td><span style="padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; background: ${badgeBg}; color: ${badgeColor};">${s.search_type}</span></td>
                            <td style="max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${escHtml(s.query)}">${escHtml(s.query)}</td>
                            <td>${s.results_count}</td>
                            <td>${s.total_tokens}</td>
                            <td>${s.search_time_ms}</td>
                        </tr>
                    `;
                }).join('')}
            </tbody>
        </table>
    `;
}

function escHtml(s) {
    if (s == null) return '';
    const d = document.createElement('div');
    d.textContent = String(s);
    return d.innerHTML;
}

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
    navigateTo('home');
});
"""

_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>localgrep Dashboard</title>
<style>
{_CSS}
</style>
</head>
<body>
<nav>
    <div class="container">
        <div class="logo"><span>&#9906;</span> localgrep</div>
        <div class="nav-links">
            <a href="#" data-page="home" onclick="navigateTo('home'); return false;">Projects</a>
            <a href="#" data-page="search" onclick="navigateTo('search'); return false;">Search</a>
            <a href="#" data-page="analytics" onclick="navigateTo('analytics'); return false;">Analytics</a>
        </div>
    </div>
</nav>
<div class="container" id="main-content"></div>

<!-- Add project modal -->
<div class="modal-overlay" id="add-modal" onclick="if(event.target===this) hideAddModal();">
    <div class="modal">
        <h3>Add Project</h3>
        <input type="text" id="add-path-input" placeholder="/path/to/project"
            onkeydown="if(event.key==='Enter') addProject()">
        <div class="modal-actions">
            <button class="btn" onclick="hideAddModal()">Cancel</button>
            <button class="btn btn-primary" onclick="addProject()">Add</button>
        </div>
    </div>
</div>

<!-- Toast -->
<div class="toast" id="toast"></div>

<script>
{_JS}
</script>
</body>
</html>"""


@dashboard_app.get("/", response_class=HTMLResponse)
async def root():
    return _HTML


# ---------------------------------------------------------------------------
# Server runner
# ---------------------------------------------------------------------------

def run_dashboard(host: str = "127.0.0.1", port: int = 8585) -> None:
    """대시보드 서버를 시작한다."""
    import uvicorn
    print(f"localgrep dashboard: http://{host}:{port}")
    uvicorn.run(dashboard_app, host=host, port=port, log_level="info")
