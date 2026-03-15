"""E2E 테스트 - 전체 파이프라인 (Ollama 실행 필요).

인덱싱 → 검색 → 결과 검증 전체 흐름을 테스트한다.
"""

import asyncio
import json
import math
from pathlib import Path

import pytest

from localgrep.chunker import SlidingWindowChunker
from localgrep.crawler import FileCrawler
from localgrep.embedder import OllamaEmbedder
from localgrep.store import VectorStore, Chunk as StoreChunk


def _create_sample_project(root: Path) -> None:
    """테스트용 샘플 프로젝트 생성."""
    (root / "src").mkdir(parents=True)

    (root / "src" / "auth.py").write_text(
        '''\
"""사용자 인증 및 권한 관리 모듈."""

from typing import Optional


class AuthMiddleware:
    """HTTP 요청에서 인증 토큰을 검증하는 미들웨어."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    async def authenticate(self, request) -> Optional[dict]:
        """Authorization 헤더에서 JWT 토큰을 추출하고 검증한다."""
        token = request.headers.get("Authorization")
        if not token:
            return None
        return self._verify_jwt(token)

    def _verify_jwt(self, token: str) -> dict:
        """JWT 토큰을 검증하고 페이로드를 반환한다."""
        # JWT 검증 로직
        pass


def require_permission(permission: str):
    """특정 권한이 필요한 엔드포인트 데코레이터."""
    def decorator(func):
        async def wrapper(request, *args, **kwargs):
            user = request.state.user
            if permission not in user.get("permissions", []):
                raise PermissionError(f"Required permission: {permission}")
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator
''',
        encoding="utf-8",
    )

    (root / "src" / "database.py").write_text(
        '''\
"""데이터베이스 연결 및 쿼리 관리."""

import asyncio
from contextlib import asynccontextmanager


class ConnectionPool:
    """비동기 데이터베이스 커넥션 풀."""

    def __init__(self, dsn: str, min_size: int = 5, max_size: int = 20):
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self._pool = None

    async def initialize(self):
        """커넥션 풀을 초기화한다."""
        # asyncpg 풀 생성
        pass

    async def close(self):
        """모든 커넥션을 정리하고 풀을 종료한다."""
        if self._pool:
            await self._pool.close()

    @asynccontextmanager
    async def acquire(self):
        """풀에서 커넥션을 하나 가져온다."""
        conn = await self._pool.acquire()
        try:
            yield conn
        finally:
            await self._pool.release(conn)


async def run_migration(pool: ConnectionPool, version: int):
    """데이터베이스 마이그레이션을 실행한다."""
    async with pool.acquire() as conn:
        await conn.execute("SELECT apply_migration($1)", version)
''',
        encoding="utf-8",
    )

    (root / "src" / "api.py").write_text(
        '''\
"""REST API 엔드포인트 정의."""

from typing import List


class UserRouter:
    """사용자 관련 API 라우터."""

    def __init__(self, db, auth):
        self.db = db
        self.auth = auth

    async def get_users(self, page: int = 1, limit: int = 20) -> List[dict]:
        """사용자 목록을 페이지네이션하여 반환한다."""
        offset = (page - 1) * limit
        return await self.db.fetch_all(
            "SELECT id, name, email FROM users LIMIT $1 OFFSET $2",
            limit, offset,
        )

    async def create_user(self, data: dict) -> dict:
        """새 사용자를 생성한다."""
        return await self.db.insert("users", data)

    async def delete_user(self, user_id: int) -> bool:
        """사용자를 삭제한다."""
        result = await self.db.execute(
            "DELETE FROM users WHERE id = $1", user_id
        )
        return result > 0


class HealthRouter:
    """헬스체크 API."""

    async def health(self) -> dict:
        """서버 상태를 반환한다."""
        return {"status": "ok", "version": "1.0.0"}
''',
        encoding="utf-8",
    )

    (root / "README.md").write_text("# Sample Project\n\nA test project.\n")


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    _create_sample_project(root)
    return root


class TestE2EPipeline:
    """전체 인덱싱 → 검색 파이프라인 테스트."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, sample_project: Path):
        """파일 크롤링 → 청킹 → 임베딩 → 인덱싱 → 검색."""
        # 1. 크롤링
        crawler = FileCrawler(sample_project)
        files = crawler.crawl()
        assert len(files) >= 3  # auth.py, database.py, api.py

        # 2. 청킹
        chunker = SlidingWindowChunker()
        all_chunks = []
        for fi in files:
            content = fi.path.read_text(encoding="utf-8")
            chunks = chunker.chunk(fi.relative_path, content)
            all_chunks.extend(chunks)
        assert len(all_chunks) > 0

        # 3. 임베딩 + 인덱싱
        db_path = sample_project / ".localgrep" / "index.db"
        store = VectorStore(db_path)

        async with OllamaEmbedder() as embedder:
            assert await embedder.health_check()

            for fi in files:
                content = fi.path.read_text(encoding="utf-8")
                chunks = chunker.chunk(fi.relative_path, content)
                if not chunks:
                    continue

                import hashlib
                h = hashlib.sha256(content.encode()).hexdigest()
                file_id = store.upsert_file(fi.relative_path, fi.mtime, h)

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

            # 4. 통계 확인
            stats = store.get_stats()
            assert stats["indexed_files"] >= 3
            assert stats["total_chunks"] > 0

            # 5. 시맨틱 검색 테스트
            # 인증 관련 검색
            q_emb = await embedder.embed("사용자 인증 미들웨어")
            results = store.search(q_emb, top_k=3, threshold=0.0)
            assert len(results) > 0
            # auth.py가 상위에 있어야 함
            top_files = [r.file for r in results[:2]]
            assert any("auth" in f for f in top_files), f"auth.py not in top results: {top_files}"

            # DB 관련 검색
            q_emb = await embedder.embed("database connection pool configuration")
            results = store.search(q_emb, top_k=5, threshold=0.0)
            assert len(results) > 0
            top_files = [r.file for r in results]
            assert any("database" in f for f in top_files), f"database.py not in results: {top_files}"

            # API 엔드포인트 검색
            q_emb = await embedder.embed("REST API endpoint user list")
            results = store.search(q_emb, top_k=5, threshold=0.0)
            assert len(results) > 0
            top_files = [r.file for r in results]
            assert any("api" in f for f in top_files), f"api.py not in results: {top_files}"

        store.close()

    @pytest.mark.asyncio
    async def test_incremental_indexing(self, sample_project: Path):
        """증분 인덱싱: 파일 추가/수정/삭제 시 정상 동작."""
        db_path = sample_project / ".localgrep" / "index.db"
        store = VectorStore(db_path)
        chunker = SlidingWindowChunker()

        async with OllamaEmbedder() as embedder:
            # 초기 인덱싱
            crawler = FileCrawler(sample_project)
            for fi in crawler.crawl():
                content = fi.path.read_text(encoding="utf-8")
                import hashlib
                h = hashlib.sha256(content.encode()).hexdigest()
                file_id = store.upsert_file(fi.relative_path, fi.mtime, h)

                chunks = chunker.chunk(fi.relative_path, content)
                if chunks:
                    texts = [c.embeddable_text for c in chunks]
                    embeddings = await embedder.embed_batch(texts)
                    store_chunks = [
                        StoreChunk(c.start_line, c.end_line, c.content, emb)
                        for c, emb in zip(chunks, embeddings)
                    ]
                    store.add_chunks(file_id, store_chunks)

            initial_stats = store.get_stats()
            initial_files = initial_stats["indexed_files"]

            # 파일 추가
            (sample_project / "src" / "new_module.py").write_text(
                "def new_function():\n    return 42\n"
            )

            # 파일 삭제
            (sample_project / "README.md").unlink()

            # 변경 감지
            known = {}
            cur = store._conn.cursor()
            cur.execute("SELECT path, mtime FROM files")
            for row in cur.fetchall():
                known[row[0]] = row[1]

            changed, deleted = crawler.get_changed_files(known)

            # 삭제 처리
            for d in deleted:
                store.delete_file(d)

            # 변경/추가 처리
            for fi in changed:
                content = fi.path.read_text(encoding="utf-8")
                import hashlib
                h = hashlib.sha256(content.encode()).hexdigest()
                file_id = store.upsert_file(fi.relative_path, fi.mtime, h)
                chunks = chunker.chunk(fi.relative_path, content)
                if chunks:
                    texts = [c.embeddable_text for c in chunks]
                    embeddings = await embedder.embed_batch(texts)
                    store_chunks = [
                        StoreChunk(c.start_line, c.end_line, c.content, emb)
                        for c, emb in zip(chunks, embeddings)
                    ]
                    store.add_chunks(file_id, store_chunks)

            final_stats = store.get_stats()
            # README 삭제 + new_module 추가 → 파일 수 동일
            assert final_stats["indexed_files"] == initial_files

        store.close()

    @pytest.mark.asyncio
    async def test_search_result_format(self, sample_project: Path):
        """검색 결과가 SPEC 출력 형식과 일치하는지 확인."""
        db_path = sample_project / ".localgrep" / "index.db"
        store = VectorStore(db_path)
        chunker = SlidingWindowChunker()

        async with OllamaEmbedder() as embedder:
            crawler = FileCrawler(sample_project)
            for fi in crawler.crawl():
                content = fi.path.read_text(encoding="utf-8")
                import hashlib
                h = hashlib.sha256(content.encode()).hexdigest()
                file_id = store.upsert_file(fi.relative_path, fi.mtime, h)
                chunks = chunker.chunk(fi.relative_path, content)
                if chunks:
                    texts = [c.embeddable_text for c in chunks]
                    embeddings = await embedder.embed_batch(texts)
                    store_chunks = [
                        StoreChunk(c.start_line, c.end_line, c.content, emb)
                        for c, emb in zip(chunks, embeddings)
                    ]
                    store.add_chunks(file_id, store_chunks)

            q_emb = await embedder.embed("authentication")
            results = store.search(q_emb, top_k=5, threshold=0.0)

            for r in results:
                # SPEC 출력 형식 검증
                assert isinstance(r.file, str)
                assert isinstance(r.start_line, int) and r.start_line >= 1
                assert isinstance(r.end_line, int) and r.end_line >= r.start_line
                assert isinstance(r.score, float) and 0.0 <= r.score <= 1.0
                assert isinstance(r.snippet, str) and len(r.snippet) > 0

        store.close()
