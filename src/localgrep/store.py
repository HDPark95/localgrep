"""SQLite + sqlite-vec 기반 벡터 저장소."""

from __future__ import annotations

import sqlite3
import struct
import time
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Sequence

import sqlite_vec


@dataclass
class Chunk:
    """청크 데이터."""

    start_line: int
    end_line: int
    content: str
    embedding: list[float]


@dataclass
class SearchResult:
    """검색 결과."""

    file: str
    start_line: int
    end_line: int
    score: float
    snippet: str


# sqlite-vec에 전달할 바이트 직렬화 (little-endian float32)
def _serialize_f32(vec: list[float] | Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


EMBEDDING_DIM = 768  # nomic-embed-text 차원


class VectorStore:
    """SQLite + sqlite-vec 벡터 저장소.

    Parameters
    ----------
    db_path : Path
        SQLite DB 파일 경로. 존재하지 않으면 새로 생성한다.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(db_path))
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._migrate()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _migrate(self) -> None:
        """DB 스키마를 생성/마이그레이션한다."""
        cur = self._conn.cursor()

        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                mtime REAL NOT NULL,
                hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
            """
        )

        # sqlite-vec 가상 테이블은 executescript 안에서 생성 불가 → 별도 실행
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[768] distance_metric=cosine
            )
            """
        )

        self._conn.commit()

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def upsert_file(self, path: str, mtime: float, hash: str) -> int:
        """파일 메타데이터를 upsert 하고 file_id를 반환한다.

        기존 파일이면 mtime/hash를 업데이트하고 관련 청크를 삭제한다.
        """
        cur = self._conn.cursor()

        cur.execute("SELECT id, hash FROM files WHERE path = ?", (path,))
        row = cur.fetchone()

        if row is not None:
            file_id, old_hash = row
            if old_hash == hash:
                # 변경 없음 — 기존 file_id 반환
                return file_id
            # 변경됨 — 기존 청크 삭제 후 메타데이터 갱신
            self._delete_chunks_for_file(file_id)
            cur.execute(
                "UPDATE files SET mtime = ?, hash = ? WHERE id = ?",
                (mtime, hash, file_id),
            )
            self._conn.commit()
            return file_id

        cur.execute(
            "INSERT INTO files (path, mtime, hash) VALUES (?, ?, ?)",
            (path, mtime, hash),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def delete_file(self, path: str) -> None:
        """파일과 관련 청크/벡터를 모두 삭제한다."""
        cur = self._conn.cursor()

        cur.execute("SELECT id FROM files WHERE path = ?", (path,))
        row = cur.fetchone()
        if row is None:
            return

        file_id = row[0]
        self._delete_chunks_for_file(file_id)
        cur.execute("DELETE FROM files WHERE id = ?", (file_id,))
        self._conn.commit()

    def get_file_hash(self, path: str) -> str | None:
        """파일의 저장된 해시를 반환한다. 없으면 None."""
        cur = self._conn.cursor()
        cur.execute("SELECT hash FROM files WHERE path = ?", (path,))
        row = cur.fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------
    # Chunk operations
    # ------------------------------------------------------------------

    def add_chunks(self, file_id: int, chunks: list[Chunk]) -> None:
        """파일에 청크와 임베딩을 저장한다."""
        if not chunks:
            return

        cur = self._conn.cursor()

        for chunk in chunks:
            emb_blob = _serialize_f32(chunk.embedding)

            cur.execute(
                "INSERT INTO chunks (file_id, start_line, end_line, content, embedding) "
                "VALUES (?, ?, ?, ?, ?)",
                (file_id, chunk.start_line, chunk.end_line, chunk.content, emb_blob),
            )
            chunk_id = cur.lastrowid

            cur.execute(
                "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, emb_blob),
            )

        self._conn.commit()

    def _delete_chunks_for_file(self, file_id: int) -> None:
        """파일의 모든 청크와 벡터 인덱스 항목을 삭제한다."""
        cur = self._conn.cursor()

        # chunks_vec에서 먼저 삭제 (가상 테이블은 CASCADE 미지원)
        cur.execute(
            "DELETE FROM chunks_vec WHERE chunk_id IN "
            "(SELECT id FROM chunks WHERE file_id = ?)",
            (file_id,),
        )
        # chunks는 files의 CASCADE로 삭제되지만, 명시적으로도 삭제
        cur.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        threshold: float = 0.3,
        file_pattern: str | None = None,
    ) -> list[SearchResult]:
        """벡터 유사도 검색을 수행한다.

        Parameters
        ----------
        query_embedding : 쿼리 임베딩 벡터
        top_k : 반환할 최대 결과 수
        threshold : 최소 유사도 점수 (0.0-1.0)
        file_pattern : 파일 필터 glob 패턴 (예: ``*.py``)

        Returns
        -------
        유사도 내림차순 정렬된 SearchResult 리스트
        """
        cur = self._conn.cursor()
        query_blob = _serialize_f32(query_embedding)

        # chunks_vec는 distance_metric=cosine으로 생성되어 코사인 거리를 반환한다.
        # cosine_distance = 1 - cosine_similarity → similarity = 1 - distance
        #
        # file_pattern 필터링 시 여유 있게 가져와서 후처리한다.
        fetch_k = max(top_k * 5, 50) if file_pattern else top_k

        cur.execute(
            """
            SELECT
                cv.chunk_id,
                cv.distance
            FROM chunks_vec cv
            WHERE cv.embedding MATCH ?
                AND k = ?
            ORDER BY cv.distance
            """,
            (query_blob, fetch_k),
        )

        vec_rows = cur.fetchall()
        if not vec_rows:
            return []

        # 거리 → 유사도 변환, threshold 이하 제거
        scored = [(cid, 1.0 - dist) for cid, dist in vec_rows if (1.0 - dist) >= threshold]
        if not scored:
            return []

        # 배치 조인으로 청크 메타데이터를 한 번에 가져온다
        chunk_ids = [cid for cid, _ in scored]
        placeholders = ",".join("?" * len(chunk_ids))
        cur.execute(
            f"""
            SELECT c.id, c.start_line, c.end_line, c.content, f.path
            FROM chunks c
            JOIN files f ON f.id = c.file_id
            WHERE c.id IN ({placeholders})
            """,
            chunk_ids,
        )
        chunk_map = {row[0]: row[1:] for row in cur.fetchall()}

        results: list[SearchResult] = []
        for chunk_id, score in scored:
            meta = chunk_map.get(chunk_id)
            if meta is None:
                continue

            start_line, end_line, content, file_path = meta

            if file_pattern and not fnmatch(file_path, file_pattern):
                continue

            results.append(
                SearchResult(
                    file=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    score=round(score, 4),
                    snippet=content,
                )
            )

            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """인덱스 통계를 반환한다.

        Returns
        -------
        dict with keys: indexed_files, total_chunks, last_updated
        """
        cur = self._conn.cursor()

        cur.execute("SELECT COUNT(*) FROM files")
        file_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cur.fetchone()[0]

        cur.execute("SELECT MAX(mtime) FROM files")
        row = cur.fetchone()
        last_updated = row[0] if row[0] is not None else None

        return {
            "indexed_files": file_count,
            "total_chunks": chunk_count,
            "last_updated": last_updated,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """DB 연결을 닫는다."""
        self._conn.close()

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
