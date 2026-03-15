"""store.py 단위 테스트."""

import math
from pathlib import Path

from localgrep.store import Chunk, SearchResult, VectorStore


def _dummy_embedding(dim: int = 1024, val: float = 0.1) -> list[float]:
    """정규화된 더미 임베딩 벡터 생성."""
    vec = [val] * dim
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


def _similar_embedding(dim: int = 1024) -> list[float]:
    """_dummy_embedding과 유사한 벡터."""
    return _dummy_embedding(dim, val=0.1)


def _different_embedding(dim: int = 1024) -> list[float]:
    """_dummy_embedding과 다른 벡터."""
    vec = [0.0] * dim
    vec[0] = 1.0  # 완전히 다른 방향
    return vec


class TestVectorStore:
    def test_create_db(self, tmp_path: Path):
        """DB 생성 및 스키마 확인."""
        db_path = tmp_path / "test.db"
        store = VectorStore(db_path)
        assert db_path.exists()
        stats = store.get_stats()
        assert stats["indexed_files"] == 0
        assert stats["total_chunks"] == 0
        store.close()

    def test_upsert_file_new(self, tmp_path: Path):
        """새 파일 등록."""
        store = VectorStore(tmp_path / "test.db")
        file_id = store.upsert_file("src/main.py", 1000.0, "abc123")
        assert file_id is not None
        assert isinstance(file_id, int)
        assert store.get_file_hash("src/main.py") == "abc123"
        store.close()

    def test_upsert_file_unchanged(self, tmp_path: Path):
        """동일 해시 → 같은 file_id 반환."""
        store = VectorStore(tmp_path / "test.db")
        id1 = store.upsert_file("main.py", 1000.0, "hash1")
        id2 = store.upsert_file("main.py", 1000.0, "hash1")
        assert id1 == id2
        store.close()

    def test_upsert_file_changed(self, tmp_path: Path):
        """해시 변경 → 기존 청크 삭제, 같은 file_id."""
        store = VectorStore(tmp_path / "test.db")
        file_id = store.upsert_file("main.py", 1000.0, "hash1")
        store.add_chunks(file_id, [
            Chunk(start_line=1, end_line=10, content="old code", embedding=_dummy_embedding()),
        ])
        assert store.get_stats()["total_chunks"] == 1

        # 해시 변경
        file_id2 = store.upsert_file("main.py", 2000.0, "hash2")
        assert file_id == file_id2
        assert store.get_stats()["total_chunks"] == 0  # 청크 삭제됨
        store.close()

    def test_add_chunks_and_search(self, tmp_path: Path):
        """청크 저장 후 벡터 검색."""
        store = VectorStore(tmp_path / "test.db")
        file_id = store.upsert_file("auth.py", 1000.0, "h1")
        emb = _dummy_embedding()
        store.add_chunks(file_id, [
            Chunk(start_line=1, end_line=20, content="auth middleware code", embedding=emb),
        ])

        results = store.search(_similar_embedding(), top_k=5, threshold=0.0)
        assert len(results) >= 1
        assert results[0].file == "auth.py"
        assert results[0].score > 0.9  # 같은 벡터 = 높은 유사도
        store.close()

    def test_search_threshold(self, tmp_path: Path):
        """threshold 이하 결과는 제외."""
        store = VectorStore(tmp_path / "test.db")
        file_id = store.upsert_file("main.py", 1000.0, "h1")
        store.add_chunks(file_id, [
            Chunk(start_line=1, end_line=10, content="some code", embedding=_dummy_embedding()),
        ])

        # 매우 다른 벡터로 검색 → 낮은 유사도
        results = store.search(_different_embedding(), top_k=5, threshold=0.9)
        assert len(results) == 0
        store.close()

    def test_search_file_pattern(self, tmp_path: Path):
        """file_pattern 필터링."""
        store = VectorStore(tmp_path / "test.db")
        emb = _dummy_embedding()

        fid1 = store.upsert_file("main.py", 1000.0, "h1")
        store.add_chunks(fid1, [
            Chunk(start_line=1, end_line=10, content="python code", embedding=emb),
        ])
        fid2 = store.upsert_file("app.js", 1000.0, "h2")
        store.add_chunks(fid2, [
            Chunk(start_line=1, end_line=10, content="js code", embedding=emb),
        ])

        results = store.search(emb, top_k=10, threshold=0.0, file_pattern="*.py")
        files = {r.file for r in results}
        assert "main.py" in files
        assert "app.js" not in files
        store.close()

    def test_delete_file(self, tmp_path: Path):
        """파일 삭제 시 청크도 삭제."""
        store = VectorStore(tmp_path / "test.db")
        file_id = store.upsert_file("temp.py", 1000.0, "h1")
        store.add_chunks(file_id, [
            Chunk(start_line=1, end_line=5, content="temp", embedding=_dummy_embedding()),
        ])
        assert store.get_stats()["total_chunks"] == 1

        store.delete_file("temp.py")
        assert store.get_stats()["indexed_files"] == 0
        assert store.get_stats()["total_chunks"] == 0
        store.close()

    def test_get_stats(self, tmp_path: Path):
        """통계 반환."""
        store = VectorStore(tmp_path / "test.db")
        fid = store.upsert_file("a.py", 1500.0, "h1")
        store.add_chunks(fid, [
            Chunk(start_line=1, end_line=10, content="c1", embedding=_dummy_embedding()),
            Chunk(start_line=11, end_line=20, content="c2", embedding=_dummy_embedding()),
        ])
        stats = store.get_stats()
        assert stats["indexed_files"] == 1
        assert stats["total_chunks"] == 2
        assert stats["last_updated"] == 1500.0
        store.close()

    def test_context_manager(self, tmp_path: Path):
        """with 문 지원."""
        with VectorStore(tmp_path / "test.db") as store:
            fid = store.upsert_file("x.py", 1000.0, "h")
            assert fid is not None
