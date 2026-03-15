"""embedder.py 단위/통합 테스트 (Ollama 실행 필요)."""

import pytest
import pytest_asyncio

from localgrep.embedder import OllamaEmbedder, OllamaEmbedderError


@pytest_asyncio.fixture
async def embedder():
    async with OllamaEmbedder() as emb:
        yield emb


class TestOllamaEmbedder:
    @pytest.mark.asyncio
    async def test_health_check(self, embedder: OllamaEmbedder):
        """Ollama 서버 연결 확인."""
        assert await embedder.health_check() is True

    @pytest.mark.asyncio
    async def test_embed_single(self, embedder: OllamaEmbedder):
        """단일 텍스트 임베딩."""
        vec = await embedder.embed("hello world")
        assert isinstance(vec, list)
        assert len(vec) == 768
        assert all(isinstance(v, float) for v in vec)

    @pytest.mark.asyncio
    async def test_embed_batch(self, embedder: OllamaEmbedder):
        """배치 임베딩."""
        texts = ["hello", "world", "test"]
        vecs = await embedder.embed_batch(texts)
        assert len(vecs) == 3
        assert all(len(v) == 768 for v in vecs)

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self, embedder: OllamaEmbedder):
        """빈 배치."""
        vecs = await embedder.embed_batch([])
        assert vecs == []

    @pytest.mark.asyncio
    async def test_similar_texts_similar_embeddings(self, embedder: OllamaEmbedder):
        """유사한 텍스트는 유사한 임베딩을 생성."""
        v1 = await embedder.embed("user authentication middleware")
        v2 = await embedder.embed("auth middleware for users")
        v3 = await embedder.embed("database connection pooling")

        # 코사인 유사도 계산
        def cosine_sim(a, b):
            import math
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb)

        sim_similar = cosine_sim(v1, v2)
        sim_different = cosine_sim(v1, v3)

        # 유사한 텍스트가 더 높은 유사도
        assert sim_similar > sim_different

    @pytest.mark.asyncio
    async def test_connection_error(self):
        """잘못된 호스트에 연결 시 에러."""
        async with OllamaEmbedder(host="http://localhost:99999", timeout=2.0) as emb:
            with pytest.raises(OllamaEmbedderError, match="연결"):
                await emb.embed("test")

    @pytest.mark.asyncio
    async def test_wrong_model_error(self):
        """존재하지 않는 모델 사용 시 에러."""
        async with OllamaEmbedder(model="nonexistent-model-xyz") as emb:
            with pytest.raises(OllamaEmbedderError):
                await emb.embed("test")
