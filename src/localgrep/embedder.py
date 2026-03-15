"""Ollama 임베딩 클라이언트 - 텍스트를 벡터로 변환한다."""

from __future__ import annotations

import httpx


class OllamaEmbedderError(Exception):
    """Ollama 임베딩 관련 에러."""


class OllamaEmbedder:
    """Ollama API를 사용하여 텍스트 임베딩을 생성하는 클라이언트.

    Attributes:
        host: Ollama 서버 주소 (기본: http://localhost:11434).
        model: 임베딩 모델 이름 (기본: nomic-embed-text).
        dimension: 임베딩 벡터 차원 수 (nomic-embed-text = 768).
    """

    DEFAULT_HOST = "http://localhost:11434"
    DEFAULT_MODEL = "nomic-embed-text"
    DIMENSION = 768

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        timeout: float = 30.0,
    ) -> None:
        """OllamaEmbedder를 초기화한다.

        Args:
            host: Ollama 서버 주소.
            model: 사용할 임베딩 모델 이름.
            timeout: HTTP 요청 타임아웃(초).
        """
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트를 lazy 초기화하여 반환한다."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.host,
                timeout=self.timeout,
            )
        return self._client

    async def embed(self, text: str) -> list[float]:
        """단일 텍스트를 임베딩 벡터로 변환한다.

        Args:
            text: 임베딩할 텍스트.

        Returns:
            768차원 float 벡터.

        Raises:
            OllamaEmbedderError: Ollama 서버 연결 실패 또는 API 에러 시.
        """
        embeddings = await self._request_embed([text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """여러 텍스트를 배치로 임베딩한다.

        Args:
            texts: 임베딩할 텍스트 목록.
            batch_size: 한 번에 처리할 텍스트 수.

        Returns:
            각 텍스트에 대한 임베딩 벡터 리스트.

        Raises:
            OllamaEmbedderError: Ollama 서버 연결 실패 또는 API 에러 시.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = await self._request_embed(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    async def _request_embed(self, inputs: list[str]) -> list[list[float]]:
        """Ollama /api/embed 엔드포인트를 호출한다."""
        client = await self._get_client()
        payload = {
            "model": self.model,
            "input": inputs,
        }
        try:
            response = await client.post("/api/embed", json=payload)
        except httpx.ConnectError as e:
            raise OllamaEmbedderError(
                f"Ollama 서버에 연결할 수 없습니다 ({self.host}). "
                "Ollama가 실행 중인지 확인하세요: ollama serve"
            ) from e
        except httpx.TimeoutException as e:
            raise OllamaEmbedderError(
                f"Ollama 요청 시간 초과 ({self.timeout}초). "
                "모델이 로딩 중일 수 있습니다. 잠시 후 다시 시도하세요."
            ) from e

        if response.status_code != 200:
            body = response.text
            if response.status_code == 404:
                raise OllamaEmbedderError(
                    f"모델 '{self.model}'을 찾을 수 없습니다. "
                    f"먼저 모델을 다운로드하세요: ollama pull {self.model}"
                )
            raise OllamaEmbedderError(
                f"Ollama API 에러 (HTTP {response.status_code}): {body}"
            )

        data = response.json()
        embeddings = data.get("embeddings")
        if embeddings is None:
            raise OllamaEmbedderError(
                f"Ollama 응답에 'embeddings' 필드가 없습니다: {data}"
            )
        return embeddings

    async def health_check(self) -> bool:
        """Ollama 서버 연결 상태를 확인한다.

        Returns:
            서버가 응답하면 True, 아니면 False.
        """
        client = await self._get_client()
        try:
            response = await client.get("/api/version")
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def close(self) -> None:
        """HTTP 클라이언트를 정리한다."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> OllamaEmbedder:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
