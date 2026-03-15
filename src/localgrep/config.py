"""설정 관리 - 프로젝트별/글로벌 설정을 로드하고 저장한다."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class OllamaConfig:
    """Ollama 서버 설정.

    Attributes:
        host: Ollama 서버 주소.
        model: 임베딩 모델 이름.
    """

    host: str = "http://localhost:11434"
    model: str = "mxbai-embed-large"


@dataclass
class IndexingConfig:
    """인덱싱 설정.

    Attributes:
        ignore: 무시할 디렉토리/패턴 목록.
        max_file_size_kb: 최대 파일 크기 (KB).
        extensions: 인덱싱할 확장자 목록 (None이면 모든 텍스트 파일).
    """

    ignore: list[str] = field(default_factory=lambda: [
        "node_modules", ".git", "dist", "build",
        "__pycache__", ".venv", "*.lock",
    ])
    max_file_size_kb: int = 512
    extensions: list[str] | None = None


@dataclass
class ChunkingConfig:
    """청킹 설정.

    Attributes:
        max_lines: 청크 최대 줄 수.
        overlap_lines: 청크 간 오버랩 줄 수.
        min_lines: 최소 청크 줄 수.
    """

    max_lines: int = 100
    overlap_lines: int = 10
    min_lines: int = 3


@dataclass
class SearchConfig:
    """검색 기본값 설정.

    Attributes:
        default_top_k: 기본 결과 수.
        default_threshold: 기본 유사도 임계값.
    """

    default_top_k: int = 5
    default_threshold: float = 0.3


@dataclass
class LocalGrepConfig:
    """localgrep 전체 설정.

    Attributes:
        ollama: Ollama 서버 설정.
        indexing: 인덱싱 설정.
        chunking: 청킹 설정.
        search: 검색 설정.
    """

    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)


def _config_path(project_root: Path) -> Path:
    return project_root / ".localgrep" / "config.json"


def load_config(project_root: Path) -> LocalGrepConfig:
    """프로젝트 설정 파일을 로드한다.

    {project_root}/.localgrep/config.json 파일이 있으면 로드하고,
    없으면 기본 설정을 반환한다.

    Args:
        project_root: 프로젝트 루트 디렉토리 경로.

    Returns:
        로드된 LocalGrepConfig 객체.
    """
    path = _config_path(project_root)
    if not path.exists():
        return LocalGrepConfig()

    data = json.loads(path.read_text(encoding="utf-8"))
    return LocalGrepConfig(
        ollama=OllamaConfig(**data.get("ollama", {})),
        indexing=IndexingConfig(**data.get("indexing", {})),
        chunking=ChunkingConfig(**data.get("chunking", {})),
        search=SearchConfig(**data.get("search", {})),
    )


def save_config(project_root: Path, config: LocalGrepConfig) -> None:
    """설정을 프로젝트 설정 파일에 저장한다.

    Args:
        project_root: 프로젝트 루트 디렉토리 경로.
        config: 저장할 설정 객체.
    """
    path = _config_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
