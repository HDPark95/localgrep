"""config.py 단위 테스트."""

import json
from pathlib import Path

from localgrep.config import (
    ChunkingConfig,
    IndexingConfig,
    LocalGrepConfig,
    OllamaConfig,
    SearchConfig,
    load_config,
    save_config,
)


def test_default_config():
    """기본 설정값이 SPEC과 일치하는지 확인."""
    cfg = LocalGrepConfig()
    assert cfg.ollama.host == "http://localhost:11434"
    assert cfg.ollama.model == "mxbai-embed-large"
    assert cfg.chunking.max_lines == 100
    assert cfg.chunking.overlap_lines == 10
    assert cfg.chunking.min_lines == 3
    assert cfg.indexing.max_file_size_kb == 512
    assert cfg.search.default_top_k == 5
    assert cfg.search.default_threshold == 0.3


def test_load_config_no_file(tmp_path: Path):
    """설정 파일 없으면 기본값 반환."""
    cfg = load_config(tmp_path)
    assert cfg.ollama.model == "mxbai-embed-large"


def test_save_and_load_config(tmp_path: Path):
    """설정 저장 후 로드하면 동일한 값."""
    cfg = LocalGrepConfig(
        ollama=OllamaConfig(host="http://custom:1234", model="custom-model"),
        chunking=ChunkingConfig(max_lines=200),
    )
    save_config(tmp_path, cfg)

    loaded = load_config(tmp_path)
    assert loaded.ollama.host == "http://custom:1234"
    assert loaded.ollama.model == "custom-model"
    assert loaded.chunking.max_lines == 200
    # 기본값은 유지
    assert loaded.search.default_top_k == 5


def test_config_file_is_json(tmp_path: Path):
    """설정 파일이 유효한 JSON인지 확인."""
    save_config(tmp_path, LocalGrepConfig())
    path = tmp_path / ".localgrep" / "config.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert "ollama" in data
    assert "chunking" in data
