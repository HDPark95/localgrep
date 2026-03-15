"""파일 크롤러 - 프로젝트 디렉토리를 순회하며 인덱싱 대상 파일을 수집한다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gitignorefile


@dataclass
class FileInfo:
    """크롤링된 파일의 메타데이터.

    Attributes:
        path: 파일의 절대 경로.
        relative_path: 프로젝트 루트 기준 상대 경로.
        mtime: 마지막 수정 시간 (Unix timestamp).
        size_bytes: 파일 크기 (바이트).
    """

    path: Path
    relative_path: str
    mtime: float
    size_bytes: int


class FileCrawler:
    """프로젝트 디렉토리를 순회하며 인덱싱 대상 파일을 수집한다.

    .gitignore 규칙을 존중하고, 설정에 따라 파일을 필터링한다.

    Attributes:
        root: 프로젝트 루트 디렉토리.
        max_file_size_kb: 최대 파일 크기 제한 (KB).
    """

    DEFAULT_IGNORE_DIRS = frozenset({
        "node_modules", ".git", "dist", "build",
        "__pycache__", ".venv", "venv", ".localgrep",
    })

    # 바이너리로 간주하는 확장자
    _BINARY_EXTENSIONS = frozenset({
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
        ".mp3", ".mp4", ".avi", ".mov", ".wav",
        ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".exe", ".dll", ".so", ".dylib", ".o", ".a",
        ".pyc", ".pyo", ".class", ".jar",
        ".woff", ".woff2", ".ttf", ".eot",
        ".sqlite", ".db",
    })

    def __init__(
        self,
        root: Path,
        max_file_size_kb: int = 512,
        extra_ignore_patterns: list[str] | None = None,
    ) -> None:
        """FileCrawler를 초기화한다.

        Args:
            root: 프로젝트 루트 디렉토리 경로.
            max_file_size_kb: 이 크기를 초과하는 파일은 무시한다.
            extra_ignore_patterns: 추가 무시 패턴 목록.
        """
        self.root = root.resolve()
        self.max_file_size_kb = max_file_size_kb
        self._extra_ignore_patterns = extra_ignore_patterns or []
        self._gitignore_match = self._load_gitignore()

    def _load_gitignore(self) -> gitignorefile.Cache:
        """프로젝트의 .gitignore 규칙을 로드한다."""
        return gitignorefile.Cache()

    def crawl(self) -> list[FileInfo]:
        """프로젝트 디렉토리를 순회하여 인덱싱 대상 파일 목록을 반환한다.

        .gitignore 규칙과 기본 무시 패턴을 적용한다.

        Returns:
            인덱싱 대상 FileInfo 리스트.
        """
        files: list[FileInfo] = []
        max_bytes = self.max_file_size_kb * 1024

        for file_path in self._walk(self.root):
            try:
                stat = file_path.stat()
            except OSError:
                continue

            if stat.st_size > max_bytes or stat.st_size == 0:
                continue

            if not self._is_text_file(file_path):
                continue

            relative = str(file_path.relative_to(self.root))
            files.append(FileInfo(
                path=file_path,
                relative_path=relative,
                mtime=stat.st_mtime,
                size_bytes=stat.st_size,
            ))

        return files

    def _walk(self, directory: Path) -> list[Path]:
        """디렉토리를 재귀 순회하며 파일 경로를 수집한다."""
        result: list[Path] = []
        try:
            entries = sorted(directory.iterdir())
        except PermissionError:
            return result

        for entry in entries:
            if entry.is_dir():
                if entry.name in self.DEFAULT_IGNORE_DIRS:
                    continue
                if entry.name.startswith(".") and entry.name != ".":
                    continue
                if self._is_ignored(entry):
                    continue
                result.extend(self._walk(entry))
            elif entry.is_file():
                if self._is_ignored(entry):
                    continue
                result.append(entry)

        return result

    def _is_ignored(self, path: Path) -> bool:
        """파일이 .gitignore 또는 추가 패턴에 의해 무시되는지 확인한다."""
        str_path = str(path)
        if self._gitignore_match(str_path):
            return True
        relative = str(path.relative_to(self.root))
        for pattern in self._extra_ignore_patterns:
            if path.match(pattern) or relative == pattern:
                return True
        return False

    def get_changed_files(self, known_files: dict[str, float]) -> tuple[list[FileInfo], list[str]]:
        """마지막 인덱싱 이후 변경/추가/삭제된 파일을 감지한다.

        Args:
            known_files: 기존 인덱싱된 파일 맵 {상대경로: mtime}.

        Returns:
            (변경/추가된 파일 리스트, 삭제된 파일 상대경로 리스트) 튜플.
        """
        current_files = self.crawl()
        current_map = {f.relative_path: f for f in current_files}

        changed: list[FileInfo] = []
        for file_info in current_files:
            old_mtime = known_files.get(file_info.relative_path)
            if old_mtime is None or file_info.mtime > old_mtime:
                changed.append(file_info)

        deleted = [p for p in known_files if p not in current_map]

        return changed, deleted

    def _is_text_file(self, path: Path) -> bool:
        """바이너리 파일이 아닌 텍스트 파일인지 판별한다.

        Args:
            path: 확인할 파일 경로.

        Returns:
            텍스트 파일이면 True.
        """
        if path.suffix.lower() in self._BINARY_EXTENSIONS:
            return False

        # 확장자가 없거나 알 수 없는 경우 첫 8KB를 읽어 바이너리 여부 판별
        try:
            with open(path, "rb") as f:
                chunk = f.read(8192)
            # null 바이트가 있으면 바이너리로 간주
            return b"\x00" not in chunk
        except (OSError, PermissionError):
            return False
