"""청킹 엔진 - 코드 파일을 의미 단위로 분할한다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    """코드 파일에서 추출된 하나의 청크.

    Attributes:
        file_path: 원본 파일의 상대 경로.
        start_line: 시작 줄 번호 (1-based).
        end_line: 끝 줄 번호 (1-based, inclusive).
        content: 청크 원본 텍스트.
    """

    file_path: str
    start_line: int
    end_line: int
    content: str

    @property
    def header(self) -> str:
        """메타데이터 접두사를 반환한다. (예: 'src/auth.py:10-45')"""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    @property
    def embeddable_text(self) -> str:
        """임베딩에 사용할 텍스트를 반환한다 (헤더 + 내용)."""
        return f"{self.header}\n{self.content}"


# SPEC Phase 1: 슬라이딩 윈도우 기본, AST 청킹 기본값 50줄
DEFAULT_WINDOW_LINES = 50

# nomic-embed-text 컨텍스트: 8192 토큰. 안전하게 문자 수 제한.
# 헤더(파일경로:줄번호) 포함하므로 여유를 둠.
DEFAULT_MAX_CHARS = 6000


class SlidingWindowChunker:
    """슬라이딩 윈도우 방식으로 텍스트를 청킹한다 (폴백 전략).

    AST 파싱이 불가능한 파일에 사용된다.

    Attributes:
        max_lines: 청크 최대 줄 수 (기본: 100).
        window_lines: 기본 윈도우 크기 (기본: 50).
        overlap_lines: 청크 간 오버랩 줄 수 (기본: 10).
        min_lines: 최소 청크 줄 수, 미만이면 이전 청크에 병합 (기본: 3).
        max_chars: 청크 최대 문자 수 (기본: 6000). 임베딩 컨텍스트 초과 방지.
    """

    def __init__(
        self,
        max_lines: int = 100,
        window_lines: int = DEFAULT_WINDOW_LINES,
        overlap_lines: int = 10,
        min_lines: int = 3,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        """SlidingWindowChunker를 초기화한다.

        Args:
            max_lines: 청크 최대 줄 수.
            window_lines: 기본 윈도우 크기.
            overlap_lines: 청크 간 오버랩 줄 수.
            min_lines: 최소 청크 줄 수.
            max_chars: 청크 최대 문자 수.
        """
        self.max_lines = max_lines
        self.window_lines = window_lines
        self.overlap_lines = overlap_lines
        self.min_lines = min_lines
        self.max_chars = max_chars

    def chunk(self, file_path: str, content: str) -> list[Chunk]:
        """텍스트를 슬라이딩 윈도우 방식으로 청크 리스트로 분할한다.

        Args:
            file_path: 파일 상대 경로 (메타데이터용).
            content: 파일 전체 텍스트.

        Returns:
            Chunk 리스트.
        """
        if not content.strip():
            return []

        lines = content.splitlines(keepends=True)
        total = len(lines)

        if total <= self.window_lines:
            # 파일이 윈도우 크기보다 작으면 하나의 청크로
            chunks = [Chunk(
                file_path=file_path,
                start_line=1,
                end_line=total,
                content=content,
            )]
        else:
            chunks = self._sliding_window(file_path, lines, total)

        # max_lines 초과 청크 강제 분할
        split_by_lines: list[Chunk] = []
        for c in chunks:
            c_lines = c.content.splitlines(keepends=True)
            if len(c_lines) <= self.max_lines:
                split_by_lines.append(c)
            else:
                for i in range(0, len(c_lines), self.max_lines):
                    sub = c_lines[i : i + self.max_lines]
                    sub_start = c.start_line + i
                    sub_end = sub_start + len(sub) - 1
                    split_by_lines.append(Chunk(
                        file_path=file_path,
                        start_line=sub_start,
                        end_line=sub_end,
                        content="".join(sub),
                    ))

        # max_chars 초과 청크 추가 분할 (minified JS 등 긴 줄 대응)
        result: list[Chunk] = []
        for c in split_by_lines:
            if len(c.content) <= self.max_chars:
                result.append(c)
            else:
                # 줄 단위로 분할하되 문자 수 제한 준수
                c_lines = c.content.splitlines(keepends=True)
                buf: list[str] = []
                buf_start = c.start_line
                buf_chars = 0
                for j, line in enumerate(c_lines):
                    if buf_chars + len(line) > self.max_chars and buf:
                        result.append(Chunk(
                            file_path=file_path,
                            start_line=buf_start,
                            end_line=buf_start + len(buf) - 1,
                            content="".join(buf),
                        ))
                        buf = []
                        buf_start = c.start_line + j
                        buf_chars = 0
                    buf.append(line)
                    buf_chars += len(line)
                if buf:
                    result.append(Chunk(
                        file_path=file_path,
                        start_line=buf_start,
                        end_line=buf_start + len(buf) - 1,
                        content="".join(buf),
                    ))
        return result

    def _sliding_window(self, file_path: str, lines: list[str], total: int) -> list[Chunk]:
        """슬라이딩 윈도우로 청크를 생성한다."""
        chunks: list[Chunk] = []
        step = self.window_lines - self.overlap_lines
        if step <= 0:
            step = 1

        pos = 0
        while pos < total:
            end = min(pos + self.window_lines, total)
            chunk_lines = lines[pos:end]
            chunk_content = "".join(chunk_lines)

            chunk = Chunk(
                file_path=file_path,
                start_line=pos + 1,  # 1-based
                end_line=end,         # 1-based, inclusive
                content=chunk_content,
            )

            # 마지막 청크가 min_lines 미만이면 이전 청크에 병합
            if len(chunk_lines) < self.min_lines and chunks:
                prev = chunks[-1]
                merged_content = "".join(lines[prev.start_line - 1 : end])
                chunks[-1] = Chunk(
                    file_path=file_path,
                    start_line=prev.start_line,
                    end_line=end,
                    content=merged_content,
                )
            else:
                chunks.append(chunk)

            if end >= total:
                break
            pos += step

        return chunks


# AST 청킹은 Phase 2+ 에서 tree-sitter로 구현 예정
# Phase 1에서는 SlidingWindowChunker만 사용

_EXTENSION_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
}


class ASTChunker:
    """AST 기반으로 코드를 함수/클래스 단위로 청킹한다.

    tree-sitter를 사용하여 구조적 경계를 인식한다.
    Phase 1 지원 언어: Python, TypeScript/JavaScript, Go.

    NOTE: Phase 1에서는 미구현. SlidingWindowChunker로 폴백한다.
    """

    SUPPORTED_LANGUAGES = frozenset({"python", "typescript", "javascript", "go"})

    def __init__(self) -> None:
        """ASTChunker를 초기화하고 tree-sitter 파서를 설정한다."""
        # Phase 1: tree-sitter 미구현, 폴백 사용
        pass

    def can_parse(self, file_path: str) -> bool:
        """주어진 파일의 언어가 AST 파싱을 지원하는지 확인한다.

        Args:
            file_path: 파일 경로 (확장자로 언어 판별).

        Returns:
            AST 파싱 가능하면 True.
        """
        # Phase 1: 항상 False 반환하여 슬라이딩 윈도우로 폴백
        ext = Path(file_path).suffix.lower()
        lang = _EXTENSION_TO_LANG.get(ext)
        return lang is not None and lang in self.SUPPORTED_LANGUAGES and False

    def chunk(self, file_path: str, content: str) -> list[Chunk]:
        """AST를 파싱하여 함수/클래스 단위로 청크를 생성한다.

        100줄을 초과하는 노드는 강제 분할한다.
        3줄 미만의 노드는 인접 청크에 병합한다.

        Args:
            file_path: 파일 상대 경로.
            content: 파일 전체 텍스트.

        Returns:
            Chunk 리스트.
        """
        # Phase 1: 미구현, SlidingWindowChunker로 폴백
        raise NotImplementedError("AST chunking is not yet implemented (Phase 2+)")


def chunk_file(file_path: str, content: str, use_ast: bool = True) -> list[Chunk]:
    """파일을 적절한 전략으로 청킹한다.

    AST 파싱이 가능하면 ASTChunker를, 아니면 SlidingWindowChunker를 사용한다.

    Args:
        file_path: 파일 상대 경로.
        content: 파일 전체 텍스트.
        use_ast: AST 청킹 시도 여부 (기본: True).

    Returns:
        Chunk 리스트.
    """
    if use_ast:
        ast_chunker = ASTChunker()
        if ast_chunker.can_parse(file_path):
            return ast_chunker.chunk(file_path, content)

    slider = SlidingWindowChunker()
    return slider.chunk(file_path, content)
