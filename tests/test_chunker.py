"""chunker.py 단위 테스트."""

from localgrep.chunker import Chunk, SlidingWindowChunker, chunk_file


def _make_content(n_lines: int) -> str:
    return "\n".join(f"line {i+1}" for i in range(n_lines)) + "\n"


class TestSlidingWindowChunker:
    def test_small_file_single_chunk(self):
        """윈도우보다 작은 파일은 청크 1개."""
        chunker = SlidingWindowChunker(window_lines=50)
        content = _make_content(10)
        chunks = chunker.chunk("test.py", content)
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 10

    def test_exact_window_size(self):
        """정확히 윈도우 크기 = 청크 1개."""
        chunker = SlidingWindowChunker(window_lines=50)
        content = _make_content(50)
        chunks = chunker.chunk("test.py", content)
        assert len(chunks) == 1

    def test_overlap(self):
        """오버랩이 제대로 동작하는지 확인."""
        chunker = SlidingWindowChunker(window_lines=10, overlap_lines=3)
        content = _make_content(25)
        chunks = chunker.chunk("test.py", content)
        assert len(chunks) > 1
        # 두 번째 청크는 첫 번째 끝에서 overlap만큼 뒤에서 시작
        assert chunks[1].start_line == 8  # 10 - 3 + 1

    def test_min_lines_merge(self):
        """min_lines 미만 마지막 청크는 이전에 병합."""
        chunker = SlidingWindowChunker(window_lines=10, overlap_lines=2, min_lines=5)
        # 12줄 → 10줄 + 2줄(잔여), 2 < 5이므로 병합
        content = _make_content(12)
        chunks = chunker.chunk("test.py", content)
        assert chunks[-1].end_line == 12

    def test_max_lines_split(self):
        """max_lines 초과 청크는 강제 분할."""
        chunker = SlidingWindowChunker(window_lines=200, max_lines=50)
        content = _make_content(150)
        chunks = chunker.chunk("test.py", content)
        for c in chunks:
            line_count = c.end_line - c.start_line + 1
            assert line_count <= 50

    def test_empty_content(self):
        """빈 파일은 빈 리스트."""
        chunker = SlidingWindowChunker()
        assert chunker.chunk("test.py", "") == []
        assert chunker.chunk("test.py", "   \n  ") == []


class TestChunk:
    def test_header(self):
        c = Chunk(file_path="src/auth.py", start_line=10, end_line=45, content="code")
        assert c.header == "src/auth.py:10-45"

    def test_embeddable_text(self):
        c = Chunk(file_path="main.py", start_line=1, end_line=5, content="hello")
        assert c.embeddable_text == "main.py:1-5\nhello"


def test_chunk_file_uses_sliding_window():
    """chunk_file은 Phase 1에서 슬라이딩 윈도우를 사용."""
    content = _make_content(100)
    chunks = chunk_file("test.py", content)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
