"""crawler.py 단위 테스트."""

from pathlib import Path

from localgrep.crawler import FileCrawler, FileInfo


def _create_project(tmp_path: Path, files: dict[str, str]) -> Path:
    """테스트용 프로젝트 디렉토리 생성."""
    for rel_path, content in files.items():
        p = tmp_path / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return tmp_path


class TestFileCrawler:
    def test_crawl_basic(self, tmp_path: Path):
        """기본 파일 크롤링."""
        root = _create_project(tmp_path, {
            "main.py": "print('hello')\n",
            "lib/utils.py": "def foo(): pass\n",
        })
        crawler = FileCrawler(root)
        files = crawler.crawl()
        paths = {f.relative_path for f in files}
        assert "main.py" in paths
        assert "lib/utils.py" in paths

    def test_ignore_git_dir(self, tmp_path: Path):
        """.git 디렉토리는 무시."""
        root = _create_project(tmp_path, {
            "main.py": "code\n",
            ".git/config": "git config\n",
        })
        crawler = FileCrawler(root)
        files = crawler.crawl()
        paths = {f.relative_path for f in files}
        assert "main.py" in paths
        assert ".git/config" not in paths

    def test_ignore_node_modules(self, tmp_path: Path):
        """node_modules 무시."""
        root = _create_project(tmp_path, {
            "index.js": "console.log('hi')\n",
            "node_modules/pkg/index.js": "module\n",
        })
        crawler = FileCrawler(root)
        files = crawler.crawl()
        paths = {f.relative_path for f in files}
        assert "index.js" in paths
        assert "node_modules/pkg/index.js" not in paths

    def test_max_file_size(self, tmp_path: Path):
        """크기 제한 초과 파일 무시."""
        root = _create_project(tmp_path, {
            "small.py": "x = 1\n",
            "big.py": "x" * (600 * 1024),  # 600KB > 512KB 기본 제한
        })
        crawler = FileCrawler(root)
        files = crawler.crawl()
        paths = {f.relative_path for f in files}
        assert "small.py" in paths
        assert "big.py" not in paths

    def test_skip_binary_files(self, tmp_path: Path):
        """바이너리 확장자 파일 무시."""
        root = _create_project(tmp_path, {
            "code.py": "print(1)\n",
        })
        # 바이너리 파일 직접 생성
        (root / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        crawler = FileCrawler(root)
        files = crawler.crawl()
        paths = {f.relative_path for f in files}
        assert "code.py" in paths
        assert "image.png" not in paths

    def test_empty_file_skipped(self, tmp_path: Path):
        """빈 파일은 무시."""
        root = _create_project(tmp_path, {
            "empty.py": "",
            "notempty.py": "x = 1\n",
        })
        crawler = FileCrawler(root)
        files = crawler.crawl()
        paths = {f.relative_path for f in files}
        assert "notempty.py" in paths
        assert "empty.py" not in paths

    def test_gitignore_respected(self, tmp_path: Path):
        """.gitignore 패턴 존중."""
        root = _create_project(tmp_path, {
            ".gitignore": "*.log\nbuild/\n",
            "main.py": "code\n",
            "debug.log": "log data\n",
            "build/output.js": "built\n",
        })
        crawler = FileCrawler(root)
        files = crawler.crawl()
        paths = {f.relative_path for f in files}
        assert "main.py" in paths
        assert "debug.log" not in paths

    def test_get_changed_files(self, tmp_path: Path):
        """변경 감지: 새 파일, 수정된 파일, 삭제된 파일."""
        root = _create_project(tmp_path, {
            "existing.py": "old\n",
            "new.py": "new\n",
        })
        crawler = FileCrawler(root)

        # existing.py는 이미 인덱싱됨 (오래된 mtime)
        known = {"existing.py": 0.0, "deleted.py": 100.0}
        changed, deleted = crawler.get_changed_files(known)

        changed_paths = {f.relative_path for f in changed}
        assert "existing.py" in changed_paths  # mtime > 0.0
        assert "new.py" in changed_paths
        assert "deleted.py" in deleted


class TestFileInfo:
    def test_fields(self, tmp_path: Path):
        root = _create_project(tmp_path, {"test.py": "hello\n"})
        crawler = FileCrawler(root)
        files = crawler.crawl()
        assert len(files) == 1
        fi = files[0]
        assert fi.relative_path == "test.py"
        assert fi.size_bytes > 0
        assert fi.mtime > 0
