# localgrep

**로컬 시맨틱 코드 검색**

> 키워드가 아닌 **의미**로 코드를 검색하세요. 로컬 임베딩 모델(Ollama)로 동작합니다.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

---

## 왜 localgrep인가?

`grep`이나 `ripgrep`은 **정확한 키워드**를 알아야 합니다. 하지만 코드를 탐색할 때 우리는 키워드가 아니라 **개념**으로 생각합니다.

```bash
# grep은 정확한 키워드를 추측해야 합니다:
grep -r "auth"           # "authenticate"일 수도?
grep -r "middleware"     # "interceptor"? "handler"?
grep -r "session"        # 어떤 단어인지 모름...

# localgrep은 자연어로 설명하면 됩니다:
localgrep search "authentication middleware"
# => src/auth/middleware.py (score: 0.87)
```

**핵심 장점:**

- **시맨틱 이해** -- 패턴 매칭이 아닌 의미로 코드를 찾습니다
- **100% 로컬** -- 클라우드 없음, API 키 없음, 데이터가 외부로 나가지 않음
- **완전 무료** -- Ollama + 오픈소스 임베딩 모델 사용
- **Claude Code 연동** -- MCP 서버로 AI 코딩 에이전트와 통합

---

## 벤치마크: Semantic Search vs Grep

localgrep 코드베이스(21개 파일, 241개 청크)에서 10개 쿼리로 실측한 결과입니다:

| 쿼리 | Grep 토큰 | Semantic 토큰 | 절감률 |
|------|----------|--------------|-------|
| vector similarity search | ~4,498 | ~1,571 | **65%** |
| file crawling with gitignore | ~7,174 | ~1,789 | **75%** |
| embedding generation from text | ~17,154 | ~2,145 | **87%** |
| chunking strategy for source code | ~19,444 | ~1,638 | **92%** |
| MCP server tool definitions | ~2,225 | ~1,557 | **30%** |
| database schema and migrations | ~1,876 | ~1,594 | **15%** |
| CLI command line interface | ~4,748 | ~1,564 | **67%** |
| configuration file loading | ~8,766 | ~1,363 | **84%** |
| incremental indexing changed files | ~8,120 | ~2,008 | **75%** |
| search result scoring and ranking | ~16,931 | ~1,984 | **88%** |
| **합계** | **~90,936** | **~17,213** | **81%** |

> **시맨틱 검색은 grep 대비 토큰을 ~81% 절감**하면서 더 관련성 높은 결과를 반환합니다.
> 평균: 쿼리당 ~1,721 토큰 (semantic) vs ~9,093 토큰 (grep).

Grep은 패턴에 매칭되는 **모든 것**을 반환합니다. 시맨틱 검색은 **가장 관련성 높은 5개 청크만** 반환하여 AI 에이전트의 컨텍스트 윈도우 소비를 대폭 줄입니다.

---

## 빠른 시작

```bash
# 1. 설치
pip install localgrep

# 2. 임베딩 모델 다운로드
ollama pull nomic-embed-text

# 3. 인덱싱 후 검색
localgrep index .
localgrep search "database connection pooling"
```

3개 명령어만으로 코드베이스 전체를 시맨틱 검색할 수 있습니다.

---

## Claude Code 연동

### 원라인 셋업

```bash
localgrep install-claude
```

Claude Code의 MCP 설정과 검색 전략 가이드를 자동으로 구성합니다.

### 수동 설정

`~/.claude/mcp.json`에 추가:

```json
{
  "mcpServers": {
    "localgrep": {
      "command": "localgrep",
      "args": ["serve"],
      "env": {
        "OLLAMA_HOST": "http://localhost:11434"
      }
    }
  }
}
```

`~/.claude/CLAUDE.md`에 추가:

```markdown
## 코드 검색 전략

1. 정확한 키워드/심볼을 아는 경우 -> Grep / Glob 사용
2. 개념이나 기능으로 찾는 경우 -> semantic_search 사용
3. Score >= 0.7: 높은 신뢰도, 0.3-0.7: 참고용, < 0.3: 무시
4. 최초 검색 전 인덱싱 필요: index_status로 확인 후 reindex 실행
```

---

## 기능

- **시맨틱 검색** -- 자연어로 코드베이스 검색
- **MCP 서버** -- `localgrep serve`로 Claude Code와 원활한 연동
- **웹 대시보드** -- `localgrep dashboard`로 시각적 분석
- **증분 인덱싱** -- 변경된 파일만 재인덱싱
- **토큰 사용량 분석** -- semantic vs grep 검색 효율 비교 추적
- **.gitignore 인식** -- 무시 규칙을 자동으로 존중
- **멀티 프로젝트** -- 여러 코드베이스의 인덱스를 관리

---

## 명령어

| 명령어 | 설명 |
|--------|------|
| `localgrep index [PATH]` | 프로젝트 디렉토리 인덱싱 |
| `localgrep index --full [PATH]` | 전체 재인덱싱 |
| `localgrep search "쿼리"` | 시맨틱 검색 |
| `localgrep search "쿼리" -k 10` | 상위 10개 결과 반환 |
| `localgrep search "쿼리" -t 0.5` | 최소 유사도 임계값 0.5 |
| `localgrep search "쿼리" -g "*.py"` | 파일 패턴 필터 |
| `localgrep search "쿼리" --json` | JSON 출력 |
| `localgrep status` | 인덱스 상태 확인 |
| `localgrep config` | 현재 설정 보기 |
| `localgrep serve` | MCP 서버 시작 (stdio) |
| `localgrep dashboard` | 웹 대시보드 시작 (http://localhost:8585) |
| `localgrep install-claude` | Claude Code 연동 설정 |
| `localgrep watch [PATH]` | 파일 변경 감시 (예정) |

---

## 대시보드

내장 웹 대시보드에서 프로젝트별 인덱스 상태와 검색 사용량을 시각적으로 확인할 수 있습니다.

```bash
localgrep dashboard
# http://localhost:8585 에서 확인
```

- **프로젝트 목록** -- 인덱싱된 파일 수, 청크 수, 상태 표시
- **검색 테스트** -- 브라우저에서 바로 시맨틱 검색
- **Analytics** -- semantic vs grep 토큰 사용량 비교, 일별 추이

---

## 설정

프로젝트별 설정 파일 `.localgrep/config.json`:

```json
{
  "ollama": {
    "host": "http://localhost:11434",
    "model": "nomic-embed-text"
  },
  "indexing": {
    "ignore": [
      "node_modules", ".git", "dist", "build",
      "__pycache__", ".venv", "*.lock"
    ],
    "max_file_size_kb": 512,
    "extensions": null
  },
  "chunking": {
    "max_lines": 100,
    "overlap_lines": 10,
    "min_lines": 3
  },
  "search": {
    "default_top_k": 5,
    "default_threshold": 0.3
  }
}
```

---

## 동작 원리

```
                        localgrep 아키텍처

  ┌──────────┐       ┌───────────┐       ┌────────────┐
  │  CLI /   │──────>│  Indexer   │──────>│  Ollama    │
  │  MCP     │       │           │       │  Embedder  │
  └──────────┘       └─────┬─────┘       └────────────┘
                           │
                    ┌──────▼──────┐
                    │  SQLite +   │
                    │  sqlite-vec │
                    └──────┬──────┘
                           │
  ┌──────────┐       ┌─────▼─────┐
  │  검색    │<──────│  벡터     │
  │  결과    │       │  저장소   │
  └──────────┘       └───────────┘
```

**파이프라인:**

1. **파일 크롤링** -- 프로젝트 트리 순회, `.gitignore` 존중
2. **청킹** -- 파일을 의미 단위로 분할 (함수/클래스 경계 또는 슬라이딩 윈도우)
3. **임베딩** -- Ollama(`nomic-embed-text`)로 벡터 임베딩 생성
4. **벡터 저장** -- SQLite + `sqlite-vec` 확장에 임베딩 저장
5. **검색** -- 쿼리를 인코딩하고 코사인 유사도로 가장 가까운 벡터를 검색

---

## 요구사항

- **Python 3.11+**
- **[Ollama](https://ollama.ai)** -- 로컬에서 실행 필요

```bash
# Ollama 설치 (macOS)
brew install ollama

# 또는 https://ollama.ai 에서 다운로드

# Ollama 시작 및 모델 다운로드
ollama serve    # 아직 실행 중이 아니면
ollama pull nomic-embed-text
```

---

## 기여

기여를 환영합니다! Pull Request를 자유롭게 보내주세요.

1. 레포지토리 Fork
2. 피처 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 Push (`git push origin feature/amazing-feature`)
5. Pull Request 생성

---

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.
