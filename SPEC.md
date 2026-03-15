# localgrep - 로컬 시맨틱 코드 검색 CLI

## 한 줄 요약

로컬 임베딩 모델(Ollama)을 사용해 코드베이스를 시맨틱 검색하는 CLI 도구.
Claude Code의 MCP 서버로 연동되어 자연어 기반 코드 탐색을 가능하게 한다.

---

## 1. 문제 정의

Claude Code가 코드를 찾을 때 사용하는 기본 도구(Grep, Glob)는 **정확한 키워드/패턴**을 알아야 한다.

- "인증 관련 미들웨어" → 정확히 `auth`인지 `authentication`인지 `session`인지 모름
- "에러 핸들링 로직" → `catch`, `error`, `handleError`, `onError` 등 패턴이 다양
- "데이터베이스 연결 설정" → `db`, `database`, `connection`, `pool`, `prisma` 등

시맨틱 검색은 **의미**로 찾으므로 이 문제를 해결한다.

---

## 2. 핵심 사용자: Claude Code

이 도구의 **1차 사용자는 Claude Code**이다. 사람은 2차 사용자.

### Claude Code가 잘 사용하려면

1. **MCP 서버로 노출** — Claude Code가 도구로 직접 호출
2. **출력 형식이 구조화** — JSON 출력, 파일 경로 + 라인 번호 + 스니펫 + 유사도 점수
3. **빠른 응답** — 인덱싱된 상태에서 검색은 1초 이내
4. **컨텍스트 효율적** — 결과가 간결해야 Claude의 컨텍스트 윈도우를 낭비하지 않음
5. **신뢰도 표시** — 유사도 점수로 결과의 관련성을 판단 가능

---

## 3. 아키텍처

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Claude Code │────▶│  MCP Server  │────▶│  localgrep  │
│  (client)   │◀────│  (JSON-RPC)  │◀────│   (core)    │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                    ┌───────────┼───────────┐
                                    ▼           ▼           ▼
                              ┌──────────┐ ┌────────┐ ┌──────────┐
                              │ Ollama   │ │ SQLite │ │ File     │
                              │ Embedder │ │ + Vec  │ │ Watcher  │
                              └──────────┘ └────────┘ └──────────┘
```

### 컴포넌트

| 컴포넌트 | 역할 | 기술 |
|---------|------|------|
| **CLI** | 사용자 인터페이스 | Python + typer |
| **MCP Server** | Claude Code 연동 | mcp python sdk |
| **Indexer** | 파일 크롤링 + 청킹 + 임베딩 | pathlib + gitignore-parser |
| **Embedder** | 텍스트 → 벡터 | Ollama (`nomic-embed-text`) |
| **Store** | 벡터 저장 + 검색 | sqlite-vec (SQLite 확장) |
| **Watcher** | 파일 변경 감지 → 재인덱싱 | watchfiles |

---

## 4. 데이터 모델

### 인덱스 저장 위치

```
{project_root}/.localgrep/
├── index.db          # SQLite DB (메타데이터 + 벡터)
└── config.json       # 프로젝트별 설정
```

### DB 스키마

```sql
-- 파일 메타데이터
CREATE TABLE files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    mtime REAL NOT NULL,           -- 마지막 수정 시간
    hash TEXT NOT NULL              -- 내용 해시 (변경 감지)
);

-- 청크 + 임베딩
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    content TEXT NOT NULL,          -- 원본 텍스트
    embedding FLOAT32[768] NOT NULL -- nomic-embed-text 차원
);

-- sqlite-vec 인덱스
CREATE VIRTUAL TABLE chunks_vec USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT32[768]
);
```

---

## 5. 청킹 전략

코드를 의미 단위로 분할한다. 단순 고정 크기가 아닌 **구조 인식 청킹**:

### 규칙

1. **함수/클래스 단위** — AST 파싱 가능한 언어는 함수/클래스 경계로 분할
2. **폴백: 슬라이딩 윈도우** — AST 파싱 불가 시 50줄 단위, 10줄 오버랩
3. **최소 크기** — 3줄 미만 청크는 이전 청크에 병합
4. **최대 크기** — 100줄 초과 시 강제 분할
5. **메타데이터 접두사** — 각 청크에 `파일경로:시작줄-끝줄` 포함

### 지원 언어 (AST 기반)

- Phase 1: Python, TypeScript/JavaScript, Go
- Phase 2: Rust, Java, C/C++, Ruby
- Fallback: 모든 텍스트 파일 (슬라이딩 윈도우)

---

## 6. MCP 서버 인터페이스

Claude Code가 호출하는 도구 정의:

### Tool: `semantic_search`

```json
{
  "name": "semantic_search",
  "description": "자연어 쿼리로 코드베이스를 시맨틱 검색합니다. 키워드를 모를 때, 개념이나 기능으로 코드를 찾을 때 사용하세요.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "검색할 자연어 쿼리 (예: '사용자 인증 처리', 'DB 커넥션 풀 설정')"
      },
      "path": {
        "type": "string",
        "description": "검색 범위를 제한할 디렉토리 경로 (선택)"
      },
      "top_k": {
        "type": "integer",
        "description": "반환할 최대 결과 수 (기본: 5)",
        "default": 5
      },
      "threshold": {
        "type": "number",
        "description": "최소 유사도 점수 0.0-1.0 (기본: 0.3)",
        "default": 0.3
      },
      "file_pattern": {
        "type": "string",
        "description": "파일 필터 glob 패턴 (예: '*.py', '*.ts')"
      }
    },
    "required": ["query"]
  }
}
```

### Tool: `index_status`

```json
{
  "name": "index_status",
  "description": "현재 인덱스 상태를 확인합니다. 인덱싱된 파일 수, 마지막 업데이트 시간 등.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "path": {
        "type": "string",
        "description": "확인할 프로젝트 경로 (기본: 현재 디렉토리)"
      }
    }
  }
}
```

### Tool: `reindex`

```json
{
  "name": "reindex",
  "description": "인덱스를 갱신합니다. 변경된 파일만 재인덱싱합니다.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "path": {
        "type": "string",
        "description": "재인덱싱할 프로젝트 경로"
      },
      "full": {
        "type": "boolean",
        "description": "전체 재인덱싱 여부 (기본: false, 변경분만)",
        "default": false
      }
    }
  }
}
```

### 출력 형식 (semantic_search)

```json
{
  "results": [
    {
      "file": "src/auth/middleware.py",
      "start_line": 45,
      "end_line": 78,
      "score": 0.87,
      "snippet": "class AuthMiddleware:\n    def __init__(self, app):\n        self.app = app\n    \n    async def __call__(self, request):\n        token = request.headers.get('Authorization')\n        ..."
    }
  ],
  "query": "인증 미들웨어",
  "indexed_files": 342,
  "search_time_ms": 120
}
```

---

## 7. CLI 인터페이스

```bash
# 인덱싱
localgrep index [PATH]              # 프로젝트 인덱싱
localgrep index --full [PATH]       # 전체 재인덱싱
localgrep watch [PATH]              # 파일 변경 감지 + 자동 재인덱싱

# 검색
localgrep search "쿼리"             # 시맨틱 검색
localgrep search "쿼리" -k 10       # 상위 10개
localgrep search "쿼리" -t 0.5      # 유사도 0.5 이상만
localgrep search "쿼리" -g "*.py"   # Python 파일만
localgrep search "쿼리" --json      # JSON 출력

# 상태
localgrep status                    # 인덱스 상태
localgrep config                    # 설정 보기/수정

# MCP 서버
localgrep serve                     # MCP 서버 시작 (stdio)
```

---

## 8. Claude Code 연동

### 설치 방법

```bash
# 1. localgrep 설치
pip install localgrep

# 2. Claude Code MCP 설정에 추가
# ~/.claude/claude_code_config.json
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

### Claude Code CLAUDE.md 가이드

```markdown
## 코드 검색 전략

1. 정확한 키워드를 아는 경우 → Grep/Glob 사용
2. 개념/기능으로 찾는 경우 → semantic_search 사용
3. 검색 결과의 score가 0.7 이상이면 높은 신뢰도
4. score 0.3-0.7은 참고용, 0.3 미만은 무시
```

---

## 9. 설정 파일

```json
// .localgrep/config.json
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

## 10. 의존성

```
python >= 3.11
ollama (외부, 별도 설치)
```

### Python 패키지

| 패키지 | 용도 |
|--------|------|
| `typer` | CLI 프레임워크 |
| `httpx` | Ollama API 호출 |
| `sqlite-vec` | SQLite 벡터 검색 확장 |
| `watchfiles` | 파일 변경 감지 |
| `gitignorefile` | .gitignore 파싱 |
| `tree-sitter` + 언어 바인딩 | AST 기반 청킹 |
| `mcp` | MCP 서버 SDK |
| `rich` | CLI 출력 포맷팅 |

---

## 11. 구현 순서

### Phase 1: MVP (핵심 동작)
- [ ] 프로젝트 구조 셋업 (pyproject.toml, src layout)
- [ ] Ollama 임베딩 클라이언트
- [ ] 파일 크롤러 (.gitignore 존중)
- [ ] 슬라이딩 윈도우 청킹 (AST 없이)
- [ ] SQLite + sqlite-vec 저장소
- [ ] `localgrep index` 명령어
- [ ] `localgrep search` 명령어
- [ ] JSON 출력 모드

### Phase 2: Claude Code 연동
- [ ] MCP 서버 구현 (`localgrep serve`)
- [ ] `semantic_search` 도구
- [ ] `index_status` 도구
- [ ] `reindex` 도구
- [ ] Claude Code 설정 자동화 (`localgrep install`)

### Phase 3: 고도화
- [ ] tree-sitter AST 기반 청킹
- [ ] `localgrep watch` 파일 감시
- [ ] 증분 인덱싱 (변경된 파일만)
- [ ] 멀티 프로젝트 지원
- [ ] 검색 결과 캐싱

---

## 12. 성능 목표

| 지표 | 목표 |
|------|------|
| 인덱싱 속도 | 1,000 파일 / 60초 이내 |
| 검색 응답 | 인덱싱 완료 후 500ms 이내 |
| 인덱스 크기 | 소스 코드 대비 2-3배 (벡터 포함) |
| 메모리 사용 | 검색 시 200MB 이하 |

---

## 13. 제약사항 및 비목표

### 제약사항
- Ollama가 로컬에 실행 중이어야 함
- 최초 인덱싱 시 시간 소요 (임베딩 생성)
- Apple Silicon Mac 최적화 우선 (Linux 호환은 유지)

### 비목표 (하지 않는 것)
- PDF/이미지 검색 (코드 전용)
- 클라우드 동기화
- 웹 UI
- 자체 임베딩 모델 학습
