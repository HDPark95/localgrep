"""검색 분석(Analytics) - 글로벌 SQLite DB 기반 검색 로그."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

ANALYTICS_DB = Path.home() / ".localgrep" / "analytics.db"


def _get_conn() -> sqlite3.Connection:
    """analytics.db 연결을 반환하고 테이블이 없으면 생성한다."""
    ANALYTICS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(ANALYTICS_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS search_logs (
            id INTEGER PRIMARY KEY,
            timestamp REAL NOT NULL,
            project_path TEXT NOT NULL,
            query TEXT NOT NULL,
            search_type TEXT NOT NULL,
            results_count INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL,
            search_time_ms REAL NOT NULL,
            top_score REAL,
            avg_score REAL
        )
        """
    )
    conn.commit()
    return conn


def _estimate_tokens(text: str) -> int:
    """텍스트의 토큰 수를 추정한다 (대략 4자당 1토큰)."""
    return len(text) // 4


def log_search(
    *,
    project_path: str,
    query: str,
    search_type: str,
    results_count: int,
    total_tokens: int,
    search_time_ms: float,
    top_score: float | None = None,
    avg_score: float | None = None,
) -> None:
    """검색 로그를 기록한다."""
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO search_logs
                (timestamp, project_path, query, search_type, results_count,
                 total_tokens, search_time_ms, top_score, avg_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                project_path,
                query,
                search_type,
                results_count,
                total_tokens,
                search_time_ms,
                top_score,
                avg_score,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_daily_stats(days: int = 7) -> list[dict]:
    """일별 검색 통계를 반환한다.

    Returns
    -------
    [{"date": "2026-03-15", "semantic": 5, "grep": 12, "semantic_tokens": 100, "grep_tokens": 200}, ...]
    """
    conn = _get_conn()
    try:
        cutoff = time.time() - days * 86400
        cur = conn.execute(
            """
            SELECT
                date(timestamp, 'unixepoch', 'localtime') AS day,
                search_type,
                COUNT(*) AS cnt,
                SUM(total_tokens) AS tokens
            FROM search_logs
            WHERE timestamp >= ?
            GROUP BY day, search_type
            ORDER BY day
            """,
            (cutoff,),
        )
        rows = cur.fetchall()

        day_map: dict[str, dict] = {}
        for day, stype, cnt, tokens in rows:
            if day not in day_map:
                day_map[day] = {"date": day, "semantic": 0, "grep": 0, "semantic_tokens": 0, "grep_tokens": 0}
            day_map[day][stype] = cnt
            day_map[day][f"{stype}_tokens"] = tokens or 0

        return list(day_map.values())
    finally:
        conn.close()


def get_token_comparison() -> dict:
    """semantic vs grep 토큰 비교를 반환한다."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            """
            SELECT
                search_type,
                COUNT(*) AS cnt,
                SUM(total_tokens) AS tokens,
                AVG(total_tokens) AS avg_tokens,
                AVG(search_time_ms) AS avg_time
            FROM search_logs
            GROUP BY search_type
            """
        )
        rows = cur.fetchall()
        result: dict = {}
        for stype, cnt, tokens, avg_tokens, avg_time in rows:
            result[stype] = {
                "count": cnt,
                "total_tokens": tokens or 0,
                "avg_tokens": round(avg_tokens or 0, 1),
                "avg_time_ms": round(avg_time or 0, 1),
            }
        return result
    finally:
        conn.close()


def get_recent_searches(limit: int = 50) -> list[dict]:
    """최근 검색 기록을 반환한다."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            """
            SELECT timestamp, project_path, query, search_type,
                   results_count, total_tokens, search_time_ms, top_score, avg_score
            FROM search_logs
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [
            {
                "timestamp": r[0],
                "project_path": r[1],
                "query": r[2],
                "search_type": r[3],
                "results_count": r[4],
                "total_tokens": r[5],
                "search_time_ms": round(r[6], 1),
                "top_score": r[7],
                "avg_score": r[8],
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_summary() -> dict:
    """전체 요약 통계를 반환한다."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN search_type='semantic' THEN 1 ELSE 0 END) AS semantic_count,
                SUM(CASE WHEN search_type='grep' THEN 1 ELSE 0 END) AS grep_count,
                SUM(total_tokens) AS total_tokens,
                AVG(total_tokens) AS avg_tokens,
                SUM(CASE WHEN search_type='semantic' THEN total_tokens ELSE 0 END) AS semantic_tokens,
                SUM(CASE WHEN search_type='grep' THEN total_tokens ELSE 0 END) AS grep_tokens,
                AVG(search_time_ms) AS avg_time
            FROM search_logs
            """
        )
        row = cur.fetchone()
        total = row[0] or 0
        semantic_count = row[1] or 0
        grep_count = row[2] or 0
        total_tokens = row[3] or 0
        avg_tokens = round(row[4] or 0, 1)
        semantic_tokens = row[5] or 0
        grep_tokens = row[6] or 0
        avg_time = round(row[7] or 0, 1)

        # 절감률: grep 대비 semantic이 얼마나 토큰을 절약하는지
        # semantic은 임베딩 기반이므로, grep 대비 평균 토큰이 적으면 절감
        savings_pct = 0.0
        if grep_count > 0 and semantic_count > 0:
            grep_avg = grep_tokens / grep_count if grep_count else 0
            semantic_avg = semantic_tokens / semantic_count if semantic_count else 0
            if grep_avg > 0:
                savings_pct = round((1 - semantic_avg / grep_avg) * 100, 1)

        return {
            "total_searches": total,
            "semantic_count": semantic_count,
            "grep_count": grep_count,
            "total_tokens": total_tokens,
            "avg_tokens": avg_tokens,
            "semantic_tokens": semantic_tokens,
            "grep_tokens": grep_tokens,
            "avg_time_ms": avg_time,
            "savings_pct": savings_pct,
        }
    finally:
        conn.close()
