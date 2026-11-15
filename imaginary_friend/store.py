"""SQLite 保存。共有ログ（全文）と、キャラごとの個人記憶。"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS utterances ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " ts REAL NOT NULL,"
            " speaker TEXT NOT NULL,"  # キャスト id、ユーザーは 'user'
            " text TEXT NOT NULL)"
        )
        # 個人記憶。owner 本人のプロンプトにだけ入る
        #   kind = 'fact' … 自分で調べた検索結果（全文）
        #   kind = 'note' … 会話の区切りで、自分の視点で書いた短い記憶（何の話だったか・どう感じたか）
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS memories ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " ts REAL NOT NULL,"
            " owner TEXT NOT NULL,"
            " kind TEXT NOT NULL,"
            " text TEXT NOT NULL)"
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS memories_owner ON memories (owner, kind, id)")
        #   kind = 'life' … 町での出来事（town.py）。1つの出来事を、その場にいた人数ぶんの行にする。
        #   ts はその出来事が起きたころの時刻。fact / note の行では下の列は NULL のまま
        cols = {row[1] for row in self._db.execute("PRAGMA table_info(memories)")}
        for name, decl in (
            ("event_id", "INTEGER"),  # 同じ出来事の行をまとめる id
            ("place_id", "TEXT"),
            ("item_id", "INTEGER"),
            ("day", "TEXT"),  # 出来事の日付（YYYY-MM-DD）
            ("openness", "TEXT"),  # open / ask / secret
            ("told", "INTEGER"),  # ユウに話したか（0 / 1）
            ("origin", "TEXT"),  # generated / spoken
        ):
            if name not in cols:
                self._db.execute(f"ALTER TABLE memories ADD COLUMN {name} {decl}")
        self._db.execute("CREATE INDEX IF NOT EXISTS memories_event ON memories (event_id)")
        # 小さな状態（今日の気分・好感度の変化など）。値は JSON 文字列
        self._db.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        # 6人目「モデルの子」が、どのモデルで動いたかの記録（meta.py）。モデル名の文字列そのものがキー
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS meta_runs ("
            " model TEXT PRIMARY KEY,"
            " engine TEXT NOT NULL DEFAULT '',"
            " first_at REAL NOT NULL,"
            " last_at REAL NOT NULL,"
            " launches INTEGER NOT NULL DEFAULT 0,"
            " tps REAL)"  # 前回の起動での1秒あたりの出力トークン数（気分の比べ先）
        )
        self._db.commit()

    # --- 6人目が動いた記録 ------------------------------------------------------

    def meta_runs(self) -> list[dict]:
        """動いた記録を、最後に動いた順（新しい順）に。"""
        with self._lock:
            rows = self._db.execute(
                "SELECT model, engine, first_at, last_at, launches, tps FROM meta_runs ORDER BY last_at DESC"
            ).fetchall()
        keys = ("model", "engine", "first_at", "last_at", "launches", "tps")
        return [dict(zip(keys, r)) for r in rows]

    def meta_launch(self, model: str, engine: str, now: float) -> None:
        """このモデルで起動した（無ければ行を作る）。"""
        with self._lock:
            self._db.execute(
                "INSERT INTO meta_runs (model, engine, first_at, last_at, launches) VALUES (?, ?, ?, ?, 1)"
                " ON CONFLICT(model) DO UPDATE SET engine = excluded.engine, last_at = excluded.last_at,"
                " launches = launches + 1",
                (model, engine, now, now),
            )
            self._db.commit()

    def meta_touch(self, model: str, now: float, tps: float | None) -> None:
        """このモデルで今も動いている（last_at と、分かっていれば tps を更新）。"""
        with self._lock:
            self._db.execute(
                "UPDATE meta_runs SET last_at = ?, tps = COALESCE(?, tps) WHERE model = ?", (now, tps, model)
            )
            self._db.commit()

    def add(self, speaker: str, text: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO utterances (ts, speaker, text) VALUES (?, ?, ?)",
                (time.time(), speaker, text),
            )
            self._db.commit()

    def recent(self, limit: int) -> list[tuple[str, str]]:
        """直近 limit 件を古い順に (speaker, text) で返す。"""
        with self._lock:
            rows = self._db.execute(
                "SELECT speaker, text FROM utterances ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return rows[::-1]

    def add_memory(self, owner: str, kind: str, text: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO memories (ts, owner, kind, text) VALUES (?, ?, ?, ?)",
                (time.time(), owner, kind, text),
            )
            self._db.commit()

    def memories(self, owner: str, kind: str, limit: int) -> list[tuple[float, str]]:
        """owner の kind の記憶を、新しい順に最大 limit 件 (ts, text) で返す。"""
        with self._lock:
            return self._db.execute(
                "SELECT ts, text FROM memories WHERE owner = ? AND kind = ? ORDER BY id DESC LIMIT ?",
                (owner, kind, limit),
            ).fetchall()

    def last_ts(self, speaker: str) -> float | None:
        """speaker の最後の発言の時刻（無ければ None）。"""
        with self._lock:
            row = self._db.execute("SELECT MAX(ts) FROM utterances WHERE speaker = ?", (speaker,)).fetchone()
        return row[0] if row and row[0] is not None else None

    # --- 町での出来事（kind = 'life'） ------------------------------------------

    def add_life(
        self,
        owners: list[str],
        ts: float,
        day: str,
        place_id: str,
        text: str,
        openness: str,
        told: bool = False,
        origin: str = "generated",
        item_id: int | None = None,
    ) -> int:
        """1つの出来事を、その場にいた人数ぶんの行として保存する。戻り値は event_id。"""
        with self._lock:
            (last,) = self._db.execute("SELECT COALESCE(MAX(event_id), 0) FROM memories").fetchone()
            event_id = last + 1
            self._db.executemany(
                "INSERT INTO memories (ts, owner, kind, text, event_id, place_id, item_id, day, openness, told, origin)"
                " VALUES (?, ?, 'life', ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (ts, o, text, event_id, place_id, item_id, day, openness, int(told), origin)
                    for o in owners
                ],
            )
            self._db.commit()
        return event_id

    def life_of(self, owner: str, limit: int) -> list[dict]:
        """owner の町での出来事を、新しい順に最大 limit 件。members はその場にいた全員（カンマ区切り）。"""
        with self._lock:
            rows = self._db.execute(
                "SELECT m.ts, m.event_id, m.place_id, m.item_id, m.day, m.openness, m.told, m.text,"
                " (SELECT GROUP_CONCAT(o.owner) FROM memories o WHERE o.kind = 'life' AND o.event_id = m.event_id)"
                " FROM memories m WHERE m.owner = ? AND m.kind = 'life' ORDER BY m.ts DESC, m.id DESC LIMIT ?",
                (owner, limit),
            ).fetchall()
        keys = ("ts", "event_id", "place_id", "item_id", "day", "openness", "told", "text", "members")
        return [dict(zip(keys, r)) for r in rows]

    def life_events(self, limit: int) -> list[dict]:
        """町での出来事を、出来事ごとに（その場にいた全員をまとめて）新しい順に最大 limit 件。"""
        with self._lock:
            rows = self._db.execute(
                "SELECT event_id, MIN(ts), place_id, day, openness, MAX(told), text, GROUP_CONCAT(owner)"
                " FROM memories WHERE kind = 'life' GROUP BY event_id ORDER BY MIN(ts) DESC LIMIT ?",
                (limit,),
            ).fetchall()
        keys = ("event_id", "ts", "place_id", "day", "openness", "told", "text", "members")
        out = [dict(zip(keys, r)) for r in rows]
        for e in out:
            e["members"] = e["members"].split(",") if e["members"] else []
        return out

    def set_told(self, event_id: int) -> None:
        """出来事をユウに話した（その場にいた全員の行をまとめて）。"""
        with self._lock:
            self._db.execute("UPDATE memories SET told = 1 WHERE kind = 'life' AND event_id = ?", (event_id,))
            self._db.commit()

    def get_state(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_state(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO state (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self._db.commit()
