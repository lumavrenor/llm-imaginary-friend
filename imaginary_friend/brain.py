"""会話の進行役（裏スレッド）。

1ラウンドの流れ:
  1. 挙手フェーズ … 5人へ並列に「どれくらい喋りたいか」を短い JSON で聞く
  2. 上位の1人が発話（ストリームを文単位に切り、VOICEVOX で先行合成して再生キューへ）
  3. 2人目は「1人目の発言を見たうえで」発話する。1人目と仲が悪いキャラほど選ばれやすい
ユーザーの発言に最初に答えるキャラと、話し込みの各ラウンドの最初の話者に、検索ツール（web_search）を渡す
（話し込み中の検索は [search] max_per_session 回まで）。
呼ばれたら「ちょっと調べるね」の再生中に検索し、結果を踏まえて続きを話す。
裏での調べもの: 挙手の JSON で「調べておきたいこと（check）」も聞き、その回の最初の話者以外の子は、
ほかの子が話している間に裏で検索しておく（[search] background）。LLM への呼び出しは増えない。
結果は本人の記憶に入り、自分の番では「さっき調べたんだけど」と使える。
調べた結果の全文は、調べた本人の個人記憶にだけ残る（memory.py）。ほかの分身は、本人が口で言った範囲しか知らない。
会話の区切りでは、各分身が自分の視点で短い記憶メモを書き、以降はその場の話題に近い自分の記憶をプロンプトに入れる。
喜怒哀楽（drama.py）: セリフの文頭の【喜】などのタグを読み取り、声と表情に使う。今日の気分・関係の変化をプロンプトに入れ、
話し込みや雑談では「口ゲンカ」「打ち明け話」「仲直り」などの演出を、その役のキャラにだけこっそり指示する。
ニュース（news.py）: 起動時の挨拶・自発的な雑談・話し込みの後半で、ときどき1人が「さっきネットで見かけたニュース」を
自分から切り出す。記事の中身はその子にだけ渡し、その子の個人記憶に残す（ほかの子は聞いた範囲しか知らない）。
天気予報（weather.py）: 今の空模様を1行、全員共通の「今の状況」に入れる（ユーザーが天気や明日のことに触れたら予報も）。
ニュースと同じく、ときどき1人が「さっき天気予報を見たら」と自分から切り出す。予報の中身はその子にだけ渡す。
6人目「モデルの子」（meta.py）: 接続中の LLM そのものが、町はずれの電気室から話す。ふだんは選ばれにくく、
きっかけ（遅かった・止まった・聞かれたなど）のあるラウンドで出やすくなり、本人にだけ本物の実行状況の数字を渡す。
ユーザーが発言したあとは「話し込み」として、session_min_sec〜session_max_sec のあいだ
ラウンドを途切れなく続ける。再生待ちが lookahead_sec を切るたびに次のセリフを作るので、
声は途切れず、作りすぎもしない。ユーザーが発言したら、まだ再生していないセリフを捨てて返答に移る。
ユーザーが黙っている間はタイマーで同じラウンドを回す（自発的な雑談。話し込みのあとは回さない）。

プロンプトは「共通部分（設定・全員の人物紹介・会話ログ）→ キャラ固有部分」の順に並べ、
並列リクエストの先頭が一致するようにしてある（プレフィックスキャッシュ用）。
"""

from __future__ import annotations

import json
import logging
import queue
import random
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

from . import clock, language
from .config import CastMember, Config
from .drama import EMOTIONS, NEUTRAL, Beat, Drama, affinity_word, emotion_label, emotion_rule, parse_emotion, take_tags
from .llm import LLMClient, LLMError, ToolCall
from .memory import Memory
from .meta import Meta, Plan, View, named
from .news import NewsFeed, NewsItem, asks_news
from .search import TOOL_NAME, Hit, SearchError, WebSearch, format_hits, tool
from .store import Store
from .town import TalkContext, Town
from .voice import SAMPLE_RATE, Player, Utterance, VoiceError
from .weather import WeatherProvider, asks_weather, focus_day

logger = logging.getLogger(__name__)

USER = "user"

# 立場の内部の名前（日本語）。英語版ではモデルに英語の名前を見せて、受け取ったら内部の名前へ戻す
STANCES = ["賛成", "反対", "共感", "補足", "質問", "茶化す", "ツッコミ"]
_STANCES_EN = {
    "agree": "賛成", "disagree": "反対", "sympathize": "共感", "add": "補足",
    "ask": "質問", "tease": "茶化す", "retort": "ツッコミ",
}


def _stance_names() -> list[str]:
    return language.pick(ja=STANCES, en=list(_STANCES_EN))


def _hand_schema(check: bool) -> dict:
    en = language.is_en()
    props = {
        "gist": {"type": "string", "maxLength": 160 if en else 60},
        "stance": {"type": "string", "enum": _stance_names()},
        "urge": {"type": "integer", "minimum": 0, "maximum": 10},
        "emotion": {"type": "string", "enum": [emotion_label(e) for e in EMOTIONS]},
    }
    required = ["gist", "stance", "urge", "emotion"]
    if check:
        # 裏での調べものを使うときの挙手。check は調べておきたい検索語（無ければ空文字）
        props["check"] = {"type": "string", "maxLength": 60 if en else 30}
        required.append("check")
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


HAND_SCHEMA = _hand_schema(False)
HAND_SCHEMA_CHECK = _hand_schema(True)
# 小さいモデルが「無い」のつもりで書いてくる言葉
_NO_CHECK = {
    "なし", "無し", "ない", "空", "空文字", "特になし", "none", "null", "-", "ー",
    "n/a", "na", "no", "nothing", "empty", "nope", "", "\"\"",
}
# 裏での調べものが終わるのを、自分の番の前に待つ上限（秒）。前の子の声の再生中なので、たいてい気づかれない
_LOOKUP_WAIT_SEC = 5.0


def _query_schema() -> dict:
    return {
        "type": "object",
        "properties": {"query": {"type": "string", "maxLength": language.pick(ja=40, en=80)}},
        "required": ["query"],
        "additionalProperties": False,
    }


# 「調べてみる」「確かめてくる」のような、調べると約束した言い方
_PROMISE_RE = re.compile(r"調べ|確かめ|検索")
_PROMISE_RE_EN = re.compile(r"\b(look (it |that |this |them |him |her )?up|search|check|google|find out)\b", re.IGNORECASE)
# 「さっき調べたんだけど」「確かめておいた」のような、もう調べた話（約束ではない）
_PAST_SEARCH_RE = re.compile(r"(調べ|確かめ|検索し)(た(?!い)|てみた(?!い)|ておいた|てあ)")
_PAST_SEARCH_RE_EN = re.compile(
    r"\b(looked (it |that |this |them |him |her )?up|searched|checked|googled|found out)\b", re.IGNORECASE
)
# 同じ文かどうかを比べるときに無視する文字
_NORM_RE = re.compile(r"[\s、。，．,.！？!?…・ー〜「」『』（）()]")
_NORM_RE_EN = re.compile(r"[^a-z0-9]")
_SENTENCE_END = re.compile(r"[。！？!?]+[」』）)]*|\n+")
# 英語の文末。後ろに空白が来て初めて区切る（ストリームの途中の "3." や "Mr." の直後で切らないため）
_SENTENCE_END_EN = re.compile(r"(?:[.!?]+|…)[\"'”’)\]]*(?=\s)|\n+")
# 文末の "." でも文が終わっていない語（Mr. Smith、a.m. など）
_ABBREV_EN = re.compile(r"(?:\b(?:mr|mrs|ms|dr|st|vs|etc|no|jr|sr|mt|prof|approx|e\.g|i\.e|a\.m|p\.m)|\b[a-z])\.$", re.IGNORECASE)


def _find_sentence_end(buf: str) -> re.Match | None:
    """buf の中で最初の文末。英語は略語の "." を飛ばす。"""
    if not language.is_en():
        return _SENTENCE_END.search(buf)
    pos = 0
    while (m := _SENTENCE_END_EN.search(buf, pos)) is not None:
        if m.group(0).startswith(".") and _ABBREV_EN.search(buf[: m.start() + 1]):
            pos = m.end()
            continue
        return m
    return None
# 話し込みが長いときは、区切りを待たずにこの発言数ごとに記憶メモを書く（履歴は history_lines*2 で切れるため）
_REFLECT_EVERY = 16

# 話し込みで、ラウンドの最初の話者に1つ渡す「話の進め方」。長く話しても同じ話の堂々巡りにしないため
_TALK_MOVES = [
    "具体的な例や場面を出して、話をふくらませる",
    "自分の性格ならではの見方で、まだ出ていない論点を出す",
    "前に誰かが言ったことに、疑問や反論をぶつける",
    "関連する別の話題に、少しだけ話を広げる",
    "ありがちな失敗談や「あるある」を話す",
    "ここまでの話を自分なりにまとめて、次の問いをみんなに投げかける",
    "黙って聞いている{user}に関係づけて、{user}ならどうかを想像して話す",
    "もし〜だったら、という仮の話をして話を転がす",
    "今の話を、自分の今日の気分やできごとにつなげて話す",
    "誰かの言ったことに、気持ちを込めて共感する",
]
_TALK_MOVES_EN = [
    "bring up a concrete example or scene and build on the topic",
    "raise a point nobody has made yet, from a view only someone with your personality would have",
    "push back on something someone said earlier, with a doubt or an objection",
    "widen the topic a little to something related",
    'share a typical blunder or a relatable "that always happens" moment',
    "sum up the conversation so far in your own way and toss a new question to everyone",
    "connect it to {user}, who is listening quietly, and imagine what {user} would do",
    'spin a "what if..." scenario to keep things rolling',
    "tie what's being said to your own mood or something that happened to you today",
    "warmly sympathize with something someone said",
]


@dataclass
class Hand:
    urge: int = 0
    stance: str = "補足"
    gist: str = ""
    emotion: str = NEUTRAL
    check: str = ""  # 裏で調べておきたい検索語（無ければ空）


@dataclass
class Turn:
    """1人のひと続きの発言。parts は (文の uid, 文) で、再生前に捨てられた文は取り除く。"""

    id: int
    speaker: str  # キャスト id か USER
    parts: list[tuple[int, str]] = field(default_factory=list)
    emotions: dict[int, str] = field(default_factory=dict)  # 文の uid → 感情
    feel: str = NEUTRAL  # 生成中の今の感情（タグが無い文はこれを引き継ぐ）

    @property
    def text(self) -> str:
        return language.pick(ja="", en=" ").join(t for _, t in self.parts)

    @property
    def mood(self) -> str:
        """発言全体の感情（最後に出た「普通」以外の感情）。"""
        for uid, _ in reversed(self.parts):
            emo = self.emotions.get(uid, NEUTRAL)
            if emo != NEUTRAL:
                return emo
        return NEUTRAL


class Brain(threading.Thread):
    def __init__(
        self,
        config: Config,
        llm: LLMClient,
        tts,  # voice.Voicevox か kokoro.Kokoro（synth_for(member, text, emotion) を持つ）
        player: Player,
        store: Store,
        emit: Callable[[tuple], None],
        search: WebSearch | None = None,
        news: NewsFeed | None = None,
        weather: WeatherProvider | None = None,
    ):
        super().__init__(name="brain", daemon=True)
        self._cfg = config
        self._talk = config.talk
        self._llm = llm
        self._tts = tts
        self._player = player
        self._store = store
        self._emit = emit
        self._search = search
        self._news = news
        self._weather = weather
        # プロンプトの中でのユーザーの書き方。呼び名が「あなた」のままだと、
        # キャラへの「あなたは〜です」と区別できず、キャラが自分のことと読んでしまう
        if language.is_en():
            self._uref = "the user" if config.user_name.lower() in ("you", "") else config.user_name
        else:
            self._uref = "ユーザー" if config.user_name in ("あなた", "") else config.user_name
        self._cast = config.cast
        self._by_id = config.cast_by_id()
        # 町で暮らす5人（6人目「モデルの子」を除く）。6人目がいなければ self._cast と同じ
        self._girls = config.town_cast
        self._meta_id = config.meta.member_id
        self._inbox: queue.Queue[str] = queue.Queue()
        self._played: queue.Queue[Utterance] = queue.Queue()
        self._pool = ThreadPoolExecutor(max_workers=len(self._cast), thread_name_prefix="hand")
        # 裏での調べもの（話していない子が、ほかの子の発言中にこっそり検索しておく）
        sc = config.search
        self._background = (
            search is not None
            and config.memory.enabled
            and sc.background
            and sc.background_per_round > 0
            and sc.background_per_session > 0
        )
        self._lookup_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="lookup")
        self._lookups: dict[str, Future] = {}  # キャラ id → 裏の検索（brain スレッドだけが触る）
        self._lookup_lock = threading.Lock()
        self._fresh: dict[str, str] = {}  # 裏で調べ終わって、まだ話していない子 → 検索語（_lookup_lock）
        self._lookup_gen = 0  # 話題が変わったら進める。古い話題の調べものを「さっき調べた」にしない
        self._bg_searches = 0
        self._searched: set[str] = set()  # この話し込みで調べた検索語（_norm 済み）
        self._uid = 0
        self._turns: list[Turn] = []
        self._auto_rounds = 0
        self._idle_at: float | None = None
        # 話し込み: いつまで続けるか（monotonic 秒）と、きっかけになったユーザーの発言
        self._session_end: float | None = None
        self._session_topic = ""
        self._session_misses = 0
        self._session_searches = 0
        self._session_news = False  # この話し込みで、もうニュースの話を出したか
        self._session_weather = False  # この話し込みで、もう天気予報の話を出したか
        # 個人記憶（調べた事実と、会話の区切りで書く記憶メモ）
        self._memory = Memory(store, llm, config.memory, self._uref) if config.memory.enabled else None
        # 喜怒哀楽・今日の気分・関係の変化・演出
        self._drama = (
            Drama(config.cast, config.affinity, store, config.drama) if config.drama.enabled else None
        )
        # 町（イマジナリーランド）での暮らし。出来事は起動時の挨拶のあと、裏で作る
        self._town = (
            Town(
                config.town,
                self._girls,
                store,
                llm,
                self._uref,
                config.memory.min_score,
                drama=self._drama,
                on_update=self._on_town_update,
            )
            if config.town.enabled
            else None
        )
        if self._drama is not None:
            # 打ち明け話・ないしょ話の材料は町の出来事（町を使わない設定では、その演出は出ない）
            self._drama.town = self._town
        # 6人目「モデルの子」。気分は町の出来事ではなく、実行状況からコードで決める
        self._meta = Meta(config, llm, store, self._uref, self._drama) if self._meta_id else None
        self._meta_plan: Plan | None = None
        self._seen_visible: set[int] = set()  # 前のラウンドで見えていた発言（外へ流れた数を数える）
        self._session_turn = 0  # 話し込みのきっかけになったユーザーの発言の turn id
        if self._meta is not None:
            self._meta.town = self._town
            if self._memory is not None:
                self._memory.intros[self._meta_id] = self._meta.memory_intro()
        self._talk_ctx = TalkContext()
        # 前にユーザーが話した時刻（「3日ぶり」と言えるように。今回の発言を保存する前に読む）
        self._last_visit = store.last_ts(USER)
        self._session_start = 0.0
        # submit() がすぐ止めた（再生前に捨てた）セリフ。履歴からの削除は brain スレッドで行う
        self._dropped: list[Utterance] = []
        self._drop_lock = threading.Lock()
        self._voice_warned = False
        self._load_history()
        # この id までの発言は記憶メモにした（前回までの会話は、前回の区切りで書いた扱い）
        self._reflected = self._uid
        # 前回の記憶メモの時刻。次の記憶メモには、これ以降に調べたことだけを見せる
        self._reflected_at = time.time()
        # 再生が始まった文をすぐ DB へ書く係（ラウンドの途中で終了しても取りこぼさない）
        threading.Thread(target=self._store_loop, name="store-writer", daemon=True).start()

    # --- 他スレッドから呼ばれる ----------------------------------------------

    def submit(self, text: str) -> None:
        """ユーザーの発言（UI スレッドから）。

        まだ再生していないセリフはここですぐ止める。brain スレッドが挙手の途中だと、
        気づくまでの数秒間、前の話が続いてしまうため。
        """
        with self._drop_lock:
            self._dropped.extend(self._player.clear())
        self._inbox.put(text)

    def on_player_event(self, kind: str, utt: Utterance) -> None:
        """再生開始の通知（オーディオスレッドから）。DB 書き込みは専用スレッドへ回す。"""
        if kind == "start":
            self._played.put(utt)

    def past_lines(self) -> list[tuple[str, str]]:
        """起動時にログへ出す、前回までの会話。"""
        return [(t.speaker, t.text) for t in self._turns]

    # --- メインループ --------------------------------------------------------

    def _on_town_update(self) -> None:
        """町の出来事を作り終えた（town スレッドから）。今日の気分が決まっていれば画面に出す。"""
        if self._drama is not None and not self._drama.needs_moods() and not self._moods_shown:
            self._moods_shown = True
            if self._cfg.drama.show_mood:
                self._emit(("system", self._drama.mood_summary()))
            self._emit(("moods", self._drama.moods_for_ui()))

    def run(self) -> None:
        self._moods_shown = False
        if self._drama is not None and not self._drama.needs_moods():
            # 今日の気分はもう決まっている（今日2回目以降の起動）。まだなら、町の出来事を作ったあとで出す
            self._on_town_update()
        if self._talk.greet_on_start:
            self._safe_round("start")
        if self._town is not None:
            # 挨拶と GPU を取り合わないよう、挨拶のあとで作る
            self._town.generate_async()
        while True:
            try:
                text = self._inbox.get(timeout=0.2)
            except queue.Empty:
                text = None
            if text is not None:
                with self._drop_lock:
                    dropped = self._dropped + self._player.clear()
                    self._dropped = []
                self._drop(dropped)
                # 前の話し込みはここで区切れる。新しい話題に移る前に記憶メモにする
                self._reflect(self._cfg.memory.reflect_min_turns)
                self._session_turn = self._add_turn(USER, [(self._next_uid(), text)]).id
                self._store.add(USER, text)
                self._auto_rounds = 0
                self._start_session(text)
                if not self._safe_round("user"):
                    self._end_session()
                self._idle_at = None
                continue
            if self._session_end is not None:
                self._continue_session()
                continue
            if not self._talk.idle_chat:
                continue
            now = time.monotonic()
            if self._player.busy:
                self._idle_at = None
            elif self._idle_at is None:
                self._idle_at = now + random.uniform(self._talk.idle_min_sec, self._talk.idle_max_sec)
            elif now >= self._idle_at and self._auto_rounds < self._talk.max_auto_rounds:
                self._auto_rounds += 1
                self._safe_round("idle")
                self._idle_at = None

    def _safe_round(self, trigger: str) -> bool:
        """1ラウンド回す。誰かが1文でも話せたら True。"""
        self._in_round = True
        try:
            return self._round(trigger)
        except LLMError as e:
            logger.warning("round failed: %s", e)
            self._emit(("system", language.pick(ja="LLM エラー: ", en="LLM error: ") + str(e)))
        except Exception as e:  # noqa: BLE001 - 裏スレッドを死なせない
            logger.exception("round crashed")
            self._emit(("system", language.pick(ja="内部エラー: ", en="Internal error: ") + f"{type(e).__name__}: {e}"))
        finally:
            self._in_round = False
            self._emit(("hands", {}))
            self._emit(("status", ""))
        return False

    def _interrupted(self) -> bool:
        return not self._inbox.empty()

    # --- 話し込み --------------------------------------------------------------

    def _start_session(self, topic: str) -> None:
        # 裏での調べものの回数と「さっき調べた」は、話し込みをしない設定でもユーザーの発言ごとに区切る
        self._bg_searches = 0
        self._searched.clear()
        with self._lookup_lock:
            self._lookup_gen += 1
            self._fresh.clear()
        lo, hi = self._talk.session_min_sec, self._talk.session_max_sec
        if says_goodbye(topic):
            # 「おやすみ」「またね」のあとは返事だけにする。話し込みや雑談を続けると、
            # 寝に行ったユウの前で口ゲンカが始まる（次の発言までは自発的な雑談も回さない）
            self._session_end = None
            self._auto_rounds = self._talk.max_auto_rounds
            return
        if hi <= 0:
            self._session_end = None
            return
        self._session_start = time.monotonic()
        self._session_end = self._session_start + random.uniform(lo, hi)
        self._session_topic = topic
        if self._drama is not None:
            self._drama.start_session()
        if self._meta is not None:
            self._meta.start_session()
        self._session_misses = 0
        self._session_searches = 0
        self._session_news = False
        self._session_weather = False

    def _end_session(self) -> None:
        if self._session_end is None:
            return
        self._session_end = None
        # 話し込みのあとは自発的な雑談を回さない（ユーザーの次の発言を待つ）
        self._auto_rounds = self._talk.max_auto_rounds
        self._reflect(self._cfg.memory.reflect_min_turns)

    def _reflect(self, min_turns: int) -> None:
        """まだ記憶にしていない発言が min_turns 以上あれば、各分身に記憶メモを書かせる（裏で）。"""
        if self._memory is None:
            return
        new = [t for t in self._turns if t.id > self._reflected and t.parts]
        if len(new) < max(1, min_turns):
            return
        self._reflected = max(t.id for t in new)
        since, self._reflected_at = self._reflected_at, time.time()
        lines = [self._log_line(t) for t in new]
        descs = "\n".join(f"- {m.name}: {m.desc}" for m in self._cast)  # 英語版は desc も英語で書く
        logger.info("記憶メモを書く（%d 発言ぶん）", len(new))
        self._memory.reflect_async(self._cast, lines, descs, since)

    def _session_progress(self) -> float:
        """話し込みの進み具合 0〜1（キューに積んだぶんの再生時間も含める）。"""
        if self._session_end is None:
            return 0.0
        span = max(1.0, self._session_end - self._session_start)
        return (time.monotonic() + self._player.pending_sec() - self._session_start) / span

    def _session_active(self) -> bool:
        """話し込みの残り時間があるか。キューに積んだぶんの再生時間も含めて数える。"""
        end = self._session_end
        return end is not None and time.monotonic() + self._player.pending_sec() < end

    def _continue_session(self) -> None:
        if not self._wait_for_room() or not self._session_active():
            if not self._interrupted():
                self._end_session()
            return
        if self._safe_round("talk"):
            self._session_misses = 0
            self._reflect(_REFLECT_EVERY)
        else:
            # LLM エラーなどで空振りが続くときは、空回りさせずに打ち切る
            self._session_misses += 1
            if self._session_misses >= 2 and not self._interrupted():
                self._end_session()

    def _wait_for_room(self) -> bool:
        """再生待ちが lookahead_sec を切るまで待つ。ユーザーが発言したら False。"""
        while self._player.pending_sec() > self._talk.lookahead_sec:
            if self._interrupted():
                return False
            time.sleep(0.1)
        return not self._interrupted()

    def _status(self, text: str) -> None:
        """「考え中…」の表示。前のセリフを再生している間は出さない（話し込みでちらつくため）。"""
        if not self._player.busy:
            self._emit(("status", text))

    # --- 1ラウンド -----------------------------------------------------------

    def _round(self, trigger: str) -> bool:
        """戻り値は、誰かが1文でも話せたか。"""
        user = self._uref
        gap = clock.gap_text(self._last_visit)
        situation = self._situation_en(trigger, gap) if language.is_en() else {
            "user": f"{user}が今、発言しました。みんながそれに反応する場面です。",
            "idle": f"{user}はしばらく黙っています。女の子どうしで直前の話題についておしゃべりするか、誰かが新しい話題を出す場面です。",
            "start": f"{user}がちょうど来たところです"
            + (f"（{user}が来るのは{gap}）" if gap else "")
            + "。誰かが短く声をかける場面です。",
            "talk": (
                f"少し前に{user}が「{self._session_topic}」と言いました。"
                f"それをきっかけに、女の子どうしで話し込んでいる場面です。{user}は黙って聞いています。"
                "これまでに出た話をくり返さず、毎回なにか新しい中身を足して、会話を途切れさせずに続けてください。"
            ),
        }[trigger]
        n_speakers = {
            "user": self._talk.speakers_per_turn,
            "idle": random.choice((1, 2, 2)),
            "start": 1,
            "talk": random.choice((2, 3, 3)),
        }[trigger]
        if trigger == "talk" and not self._wait_for_room():
            return False

        # ユーザーが名前を呼んだキャラ（「アカリちゃん、…」など）は最初に答えさせる
        addressed: list[str] = []
        if trigger == "user":
            last = self._turns[-1].text
            addressed = [m.id for m in self._cast if named(m.name, last)]
            if addressed:
                if language.is_en():
                    names = " and ".join(self._by_id[c].name for c in addressed)
                    situation += f" {user} is talking to {names}."
                else:
                    names = "と".join(self._by_id[c].name for c in addressed)
                    situation += f"{user}は{names}に話しかけています。"
        # 時間を意識させる（「おはよう」と「夜ふかし？」を言い分けられるように）。全員に同じ文なので共通部分に入る
        situation = f"{clock.now_text()}{self._weather_text(trigger)}{situation}"
        # 町の出来事を、このラウンドでどこまで話してよいか
        if trigger == "user":
            self._talk_ctx = TalkContext.from_user(self._turns[-1].text, addressed)
        elif trigger == "talk":
            topic = self._session_topic
            self._talk_ctx = TalkContext.from_user(topic, [m.id for m in self._cast if (named(m.name, topic) if language.is_en() else m.name in topic)]
            )
        else:
            self._talk_ctx = TalkContext(idle=True)

        # 演出（口ゲンカ・打ち明け話・仲直りなど）。役のキャラを決まった順に話させ、指示はその子にだけ渡す
        beat: Beat | None = None
        if self._drama is not None and trigger in ("talk", "idle"):
            beat = self._drama.beat_for(trigger, self._session_progress())
            if beat is not None:
                n_speakers = max(n_speakers, beat.min_speakers)
        # 演出が無いラウンドでは、ときどき誰かが見かけたニュースか天気予報の話を切り出す
        # （起動時の挨拶は天気の話のほうが自然なので先に試す）
        if beat is None and trigger in ("start", "idle", "talk"):
            pickers = [self._news_beat, self._weather_beat]
            if trigger == "start":
                pickers.reverse()
            for pick in pickers:
                beat = pick(trigger)
                if beat is not None:
                    n_speakers = max(n_speakers, beat.min_speakers)
                    break
        # ユウに「最近なにかニュースあった？」と聞かれたら、見かけたニュースを1人に渡して答えさせる
        # （渡さないと、小さいモデルは「ちょっと探してみるね」と言ったまま何も出さない）
        if beat is None and trigger == "user" and asks_news(self._turns[-1].text):
            beat = self._news_beat(trigger, prefer=[c for c in addressed if c != self._meta_id])
            if beat is not None:
                n_speakers = max(n_speakers, beat.min_speakers)
        priority = beat.order if beat is not None else addressed
        # 6人目: きっかけ・選ばれやすさ・本人に渡す実行状況（挙手の前に決める。挙手と発話で同じものを見せる）
        self._meta_plan = self._meta.plan(self._meta_view(trigger)) if self._meta is not None else None

        # 「考え中…」は、話し込みの途中（前のセリフの再生中）には出さない
        status = self._status if trigger == "talk" else (lambda s: self._emit(("status", s)))
        status(language.pick(ja="みんな考え中…", en="Everyone's thinking…"))
        t0 = time.monotonic()
        hands = self._raise_hands(situation)
        logger.info(
            "hands %s (%.1fs): %s",
            trigger,
            time.monotonic() - t0,
            {k: (h.urge, h.stance, h.gist) for k, h in hands.items()},
        )
        self._emit(("hands", {k: h.urge for k, h in hands.items()}))

        spoken: list[str] = []
        for _ in range(n_speakers):
            # 前のセリフを再生している間に、次のセリフを作りすぎないよう待つ
            if spoken and not self._wait_for_room():
                break
            if self._interrupted() or (trigger == "talk" and not self._session_active()):
                break
            speaker = self._choose(hands, spoken, priority)
            if speaker is None:
                break
            member = self._by_id[speaker]
            if not spoken:
                # 最初の話者が話している間に、ほかの子は裏で調べておく
                self._lookup_behind(hands, speaker, self._has_tool(trigger, spoken))
            status(language.pick(ja=f"{member.name} が話そうとしている…", en=f"{member.name} is about to speak…"))
            # 検索結果は本人の記憶に入るだけ。2人目以降は、本人が口で言った範囲（会話ログ）で知る
            if not self._speak(member, situation, hands[speaker], spoken, trigger, beat):
                break
            spoken.append(speaker)
            self._emit(("hands", {k: h.urge for k, h in hands.items() if k not in spoken}))
        if beat is not None and self._drama is not None and spoken and spoken[0] == beat.lead:
            self._drama.apply_beat(beat)
            self._emit(("moods", self._drama.moods_for_ui()))
        if self._meta is not None:
            self._meta.after_round(trigger, spoken)
            if self._drama is not None:
                self._emit(("moods", self._drama.moods_for_ui()))
        return bool(spoken)

    def _situation_en(self, trigger: str, gap: str) -> str:
        user = self._uref
        return {
            "user": f"{user} just said something. Everyone is reacting to it.",
            "idle": f"{user} has been quiet for a while. The girls chat among themselves about the last topic, "
            "or someone brings up something new.",
            "start": f"{user} has just come by"
            + (f" ({gap} that {user} has come by)" if gap else "")
            + ". Someone says a quick hello.",
            "talk": (
                f'A little while ago {user} said "{self._session_topic}". That got the girls talking among themselves, '
                f"and they're deep in conversation. {user} is listening quietly. "
                "Don't repeat what has already been said: add something new every time and keep the conversation going."
            ),
        }[trigger]

    def _meta_view(self, trigger: str) -> View:
        """6人目のきっかけの判定に渡す、今の会話の見え方。"""
        visible = {t.id for t in self._visible_turns()}
        dropped = len(self._seen_visible - visible) if self._seen_visible else 0
        self._seen_visible = visible
        searches_left = None
        if trigger == "talk" and self._search is not None:
            searches_left = max(0, self._cfg.search.max_per_session - self._session_searches)
        return View(
            trigger=trigger,
            user_text=self._turns[-1].text if trigger == "user" and self._turns else "",
            recent=[(t.id, t.speaker, t.text) for t in self._turns[-4:] if t.parts],
            visible=len(visible),
            dropped=dropped,
            topic_visible=not self._session_turn or self._session_turn in visible,
            searches_left=searches_left,
        )

    def _news_beat(self, trigger: str, prefer: list[str] | None = None) -> Beat | None:
        """見かけたニュースを1人に切り出させる演出（今回は出さないなら None）。

        記事の中身はその子の指示と個人記憶にだけ入れる。ほかの子は、その子が口で言った範囲しか知らない。
        trigger == "user" は、ユウにニュースを聞かれたとき（必ず出す。prefer はユウが名前を呼んだ子）。
        """
        if self._news is None:
            return None
        cfg = self._cfg.news
        chance = {"start": cfg.start_chance, "idle": cfg.idle_chance, "user": 1.0}.get(trigger, 0.0)
        if trigger == "talk":
            # 話し込みの前半はユーザーの話題を優先する。後半、話が尽きてきたころに1回だけ
            chance = cfg.talk_chance if not self._session_news and self._session_progress() >= 0.5 else 0.0
        if chance <= 0 or random.random() >= chance:
            return None
        self._status(language.pick(ja="みんな考え中…", en="Everyone's thinking…"))
        item = self._news.pick()
        if item is None:
            return None
        if trigger == "talk":
            self._session_news = True
        last = self._last_cast_speaker()
        # 6人目は町でニュースを見かけないので、切り出す役にしない
        girls = [m.id for m in self._girls]
        lead = random.choice([c for c in (prefer or []) if c in girls] or [c for c in girls if c != last] or girls)
        member = self._by_id[lead]
        en = language.is_en()
        when = language.pick(ja="・", en=", ").join(x for x in (item.date_label, item.source) if x)
        cue = self._news_cue_en(item, when, trigger) if en else (
            ("まず" + self._uref + "に短く声をかけてから、" if trigger == "start" else "")
            + (
                self._uref + "に最近のニュースのことを聞かれたので、さっきネットで見かけたニュースの話をしてください。"
                if trigger == "user"
                else "さっきネットで見かけたニュースの話を、自分から切り出してください。"
            )
            + f"\n見出し: {item.title}"
            + (f"\n内容: {item.summary}" if item.summary else "")
            + (f"\n（{when}）" if when else "")
            + "\n書いてあることだけを話し、書いていない細部は言い切らない。"
            "ニュースを読み上げるのではなく、自分の性格らしい感想を言って、みんなにも話を振る。"
            "URL やサイト名は読み上げない。"
        )
        logger.info("news: %s が見かけた: %s（%s）", member.name, item.title, when)
        if en:
            self._emit(("system", f"{member.name} saw online: {item.title}" + (f" ({item.source})" if item.source else "")))
        else:
            self._emit(("system", f"{member.name}がネットで見かけた: {item.title}" + (f"（{item.source}）" if item.source else "")))
        if self._memory is not None:
            self._memory.add_fact(
                lead,
                item.title,
                [Hit(item.title, item.summary or when, item.url)],
                heading=(
                    f"A news story you saw online ({when or 'date unknown'}):"
                    if en
                    else f"ネットで見かけたニュース（{when or '日付不明'}）:"
                ),
            )
        return Beat(
            kind="news",
            lead=lead,
            partner=None,
            cues={lead: cue},
            min_speakers=1 if trigger == "start" else 2,
            label=f"ニュース: {item.title}",
        )

    def _news_cue_en(self, item: NewsItem, when: str, trigger: str) -> str:
        return (
            (f"First say a quick hello to {self._uref}, then " if trigger == "start" else "")
            + (
                f"{self._uref} asked about the news, so tell them about a news story you just saw online."
                if trigger == "user"
                else "bring up, on your own, a news story you just saw online."
            )
            + f"\nHeadline: {item.title}"
            + (f"\nSummary: {item.summary}" if item.summary else "")
            + (f"\n({when})" if when else "")
            + "\nOnly talk about what's written there; don't state details that aren't written as fact. "
            "Don't read the news out: give your own reaction, the way someone with your personality would, "
            "and ask the others what they think. Don't read out URLs or site names."
        )

    def _weather_text(self, trigger: str) -> str:
        """「今の状況」に入れる空模様（全員共通）。ユーザーが天気に触れたら予報も入れる。"""
        if self._weather is None or not self._cfg.weather.in_situation:
            return ""
        w = self._weather.current()
        if w is None:
            return ""
        text = w.now_line()
        if trigger == "user" and self._turns and asks_weather(self._turns[-1].text):
            text += w.forecast_text()
        return text

    def _weather_beat(self, trigger: str) -> Beat | None:
        """さっき見た天気予報の話を1人に切り出させる演出（今回は出さないなら None）。"""
        if self._weather is None:
            return None
        cfg = self._cfg.weather
        chance = {"start": cfg.start_chance, "idle": cfg.idle_chance}.get(trigger, 0.0)
        if trigger == "talk":
            # 話し込みの前半はユーザーの話題を優先する。後半、話が尽きてきたころに1回だけ
            chance = cfg.talk_chance if not self._session_weather and self._session_progress() >= 0.5 else 0.0
        if chance <= 0 or random.random() >= chance:
            return None
        # 同じ予報の話を何度もしない（再起動をまたいで間を空ける）
        last = float(self._store.get_state("weather_beat_at") or 0)
        if time.time() - last < cfg.min_gap_min * 60:
            return None
        w = self._weather.current()
        if w is None:
            return None
        if trigger == "talk":
            self._session_weather = True
        self._store.set_state("weather_beat_at", str(time.time()))
        prev = self._last_cast_speaker()
        # 6人目は町でスマホを見ないので、切り出す役にしない
        lead = random.choice([m.id for m in self._girls if m.id != prev] or [m.id for m in self._girls])
        member = self._by_id[lead]
        day = focus_day()
        cue = self._weather_cue_en(w, day, trigger) if language.is_en() else (
            ("まず" + self._uref + "に短く声をかけてから、" if trigger == "start" else "")
            + f"さっき{self._uref}の住んでいる{w.location}の天気予報を見たので、その話を自分から切り出してください。"
            + "\n" + "\n".join(w.forecast_lines())
            + f"\n今の時間なら、{day}の予報の話を中心に。"
            "書いてある天気と数字だけを使い、書いていない予報（週末・来週など）は言い切らない。"
            "予報を読み上げるのではなく、それで何をしたいか・何を着るか・どんな気分かを、自分の性格らしく話して、"
            "みんなにも話を振る。"
        )
        logger.info("weather: %s が天気予報を見た: %s", member.name, " / ".join(w.forecast_lines()))
        self._emit(
            (
                "system",
                language.pick(
                    ja=f"{member.name}が天気予報を見た: {w.location}・{w.description}",
                    en=f"{member.name} checked the forecast: {w.location}, {w.description}",
                ),
            )
        )
        return Beat(
            kind="weather",
            lead=lead,
            partner=None,
            cues={lead: cue},
            min_speakers=1 if trigger == "start" else 2,
            label=f"天気予報: {w.location}",
        )

    def _weather_cue_en(self, w, day: str, trigger: str) -> str:
        return (
            (f"First say a quick hello to {self._uref}, then " if trigger == "start" else "")
            + f"you just checked the weather forecast for {w.location}, where {self._uref} lives; "
            "bring it up on your own."
            + "\n" + "\n".join(w.forecast_lines())
            + f"\nAt this time of day, focus on the forecast for {day}. "
            "Only use the weather and numbers written here; don't state forecasts that aren't written "
            "(the weekend, next week, and so on) as fact. Don't read the forecast out: talk, in your own way, "
            "about what you want to do because of it, what you'll wear, or how it makes you feel, "
            "and ask the others too."
        )

    # --- 挙手フェーズ（並列） --------------------------------------------------

    def _raise_hands(self, situation: str) -> dict[str, Hand]:
        futures = {m.id: self._pool.submit(self._ask_hand, m, situation) for m in self._cast}
        hands: dict[str, Hand] = {}
        errors: list[Exception] = []
        for cid, fut in futures.items():
            try:
                hands[cid] = fut.result()
            except (LLMError, ValueError, TypeError) as e:
                errors.append(e)
                hands[cid] = Hand()
        if len(errors) == len(futures):
            raise LLMError(str(errors[0]))
        return hands

    def _ask_hand(self, member: CastMember, situation: str) -> Hand:
        en = language.is_en()
        if en:
            instruction = (
                f'As "{member.name}", answer in JSON how you want to react to the conversation right now.\n'
                f"- gist: the gist of what you want to say (under 15 words)\n"
                f"- stance: your stance toward the last thing said ({' / '.join(_stance_names())})\n"
                f"- urge: how badly you want to speak right now (0 = would rather stay quiet, 10 = just have to say it). "
                f"Keep it low if the topic has little to do with your personality\n"
                f"- emotion: how you feel right now ({' / '.join(emotion_label(e) for e in EMOTIONS)})"
            )
            if self._background:
                instruction += (
                    "\n- check: if what you want to say involves a specific fact you're unsure of (a person, a work, "
                    "a year, an event...) and you'd like to look it up while the others are talking, the search terms "
                    "(1 to 3 noun keywords, not a sentence, separated by spaces). If you can talk without looking "
                    "anything up, an empty string. When in doubt, leave it empty"
                )
        else:
            instruction = (
                f"「{member.name}」として、今この会話にどう反応したいかを JSON で答えてください。\n"
                f"- gist: 言いたいことの要点（30字以内）\n"
                f"- stance: 直前の発言に対する立場（{' / '.join(STANCES)}）\n"
                f"- urge: 今すぐ発言したい度合い（0 = 黙っていたい、10 = どうしても言いたい）。"
                f"自分の性格に関係の薄い話題なら低くする\n"
                f"- emotion: 今の気持ち（{' / '.join(EMOTIONS)}）"
            )
            if self._background:
                instruction += (
                    "\n- check: 言いたいことに、自信のない具体的な事実（人物・作品・年・出来事など）があって、"
                    "ほかの子が話している間に調べておきたいなら、その検索語"
                    "（文ではなく名詞のキーワード1〜3語、空白区切り）。調べなくても話せるなら空文字。迷ったら空にする"
                )
        raw = self._llm.chat(
            self._messages(member, situation, instruction),
            schema=_hand_schema(self._background),
            max_tokens=(180 if self._background else 150) if en else (150 if self._background else 120),
            temperature=0.7,
            tag=f"{member.name}の挙手",
        )
        data = json.loads(raw)
        stance = data.get("stance")
        if en:
            stance = _STANCES_EN.get(str(stance).strip().lower())
        emotion = data.get("emotion")
        if en:
            emotion = parse_emotion(str(emotion or ""))
        check = str(data.get("check", "") or "").strip()[: 80 if en else 40]
        if check.lower().strip(".") in _NO_CHECK:
            check = ""
        return Hand(
            urge=max(0, min(10, int(data.get("urge", 0)))),
            stance=stance if stance in STANCES else "補足",
            gist=str(data.get("gist", ""))[: 200 if en else 80],
            emotion=emotion if emotion in EMOTIONS else NEUTRAL,
            check=check,
        )

    # --- 裏での調べもの（並列） --------------------------------------------------

    def _has_tool(self, trigger: str, spoken: list[str]) -> bool:
        """検索ツールを渡すか。ユーザーに最初に答える分身と、話し込みの各ラウンドの最初の話者（回数制限あり）。"""
        return (
            self._search is not None
            and not spoken
            and (
                trigger == "user"
                or (trigger == "talk" and self._session_searches < self._cfg.search.max_per_session)
            )
        )

    def _lookup_behind(self, hands: dict[str, Hand], speaker: str, speaker_searches: bool) -> None:
        """その回の最初の話者以外で、調べたいことがある子に裏で検索させる（挙手の強い順に）。

        speaker_searches: 最初の話者が検索ツールを持っている。そのときは、話者と同じことは裏で調べない
        （「東京タワー 完成年」を話者と裏の子が二重に引かないため）。
        """
        if not self._background:
            return
        sc = self._cfg.search
        budget = min(sc.background_per_round, sc.background_per_session - self._bg_searches)
        if budget <= 0:
            return
        taken = set(self._searched)
        if speaker_searches and hands[speaker].check:
            taken.add(_norm(hands[speaker].check))
        busy = {cid for cid, fut in self._lookups.items() if not fut.done()}
        order = sorted(
            (cid for cid, h in hands.items() if h.check and cid != speaker and cid not in busy),
            key=lambda cid: hands[cid].urge,
            reverse=True,
        )
        for cid in order:
            if budget <= 0:
                break
            query = hands[cid].check
            key = _norm(query)
            if not key or key in taken:
                continue
            taken.add(key)
            self._searched.add(key)
            self._bg_searches += 1
            budget -= 1
            logger.info("%s: 裏で「%s」を調べる", self._by_id[cid].name, query)
            self._lookups[cid] = self._lookup_pool.submit(
                self._lookup, self._by_id[cid], query, self._lookup_gen
            )

    def _lookup(self, member: CastMember, query: str, gen: int) -> None:
        """裏の検索（lookup スレッド）。結果は本人の記憶にだけ入れる。"""
        assert self._search is not None and self._memory is not None
        t0 = time.monotonic()
        try:
            hits = self._search.search(query)
        except SearchError as e:
            logger.warning("%s: 裏の検索に失敗: %s", member.name, e)
            return
        logger.info("lookup (%.1fs) %s %r -> %d hits", time.monotonic() - t0, member.name, query, len(hits))
        if not hits:
            return
        self._memory.add_fact(member.id, query, hits)
        with self._lookup_lock:
            if gen != self._lookup_gen:
                return  # もう話題が変わった。記憶には残すが「さっき調べた」とは言わせない
            self._fresh[member.id] = query
        if self._cfg.search.show_background:
            self._emit(
                (
                    "system",
                    language.pick(
                        ja=f"{member.name}がこっそり「{query}」を調べた（{self._search.label}）",
                        en=f'{member.name} quietly looked up "{query}" ({self._search.label})',
                    ),
                )
            )

    def _take_lookup(self, member: CastMember) -> str:
        """自分の番の前に、裏の検索が終わるのを少し待つ。調べ終わっていたら検索語（無ければ空）。"""
        fut = self._lookups.pop(member.id, None)
        if fut is not None and not fut.done():
            self._status(
                language.pick(
                    ja=f"{member.name} が調べたことを見ている…", en=f"{member.name} is reading what she looked up…"
                )
            )
            try:
                fut.result(timeout=_LOOKUP_WAIT_SEC)
            except FutureTimeout:
                logger.info("%s: 裏の検索が間に合わなかった", member.name)
            except Exception:  # noqa: BLE001 - 裏の検索の失敗で発話を止めない
                logger.exception("lookup crashed")
        with self._lookup_lock:
            return self._fresh.pop(member.id, "")

    def _choose(self, hands: dict[str, Hand], spoken: list[str], priority: list[str]) -> str | None:
        """次の発話者。挙手の強さ＋反対意見＋（2人目以降は）直前の話者との仲の悪さで決める。

        priority（名指しされた子・演出の役）は、その並び順で先に話させる。
        """
        recent = [t.speaker for t in self._turns if t.speaker != USER][-2:]
        plan = self._meta_plan
        best, best_score = None, float("-inf")
        for cid, hand in hands.items():
            if cid in spoken:
                continue
            if cid == self._meta_id and plan is not None and plan.weight <= 0 and cid not in priority:
                continue  # 6人目が続けて話しすぎている
            score = hand.urge + random.uniform(0.0, 2.0)
            if cid in priority:
                score += 100.0 * (len(priority) - priority.index(cid))
            if hand.stance in ("反対", "茶化す", "ツッコミ"):
                score += 1.5
            if hand.emotion not in (NEUTRAL, "楽"):
                score += 0.8  # 気持ちが動いている子は口を開きやすい
            if cid in self._fresh:
                score += 2.5  # 裏で調べ終わった子は、調べたことを言いたい
            if spoken:
                score -= self._affinity(cid, spoken[-1]) * 1.2
            elif recent:
                # ラウンドをまたいで同じキャラばかり喋らないようにする
                if cid == recent[-1]:
                    score -= 3.0
                elif cid in recent:
                    score -= 1.5
            if cid == self._meta_id and plan is not None and cid not in priority:
                # 6人目は、ふだん選ばれにくく、きっかけがあると選ばれやすい
                score = score * plan.weight if score > 0 else score - 5.0 * (1.0 - min(1.0, plan.weight))
            if score > best_score:
                best, best_score = cid, score
        return best

    # --- 発話（直列） ----------------------------------------------------------

    def _speak(
        self,
        member: CastMember,
        situation: str,
        hand: Hand,
        spoken: list[str],
        trigger: str,
        beat: Beat | None = None,
    ) -> bool:
        """1人ぶんを生成して再生キューへ積む。戻り値は、1文でも出せたか。"""
        n_max, max_chars = self._limits(trigger)
        looked_up = self._take_lookup(member) if self._background else ""
        if beat is not None and beat.kind in ("news", "weather") and beat.lead == member.id:
            # ニュース・天気予報の話は挨拶でも短くしすぎない
            n_max, max_chars = self._talk.max_sentences, self._talk.max_chars
        # 検索できるのは、ユーザーに最初に答える分身と、話し込みの各ラウンドの最初の話者（回数制限あり）
        use_search = self._has_tool(trigger, spoken)
        # 挙手で自分から調べたいと言っていて、まだ誰も調べていないこと（ツールを呼ばなければ代わりに調べる）
        must_check = use_search and bool(hand.check) and _norm(hand.check) not in self._searched
        beat_cue = beat.cue_for(member.id) if beat is not None else ""
        if language.is_en():
            instruction = self._speak_instruction_en(
                member, hand, spoken, trigger, beat_cue, use_search, must_check, looked_up, n_max, max_chars
            )
        else:
            instruction = self._speak_instruction_ja(
                member, hand, spoken, trigger, beat_cue, use_search, must_check, looked_up, n_max, max_chars
            )
        messages = self._messages(member, situation, instruction)
        # 直前に話した子（発言の相手を決めるのに使う）。今の発言を足す前に調べる
        prev_speaker = spoken[-1] if spoken else self._last_cast_speaker()
        turn = self._add_turn(member.id, [])
        turn.feel = hand.emotion
        tools = [tool()] if use_search else None
        n_sent, calls = self._stream_lines(member, turn, messages, tools, n_max, max_chars)
        if n_sent == 0 and not calls and not self._interrupted():
            # 空振りや、指示の言い換えを捨てたあとは1回だけ引き直す
            n_sent, calls = self._stream_lines(member, turn, messages, tools, n_max, max_chars)
        return self._finish_speak(member, turn, messages, calls, n_sent, n_max, max_chars, hand, trigger,
                                  use_search, must_check, prev_speaker)

    def _speak_instruction_en(
        self,
        member: CastMember,
        hand: Hand,
        spoken: list[str],
        trigger: str,
        beat_cue: str,
        use_search: bool,
        must_check: bool,
        looked_up: str,
        n_max: int,
        max_chars: int,
    ) -> str:
        u = self._uref
        if spoken:
            prev = self._by_id[spoken[-1]].name
            cue = f"Respond to what {prev} just said."
            if beat_cue:
                cue += f"\nWhat happens this time: {beat_cue}"
            else:
                cue += (
                    f" If you agree, say so with feeling; if you think she's wrong, don't hold back and talk back to {prev}; "
                    "if she said something silly, call it out."
                )
        elif beat_cue:
            # 演出のリード。挙手の要点より演出を優先する
            cue = f"What happens this time: {beat_cue}"
            if trigger == "talk":
                cue += f"\nIf you can tie it to {u}'s topic, do."
        else:
            stance = next((k for k, v in _STANCES_EN.items() if v == hand.stance), "add")
            cue = f"What you were going to say: {hand.gist} (stance: {stance}, feeling: {emotion_label(hand.emotion)})"
            if trigger == "talk":
                move = random.choice(_TALK_MOVES_EN).format(user=u)
                cue += f"\nHow to move the conversation along: {move}."
            if trigger == "user":
                # 分身どうしの言い合いに流れて、ユーザーが置き去りになるのを防ぐ
                cue += (
                    f"\nFirst answer what {u} just said, speaking directly to {u}. "
                    "Any comebacks to the other girls come after that."
                )
        if use_search:
            if trigger == "user":
                cue += (
                    f"\nWhen {u} asks about facts (people, works, history, places, recent events and so on), "
                    f"look it up with the {TOOL_NAME} tool before answering."
                )
            else:
                cue += (
                    "\nIf you want to talk about a specific fact (a person, a work, a year, an event...) but it isn't "
                    f'in "Your memories" and you aren\'t sure, look it up with the {TOOL_NAME} tool before you talk about it.'
                )
            # 例文を固定すると全員が同じ一言を写すので、キャラごとの search_line を手本にする
            cue += (
                " When you look something up, before calling the tool say only one short line, in your own voice, "
                f'saying you\'re going to check (e.g. "{member.search_line}"; don\'t say it the same way every time).'
            )
            if must_check:
                cue += (
                    f'\nYou don\'t really know about "{hand.check}", so don\'t pretend you do: say you\'ll check, '
                    f"then look it up with the {TOOL_NAME} tool."
                )
        if looked_up:
            cue += (
                f'\nWhile the others were talking, you quietly looked up "{looked_up}" '
                '(the results are under "Things you looked up yourself" in "Your memories"). '
                'If it fits the flow, you can use it in your own words, like "I just looked it up, and...". '
                "Don't state anything as fact that isn't in the results."
            )
        return (
            f'As "{member.name}", write only the line you say next. {cue}\n'
            f"{_length_rule(n_max, max_chars)} No name label and no quotation marks. "
            "Speak as yourself, and don't repeat phrases or catchphrases you just used."
            + (f"\n{emotion_rule()}" if self._drama is not None else "")
        )

    def _speak_instruction_ja(
        self,
        member: CastMember,
        hand: Hand,
        spoken: list[str],
        trigger: str,
        beat_cue: str,
        use_search: bool,
        must_check: bool,
        looked_up: str,
        n_max: int,
        max_chars: int,
    ) -> str:
        if spoken:
            prev = self._by_id[spoken[-1]].name
            cue = f"直前の{prev}の発言を受けて話してください。"
            if beat_cue:
                cue += f"\n今回の展開: {beat_cue}"
            else:
                cue += (
                    f"共感したら気持ちを込めて、違うと思ったら遠慮せず{prev}に言い返し、"
                    "ボケていたらツッコんでください。"
                )
        elif beat_cue:
            # 演出のリード。挙手の要点より演出を優先する
            cue = f"今回の展開: {beat_cue}"
            if trigger == "talk":
                cue += f"\n{self._uref}の話題とつなげられるなら、つなげてください。"
        else:
            cue = f"言おうとしていたこと: {hand.gist}（立場: {hand.stance}、気持ち: {hand.emotion}）"
            if trigger == "talk":
                move = random.choice(_TALK_MOVES).format(user=self._uref)
                cue += f"\n話の進め方: {move}。"
            if trigger == "user":
                # 分身どうしの言い合いに流れて、ユーザーが置き去りになるのを防ぐ
                cue += (
                    f"\nまず{self._uref}の直前の発言に、{self._uref}へ向けて直接答えてください。"
                    "ほかの子への言い返しはその後にしてください。"
                )
        if use_search:
            if trigger == "user":
                cue += (
                    f"\n人物・作品・歴史・場所・最新の出来事など、事実について{self._uref}に聞かれたときは、"
                    f"{TOOL_NAME} ツールで調べてから答えてください。"
                )
            else:
                cue += (
                    "\n人物・作品・年・出来事など具体的な事実を話したいのに、"
                    "「あなたの記憶」に無くて自信がないときは、"
                    f"{TOOL_NAME} ツールで調べてから話してください。"
                )
            # 例文を固定すると全員が同じ一言を写すので、キャラごとの search_line を手本にする
            cue += (
                "調べるときは、ツールを呼ぶ前に、これから調べると伝える短い一言だけを自分の口調で言ってください"
                f"（例: {member.search_line}。毎回同じ言い方にはしない）。"
            )
            if must_check:
                # 挙手で自分から「調べたい」と言っていた。小さいモデルはツールを呼ばずに「誰だろう」で済ませがちなので念を押す
                cue += (
                    f"\nあなたは「{hand.check}」のことをよく知らないので、知ったかぶりをせず、"
                    f"調べると一言言ってから {TOOL_NAME} ツールで調べてください。"
                )
        if looked_up:
            cue += (
                f"\nほかの子が話している間に、あなたはこっそり「{looked_up}」を調べておいた"
                "（結果は「あなたの記憶」の「自分で調べたこと」にある）。"
                "話の流れに合えば「さっき調べたんだけど」のように自分の口調で使ってよい。"
                "結果に書いていないことは言い切らない。"
            )
        return (
            f"「{member.name}」として、次に言うセリフだけを書いてください。{cue}\n"
            f"{_length_rule(n_max, max_chars)}名前やカギ括弧は付けない。"
            "一人称は自分のものを使い、直前に使った言い回しや決まり文句は繰り返さない。"
            + (f"\n{emotion_rule()}" if self._drama is not None else "")
        )

    def _finish_speak(
        self,
        member: CastMember,
        turn: Turn,
        messages: list[dict],
        calls: list[ToolCall],
        n_sent: int,
        n_max: int,
        max_chars: int,
        hand: Hand,
        trigger: str,
        use_search: bool,
        must_check: bool,
        prev_speaker: str | None,
    ) -> bool:
        """ストリームのあと（検索の代わりの呼び出し・検索して続き・関係の変化）。"""
        if use_search and not calls and not self._interrupted():
            if _promised_search(turn):
                # 「ちょっと調べてみる」と言っただけでツールを呼ばなかったときは、検索語だけ聞き直して調べる
                query = self._ask_query(member, messages, turn.text)
                if query:
                    logger.info("%s: 調べると言ったのにツールを呼ばなかったので検索語を聞き直した", member.name)
                    calls = [ToolCall("call_promised", TOOL_NAME, {"query": query})]
            elif must_check and turn.parts:
                # 挙手で調べたいと言っていたのに、ツールを呼ばずに「誰だろう」で済ませた。挙手の検索語で調べて続ける
                logger.info("%s: 調べたいと言っていたのにツールを呼ばなかったので「%s」を調べる", member.name, hand.check)
                # 途中で「調べてみるね」とすでに言っていたら、決まり文句は重ねない
                promise = language.pick(ja=_PROMISE_RE, en=_PROMISE_RE_EN)
                if member.search_line and not any(promise.search(text) for _, text in turn.parts):
                    self._say(member, turn, member.search_line)
                    n_sent += 1
                calls = [ToolCall("call_check", TOOL_NAME, {"query": hand.check})]
        call = next((c for c in calls if c.name == TOOL_NAME), None)
        if call is not None and not self._interrupted():
            if trigger == "talk":
                self._session_searches += 1
            self._search_and_continue(member, turn, messages, call, n_sent, n_max, max_chars)
        if not turn.parts:
            self._turns.remove(turn)
            return False
        if self._town is not None and member.id != self._meta_id:
            self._town.on_spoken(member.id, turn.text)
        if self._drama is not None:
            target = self._drama.target_of(member.id, turn.text, prev_speaker)
            self._drama.on_utterance(member.id, turn.mood, target)
            self._emit(("moods", self._drama.moods_for_ui()))
        return True

    def _last_cast_speaker(self) -> str | None:
        for t in reversed(self._turns):
            if t.speaker != USER and t.parts:
                return t.speaker
        return None

    def _affinity(self, a: str, b: str) -> float:
        if self._drama is not None:
            return self._drama.affinity(a, b)
        return float(self._cfg.affinity.get(a, {}).get(b, 0))

    def _limits(self, trigger: str) -> tuple[int, int]:
        """1回の発言の (文の数の上限, 字数の目安)。挨拶だけは短く。"""
        if trigger == "start":
            # 英語は同じ中身で字数が2倍強になる
            return min(2, self._talk.max_sentences), min(language.pick(ja=60, en=140), self._talk.max_chars)
        return self._talk.max_sentences, self._talk.max_chars

    def _ask_query(self, member: CastMember, messages: list[dict], said: str) -> str:
        try:
            raw = self._llm.chat(
                [
                    *messages,
                    {"role": "assistant", "content": said},
                    {
                        "role": "user",
                        "content": language.pick(
                            ja="今言ったとおりに調べます。検索語を JSON で答えてください。"
                            "query: 文ではなく名詞のキーワードを1〜3語、空白区切りで（例: VOICEVOX 開発者）。年号や推測した細かい条件は入れない",
                            en="You'll look it up, just as you said. Answer with the search terms in JSON. "
                            "query: 1 to 3 noun keywords, not a sentence, separated by spaces (e.g. Kokoro TTS developer). "
                            "Don't add years or details you are only guessing",
                        ),
                    },
                ],
                schema=_query_schema(),
                max_tokens=60,
                temperature=0.2,
            )
            return str(json.loads(raw).get("query", "")).strip()
        except (LLMError, ValueError, TypeError, AttributeError) as e:
            logger.warning("%s: 検索語を聞き直せなかった: %s", member.name, e)
            return ""

    def _stream_lines(
        self,
        member: CastMember,
        turn: Turn,
        messages: list[dict],
        tools: list[dict] | None,
        n_max: int,
        max_chars: int,
    ) -> tuple[int, list[ToolCall]]:
        """ストリームを文に切って再生キューへ積む。戻り値は (積んだ文の数, ツール呼び出し)。"""
        buf = ""
        total = 0
        n_sent = 0
        calls: list[ToolCall] = []
        # 日本語はおおよそ1字1トークン強。字数の目安を超えて打ち切るぶんの余裕を持たせる
        stream = self._llm.stream(messages, max_tokens=int(max_chars * 2) + 80, tools=tools, tag=member.name)
        meta = False
        try:
            for chunk in stream:
                if isinstance(chunk, ToolCall):
                    calls.append(chunk)
                    continue
                buf += chunk
                while (m := _find_sentence_end(buf)) is not None:
                    raw, buf = buf[: m.end()], buf[m.end() :]
                    emo, raw = take_tags(raw)
                    if emo is not None:
                        turn.feel = emo
                    if _is_meta(raw, member.name):
                        meta = True
                        break
                    sentence = _clean(raw, member.name)
                    if not sentence or self._is_repeat(member, sentence):
                        continue
                    self._say(member, turn, sentence)
                    n_sent += 1
                    total += len(sentence)
                if meta or n_sent >= n_max or total >= max_chars * 1.5 or self._interrupted():
                    buf = ""
                    break
        finally:
            stream.close()
        if meta:
            logger.warning("%s: 指示の言い換えのような出力を捨てた", member.name)
        emo, buf = take_tags(buf)
        if emo is not None:
            turn.feel = emo
        tail = "" if _is_meta(buf, member.name) else _clean(buf, member.name)
        if tail and not self._interrupted() and not self._is_repeat(member, tail):
            self._say(member, turn, tail)
            n_sent += 1
        return n_sent, calls

    def _is_repeat(self, member: CastMember, sentence: str) -> bool:
        """直近の会話（今の発言を含む）で、すでに誰かが言った文か。

        検索のあとの「続き」で、モデルが言い終えた文をそのまま書き直したり、
        2人目が1人目のセリフを丸写ししたりすることがあるため。
        頭に「ユウ、」「うーん、〜けど、」が付いただけの言い直しも同じ文とみなす。
        「うん。」のような短い相づちは重なっても自然なので見逃す。
        """
        key = _norm(sentence)
        if len(key) < language.pick(ja=6, en=15):
            return False
        for t in self._turns[-8:]:
            for _, text in t.parts:
                if _same_sentence(key, _norm(text)):
                    logger.info("%s: 直前に言った文とほぼ同じなので読み上げない: %s", member.name, sentence)
                    return True
        return False

    def _search_and_continue(
        self,
        member: CastMember,
        turn: Turn,
        messages: list[dict],
        call: ToolCall,
        n_sent: int,
        n_max: int,
        max_chars: int,
    ) -> None:
        """検索して、その結果を踏まえた続きを話す。結果の全文は、調べた本人の記憶にだけ残す。"""
        assert self._search is not None
        query = str(call.arguments.get("query", "")).strip()
        self._searched.add(_norm(query))  # 同じことを裏でもう一度調べない
        if n_sent == 0 and member.search_line:
            # 先に一言も言わずにツールを呼んだときは、決まり文句でつなぐ
            self._say(member, turn, member.search_line)
        en = language.is_en()
        self._emit(("status", f"{member.name} is looking it up…" if en else f"{member.name} が調べている…"))
        self._emit(
            (
                "system",
                f'{member.name} searched for "{query}" ({self._search.label})'
                if en
                else f"{member.name}が「{query}」を検索（{self._search.label}）",
            )
        )
        t0 = time.monotonic()
        try:
            hits = self._search.search(query)
            result = format_hits(query, hits)
            note = "Based on these search results" if en else "この検索結果を踏まえて"
        except SearchError as e:
            logger.warning("%s", e)
            self._emit(("system", str(e)))
            hits = []
            result = f'Couldn\'t look up "{query}" (the search failed).' if en else f"「{query}」は調べられませんでした（検索が失敗）。"
            note = "Honestly say you couldn't look it up, then" if en else "調べられなかったことを正直に伝えたうえで"
        logger.info("search (%.1fs) %r -> %d hits", time.monotonic() - t0, query, len(hits))
        if hits and self._memory is not None:
            self._memory.add_fact(member.id, query, hits)
        if self._interrupted():
            return
        # 続きは残りの文数だけ。言い切ったあとに上限いっぱい頼むと、同じ文を書き直してくる
        rest = max(2, n_max - n_sent)
        rest_chars = max(language.pick(ja=60, en=140), max_chars * rest // n_max)
        # 言い終えた文を指示の中で引用して、書き直さないよう念を押す。assistant の発言として渡すだけだと、
        # 言い終えたあとで調べたとき（挙手の check の代わりの検索）に、gemma は6回中6回とも頭から言い直した（引用すると0回）
        said = turn.text.strip()
        if en:
            content = (
                f"{result}\n\n"
                + (f'(You have already said: "{said}". Don\'t rewrite any of it.) ' if said else "")
                + f"({note} write only the rest of your line, continuing from what you just said, "
                f'in {member.name}\'s voice: what you found out. {_length_rule(rest, rest_chars)} Don\'t read out URLs or site names. '
                "Don't repeat what you already said. If you had already given an answer, say so briefly if it was "
                "right, or correct it if it was wrong. Don't state anything as fact that isn't in the results. "
                "You can't search any more, so if you didn't find the answer, honestly say you couldn't find out."
                + (" Start with a feeling tag like [happy]." if self._drama is not None else "")
                + ")"
            )
        else:
            content = (
                f"{result}\n\n"
                + (f"（あなたはもう「{said}」と言い終えています。これは一字も書き直さない）" if said else "")
                + f"（{note}、さっきの一言の続きとして、調べてわかったことを話すセリフだけを「{member.name}」の口調で書いてください。"
                f"{_length_rule(rest, rest_chars)}URL やサイト名は読み上げない。"
                "さっき言ったことはくり返さない。先に答えを言っていたなら、"
                "合っていれば短くそう言い、違っていれば訂正する。"
                "結果に書いていないことは言い切らない。これ以上は調べられないので、"
                "答えが見つからなければ、わからなかったと正直に言う。"
                + ("文頭には気持ちのタグ【喜】などを付ける。" if self._drama is not None else "")
                + "）"
            )
        followup = [
            *messages,
            {"role": "assistant", "content": turn.text, "tool_calls": [call.as_message()]},
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": content,
            },
        ]
        self._emit(("status", f"{member.name} is about to speak…" if en else f"{member.name} が話そうとしている…"))
        self._stream_lines(member, turn, followup, None, rest, rest_chars)

    def _say(self, member: CastMember, turn: Turn, sentence: str) -> None:
        uid = self._next_uid()
        emotion = turn.feel
        try:
            # 感情専用の声（VOICEVOX の「泣き」「ツンツン」、Kokoro の配合）があればその声で
            pcm, times, opens = self._tts.synth_for(member, sentence, emotion)
        except VoiceError as e:
            if not self._voice_warned:
                self._voice_warned = True
                self._emit(("system", f"{e}" + language.pick(ja="（音声なしで続けます）", en=" (continuing without voice)")))
            # 声が出せなくても、読める速さでログと口パクだけ進める（英語は1字あたりが短い）
            per_char = language.pick(ja=0.13, en=0.065)
            pcm = np.zeros(int(SAMPLE_RATE * max(0.8, len(sentence) * per_char)), dtype=np.float32)
            times, opens = [0.0], [0.0]
        turn.parts.append((uid, sentence))
        turn.emotions[uid] = emotion
        self._player.enqueue(
            Utterance(member.id, sentence, pcm, times, opens, uid=uid, turn=turn.id, emotion=emotion)
        )

    # --- プロンプト ------------------------------------------------------------

    def _visible_turns(self) -> list[Turn]:
        """プロンプトの会話ログに入る発言（直近から history_lines 行まで、合計 history_chars 字まで）。古い順。"""
        out: list[Turn] = []
        budget = self._talk.history_chars
        for t in reversed([t for t in self._turns if t.parts][-self._talk.history_lines :]):
            line = self._log_line(t)
            if out and len(line) > budget:
                break
            out.append(t)
            budget -= len(line)
        return out[::-1]

    def _system_en(self) -> str:
        user = self._uref
        roster = "\n".join(f"- {m.name}: {m.desc}" for m in self._cast)
        n = len(self._girls)
        meta = self._meta
        tags = " (the feeling tags in [ ] are the only exception)" if self._drama is not None else ""
        who = f"{n} girls" + (f", plus {meta.name} in {meta.place}" if meta is not None else "")
        return (
            f"This is a conversation between {user} and {user}'s imaginary friends: {who}. "
            f"They talk with {user} and with each other.\n"
            + (
                f"The {n} girls were each born from a different side of {user}'s heart.\n"
                if self._town is not None
                else f"The {n} girls were each born from a different side of {user}'s heart, and they live together "
                f"in a little town inside {user}'s head.\n"
            )
            + "The mood is a cozy slice-of-life four-panel comic: laid-back, cute everyday life, silly jokes and "
            "comebacks, small misunderstandings and making up.\n"
            "Everyone wears their feelings on their sleeve: they laugh, sulk, squabble, and cry happy tears. "
            f"Even when they fight they never truly hate each other, and they never say anything cruel or really mean. "
            f"Every one of them adores {user} and is on {user}'s side.\n"
            + (f"{self._town.system_text()}\n\n" if self._town is not None else "\n")
            + (f"{meta.common_text()}\n\n" if meta is not None else "")
            + f"# The girls\n{roster}\n\n"
            "# How to talk\n"
            "- Speak natural, casual spoken English, and keep it short.\n"
            f"- Every line is read aloud by a speech synthesizer, so no emoji, emoticons, symbols, stage directions, "
            f"asterisks or parentheses{tags}.\n"
            '- Write times and scores in words ("two thirty", "seven to three"), never with colons, slashes or '
            'hyphens between numbers. Write units as words ("333 meters", "40 percent").\n'
            "- Stick to your own personality and way of talking. Don't just repeat what another girl said.\n"
            f'- When you talk to {user}, call {user} "{self._cfg.user_name}". The girls call each other by name or nickname.\n'
            # 小さいモデルは直前のログの口調を強く真似るので、崩れたセリフが連鎖しないようにする
            "- Some earlier lines in the conversation slipped out of character. Match your own character "
            "description, not the log.\n"
            # 分身どうしの言い返しが続くと、ユーザーの話題から離れていくため
            f"- The heart of the conversation is the topic {user} brought up. It's fine for it to grow into talk "
            "about your own feelings and relationships, but if unrelated talk drags on, steer back to the topic "
            "in your own way.\n"
            "- If you know a fact, feel free to be specific, but don't state numbers or events you're unsure of.\n"
            # 調べた結果の全文は本人しか持っていない。聞いただけの分身が細部を作らないようにする
            "- You only know what the other girls looked up as far as you heard it in the conversation. "
            "Don't add names or numbers you didn't hear."
        )

    def _messages(self, member: CastMember, situation: str, instruction: str) -> list[dict]:
        if language.is_en():
            return self._messages_en(member, situation, instruction)
        user = self._uref
        roster = "\n".join(f"- {m.name}: {m.desc}" for m in self._cast)
        n = len(self._girls)
        meta = self._meta
        tags = "（気持ちのタグ【】だけは例外）" if self._drama is not None else ""
        who = f"{n}人の女の子" + (f"と、{meta.place}にいる{meta.name}" if meta is not None else "")
        system = (
            f"これは、{user}のイマジナリーフレンドである{who}が、{user}や互いと話す会話です。\n"
            + (
                f"{n}人は{user}の心の一面から生まれた女の子たちです。\n"
                if self._town is not None
                else f"{n}人は{user}の心の一面から生まれた女の子たちで、{user}の頭の中の小さな町で、いっしょに暮らしています。\n"
            )
            + "雰囲気は、日常ものの4コマ漫画です。ゆるくてかわいい日常、ボケとツッコミ、ちょっとしたすれ違いと仲直り。\n"
            "みんな喜怒哀楽が豊かで、笑ったり、すねたり、ケンカしたり、うれし泣きしたりします。"
            f"ケンカをしても本気で嫌い合うことはなく、意地悪や本気の悪口は言いません。全員{user}のことが大好きな味方です。\n"
            + (f"{self._town.system_text()}\n\n" if self._town is not None else "\n")
            + (f"{meta.common_text()}\n\n" if meta is not None else "")
            + f"# 女の子たち\n{roster}\n\n"
            "# 話し方の決まり\n"
            "- 自然な話し言葉の日本語で、短く話す。\n"
            f"- 音声で読み上げるので、絵文字・顔文字・記号・ト書き・カッコ書きは使わない{tags}。\n"
            "- 数や年は算用数字で書く（例: 333メートル、1958年）。漢数字にしない。\n"
            "- 自分の性格と口調を守る。ほかの子の言ったことをそのまま繰り返さない。\n"
            f"- {user}に話すときは「{self._cfg.user_name}」と呼びかける。女の子どうしは名前やあだ名で呼び合う。\n"
            # 小さいモデルは直前のログの口調を強く真似るので、崩れたセリフが連鎖しないようにする
            "- これまでの会話には、口調が崩れたセリフも混ざっている。口調はログではなく、自分の人物説明に合わせる。\n"
            # 分身どうしの言い返しが続くと、ユーザーの話題から離れていくため
            f"- 話の中心は{user}が出した話題。そこから自分たちの気持ちや関係の話にふくらむのはよいが、"
            "関係のない話が続いたら、自分らしいやり方で話題に戻す。\n"
            "- 知っている事実があれば具体的に入れてよいが、自信のない数字や出来事は言わない。\n"
            # 調べた結果の全文は本人しか持っていない。聞いただけの分身が細部を作らないようにする
            "- ほかの子が調べたことは、会話で聞いた範囲しか知らない。聞いていない名前や数字を足さない。"
        )
        log, is_meta, meta_text, relations, feeling, life, memory = self._personal(member)
        content = (
            f"# これまでの会話\n{log or '（まだ誰も話していない）'}\n\n"
            f"# 今の状況\n{situation}\n\n"
            f"# あなた\nあなたは「{member.name}」です。{member.desc}\n"
            + (f"{relations}。\n" if relations else "")
            + (
                "話し方の例（内容ではなく口調の参考）:\n" + "".join(f"- {s}\n" for s in member.examples)
                if member.examples
                else ""
            )
            + (meta_text if meta_text and not is_meta else "")
            + (f"\n{feeling}" if feeling else "")
            + (f"\n{meta_text}" if meta_text and is_meta else "")
            + (f"\n{life}" if life else "")
            + (f"\n{memory}" if memory else "")
            + f"\n# 指示\n{instruction}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": content}]

    def _messages_en(self, member: CastMember, situation: str, instruction: str) -> list[dict]:
        log, is_meta, meta_text, relations, feeling, life, memory = self._personal(member)
        content = (
            f"# The conversation so far\n{log or '(nobody has said anything yet)'}\n\n"
            f"# The current situation\n{situation}\n\n"
            f'# You\nYou are "{member.name}". {member.desc}\n'
            + (f"{relations}.\n" if relations else "")
            + (
                "Examples of how you talk (for your tone, not the content):\n"
                + "".join(f"- {s}\n" for s in member.examples)
                if member.examples
                else ""
            )
            + (meta_text if meta_text and not is_meta else "")
            + (f"\n{feeling}" if feeling else "")
            + (f"\n{meta_text}" if meta_text and is_meta else "")
            + (f"\n{life}" if life else "")
            + (f"\n{memory}" if memory else "")
            + f"\n# Instructions\n{instruction}"
        )
        return [{"role": "system", "content": self._system_en()}, {"role": "user", "content": content}]

    def _personal(self, member: CastMember) -> tuple[str, bool, str, str, str, str, str]:
        """キャラ固有部分の材料（会話ログ・6人目の文・関係・気持ち・町の出来事・記憶）。"""
        meta = self._meta
        # 直近から history_lines 行まで、合計 history_chars 字まで（長いセリフでコンテキストを溢れさせない）
        lines = [self._log_line(t) for t in reversed(self._visible_turns())]  # 新しい順
        log = "\n".join(reversed(lines))
        # 本人の記憶のうち、今の話題（話し込みのきっかけと直近の発言）に近いもの
        context = "\n".join([self._session_topic, *lines[:3]])
        memory = ""
        if self._memory is not None:
            memory = self._memory.recall(member, context)
        # 町での出来事（本人が体験したこと）。「# あなたの記憶」とは見出しを分ける
        is_meta = meta is not None and member.id == self._meta_id
        life = (
            self._town.life_text(member, context, self._talk_ctx)
            if self._town is not None and not is_meta
            else ""
        )
        # 6人目には本人だけが知っていること（と、このラウンドの実行状況）、5人には6人目の扱い方
        meta_text = ""
        if meta is not None:
            meta_text = meta.self_text(self._meta_plan) if is_meta else meta.others_note() + "\n"
        if self._drama is not None:
            relations = self._drama.relations_text(member.id)
            feeling = self._drama.feeling_text(member.id)
        else:
            parts = []
            for other, v in self._cfg.affinity.get(member.id, {}).items():
                word = affinity_word(float(v))
                if word and other in self._by_id:
                    name = self._by_id[other].name
                    parts.append(language.pick(ja=f"{name}とは{word}", en=f"{name} is {word}"))
            relations = language.pick(ja="、", en="; ").join(parts)
            feeling = ""
        return log, is_meta, meta_text, relations, feeling, life, memory

    def _display_name(self, speaker: str) -> str:
        return self._uref if speaker == USER else self._by_id[speaker].name

    def _log_line(self, t: Turn) -> str:
        """会話ログの1行。感情が出ていれば「アカリ（怒）: …」のように添える。"""
        mood = t.mood
        name = self._display_name(t.speaker)
        if mood == NEUTRAL:
            return f"{name}: {t.text}"
        return language.pick(ja=f"{name}（{mood}）: {t.text}", en=f"{name} ({emotion_label(mood)}): {t.text}")

    # --- 履歴 ------------------------------------------------------------------

    def _next_uid(self) -> int:
        self._uid += 1
        return self._uid

    def _add_turn(self, speaker: str, parts: list[tuple[int, str]]) -> Turn:
        turn = Turn(self._next_uid(), speaker, parts)
        self._turns.append(turn)
        del self._turns[: -self._talk.history_lines * 2]
        return turn

    def _drop(self, utts: list[Utterance]) -> None:
        """再生されずに捨てられた文を履歴から消す（言っていないことを言ったことにしない）。"""
        dropped = {u.uid for u in utts}
        if not dropped:
            return
        for t in self._turns:
            t.parts = [p for p in t.parts if p[0] not in dropped]

    def _store_loop(self) -> None:
        while True:
            utt = self._played.get()
            self._store.add(utt.speaker_id, utt.text)

    def _load_history(self) -> None:
        for speaker, text in self._store.recent(self._talk.history_lines * 3):
            if speaker != USER and speaker not in self._by_id:
                continue  # キャストから外れたキャラの過去発言
            if self._turns and self._turns[-1].speaker == speaker and speaker != USER:
                self._turns[-1].parts.append((self._next_uid(), text))
            else:
                self._turns.append(Turn(self._next_uid(), speaker, [(self._next_uid(), text)]))
        del self._turns[: -self._talk.history_lines]


def _promised_search(turn: Turn) -> bool:
    """「ちょっと調べてみる」で発言を終えたか（ツールを呼ばなかったときに代わりに調べる判定）。

    最後の文だけを見る。途中の「さっき調べたんだけど」に反応すると、
    言い終えた発言のあとに検索して、同じ内容を言い直してしまうため。
    """
    if not turn.parts:
        return False
    last = turn.parts[-1][1]
    promise = language.pick(ja=_PROMISE_RE, en=_PROMISE_RE_EN)
    past = language.pick(ja=_PAST_SEARCH_RE, en=_PAST_SEARCH_RE_EN)
    return bool(promise.search(last)) and not past.search(last)


# ユウが去るときのあいさつ（「おやすみ」「またね」「行ってきます」）。「おやすみの日」のような話題は拾わない
_GOODBYE_RE = re.compile(
    r"おやすみ(?!の日|だった|なの|中)|(?:そろそろ|もう)(?:寝る|寝よう|落ちる|行く|出かける|帰る)|"
    r"また(?:ね|明日|あした|あとで|後で)|バイバイ|ばいばい|じゃあね|行ってきます|いってきます"
)
_GOODBYE_RE_EN = re.compile(
    r"\b(good ?night|night night|bye|goodbye|see you|see ya|talk later|gotta go|heading out|"
    r"(?:going|off) to (?:bed|sleep))\b",
    re.IGNORECASE,
)


def says_goodbye(text: str) -> bool:
    """ユウの発言が、寝る・出かけるなどで会話を切り上げるあいさつか。"""
    return bool(language.pick(ja=_GOODBYE_RE, en=_GOODBYE_RE_EN).search(text or ""))


def _norm(text: str) -> str:
    if language.is_en():
        return _NORM_RE_EN.sub("", text.lower())
    return _NORM_RE.sub("", text)


def _same_sentence(a: str, b: str) -> bool:
    """正規化した2文がほぼ同じか。完全一致、または短いほうの8割以上（12字以上）が1続きで重なる。

    長さが倍以上ちがう2文は別の文とみなす（短い相づちが長い文の一部に入っているだけのことが多い）。
    """
    if a == b:
        return True
    short, long_ = sorted((len(a), len(b)))
    if short < language.pick(ja=12, en=30) or short * 2 < long_:
        return False
    m = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size >= short * 0.8


def _length_rule(n_max: int, max_chars: int) -> str:
    """発言の長さの指示。下限も書かないと、モデルは1文で済ませがち。

    英語版の max_chars も字数で持ち（打ち切りの判定と同じ単位）、モデルには語数で見せる。
    """
    if language.is_en():
        words = max(8, round(max_chars / 5.5))
        if n_max <= 2:
            return f"1 to {n_max} sentences, under {words} words."
        return f"{n_max - 1} to {n_max} sentences, up to about {words} words."
    if n_max <= 2:
        return f"1〜{n_max}文、{max_chars}字以内。"
    return f"{n_max - 1}〜{n_max}文、{max_chars}字くらいまで。"


def _is_meta(text: str, name: str) -> bool:
    """セリフではなく、指示やプロンプトを言い換えた文か。

    例: 「ユーザーは『レン』として…」「- 指示：…」、
        ツールの説明を受けた「固有名詞の詳しい情報に該当する可能性が高いので…」
    """
    text = text.strip()
    if language.is_en():
        return bool(
            re.match(r"[-•*]\s", text)
            or re.search(rf"\bas [\"“]{re.escape(name)}[\"”]", text, re.IGNORECASE)
            or re.search(r"\binstructions?\s*:|\bproper nouns?\b|\bsearch tool\b|web_search|^the user\b", text, re.IGNORECASE)
        )
    return bool(
        re.match(r"[-・*]\s", text)
        or f"「{name}」として" in text
        or re.search(r"指示\s*[:：]|固有名詞|検索ツール|web_search", text)
    )


def _clean(text: str, name: str) -> str:
    """LLM の1文を読み上げ・表示用に整える。"""
    if language.is_en():
        return _clean_en(text, name)
    text = text.strip()
    text = re.sub(rf"^(?:{re.escape(name)}|「{re.escape(name)}」)\s*[:：]\s*", "", text)
    text = text.strip("「」『』\"' 　")
    text = re.sub(r"[*_#`]+", "", text)
    # 記号だけ（「。」「…」など）の断片は読み上げない
    if not re.search(r"[0-9A-Za-zぁ-んァ-ヶ一-龠々]", text):
        return ""
    return text


def _clean_en(text: str, name: str) -> str:
    text = text.strip()
    text = re.sub(rf"^[\"“]?{re.escape(name)}[\"”]?\s*:\s*", "", text, flags=re.IGNORECASE)
    # *giggles* や (sighs) のようなト書きは読み上げない（感情として読めるものは take_tags が先に外している）
    text = re.sub(r"\*[^*]*\*|\([^)]*\)", "", text)
    text = re.sub(r"[*_#`]+", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = text.strip("\"“”「」 ")
    if not re.search(r"[0-9A-Za-z]", text):
        return ""
    return text
