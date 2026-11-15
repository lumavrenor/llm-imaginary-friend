"""町（イマジナリーランド）と、そこでの5人の暮らし。仕様は docs/2026-11-14-イマジナリーランド.md。

- 場所は config.toml の [[town.places]] に固定。モデルには「そこで何があったか」だけを書かせる
- 起動時に、前回作ったときから今までのあいだに起きた出来事を、会話の裏で1回の LLM 呼び出しで作る（structured output）。
  1つの出来事は、その場にいた子の数だけ個人記憶（memories の kind = 'life'）の行になる。いなかった子は知らない。
  その日の気分がまだ決まっていなければ、同じ呼び出しで5人の今日の気分も作り、drama.py に渡す（決まった表は持たない）
- drama.py の「打ち明け話」「ないしょ話の発覚」は、ここのまだ話していない出来事を材料にする
- 話すときは、本人の出来事のうち、まだユウに話していないものを話しやすさ（openness）に応じて少しだけ入れる
    open   … 話題に近ければ入れる。ユウが黙っている間の雑談では、話題に関係なく1件入れてよい
    ask    … ユウが「最近どう？」「昨日何してた？」と聞いたとき、またはユウの質問の話題に近いとき
    secret … ユウがその子を名指しして聞いたときだけ（「言いにくそうに話す」を添える）
- 話したかは、プロンプトに入れた出来事とセリフの2文字の重なりで判定する。1人が話したら、同じ出来事の全員の行を「話した」にする
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import random
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from . import clock, language
from .config import CastMember, TownConfig
from .llm import LLMClient, LLMError
from .memory import _bigrams, _score
from .store import Store

if TYPE_CHECKING:
    from .drama import Drama

logger = logging.getLogger(__name__)

OPENNESS = ("open", "ask", "secret")
# これより長く空いても、作るのは直近のこの日数ぶんだけ
_MAX_SPAN_DAYS = 7
# 前回作った時刻より、これ以上前の時間帯の出来事は捨てる（前回の会話と食い違わないように）
_BEFORE_SLACK_SEC = 4 * 3600
# 生成に渡す、直近の出来事の件数（くり返しを避け、続きの話にできるように）
_RECENT_EVENTS = 10
_STATE_KEY = "town"
# 新しい出来事の文の2文字のうち、この割合以上が最近の出来事と同じなら、写しとみなして捨てる
_COPY_OVERLAP = 0.6

# 「最近どう？」「昨日何してた？」のような、暮らしを聞く言い方
_LIFE_RE = re.compile(
    r"最近|さいきん|何して|なにして|何かあ|なにかあ|どうしてた|どうだった|どう過ご|何があ|なにがあ"
    r"|昨日|きのう|おととい|けさ|今朝|今日は|町で|町の|どこ(?:か|に|へ)?行"
)
_QUESTION_RE = re.compile(r"[？?]|(?:か|の|かな|っけ)\s*$|何|なに|どう|どこ|だれ|誰|いつ")
_LIFE_RE_EN = re.compile(
    r"\b(lately|recently|these days|what have you been|what did you do|what were you|up to|anything (new|happen\w*|fun)"
    r"|how (was|were|have|has|are|is) (your|you|things|everyone|it going)|what happened|yesterday|this morning"
    r"|last night|today|in town|around town|did you go|been anywhere)\b",
    re.IGNORECASE,
)
_QUESTION_RE_EN = re.compile(
    r"\?|^\s*(what|how|where|who|when|why|did|do|does|are|is|was|were|have|has|can|could|would|will)\b", re.IGNORECASE
)


@dataclass
class TalkContext:
    """そのラウンドで、町の出来事をどこまで話してよいかを決める材料（brain が毎ラウンド作る）。"""

    idle: bool = False  # ユウが黙っている（雑談・来たときの挨拶）
    question: bool = False  # ユウの発言が質問
    life_question: bool = False  # 「最近どう？」「昨日何してた？」のような聞き方
    addressed: frozenset[str] = field(default_factory=frozenset)  # ユウが名指しした子

    @classmethod
    def from_user(cls, text: str, addressed: list[str]) -> TalkContext:
        question_re = language.pick(ja=_QUESTION_RE, en=_QUESTION_RE_EN)
        life_re = language.pick(ja=_LIFE_RE, en=_LIFE_RE_EN)
        q = bool(question_re.search(text))
        return cls(question=q, life_question=q and bool(life_re.search(text)), addressed=frozenset(addressed))


def _schema(place_ids: list[str], member_ids: list[str], moods: bool) -> dict:
    schema: dict = {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "day": {"type": "string"},
                        "time": {"type": "string", "enum": list(clock.bands())},
                        "place_id": {"type": "string", "enum": place_ids},
                        "members": {
                            "type": "array",
                            "items": {"type": "string", "enum": member_ids},
                            "minItems": 1,
                            "maxItems": 3,
                        },
                        "text": {"type": "string", "maxLength": language.pick(ja=90, en=220)},
                        "openness": {"type": "string", "enum": list(OPENNESS)},
                    },
                    "required": ["day", "time", "place_id", "members", "text", "openness"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["events"],
        "additionalProperties": False,
    }
    if moods:
        schema["properties"]["moods"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "member": {"type": "string", "enum": member_ids},
                    "label": {"type": "string", "maxLength": language.pick(ja=12, en=30)},
                    "text": {"type": "string", "maxLength": language.pick(ja=60, en=160)},
                    "valence": {"type": "number", "minimum": -2, "maximum": 2},
                },
                "required": ["member", "label", "text", "valence"],
                "additionalProperties": False,
            },
        }
        schema["required"] = ["moods", "events"]
    return schema


class Town:
    def __init__(
        self,
        config: TownConfig,
        cast: list[CastMember],
        store: Store,
        llm: LLMClient,
        user_ref: str,
        min_score: float,
        drama: Drama | None = None,
        on_update: Callable[[], None] | None = None,
    ):
        self._cfg = config
        self._cast = cast
        self._ids = [m.id for m in cast]
        self._names = {m.id: m.name for m in cast}
        self._store = store
        self._llm = llm
        self._uref = user_ref
        self._min_score = min_score
        # 今の関係を生成に渡し、作った今日の気分を返す相手。on_update は作り終えたあと（画面の更新用）
        self._drama = drama
        self._on_update = on_update
        self._places = {p.id: p for p in config.places}
        # キャラ id → 直近の発話のプロンプトに入れた、まだ話していない出来事（「話した」の判定用）
        self._offered: dict[str, list[dict]] = {}
        self._gen_lock = threading.Lock()
        # この起動で作った出来事の件数（6人目には件数だけを渡す）
        self.generated = 0

    # --- 共通部分（system） -------------------------------------------------------

    def system_text(self) -> str:
        u, name = self._uref, self._cfg.name
        places = "\n".join(f"- {p.name}: {p.desc}" if p.desc else f"- {p.name}" for p in self._cfg.places)
        if language.is_en():
            return (
                f'Everyone lives together in "{name}", a little town inside {u}\'s heart.\n'
                f"But they are not {u}. They have lives of their own, and all sorts of things happen every day that {u} doesn't know about.\n"
                f"They only know as much about {u} as {u} has told them. They never assume {u}'s feelings or speak for {u}.\n"
                f'{u} can\'t come into the town. It\'s fine to say "let\'s go together", but only the girls actually go.\n'
                'The only places in town are the ones listed under "# The town". Never invent other places or shops.\n\n'
                f"# The town ({name})\n{places}"
            )
        return (
            f"みんなは、{u}の心の中にある小さな町「{name}」で、いっしょに暮らしています。\n"
            f"でも{u}本人ではありません。自分の暮らしがあり、{u}の知らないところで毎日いろいろなことが起きています。\n"
            f"{u}のことは、{u}が話してくれた範囲しか知りません。{u}の気持ちを決めつけたり、代わりに言ったりしません。\n"
            f"{u}は町には来られません。「いっしょに行こう」と言うのはよいけれど、実際に行くのは自分たちだけです。\n"
            "町にあるのは「# 町のようす」に書いた場所だけです。ほかの場所や店を作りません。\n\n"
            f"# 町のようす（{name}）\n{places}"
        )

    # --- 出来事を先に作る（起動時、裏で） --------------------------------------------

    def generate_async(self) -> None:
        threading.Thread(target=self._generate_safe, name="town", daemon=True).start()

    def _generate_safe(self) -> None:
        try:
            self.generate()
        except Exception:  # noqa: BLE001 - 裏スレッドの失敗で会話を止めない
            logger.exception("町の出来事を作れなかった")

    def generate(self, now: dt.datetime | None = None) -> int:
        """前回作ったときから now までのあいだの出来事を作って保存する。戻り値は保存した件数。"""
        with self._gen_lock:
            return self._generate(now or dt.datetime.now())

    def _generate(self, now: dt.datetime) -> int:
        state = self._load_state()
        last = state.get("last_gen")
        start = dt.datetime.fromtimestamp(last) if last else now - dt.timedelta(days=1)
        start = max(start, now - dt.timedelta(days=_MAX_SPAN_DAYS))
        hours = (now - start).total_seconds() / 3600
        n = min(self._cfg.max_events_per_launch, round(self._cfg.events_per_day * hours / 24))
        drama = self._drama
        need_moods = drama is not None and drama.needs_moods() and drama.date == now.date().isoformat()
        if n < 1:
            if not need_moods:
                logger.info("町: 前回から %.1f 時間なので、新しい出来事は作らない", hours)
                return 0
            n = 1  # 日付が変わって気分を決めるついでに、1件だけ
        # ないしょの数は先に決めて頼む（全体の secret_ratio まで。端数は確率で）。頼まないとモデルはほとんど出さない
        quota = n * self._cfg.secret_ratio
        secrets = math.floor(quota) + (1 if random.random() < quota - math.floor(quota) else 0)
        recent = self._store.life_events(_RECENT_EVENTS)
        t0 = dt.datetime.now()
        try:
            raw = self._llm.chat(
                self._prompt(start, now, n, secrets, recent, need_moods),
                schema=_schema(list(self._places), self._ids, need_moods),
                max_tokens=language.pick(ja=160, en=200) * n + (60 * len(self._ids) if need_moods else 0) + 200,
                temperature=1.0,
            )
            data = json.loads(raw)
            events = data.get("events", [])
        except (LLMError, ValueError, AttributeError) as e:
            logger.warning("町: 出来事を作れなかった: %s", e)
            return 0
        if need_moods and drama is not None:
            moods = self._moods(data.get("moods"))
            if moods:
                drama.set_moods(moods)
        kept = self._validate(events, start, now, n, recent, secrets)
        for e in kept:
            self._store.add_life(e["members"], e["ts"], e["day"], e["place_id"], e["text"], e["openness"])
            logger.info(
                "町の出来事: %s %s（%s）[%s] %s",
                clock.when_text(e["ts"], now),
                self._places[e["place_id"]].name,
                "・".join(self._names[m] for m in e["members"]),
                e["openness"],
                e["text"],
            )
        logger.info(
            "町: %s〜今の出来事を %d/%d 件作った（%.0f秒）",
            clock.date_label(start.date()),
            len(kept),
            len(events),
            (dt.datetime.now() - t0).total_seconds(),
        )
        self.generated += len(kept)
        state["last_gen"] = now.timestamp()
        self._store.set_state(_STATE_KEY, json.dumps(state, ensure_ascii=False))
        if self._on_update is not None:
            self._on_update()
        return len(kept)

    def _moods(self, raw: object) -> dict:
        """作った今日の気分を Mood にする。名前が重なったり抜けたりした子は渡さない（「ふつう」のまま）。"""
        from .drama import Mood

        out: dict[str, Mood] = {}
        for m in raw if isinstance(raw, list) else []:
            if not isinstance(m, dict) or m.get("member") not in self._names or m["member"] in out:
                continue
            label = str(m.get("label", "")).strip()
            text = str(m.get("text", "")).strip()
            try:
                valence = max(-2.0, min(2.0, float(m.get("valence", 0))))
            except (TypeError, ValueError):
                continue
            if label and text:
                out[m["member"]] = Mood(label, text, valence)
        missing = [self._names[c] for c in self._ids if c not in out]
        if missing:
            logger.warning("町: 今日の気分が作れなかった子: %s", "・".join(missing))
        return out

    def _prompt(
        self, start: dt.datetime, now: dt.datetime, n: int, secrets: int, recent: list[dict], moods: bool
    ) -> list[dict]:
        if language.is_en():
            return self._prompt_en(start, now, n, secrets, recent, moods)
        u, name = self._uref, self._cfg.name
        places = "\n".join(f"- {p.id}（{p.name}）: {p.desc}" for p in self._cfg.places)
        girls = "\n".join(f"- {m.id}（{m.name}）: {m.desc}" for m in self._cast)
        past = "\n".join(
            f"- {clock.when_text(e['ts'], now)}、{self._place_name(e['place_id'])}で"
            f"（{'・'.join(self._names.get(m, m) for m in e['members'])}）: {e['text']}"
            for e in reversed(recent)
        )
        hint = self._drama.town_hint(self._ids) if self._drama is not None else ""
        span = f"{clock.date_label(start.date())}の{clock.band(start)}から、今（{clock.date_label(now.date())}の{clock.band(now)}）まで"
        system = (
            f"あなたは、ゆるい日常ものの4コマ漫画の脚本家です。{u}の心の中にある小さな町「{name}」で暮らす、"
            f"{len(self._cast)}人の女の子の、何気ない日常の出来事を考えます。"
        )
        content = (
            f"# 町の場所（これ以外の場所は使わない）\n{places}\n\n"
            f"# 女の子たち（members には id を書く）\n{girls}\n\n"
            + (f"# いまの関係\n{hint}\n\n" if hint else "")
            + (f"# 最近の出来事（同じ話はくり返さない。続きの話にするのはよい）\n{past}\n\n" if past else "")
            + "# 指示\n"
            f"{span}のあいだに、町で起きた出来事を{n}件考えてください。\n"
            f"- day は YYYY-MM-DD で、{start.date().isoformat()} から {now.date().isoformat()} まで。"
            f"time は {' / '.join(clock.BANDS)}。今日のぶんは、今（{now.hour}時ごろ）より前の時間帯だけ\n"
            "- members はその場にいた子（1〜3人）。出来事ごとに場所と顔ぶれを変え、同じ子ばかりにしない\n"
            "- text は何があったかを、過去形の1文・60字以内で。だれのことか分かるように名前で書く\n"
            "- 日常の小さなこと（ちょっとした失敗、発見、すれ違い、親切、ないしょの準備など）。"
            "その子の性格が出るものにする。大事件・けが・病気・不幸は入れない\n"
            f"- {u}は町に来られないので、出来事に{u}を出さない\n"
            "- 現実の店名・作品名・人名などの固有名詞、具体的な数字、ニュースは入れない\n"
            f"- openness はその子たちが{u}に話すかどうか。open = すぐ話したくなる / ask = 聞かれたら話す / "
            "secret = ないしょにしたい\n"
            + (
                f"- secret はちょうど{secrets}件。話すと一騒動になる、ちょっとした秘密にする"
                "（照れくさいこと、サプライズの準備、その場にいなかった子に関係すること＝だれかの物をこっそり使った・なくした、など）。"
                "members には、その場にいて秘密を知っている子だけを書く"
                if secrets
                else "- secret は使わない"
            )
            + (
                f"\n- moods は、今日（{clock.date_label(now.date())}）の{len(self._cast)}人全員の気分。"
                "label は短い言葉（8字以内）、text はその気分を1文で、valence は -2（落ちこみ）〜 +2（上機嫌）。"
                "性格と、きのうから今日の出来事から決める。全員を上機嫌にせず、少し元気がない子も1〜2人入れる。"
                "出来事の文には気分をそのまま書かない"
                if moods
                else ""
            )
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": content}]

    def _prompt_en(
        self, start: dt.datetime, now: dt.datetime, n: int, secrets: int, recent: list[dict], moods: bool
    ) -> list[dict]:
        u, name = self._uref, self._cfg.name
        places = "\n".join(f"- {p.id} ({p.name}): {p.desc}" for p in self._cfg.places)
        girls = "\n".join(f"- {m.id} ({m.name}): {m.desc}" for m in self._cast)
        past = "\n".join(
            f"- {clock.when_text(e['ts'], now)}, at {self._place_name(e['place_id'])} "
            f"({', '.join(self._names.get(m, m) for m in e['members'])}): {e['text']}"
            for e in reversed(recent)
        )
        hint = self._drama.town_hint(self._ids) if self._drama is not None else ""
        span = (
            f"from {clock.date_label(start.date())} ({clock.band(start)}) "
            f"until now ({clock.date_label(now.date())}, {clock.band(now)})"
        )
        system = (
            f"You are a writer for a cozy slice-of-life four-panel comic. You come up with small, everyday events "
            f'in the lives of {len(self._cast)} girls who live in "{name}", a little town inside {u}\'s heart.'
        )
        content = (
            f"# Places in town (use no other places)\n{places}\n\n"
            f"# The girls (write their ids in members)\n{girls}\n\n"
            + (f"# Current relationships\n{hint}\n\n" if hint else "")
            + (f"# Recent events (don't repeat them; continuing one is fine)\n{past}\n\n" if past else "")
            + "# Instructions\n"
            f"Come up with {n} events that happened in town {span}.\n"
            f"- day is YYYY-MM-DD, from {start.date().isoformat()} to {now.date().isoformat()}. "
            f"time is one of {' / '.join(clock.bands())}. For today, only time slots before now (around {now.hour}:00)\n"
            "- members are the girls who were there (1 to 3). Vary the place and the faces from event to event; "
            "don't keep using the same girls\n"
            "- text says what happened in one past-tense sentence, under 30 words. Use names so it's clear who it's about\n"
            "- Small everyday things (a little mistake, a discovery, a misunderstanding, a kindness, secretly "
            "preparing something, and so on) that show the girl's personality. No big incidents, injuries, "
            "illness or misfortune\n"
            f"- {u} can't come to town, so {u} never appears in the events\n"
            "- No real shop names, titles of works, people's names or other proper nouns, no specific numbers, no news\n"
            f"- openness is whether the girls would tell {u} about it: open = can't wait to tell / "
            "ask = will tell if asked / secret = wants to keep it secret\n"
            + (
                f"- Make exactly {secrets} of them secret: a small secret that would cause a bit of a stir if it came out "
                "(something embarrassing, preparing a surprise, something involving a girl who wasn't there, like "
                "secretly borrowing or losing someone's things). In members, list only the girls who were there "
                "and know the secret"
                if secrets
                else "- Don't use secret"
            )
            + (
                f"\n- moods is today's ({clock.date_label(now.date())}) mood for all {len(self._cast)} girls. "
                "label is a short word or two, text is one sentence describing the mood, valence is from -2 (down) "
                "to +2 (in high spirits). Decide from their personalities and what happened yesterday and today. "
                "Don't put everyone in high spirits; include one or two who are a little down. "
                "Don't write the moods straight into the event text"
                if moods
                else ""
            )
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": content}]

    def _validate(
        self, events: list, start: dt.datetime, now: dt.datetime, n: int, recent: list[dict], secrets: int
    ) -> list[dict]:
        """場所・顔ぶれ・日時がリストと期間に収まる出来事だけ残す。最近の出来事の写し・ないしょの数はコードで切る。"""
        seen = [_bigrams(e["text"]) for e in recent]
        kept: list[dict] = []
        for e in events:
            if not isinstance(e, dict):
                continue
            members = [m for m in dict.fromkeys(e.get("members") or []) if m in self._names]
            place = e.get("place_id")
            text = str(e.get("text", "")).strip()
            time = e.get("time")
            bands = clock.bands()
            if not members or len(members) > 3 or place not in self._places or not text or time not in bands:
                logger.info("町: 出来事を捨てた（場所か顔ぶれがリストに無い）: %s", e)
                continue
            try:
                day = dt.date.fromisoformat(str(e.get("day", "")))
            except ValueError:
                logger.info("町: 出来事を捨てた（日付が読めない）: %s", e)
                continue
            ts = dt.datetime.combine(day, dt.time(bands[time])).timestamp()
            if ts > now.timestamp() or ts < start.timestamp() - _BEFORE_SLACK_SEC:
                logger.info("町: 出来事を捨てた（期間の外: %s %s）: %s", day, time, text)
                continue
            words = _bigrams(text)
            if any(words and len(words & s) / len(words) >= _COPY_OVERLAP for s in seen):
                logger.info("町: 出来事を捨てた（最近の出来事とほぼ同じ）: %s", text)
                continue
            seen.append(words)
            openness = e.get("openness") if e.get("openness") in OPENNESS else "open"
            kept.append(
                {"members": members, "place_id": place, "text": text, "day": day.isoformat(), "ts": ts, "openness": openness}
            )
        kept = sorted(kept, key=lambda e: e["ts"])[:n]
        # ないしょは頼んだ数まで。超えたぶんは「聞かれたら話す」にする
        allowed = secrets
        for e in kept:
            if e["openness"] == "secret":
                if allowed > 0:
                    allowed -= 1
                else:
                    e["openness"] = "ask"
        return kept

    # --- 発話のプロンプトに入れる ------------------------------------------------------

    def life_text(self, member: CastMember, context: str, talk: TalkContext) -> str:
        """キャラ固有部分に入れる「# 町でのできごと」（入れるものが無ければ空）。"""
        now = dt.datetime.now()
        rows = self._store.life_of(member.id, 60)
        q = _bigrams(context)
        fresh = now.timestamp() - self._cfg.fresh_days * 86400
        named = member.id in talk.addressed and (talk.question or talk.life_question)
        picks: list[tuple[float, dict]] = []
        for r in rows:
            if r["told"] or r["ts"] < fresh:
                continue
            s = _score(q, self._event_words(r))
            near = s >= self._min_score
            o = r["openness"]
            if o == "secret":
                ok = named
                prio = 3.0
            elif o == "ask":
                ok = talk.life_question or (talk.question and near)
                prio = 2.0 if talk.life_question else 1.0
            else:
                ok = near or talk.life_question
                prio = 2.0 if talk.life_question else 1.0
            if ok:
                picks.append((prio + s, r))
        picks.sort(key=lambda x: (x[0], x[1]["ts"]), reverse=True)
        chosen = [r for _, r in picks[: self._cfg.life_in_prompt]]
        if not chosen and talk.idle:
            # ユウが黙っている間は、話題に関係なく、いちばん新しい「すぐ話したい」出来事を1件
            chosen = [r for r in rows if not r["told"] and r["ts"] >= fresh and r["openness"] == "open"][:1]
        self._offered[member.id] = chosen
        told = sorted(
            ((_score(q, self._event_words(r)), r) for r in rows if r["told"]), key=lambda x: x[0], reverse=True
        )
        told = [r for s, r in told[:1] if s >= self._min_score]
        if not chosen and not told:
            return ""
        u = self._uref
        out: list[str] = []
        if language.is_en():
            if chosen:
                out.append(f"# Things that happened in town (you were there; {u} doesn't know yet)")
                out.append(
                    "Only bring them up, in your own words, when they fit the conversation. You don't have to. "
                    "Don't tell everything at once."
                )
                for r in sorted(chosen, key=lambda r: r["ts"]):
                    line = f"- {self._line(r, member, now)}"
                    if r["openness"] == "secret":
                        line += f" (you wanted to keep this secret; {u} asked, so tell it bit by bit, reluctantly)"
                    elif r["openness"] == "ask":
                        line += " (you won't bring it up yourself, but you'll tell if asked)"
                    out.append(line)
            if told:
                out.append(
                    f'# Things from town you already told {u} about (if you repeat one, say "like I told you before")'
                )
                out.extend(f"- {self._line(r, member, now)}" for r in told)
            return "\n".join(out) + "\n"
        if chosen:
            out.append(f"# 町でのできごと（あなたが体験したこと。{u}はまだ知らない）")
            out.append("話の流れに合うときだけ、自分の言葉で話す。無理に話さなくてよい。一度に全部は話さない。")
            for r in sorted(chosen, key=lambda r: r["ts"]):
                line = f"- {self._line(r, member, now)}"
                if r["openness"] == "secret":
                    line += f"（ないしょにしておきたかったこと。{u}に聞かれたので、言いにくそうに少しずつ話す）"
                elif r["openness"] == "ask":
                    line += "（自分からは言わないけれど、聞かれたら話す）"
                out.append(line)
        if told:
            out.append(f"# 前に{u}に話した、町でのできごと（もう話したので、くり返すなら「前に話したけど」と言う）")
            out.extend(f"- {self._line(r, member, now)}" for r in told)
        return "\n".join(out) + "\n"

    def _line(self, r: dict, member: CastMember, now: dt.datetime) -> str:
        others = [self._names[m] for m in (r.get("members") or "").split(",") if m and m != member.id and m in self._names]
        if language.is_en():
            who = f"(with {' and '.join(others)})" if others else "(on your own)"
            return f"{clock.when_text(r['ts'], now)}, at {self._place_name(r['place_id'])} {who}: {r['text']}"
        who = f"（{'・'.join(others)}といっしょに）" if others else "（ひとりで）"
        return f"{clock.when_text(r['ts'], now)}、{self._place_name(r['place_id'])}で{who}: {r['text']}"

    def _event_words(self, r: dict) -> str:
        return f"{self._place_name(r['place_id'])} {r['text']}"

    def _place_name(self, pid: str | None) -> str:
        p = self._places.get(pid or "")
        return p.name if p else language.pick(ja="町", en="town")

    # --- 演出の材料（drama.py から） ------------------------------------------------

    def untold(self, openness: tuple[str, ...]) -> list[dict]:
        """まだユウに話していない、新しめの出来事（出来事ごと。members はその場にいた全員）。"""
        fresh = dt.datetime.now().timestamp() - self._cfg.fresh_days * 86400
        return [
            e
            for e in self._store.life_events(60)
            if not e["told"]
            and e["ts"] >= fresh
            and e["openness"] in openness
            and e["members"]
            and all(m in self._names for m in e["members"])
        ]

    def describe(self, e: dict) -> str:
        """「ゆうべ、商店街で: …」"""
        if language.is_en():
            return f"{clock.when_text(e['ts'])}, at {self._place_name(e['place_id'])}: {e['text']}"
        return f"{clock.when_text(e['ts'])}、{self._place_name(e['place_id'])}で: {e['text']}"

    def mark_told(self, event_id: int) -> None:
        self._store.set_told(event_id)
        for rows in list(self._offered.values()):
            for r in rows:
                if r["event_id"] == event_id:
                    r["told"] = 1

    # --- 話したかの判定 ----------------------------------------------------------

    def on_spoken(self, member_id: str, text: str) -> None:
        """発話のあと。プロンプトに入れた出来事をセリフがなぞっていたら「ユウに話した」にする（全員ぶん）。"""
        said = _bigrams(text)
        if not said:
            return
        for r in self._offered.get(member_id, []):
            if r["told"]:
                continue
            # 出来事の文は「アカリが〜」と三人称なので、本人の名前は数えない
            words = _bigrams(r["text"].replace(self._names[member_id], ""))
            if not words:
                continue
            overlap = len(words & said) / len(words)
            if overlap >= 0.2:
                logger.info("町: %s のセリフと出来事の重なり %.2f: %s", self._names[member_id], overlap, r["text"])
            if overlap >= self._cfg.told_overlap:
                r["told"] = 1
                self._store.set_told(r["event_id"])
                logger.info("町: %s が出来事を話した: %s", self._names[member_id], r["text"])

    # --- 状態 ------------------------------------------------------------------

    def _load_state(self) -> dict:
        raw = self._store.get_state(_STATE_KEY)
        try:
            return json.loads(raw) if raw else {}
        except ValueError:
            return {}
