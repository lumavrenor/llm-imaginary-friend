"""喜怒哀楽と、会話を盛り上げる仕掛け。

- 感情: セリフの文頭に付けさせる【喜】などのタグ。声（VOICEVOX のスタイルと抑揚）と 3D の表情に使う
- 今日の気分: 町の出来事といっしょに、LLM がその日の各キャラの気分を作る（town.py が set_moods で渡す）。
  決まった表は持たない。決まるまでと、町を使わない設定では「ふつう」。同じ日なら、再起動しても同じ気分（DB に保存）
- 小ネタ: 「打ち明け話」「ないしょ話の発覚」の材料は、町でのまだユウに話していない出来事（town.py）。
  ないしょの出来事が、その場にいなかった子に関係する話なら、その子が相手役になり、すれ違い・仲直りの種になる
- 関係と気分の変化: 発言の感情と、誰に向けた発言かから、好感度と気分を少しずつ動かす。
  怒りが続いて好感度が基準より下がると「気まずい」状態になり、仲直りするまで残る。DB に保存し、日をまたぐと基準へ半分戻る
- 演出（ビート）: 話し込みを起承転結に見立て、「転」で口ゲンカや打ち明け話、「結」で仲直りや共感を1回ずつ仕掛ける。
  演出の指示は、その役のキャラのプロンプトにだけ入れる（ほかのキャラには、ないしょ話が漏れない）
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import random
import re
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import language

if TYPE_CHECKING:
    from .config import CastMember, DramaConfig
    from .store import Store
    from .town import Town

logger = logging.getLogger(__name__)

# 感情の内部の名前は、英語版でも日本語の1字のまま（声・表情・好感度の表がこれを引く）。
# モデルに見せる名前だけを言語で変える（emotion_label）
NEUTRAL = "普通"
EMOTIONS = (NEUTRAL, "喜", "楽", "怒", "哀", "驚", "照")
# 英語版でモデルに見せる名前（タグ [happy]、挙手の emotion、会話ログの「Akari (angry):」）
EMOTION_EN = {
    NEUTRAL: "neutral", "喜": "happy", "楽": "cheerful", "怒": "angry",
    "哀": "sad", "驚": "surprised", "照": "shy",
}
# モデルがタグに書きがちな別の言い方
_ALIASES = {
    "笑": "喜", "嬉": "喜", "うれ": "喜", "喜び": "喜", "わくわく": "楽", "楽しい": "楽", "穏": "楽",
    "泣": "哀", "悲": "哀", "寂": "哀", "淋": "哀", "落": "哀", "しょんぼり": "哀", "涙": "哀",
    "苛": "怒", "怒り": "怒", "ムッ": "怒", "むっ": "怒", "不満": "怒",
    "驚き": "驚", "びっくり": "驚", "焦": "驚", "怖": "驚", "不安": "哀",
    "恥": "照", "照れ": "照", "てれ": "照",
    "普": NEUTRAL, "平": NEUTRAL, "ふつう": NEUTRAL, "通常": NEUTRAL,
}
# 英語のタグ（小文字で前方一致）。(laughs) のようなト書きも、感情として読めれば外す
_ALIASES_EN = {
    "happy": "喜", "glad": "喜", "joy": "喜", "laugh": "喜", "giggl": "喜", "delight": "喜", "thrill": "喜",
    "cheer": "楽", "fun": "楽", "playful": "楽", "relax": "楽", "content": "楽", "excit": "楽", "amused": "楽",
    "calm": "楽", "smil": "楽",
    "angry": "怒", "anger": "怒", "annoy": "怒", "irrita": "怒", "mad": "怒", "grump": "怒", "huff": "怒",
    "frustrat": "怒", "pout": "怒", "indignant": "怒",
    "sad": "哀", "upset": "哀", "tear": "哀", "cry": "哀", "sob": "哀", "lonely": "哀", "gloom": "哀",
    "worri": "哀", "anxious": "哀", "sniff": "哀", "sigh": "哀", "down": "哀", "hurt": "哀",
    "surpris": "驚", "shock": "驚", "amaz": "驚", "startl": "驚", "gasp": "驚", "scared": "驚",
    "nervous": "驚", "panic": "驚",
    "shy": "照", "embarrass": "照", "fluster": "照", "blush": "照", "bashful": "照", "sheepish": "照",
    "neutral": NEUTRAL, "normal": NEUTRAL, "plain": NEUTRAL, "flat": NEUTRAL,
}
# 【喜】（笑）[怒] など。【】は感情でなくても外す（読み上げない）。( ) は感情のときだけ外す
_TAG_RE = re.compile(r"[【\[［(（]\s*([^】\]］)）]{1,6}?)\s*[】\]］)）]")
# 英語版は [surprised] のように語が長いので、長めまで読む
_TAG_RE_EN = re.compile(r"[【\[［(（]\s*([^】\]］)）]{1,14}?)\s*[】\]］)）]")


def emotion_label(emotion: str) -> str:
    """モデルに見せる感情の名前（日本語版は内部の名前そのまま）。"""
    return language.pick(ja=emotion, en=EMOTION_EN.get(emotion, emotion))


def emotion_rule() -> str:
    return language.pick(
        ja=(
            "セリフの最初に、そのときの気持ちを【喜】【楽】【怒】【哀】【驚】【照】【普通】のどれか1つで付ける。"
            "途中で気持ちが変わったら、その文の頭にも付ける。"
            "笑うときは「あはは」「ふふっ」、泣きそうなときは「うぅ」「ぐすっ」のように、声に出して表してよい。"
        ),
        en=(
            "Start your line with exactly one feeling tag: [happy] [cheerful] [angry] [sad] [surprised] [shy] [neutral]. "
            "If your feeling changes partway through, put a tag at the start of that sentence too. "
            "The tags are not read aloud. To laugh or cry, say it out loud as sounds "
            '("ahaha", "hehe", "uuu", "sniff"), never as stage directions like (laughs) or *giggles*.'
        ),
    )


# 互換用（日本語版の文）。英語版では emotion_rule() を使う
EMOTION_RULE = emotion_rule()


def parse_emotion(label: str) -> str | None:
    label = label.strip()
    if label in EMOTIONS:
        return label
    if language.is_en():
        low = label.lower().lstrip("*")
        for k, v in _ALIASES_EN.items():
            if low.startswith(k):
                return v
    for k, v in _ALIASES.items():
        if label.startswith(k):
            return v
    if label and label[0] in EMOTIONS:
        return label[0]
    return None


def take_tags(text: str) -> tuple[str | None, str]:
    """文から気持ちのタグを外す。戻り値は (最初のタグの感情か None, タグを外した文)。"""
    found: str | None = None

    def repl(m: re.Match) -> str:
        nonlocal found
        emo = parse_emotion(m.group(1))
        if emo is None:
            return "" if m.group(0)[0] == "【" else m.group(0)
        if found is None:
            found = emo
        return ""

    rest = language.pick(ja=_TAG_RE, en=_TAG_RE_EN).sub(repl, text)  # found は sub の中で決まる
    return found, rest


# --- 今日の気分とできごと -----------------------------------------------------


@dataclass(frozen=True)
class Mood:
    label: str
    text: str
    valence: float  # -2（落ちこみ）〜 +2（上機嫌）


# その日の気分がまだ決まっていないとき（町の出来事を作る前・町を使わない設定）
_PLAIN_JA = Mood("ふつう", "いつもどおりの気分", 0.0)
_PLAIN_EN = Mood("so-so", "feeling the same as always", 0.0)


def plain_mood() -> Mood:
    return language.pick(ja=_PLAIN_JA, en=_PLAIN_EN)


PLAIN_MOOD = _PLAIN_JA  # 互換用（日本語版）
# 3D の表情で「眠そうな顔」にする気分の言い方（display/expression.py は「ねむい」で判定する）
_SLEEPY_RE = re.compile(r"ねむ|眠|寝不足|うとうと|sleep|drows|tired|yawn|groggy|dozy", re.IGNORECASE)


# --- 演出 --------------------------------------------------------------------


@dataclass
class Beat:
    kind: str
    lead: str
    partner: str | None
    cues: dict[str, str]  # キャラ id → その人にだけ渡す演出の指示
    others: str = ""  # 先頭2人以外に渡す演出
    min_speakers: int = 2
    label: str = ""
    event_id: int | None = None  # 材料にした町の出来事（打ち明けたら「ユウに話した」にする）
    strained: bool = False  # ないしょ話が、その場にいなかった相手に関係する（打ち明けると気まずくなる）

    @property
    def order(self) -> list[str]:
        return [self.lead] + ([self.partner] if self.partner else [])

    def cue_for(self, cid: str) -> str:
        return self.cues.get(cid, self.others)


@dataclass
class _Arc:
    turn_at: float
    resolve_at: float
    turned: bool = False
    resolved: bool = False


# 感情 → (話した本人の気分, 本人→相手の好感度, 相手の気分, 相手→本人の好感度)
_EFFECTS = {
    "喜": (0.25, 0.2, 0.1, 0.1),
    "楽": (0.2, 0.15, 0.1, 0.05),
    "照": (0.15, 0.15, 0.05, 0.1),
    "怒": (-0.3, -0.35, -0.2, -0.2),
    "哀": (-0.35, -0.1, -0.1, 0.0),
    "驚": (0.0, 0.0, 0.0, 0.0),
    NEUTRAL: (0.0, 0.0, 0.0, 0.0),
}
_FEELINGS = {
    "喜": "{t}とのやりとりがうれしかった",
    "楽": "{t}と話すのが楽しかった",
    "照": "{t}の前で照れてしまった",
    "怒": "{t}にムッとした",
    "哀": "{t}の言葉に、しゅんとした",
    "驚": "{t}の話にびっくりした",
}
_FEELINGS_EN = {
    "喜": "you were happy about the exchange with {t}",
    "楽": "you had fun talking with {t}",
    "照": "you got flustered in front of {t}",
    "怒": "you got annoyed at {t}",
    "哀": "you felt a little hurt by what {t} said",
    "驚": "you were surprised by what {t} said",
}
_TENSION_GAP = 0.6  # 好感度が基準からこれ以上下がったら「気まずい」


def _mood_word(v: float) -> str:
    if v >= 1.5:
        return language.pick(ja="とてもごきげん", en="in a really great mood")
    if v >= 0.6:
        return language.pick(ja="ごきげん", en="in a good mood")
    if v > -0.6:
        return language.pick(ja="ふつう", en="feeling okay")
    if v > -1.4:
        return language.pick(ja="ちょっと元気がない", en="a little down")
    return language.pick(ja="かなり落ちこんでいる", en="pretty down")


def affinity_word(v: float) -> str:
    if v <= -1.5:
        return language.pick(
            ja="ケンカ友だちで、すぐ言い合いになる（でも本当は嫌いじゃない）",
            en="a sparring partner; you bicker at the drop of a hat (but you don't really dislike her)",
        )
    if v <= -0.5:
        return language.pick(ja="ちょっとそりが合わない", en="someone you don't quite click with")
    if v >= 1.5:
        return language.pick(ja="大の仲良し", en="your best friend")
    if v >= 0.5:
        return language.pick(ja="気が合う", en="someone you get along with")
    return ""


_affinity_word = affinity_word


class Drama:
    def __init__(
        self,
        cast: list[CastMember],
        base_affinity: dict[str, dict[str, int]],
        store: Store,
        config: DramaConfig,
        today: dt.date | None = None,
    ):
        self._cfg = config
        self._store = store
        self._ids = [m.id for m in cast]
        self._names = {m.id: m.name for m in cast}
        self._base = {
            a: {b: float(base_affinity.get(a, {}).get(b, 0)) for b in self._ids if b != a} for a in self._ids
        }
        self._date = (today or dt.date.today()).isoformat()
        # 今日の気分。町の出来事といっしょに LLM が作る（town.py が set_moods で渡す）。決まるまでは「ふつう」
        plain = plain_mood()
        self.moods: dict[str, Mood] = {cid: plain for cid in self._ids}
        self._moods_set = False
        # 打ち明け話・ないしょ話の材料（町の出来事）。brain が Town を作ったあとで入れる
        self.town: Town | None = None
        self.aff = {a: dict(row) for a, row in self._base.items()}
        self.mood = {cid: plain.valence for cid in self._ids}
        self.feelings: dict[str, list[str]] = {cid: [] for cid in self._ids}
        # 気まずい2人: [怒った側, 相手, 始まった日付]
        self.tension: list[list[str]] = []
        self._arc: _Arc | None = None
        self._last_quarrel: frozenset[str] | None = None  # 直前の口ゲンカの2人（続けて同じ組にしない）
        # 町の出来事を作るスレッドから set_moods が呼ばれるので、状態を動かすところはこれで守る
        self._lock = threading.RLock()
        self._load()

    # --- 今日の気分 ------------------------------------------------------------

    @property
    def date(self) -> str:
        return self._date

    def needs_moods(self) -> bool:
        """今日の気分がまだ決まっていないか。"""
        return not self._moods_set

    def set_moods(self, moods: dict[str, Mood]) -> None:
        """今日の気分を決める（town.py から）。それまでの会話で動いたぶんは残す。"""
        with self._lock:
            for cid, m in moods.items():
                if cid in self.moods:
                    self.mood[cid] = max(-2.0, min(2.0, self.mood[cid] + m.valence - self.moods[cid].valence))
                    self.moods[cid] = m
            self._moods_set = True
            self._save()
        logger.info("%s", self.mood_summary())

    def set_own_mood(self, cid: str, mood: Mood) -> None:
        """町の出来事に出ない子（6人目）の気分を、コードから決める（meta.py から）。"""
        with self._lock:
            if cid not in self.moods:
                return
            self.mood[cid] = max(-2.0, min(2.0, self.mood[cid] + mood.valence - self.moods[cid].valence))
            self.moods[cid] = mood
            self._save()
        logger.info("今日の気分: %s=%s", self._names[cid], mood.label)

    def mood_summary(self) -> str:
        return language.pick(ja="今日の気分: ", en="Today's moods: ") + " / ".join(
            f"{self._names[c]}={self.moods[c].label}" for c in self._ids
        )

    def moods_for_ui(self) -> dict[str, tuple[float, str]]:
        # 3D の表情は「ねむい」で眠そうな顔にするので、作った気分の「寝不足」「眠たい」などもそろえる
        return {
            c: (self.mood[c], "ねむい" if _SLEEPY_RE.search(self.moods[c].label) else self.moods[c].label)
            for c in self._ids
        }

    # --- 保存 ------------------------------------------------------------------

    def _load(self) -> None:
        raw = self._store.get_state("drama")
        if not raw:
            self._save()
            return
        try:
            data = json.loads(raw)
            aff = data.get("affinity", {})
            same_day = data.get("date") == self._date
            for a, row in aff.items():
                for b, v in row.items():
                    if a in self.aff and b in self.aff[a]:
                        base = self._base[a][b]
                        # 日をまたいだら基準へ半分戻す（昨日のケンカは少しだけ引きずる）
                        self.aff[a][b] = float(v) if same_day else base + (float(v) - base) * 0.5
            self.tension = [
                t for t in data.get("tension", []) if len(t) == 3 and t[0] in self.aff and t[1] in self.aff
            ]
            if same_day:
                for c, m in (data.get("moods") or {}).items():
                    if c in self.moods and isinstance(m, list) and len(m) == 3:
                        self.moods[c] = Mood(str(m[0]), str(m[1]), float(m[2]))
                        self._moods_set = True
                for c, v in data.get("mood", {}).items():
                    if c in self.mood:
                        self.mood[c] = float(v)
                for c, fs in data.get("feelings", {}).items():
                    if c in self.feelings:
                        self.feelings[c] = list(fs)[-3:]
            else:
                self._settle_tension()
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning("drama の状態を読めなかったので初期化する: %s", e)
        self._save()

    def _save(self) -> None:
        with self._lock:
            data = {
                "date": self._date,
                "affinity": self.aff,
                "moods": {c: [m.label, m.text, m.valence] for c, m in self.moods.items()} if self._moods_set else {},
                "mood": self.mood,
                "feelings": self.feelings,
                "tension": self.tension,
            }
            self._store.set_state("drama", json.dumps(data, ensure_ascii=False))

    # --- 関係 ------------------------------------------------------------------

    def affinity(self, a: str, b: str) -> float:
        return self.aff.get(a, {}).get(b, 0.0)

    def _tense(self, a: str, b: str) -> list[str] | None:
        return next((t for t in self.tension if {t[0], t[1]} == {a, b}), None)

    def _add_tension(self, a: str, b: str) -> None:
        if self._tense(a, b) is None:
            self.tension.append([a, b, self._date])
            logger.info("気まずい: %s と %s", self._names[a], self._names[b])

    def _resolve_tension(self, a: str, b: str) -> None:
        t = self._tense(a, b)
        if t is not None:
            self.tension.remove(t)
            logger.info("仲直り: %s と %s", self._names[a], self._names[b])

    def _settle_tension(self) -> None:
        """好感度が基準近くまで戻った2人は、気まずさを解く。"""
        for a, b, _ in list(self.tension):
            if self.aff[a][b] >= self._base[a][b] - 0.15 and self.aff[b][a] >= self._base[b][a] - 0.15:
                self._resolve_tension(a, b)

    def _bump(self, a: str, b: str, d: float) -> None:
        if a in self.aff and b in self.aff[a]:
            self.aff[a][b] = max(-3.0, min(3.0, self.aff[a][b] + d))

    def _cheer(self, c: str, d: float) -> None:
        if c in self.mood:
            self.mood[c] = max(-2.0, min(2.0, self.mood[c] + d))

    def target_of(self, speaker: str, text: str, prev: str | None) -> str | None:
        """発言が誰に向いているか。名前を呼んでいればその子、なければ直前に話した子。"""
        for cid in self._ids:
            if cid == speaker:
                continue
            name = re.escape(self._names[cid])
            pattern = language.pick(ja=name + r"(?![ァ-ヶー])", en=rf"\b{name}\b")
            if re.search(pattern, text, re.IGNORECASE if language.is_en() else 0):
                return cid
        return prev if prev in self._names and prev != speaker else None

    def on_utterance(self, speaker: str, emotion: str, target: str | None) -> None:
        """1人ぶんの発言のあとに、気分と好感度を動かす。"""
        if speaker not in self.mood:
            return
        with self._lock:
            self._on_utterance(speaker, emotion, target)

    def _on_utterance(self, speaker: str, emotion: str, target: str | None) -> None:
        for c in self._ids:
            # 気分は少しずつ今日の基準へ戻る
            self.mood[c] += (self.moods[c].valence - self.mood[c]) * 0.04
        me, to, them, back = _EFFECTS.get(emotion, _EFFECTS[NEUTRAL])
        self._cheer(speaker, me)
        if target is not None:
            self._bump(speaker, target, to)
            self._cheer(target, them)
            self._bump(target, speaker, back)
            feelings = language.pick(ja=_FEELINGS, en=_FEELINGS_EN)
            if emotion in feelings:
                self._feel(speaker, feelings[emotion].format(t=self._names[target]))
            if emotion == "怒" and self.aff[speaker][target] < self._base[speaker][target] - _TENSION_GAP:
                self._add_tension(speaker, target)
        self._settle_tension()
        self._save()

    def _feel(self, cid: str, text: str) -> None:
        fs = self.feelings[cid]
        if not fs or fs[-1] != text:
            fs.append(text)
            del fs[:-3]

    # --- プロンプト ------------------------------------------------------------

    def relations_text(self, cid: str) -> str:
        parts = []
        for other in self._ids:
            if other == cid:
                continue
            word = _affinity_word(self._base[cid][other])
            if word:
                parts.append(language.pick(ja=f"{self._names[other]}とは{word}", en=f"{self._names[other]} is {word}"))
        return language.pick(ja="、", en="; ").join(parts)

    def town_hint(self, ids: list[str] | None = None) -> str:
        """町の出来事を作るときに渡す、今の関係（と、もう決まっていれば今日の気分）。town.py から。

        ids: 町で暮らす子だけ（6人目は町の出来事に出ないので入れない）。
        """
        ids = [c for c in (ids or self._ids) if c in self.aff]
        en = language.is_en()
        n = self._names
        lines = []
        if self._moods_set:
            moods = " / ".join(f"{n[c]}={self.moods[c].label}" for c in ids)
            lines.append(f"Today's moods: {moods} (already decided)" if en else f"今日の気分: {moods}（もう決まっている）")
        pairs = []
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                v = (self.aff[a][b] + self.aff[b][a]) / 2
                if v >= 1.0:
                    pairs.append(f"{n[a]} and {n[b]} are close" if en else f"{n[a]}と{n[b]}は仲良し")
                elif v <= -1.0:
                    pairs.append(f"{n[a]} and {n[b]} often bicker" if en else f"{n[a]}と{n[b]}はよく言い合いになる")
        if pairs:
            lines.append(("Relationships: " + "; ".join(pairs)) if en else ("関係: " + "、".join(pairs)))
        for a, b, _since in self.tension:
            if a not in ids or b not in ids:
                continue
            lines.append(
                f"{n[a]} and {n[b]} had a fight in conversation and things are still awkward between them "
                "(if they appear in the same event, make it a chance to make up, or show they're still a bit stiff)"
                if en
                else f"{n[a]}と{n[b]}は、会話でケンカしたまま気まずい"
                "（同じ出来事に出すなら、仲直りのきっかけか、まだぎこちない様子にする）"
            )
        return "\n".join(lines)

    def feeling_text(self, cid: str) -> str:
        """キャラ固有部分に入れる「今の気持ち」。"""
        if language.is_en():
            return self._feeling_text_en(cid)
        # 今日のできごと（まだ話していない小ネタ）は、町の出来事として town.py が別の見出しで入れる
        lines = [f"今日の気分: {self.moods[cid].text.rstrip('。')}。今は{_mood_word(self.mood[cid])}。"]
        if self.feelings[cid]:
            lines.append("さっきから: " + "、".join(self.feelings[cid]) + "。")
        for a, b, since in self.tension:
            if cid in (a, b):
                other = self._names[b if cid == a else a]
                when = "さっき" if since == self._date else "前"
                lines.append(f"{when}の{other}とのことで、今ちょっと気まずい。")
        for other in self._ids:
            if other == cid:
                continue
            delta = self.aff[cid][other] - self._base[cid][other]
            if delta >= 0.8:
                lines.append(f"今日は{self._names[other]}のことが、いつもよりもっと好き。")
        return "# あなたの今の気持ち\n" + "\n".join(lines) + "\n"

    def _feeling_text_en(self, cid: str) -> str:
        lines = [f"Today's mood: {self.moods[cid].text.rstrip('.')}. Right now you're {_mood_word(self.mood[cid])}."]
        if self.feelings[cid]:
            lines.append("Earlier: " + "; ".join(self.feelings[cid]) + ".")
        for a, b, since in self.tension:
            if cid in (a, b):
                other = self._names[b if cid == a else a]
                when = "earlier today" if since == self._date else "a while ago"
                lines.append(f"Things are a bit awkward with {other} after what happened {when}.")
        for other in self._ids:
            if other == cid:
                continue
            delta = self.aff[cid][other] - self._base[cid][other]
            if delta >= 0.8:
                lines.append(f"Today you like {self._names[other]} even more than usual.")
        return "# How you feel right now\n" + "\n".join(lines) + "\n"

    # --- 演出 ------------------------------------------------------------------

    def start_session(self) -> None:
        if not self._cfg.beats:
            self._arc = None
            return
        self._arc = _Arc(turn_at=random.uniform(0.3, 0.5), resolve_at=random.uniform(0.65, 0.85))

    def beat_for(self, trigger: str, progress: float) -> Beat | None:
        """このラウンドで仕掛ける演出（無ければ None）。progress は話し込みの進み具合 0〜1。"""
        if not self._cfg.beats or len(self._ids) < 2:
            return None
        if trigger == "idle":
            if random.random() >= self._cfg.idle_beat_chance:
                return None
            return self._pick(
                [
                    (3.0 if self.tension else 0.0, self._makeup),
                    (2.0, self._secret),
                    (2.0, self._confide),
                    (1.0, self._boke),
                    (1.0, self._praise),
                    (1.0, self._tease),
                ]
            )
        if trigger != "talk" or self._arc is None:
            return None
        arc = self._arc
        if not arc.turned and progress >= arc.turn_at:
            arc.turned = True
            irritable = any(self.mood[c] < -0.6 for c in self._ids)
            return self._pick(
                [
                    (3.0, self._secret),
                    (2.0 + (1.0 if irritable else 0.0), self._quarrel),
                    (2.0, self._confide),
                    (1.0, self._boke),
                ]
            )
        if arc.turned and not arc.resolved and progress >= arc.resolve_at:
            arc.resolved = True
            return self._pick(
                [
                    (6.0 if self.tension else 0.0, self._makeup),
                    (2.0, self._empathy),
                    (1.0, self._laugh),
                ]
            )
        if random.random() < self._cfg.light_beat_chance:
            return self._pick([(1.0, self._boke), (1.0, self._praise), (1.0, self._tease)])
        return None

    def _pick(self, options: list[tuple[float, object]]) -> Beat | None:
        """重みつきで演出を選ぶ。その演出が作れなければ（ネタ切れなど）ほかを試す。"""
        options = [(w, f) for w, f in options if w > 0]
        while options:
            w, f = random.choices(options, weights=[w for w, _ in options])[0]
            options = [o for o in options if o[1] is not f]
            beat = f()  # type: ignore[operator]
            if beat is not None:
                logger.info("演出: %s", beat.label)
                return beat
        return None

    def _n(self, cid: str) -> str:
        return self._names[cid]

    def _friend_of(self, cid: str, exclude: tuple[str, ...] = ()) -> str:
        cands = [c for c in self._ids if c != cid and c not in exclude]
        return max(cands, key=lambda c: self.aff[c][cid] + random.uniform(0, 1.0))

    def _quarrel(self) -> Beat | None:
        pairs = [(a, b) for i, a in enumerate(self._ids) for b in self._ids[i + 1 :]]
        # 仲の悪い組ほど起きやすいが、いちばん悪い組だけにはしない。口ゲンカで好感度が下がると
        # その組がますます選ばれて、毎回同じ2人のケンカになるため（直前と同じ組も避ける）
        ranked = sorted(
            (p for p in pairs if frozenset(p) != self._last_quarrel),
            key=lambda p: self.aff[p[0]][p[1]] + self.aff[p[1]][p[0]] + random.uniform(0, 1.5),
        ) or pairs
        a, b = random.choices(ranked[:3], weights=(3, 2, 1)[: len(ranked[:3])])[0]
        self._last_quarrel = frozenset((a, b))
        if self.mood[b] < self.mood[a]:
            a, b = b, a  # 機嫌の悪いほうが突っかかる
        na, nb = self._n(a), self._n(b)
        if language.is_en():
            return Beat(
                "quarrel",
                a,
                b,
                {
                    a: f"Something about the way {nb} just said that, or one of {nb}'s usual habits, rubs you the wrong way "
                    "and you snap at her (a slice-of-life comic squabble, nothing truly mean). Your feeling is [angry].",
                    b: f"{na} just snapped at you. Don't just take it: talk back, or get a little hurt, in your own way.",
                },
                others=f"{na} and {nb} are squabbling. In your own way, step in to stop it, joke about it, or take a side.",
                min_speakers=3,
                label=f"Squabble ({na} -> {nb})",
            )
        return Beat(
            "quarrel",
            a,
            b,
            {
                a: f"{nb}のさっきの言い方か、{nb}のいつものクセに、ついムッとして突っかかる"
                "（日常ものの4コマ漫画の口ゲンカくらい。本気の悪口は言わない）。気持ちは【怒】。",
                b: f"{na}に突っかかられた。言われっぱなしにせず言い返すか、しゅんとするか、自分らしく反応する。",
            },
            others=f"{na}と{nb}が言い合いになっている。自分らしく、止めに入るか、茶化すか、どちらかの味方をする。",
            min_speakers=3,
            label=f"口ゲンカ（{na}→{nb}）",
        )

    def _makeup(self) -> Beat | None:
        if not self.tension:
            return None
        a, b, since = random.choice(self.tension)
        na, nb = self._n(a), self._n(b)
        if language.is_en():
            when = "earlier today" if since == self._date else "the other day"
            return Beat(
                "makeup",
                a,
                b,
                {
                    a: f"What happened with {nb} {when} (the squabble, the awkwardness) has been bugging you this whole time. "
                    f"Even if it's hard to be honest, apologize to {nb} or reach out to her in your own way.",
                    b: f"{na} is reaching out to you. Take it in your own way: relieved, embarrassed, or close to tears.",
                },
                others=f"{na} and {nb} are trying to make up. Watch over them warmly, or add a word of your own.",
                label=f"Making up ({na} and {nb})",
            )
        when = "さっき" if since == self._date else "前"
        return Beat(
            "makeup",
            a,
            b,
            {
                a: f"{when}の{nb}とのこと（言い合いや気まずさ）が、ずっと気になっていた。"
                f"素直になれないなりに、自分らしいやり方で{nb}に謝るか、歩み寄る。",
                b: f"{na}が歩み寄ってきた。ほっとしたり、照れたり、泣きそうになったりしながら、自分らしく受け止める。",
            },
            others=f"{na}と{nb}が仲直りしようとしている。あたたかく見守るか、ひと言そえる。",
            label=f"仲直り（{na}と{nb}）",
        )

    def _secret(self) -> Beat | None:
        """町の、ないしょの出来事を打ち明ける。その場にいなかった子に関係する話なら、その子が相手役になる。"""
        if self.town is None:
            return None
        events = self.town.untold(("secret",))
        if not events:
            return None
        e = random.choice(events)
        a = random.choice(e["members"])
        mentioned = [c for c in self._ids if c not in e["members"] and self._mentions(c, e["text"])]
        others = [c for c in e["members"] if c != a]
        b = mentioned[0] if mentioned else (others[0] if others else self._friend_of(a))
        na, nb = self._n(a), self._n(b)
        what = self.town.describe(e)
        if language.is_en():
            if b in e["members"]:
                cue_b = (
                    f"{na} has started talking about the secret you were in on too ({what}). "
                    "Panic, try to stop her, or give up and tell it together, in your own way."
                )
            elif mentioned:
                cue_b = (
                    f"What {na} is telling turns out to involve you (you weren't there, so this is the first you've heard of it). "
                    "React in your own way: surprised, angry, forgiving, or embarrassed."
                )
            else:
                cue_b = f"{na} just confessed something she'd been keeping secret. Take it in your own way."
            return Beat(
                "secret",
                a,
                b,
                {
                    a: f"Work up the nerve to bring up something from town you'd been keeping secret ({what}) "
                    "(own up to it, ask for advice, or let it slip by accident, in your own way. "
                    "If it has already come up, continue from there and say how you really feel).",
                    b: cue_b,
                },
                others=f"{na} just brought up something she'd been keeping secret. React in your own way.",
                min_speakers=3,
                label=f"Secret comes out ({na} and {nb})",
                event_id=e["event_id"],
                strained=bool(mentioned),
            )
        if b in e["members"]:
            cue_b = (
                f"{na}が、あなたもいっしょだったないしょのできごと（{what}）を話し始めた。"
                "慌てる、止める、観念していっしょに話すなど、自分らしく反応する。"
            )
        elif mentioned:
            cue_b = (
                f"{na}の話は、あなたに関係のあることだった（あなたはその場にいなかったので、今はじめて知る）。"
                "驚く、怒る、許す、照れるなど、自分らしく反応する。"
            )
        else:
            cue_b = f"{na}が、ないしょにしていたことを打ち明けた。自分らしく受け止める。"
        return Beat(
            "secret",
            a,
            b,
            {
                a: f"ないしょにしていた町でのできごと（{what}）のことを、思い切って切り出す"
                "（白状する、相談する、ぽろっと口をすべらせるなど、自分らしく。"
                "もう話に出ているなら、その続きとして、自分の本当の気持ちを伝える）。",
                b: cue_b,
            },
            others=f"{na}が何かないしょにしていたことを切り出した。自分らしく反応する。",
            min_speakers=3,
            label=f"ないしょ話の発覚（{na}と{nb}）",
            event_id=e["event_id"],
            strained=bool(mentioned),
        )

    def _mentions(self, cid: str, text: str) -> bool:
        name = self._names[cid]
        if language.is_en():
            return re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE) is not None
        return name in text

    def _confide(self) -> Beat | None:
        """町の出来事のうち、まだユウに話していないものを、自分から打ち明ける。"""
        if self.town is None:
            return None
        cands = [(c, e) for e in self.town.untold(("open", "ask")) for c in e["members"]]
        if not cands:
            return None
        # 元気のない子ほど、話を聞いてほしい
        a, e = min(cands, key=lambda ce: self.mood[ce[0]] + random.uniform(0, 1.5))
        b = self._friend_of(a, exclude=tuple(c for c in e["members"] if c != a))
        na = self._n(a)
        if language.is_en():
            return Beat(
                "confide",
                a,
                b,
                {
                    a: f"Tell everyone about something that happened in town ({self.town.describe(e)}), "
                    "with the feelings you had at the time "
                    "(if it has already come up, continue from there and say more about how you felt).",
                    b: f"Respond to {na}'s story with real sympathy (be happy with her, comfort her, laugh with her, and so on).",
                },
                label=f"Opening up ({na})",
                event_id=e["event_id"],
            )
        return Beat(
            "confide",
            a,
            b,
            {
                a: f"町でのできごと（{self.town.describe(e)}）のことを、みんなに打ち明ける。そのときの気持ちを込めて話す"
                "（もう話に出ているなら、その続きとして、もっと詳しく気持ちを話す）。",
                b: f"{na}の話に、心から共感して寄り添う（いっしょに喜ぶ、なぐさめる、笑うなど）。",
            },
            label=f"打ち明け話（{na}）",
            event_id=e["event_id"],
        )

    def _empathy(self) -> Beat | None:
        m = min(self._ids, key=lambda c: self.mood[c] + random.uniform(0, 0.8))
        a = self._friend_of(m)
        nm, na = self._n(m), self._n(a)
        if language.is_en():
            return Beat(
                "empathy",
                a,
                m,
                {
                    a: f"You notice {nm} seems a little down, and you cheer {nm} up in your own way.",
                    m: f"{na} is cheering you up. Say how you honestly feel, happy and a bit embarrassed, maybe teary.",
                },
                others=f"{na} is cheering {nm} up. Add a word of your own to lift {nm}'s spirits.",
                label=f"Cheering up ({na} -> {nm})",
            )
        return Beat(
            "empathy",
            a,
            m,
            {
                a: f"{nm}が少し元気がないことに気づいて、{nm}を自分らしいやり方で励ます。",
                m: f"{na}に励まされた。うれしくて照れたり、涙ぐんだりしながら、素直な気持ちを言う。",
            },
            others=f"{na}が{nm}を励ましている。自分も、自分らしいひと言で{nm}を元気づける。",
            label=f"励まし（{na}→{nm}）",
        )

    def _boke(self) -> Beat | None:
        a, b = random.sample(self._ids, 2)
        na = self._n(a)
        if language.is_en():
            return Beat(
                "boke",
                a,
                b,
                {
                    a: "Say something a little airheaded, or get hold of the wrong end of the stick, and lighten the mood. "
                    "Your feeling is [cheerful] or [happy].",
                    b: f"Immediately call out {na}'s silly mix-up with a quick comeback.",
                },
                label=f"Airhead and straight man ({na} and {self._n(b)})",
            )
        return Beat(
            "boke",
            a,
            b,
            {
                a: "ちょっと天然なボケか、ずれた勘違いを言って、場をなごませる。気持ちは【楽】か【喜】。",
                b: f"{na}のボケに、すかさずツッコミを入れる。",
            },
            label=f"ボケとツッコミ（{na}と{self._n(b)}）",
        )

    def _praise(self) -> Beat | None:
        a = random.choice(self._ids)
        b = self._friend_of(a)
        na, nb = self._n(a), self._n(b)
        if language.is_en():
            return Beat(
                "praise",
                a,
                b,
                {
                    a: f"Out of nowhere, sincerely praise something good about {nb}.",
                    b: f"{na} suddenly praised you and you get flustered. Your feeling is [shy].",
                },
                label=f"Praise ({na} -> {nb})",
            )
        return Beat(
            "praise",
            a,
            b,
            {
                a: f"{nb}の良いところを、急にまっすぐほめる。",
                b: f"{na}に急にほめられて、照れる。気持ちは【照】。",
            },
            label=f"ほめ（{na}→{nb}）",
        )

    def _tease(self) -> Beat | None:
        a, b = random.sample(self._ids, 2)
        na, nb = self._n(a), self._n(b)
        if language.is_en():
            return Beat(
                "tease",
                a,
                b,
                {
                    a: f"Tease {nb} lightly (never to the point of being mean).",
                    b: f"{na} is teasing you. Get flustered, or get worked up and fire back.",
                },
                label=f"Teasing ({na} -> {nb})",
            )
        return Beat(
            "tease",
            a,
            b,
            {
                a: f"{nb}のことを、軽くからかう（意地悪にならない程度）。",
                b: f"{na}にからかわれた。照れるか、ムキになって言い返す。",
            },
            label=f"からかい（{na}→{nb}）",
        )

    def _laugh(self) -> Beat | None:
        a = max(self._ids, key=lambda c: self.mood[c] + random.uniform(0, 1.5))
        b = self._friend_of(a)
        if language.is_en():
            return Beat(
                "laugh",
                a,
                b,
                {
                    a: "Turn the conversation so far into a laugh with one line that's very you, so everyone can laugh together.",
                    b: f"{self._n(a)}'s line makes you burst out laughing; say how much fun this is. Your feeling is [happy].",
                },
                label=f"Ending on a laugh ({self._n(a)})",
            )
        return Beat(
            "laugh",
            a,
            b,
            {
                a: "ここまでの話を、自分らしい一言で笑いに変えて、みんなで笑える空気にする。",
                b: f"{self._n(a)}の一言に思わず笑って、楽しい気持ちを言う。気持ちは【喜】。",
            },
            label=f"笑って締め（{self._n(a)}）",
        )

    def apply_beat(self, beat: Beat) -> None:
        """演出のリードが話せたら、その展開に合わせて関係と気分を動かす。"""
        if beat.partner is None:
            return
        with self._lock:
            self._apply_beat(beat)

    def _told(self, beat: Beat) -> None:
        if self.town is not None and beat.event_id is not None:
            self.town.mark_told(beat.event_id)

    def _apply_beat(self, beat: Beat) -> None:
        a, b = beat.lead, beat.partner
        assert b is not None
        if beat.kind == "quarrel":
            self._bump(a, b, -0.5)
            self._bump(b, a, -0.5)
            self._cheer(b, -0.3)
            self._add_tension(a, b)
        elif beat.kind == "makeup":
            self._bump(a, b, 0.9)
            self._bump(b, a, 0.9)
            self._cheer(a, 0.5)
            self._cheer(b, 0.5)
            self._resolve_tension(a, b)
            self._feel(a, language.pick(ja=f"{self._n(b)}と仲直りできてほっとした", en=f"you felt relieved to make up with {self._n(b)}"))
            self._feel(b, language.pick(ja=f"{self._n(a)}と仲直りできてうれしかった", en=f"you were glad to make up with {self._n(a)}"))
        elif beat.kind == "secret":
            self._told(beat)
            if beat.strained:
                self._bump(a, b, -0.3)
                # 相手に関係のあるないしょだった。打ち明けたあとは、仲直りの演出で決着をつける
                self._add_tension(a, b)
            else:
                self._bump(a, b, 0.2)
                self._bump(b, a, 0.2)
        elif beat.kind == "confide":
            self._told(beat)
            self._bump(a, b, 0.4)
            self._bump(b, a, 0.3)
            self._cheer(a, 0.3)
            self._feel(a, language.pick(ja=f"{self._n(b)}が話を聞いてくれてうれしかった", en=f"you were glad {self._n(b)} listened"))
        elif beat.kind == "empathy":
            self._bump(a, b, 0.3)
            self._bump(b, a, 0.5)
            self._cheer(b, 0.6)
            self._feel(b, language.pick(ja=f"{self._n(a)}に励まされて、元気が出た", en=f"you felt better after {self._n(a)} cheered you up"))
        elif beat.kind in ("boke", "laugh"):
            for c in self._ids:
                self._cheer(c, 0.15)
        elif beat.kind == "praise":
            self._bump(b, a, 0.4)
            self._cheer(b, 0.3)
        elif beat.kind == "tease":
            self._bump(a, b, 0.1)
            self._bump(b, a, -0.1)
        self._save()
