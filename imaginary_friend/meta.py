"""6人目「モデルの子」。仕様は docs/2026-11-15-llmイマジナリーフレンド.md。

接続中の LLM そのものが、町はずれの電気室に来る子。どのモデルで動いていても同じ1人（[meta.cast]）で、
記憶・好感度・気分の id も1つ。モデル名は接続先から自動で取り、本人に「今日の事実」として渡すだけ
（モデルごとの設定の表は持たない。gemma5 や qwen4 が出ても、そのまま来る）。

- 起動時: meta_runs（モデル名ごとの動いた記録）と比べ、「初めて」「◯日ぶり」「前回は別のモデル」を決める
- 実行状況: llm.py の RunStats（生成の秒数・トークン数・失敗）と、設定の数字を、本人のプロンプトにだけ入れる。
  入れるのは、きっかけ（遅かった・止まった・忘れた・聞かれた・入れ替わった・調べものが尽きた）のあるラウンドと、
  stats_chance に当たったラウンドだけ。材料を渡さないことで、メタ発言を口癖にさせない
- 選ばれやすさ: ふだんは appear_weight 倍、きっかけがあれば cue_weight 倍。続けて話すのは max_streak ラウンドまで
- 気分: 町の出来事に出ないので、実行状況からコードで決める（drama.set_own_mood）
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from . import clock, language
from .drama import Mood
from .llm import short_model_name

if TYPE_CHECKING:
    from .config import Config
    from .drama import Drama
    from .llm import LLMClient
    from .store import Store

logger = logging.getLogger(__name__)

# ユウがしくみのことを聞いている言い方
_ASK_RE = re.compile(
    r"AI|ＡＩ|人工知能|モデル|LLM|ＬＬＭ|GPU|ＧＰＵ|記憶|しくみ|仕組み|プログラム|コンピュータ|計算機|機械|ロボット|トークン"
)
# だれかが忘れたことを話している
_FORGOT_RE = re.compile(r"覚えてな|おぼえてな|忘れ|わすれ|思い出せな")
_ASK_RE_EN = re.compile(
    r"\bAI\b|artificial intelligence|\bmodels?\b|\bLLMs?\b|\bGPUs?\b|\bmemory\b|\bmemories\b|how (do|does) (you|it|this) work"
    r"|\bprogram\w*|\bcomputer\w*|\bmachines?\b|\brobots?\b|\btokens?\b|neural|\bweights\b",
    re.IGNORECASE,
)
_FORGOT_RE_EN = re.compile(
    r"\b(forg[eo]t\w*|don'?t remember|can'?t remember|doesn'?t remember|can'?t recall|slipped my mind)\b", re.IGNORECASE
)
_MEMORY_TALK_RE = re.compile(r"記憶|覚え|忘れ")
_MEMORY_TALK_RE_EN = re.compile(r"\b(memor\w*|remember\w*|forg[eo]t\w*|recall)\b", re.IGNORECASE)
# モデル名の中のパラメータ数（31b、27B、e2b の 2b、30b-a3b の 30b など）
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")
# 生成の速さを比べるのに要る、トークン数の分かった生成の数
_MOOD_SAMPLES = 5

_MOODS = {
    "ぼんやり": Mood("ぼんやり", "さっき一度止まってしまって、頭が少しぼんやりする", -0.6),
    "調子がいい": Mood("調子がいい", "今日は言葉がすらすら出てくる", 0.8),
    "そわそわ": Mood("そわそわ", "前と中身が替わった気がして、少し落ちつかない", -0.2),
    "ふつう": Mood("ふつう", "いつもどおり", 0.0),
}
_MOODS_EN = {
    "ぼんやり": Mood("hazy", "I stalled once a moment ago, and my head feels a little hazy", -0.6),
    "調子がいい": Mood("on a roll", "the words are coming easily today", 0.8),
    "そわそわ": Mood("restless", "I feel like what's inside me changed since last time, and I can't quite settle", -0.2),
    "ふつう": Mood("as usual", "the same as always", 0.0),
}

CUES = ("slow", "fail", "forgot", "asked", "switched", "search_out")


@dataclass
class View:
    """ラウンドの始めに brain が渡す、今の会話の見え方。"""

    trigger: str  # user / idle / start / talk
    user_text: str = ""  # trigger == "user" のときのユウの発言
    recent: list[tuple[int, str, str]] = field(default_factory=list)  # 直近の (turn id, 話者 id, 本文)
    visible: int = 0  # 今プロンプトに入っている発言の数
    dropped: int = 0  # 前のラウンドから、見えている範囲の外へ流れた発言の数
    topic_visible: bool = True  # 話し込みのきっかけになったユウの発言が、まだ見えているか
    searches_left: int | None = None  # 話し込みの検索の残り（話し込みでなければ None）


@dataclass
class Plan:
    """このラウンドの6人目の扱い。"""

    weight: float  # 発話者の選び方で掛ける倍率（0 なら、名指し・演出の役でない限り選ばない）
    cues: list[str] = field(default_factory=list)
    stats: str = ""  # 本人のプロンプトに入れる「# いまの実行状況」（入れないラウンドは空）


def named(name: str, text: str) -> bool:
    """text に呼び名が出てくるか。前後がカタカナなら別の語の一部とみなす（「ライオン」の中の「イオ」など）。

    英語版は語の境界で見る（"Io" が "Ionic" や "radio" の一部に当たらないように）。
    """
    if not name:
        return False
    if language.is_en():
        return re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE) is not None
    return re.search(rf"(?<![ァ-ヶー]){re.escape(name)}(?![ァ-ヶ])", text) is not None


def model_size(name: str) -> float | None:
    """モデル名から拾ったパラメータ数（B）。拾えなければ None。"""
    vals = [float(v) for v in _SIZE_RE.findall(name)]
    return max(vals) if vals else None


def _fmt_size(v: float) -> str:
    return f"{v:g}B"


class Meta:
    def __init__(self, config: Config, llm: LLMClient, store: Store, user_ref: str, drama: Drama | None = None):
        mc = config.meta
        self._cfg = mc
        self._talk = config.talk
        self._llm_cfg = config.llm
        self._llm = llm
        self._store = store
        self._uref = user_ref
        self._drama = drama
        self.member = config.cast_by_id()[mc.member_id]
        self.id = self.member.id
        self.name = self.member.name
        place = next((p for p in config.town.places if p.id == mc.place), None)
        self.place = place.name if place else language.pick(ja="電気室", en="the power room")
        self.model = llm.model_name or llm.profile.model
        # 声に出すときの短い名前（gemma-4-31B-it-NVFP4-RedHat → gemma4-31B）。画面の名前欄と同じ
        self.spoken_model = short_model_name(self.model)
        self.engine = llm.profile.engine
        self.town = None  # brain が Town を作ったあとで入れる（出来事の件数だけを見る）
        llm.stats.set_window(mc.stats_window)
        self._stats = llm.stats

        # --- 起動時: 前回と比べる -------------------------------------------------
        runs = store.meta_runs()
        prev = runs[0] if runs else None
        mine = next((r for r in runs if r["model"] == self.model), None)
        self.first_ever = not runs
        self.switched = prev is not None and prev["model"] != self.model
        self.new_model = mine is None
        self.prev_model = prev["model"] if prev else ""
        self._prev_last = prev["last_at"] if prev else None
        self._prev_tps = mine["tps"] if mine else None
        store.meta_launch(self.model, self.engine, time.time())
        self.arrival = self._arrival_text()
        logger.info(
            "6人目: %s（%s / %s）%s",
            self.name,
            self.model,
            self.engine,
            "初めて" if self.first_ever else (f"前回は {self.prev_model}" if self.switched else "前回と同じモデル"),
        )

        # --- ラウンドをまたいで持つ状態 ------------------------------------------------
        self._switch_cue = self.switched or self.first_ever  # 挨拶のあとの最初のラウンドで使う
        self._slow_seen = 0  # 「遅かった」に使った記録の seq
        self._fail_seen = 0  # 「止まった」に使った失敗の数
        self._forgot_seen: int | None = None  # 「忘れた」を見終えた turn id（None は、まだ1回も見ていない）
        self._session_cues: set[str] = set()  # 話し込みで1回だけのきっかけ
        self._streak = 0  # 続けて話したラウンドの数
        self._mood_final = False
        self._mood_label = ""
        self._update_mood()

    # --- 起動時の文 ------------------------------------------------------------------

    def _arrival_text(self) -> str:
        if language.is_en():
            return self._arrival_text_en()
        if self.first_ever:
            return "あなたは今日、初めて動いた"
        parts: list[str] = []
        if self.switched:
            parts.append(f"前回は {self.prev_model} で動いていた。今日は {self.model} で動いている。覚えていることは同じ")
            if self.new_model:
                parts.append("このモデルで動くのは初めて")
            a, b = model_size(self.prev_model), model_size(self.model)
            if a and b and a != b:
                parts.append(
                    f"名前から見ると、前回より{'大きい' if b > a else '小さい'}（{_fmt_size(a)} → {_fmt_size(b)}）"
                )
        if self._cfg.absent_note and self._prev_last and clock.gap_text(self._prev_last):
            parts.append(f"前に動いていたのは{clock.when_text(self._prev_last)}。そのあいだは止まっていた")
        return "。".join(parts)

    def _arrival_text_en(self) -> str:
        if self.first_ever:
            return "Today is the first time you have ever run"
        parts: list[str] = []
        if self.switched:
            parts.append(
                f"Last time you ran on {self.prev_model}. Today you are running on {self.model}. "
                "What you remember is the same"
            )
            if self.new_model:
                parts.append("This is the first time you have run on this model")
            a, b = model_size(self.prev_model), model_size(self.model)
            if a and b and a != b:
                parts.append(
                    f"Going by the names, you are {'bigger' if b > a else 'smaller'} than last time "
                    f"({_fmt_size(a)} -> {_fmt_size(b)})"
                )
        if self._cfg.absent_note and self._prev_last and clock.gap_text(self._prev_last):
            parts.append(f"The last time you ran was {clock.when_text(self._prev_last)}. You were stopped in between")
        return ". ".join(parts)

    # --- プロンプトの文 ----------------------------------------------------------------

    def common_text(self) -> str:
        """全員共通の system に入れる、6人目の紹介（しくみのことは書かない）。"""
        n, u = self.name, self._uref
        if language.is_en():
            return (
                f"In {self.place} at the edge of town, there is {n}. {n} wasn't born from {u}'s heart; "
                f"she came from a faraway computer. {n} doesn't live in town; her voice only reaches you "
                f"from {self.place} while she is talking."
            )
        return (
            f"町はずれの{self.place}に、{n}がいます。{n}は{u}の心から生まれた子ではなく、遠くの計算機から来た子です。"
            f"{n}は町では暮らしておらず、話しているときだけ、{self.place}から声が届きます。"
        )

    def others_note(self) -> str:
        """5人のキャラ固有部分に入れる文。"""
        n = self.name
        if language.is_en():
            return (
                f"{n} sometimes says difficult things about how the town works. You don't really understand them. "
                f"You are an ordinary girl who lives in this town. Don't copy the way {n} talks."
            )
        return (
            f"{n}は、ときどき町のしくみについて、むずかしいことを言う。あなたにはよくわからない。"
            f"あなたは、この町で暮らす、ふつうの女の子。{n}の言い方はまねない。"
        )

    def self_text(self, plan: Plan | None) -> str:
        """6人目のキャラ固有部分に入れる文（本人だけが知っていること・決まり・実行状況）。"""
        u = self._uref
        if language.is_en():
            out = [
                "# What only you know",
                self._cfg.self_desc.strip()
                or "You are the language model that is running this conversation. The other five girls' lines come "
                "from the same model as you. They don't know that.",
                f"{u} is the person running you on their own computer. You have no life in town. "
                "While you aren't running, you are stopped.",
                "Talk about how things work at most once per turn. Say it briefly, then go back to normal conversation.",
                'Only say numbers that are written under "# Current run status". Don\'t make up detailed explanations '
                "of how things work.",
                "Never call the five girls fake. Their lives are real to them.",
            ]
            if plan is not None and plan.stats:
                out.append("")
                out.append(plan.stats)
            return "\n".join(out) + "\n"
        out = [
            "# あなただけが知っていること",
            self._cfg.self_desc.strip()
            or "あなたは、この会話を動かしている言語モデルそのもの。5人のセリフも、あなたと同じモデルが出している。5人はそれを知らない。",
            f"{u}は、あなたを自分の計算機で動かしている人。町での暮らしは無い。動いていないあいだは止まっている。",
            "しくみの話は、1回の発言で1つまで。短く言って、ふつうの話に戻る。",
            "数字を言うのは「# いまの実行状況」に書いてあるものだけ。しくみの細かい説明を作らない。",
            "5人を作り物だと言わない。5人の暮らしは、5人にとって本当のこと。",
        ]
        if plan is not None and plan.stats:
            out.append("")
            out.append(plan.stats)
        return "\n".join(out) + "\n"

    def memory_intro(self) -> str:
        """記憶メモを書くときの名乗り（memory.py）。"""
        if language.is_en():
            return (
                f'You are "{self.name}". You weren\'t born from {self._uref}\'s heart; you came from a faraway computer. '
                f"{self.member.desc} {self._cfg.self_desc}"
            )
        return (
            f"あなたは「{self.name}」。{self._uref}の心から生まれた子ではなく、遠くの計算機から来た子です。"
            f"{self.member.desc}{self._cfg.self_desc}"
        )

    # --- ラウンド ----------------------------------------------------------------------

    def start_session(self) -> None:
        self._session_cues.clear()

    def plan(self, view: View) -> Plan:
        """ラウンドの始めに、きっかけ・選ばれやすさ・渡す実行状況を決める。"""
        cues = self._cues(view)
        if self._streak >= self._cfg.max_streak:
            weight = 0.0
        elif cues:
            weight = self._cfg.cue_weight
        else:
            weight = self._cfg.appear_weight
        stats = ""
        if cues or random.random() < self._cfg.stats_chance:
            stats = self._stats_text(view, cues)
        if cues or stats:
            logger.info("6人目: きっかけ=%s 倍率=%.1f 実行状況=%s", cues or "-", weight, "あり" if stats else "なし")
        return Plan(weight, cues, stats)

    def after_round(self, trigger: str, spoken: list[str]) -> None:
        self._streak = self._streak + 1 if self.id in spoken else 0
        if trigger != "start":
            self._switch_cue = False
        calls = [c for c in self._stats.calls() if c.kind == "stream" and c.tps]
        tps = sum(c.tps for c in calls) / len(calls) if calls else None  # type: ignore[misc]
        try:
            self._store.meta_touch(self.model, time.time(), tps)
        except Exception:  # noqa: BLE001 - 記録の失敗で会話を止めない
            logger.exception("meta_runs を更新できなかった")
        # 気分は、生成が _MOOD_SAMPLES 回たまったところで1回だけ決め直す（止まったら、いつでも「ぼんやり」）
        if not self._mood_final and len(calls) >= _MOOD_SAMPLES:
            self._mood_final = True
            self._update_mood(tps)
        elif self._stats.failures:
            self._update_mood(tps)

    def _cues(self, view: View) -> list[str]:
        st = self._stats
        cues: list[str] = []
        streams = [c for c in st.calls() if c.kind == "stream"]
        last = streams[-1] if streams else None
        if last is not None and last.seq > self._slow_seen and last.ok and last.sec > self._cfg.slow_sec:
            self._slow_seen = last.seq
            cues.append("slow")
        if st.failures > self._fail_seen:
            self._fail_seen = st.failures
            cues.append("fail")
        # 起動前の会話（DB から読んだ前回のログ）の「忘れた」は数えない
        seen = self._forgot_seen
        forgot_re = language.pick(ja=_FORGOT_RE, en=_FORGOT_RE_EN)
        forgot = [
            tid
            for tid, spk, text in view.recent
            if seen is not None and tid > seen and spk != self.id and forgot_re.search(text)
        ]
        self._forgot_seen = max([seen or 0, *(tid for tid, _, _ in view.recent)])
        if forgot or (view.trigger == "talk" and not view.topic_visible and "forgot" not in self._session_cues):
            self._session_cues.add("forgot")
            cues.append("forgot")
        ask_re = language.pick(ja=_ASK_RE, en=_ASK_RE_EN)
        if view.trigger == "user" and (named(self.name, view.user_text) or ask_re.search(view.user_text)):
            cues.append("asked")
        if self._switch_cue and view.trigger != "start":
            cues.append("switched")
        if view.trigger == "talk" and view.searches_left == 0 and "search_out" not in self._session_cues:
            self._session_cues.add("search_out")
            cues.append("search_out")
        return cues

    # --- 実行状況 ----------------------------------------------------------------------

    def _lines(self, view: View) -> dict[str, str]:
        """渡せる行の候補（材料が無い行は入れない）。"""
        if language.is_en():
            return self._lines_en(view)
        st = self._stats
        out: dict[str, str] = {}
        size = model_size(self.model)
        out["model"] = (
            f"あなたを動かしているモデル: {self.model}（{self.engine}）。5人のセリフも、同じモデルが出している"
            + (f"。名前から見ると、大きさは {_fmt_size(size)}" if size else "")
            + (f"。声に出して言うときは「{self.spoken_model}」と短く言う（後ろの量子化や配布元の部分は読み上げない）"
               if self.spoken_model != self.model else "")
        )
        out["temp"] = f"文字の選び方のゆらぎ（temperature）: {self._llm_cfg.temperature:g}"
        window = (
            f"一度に見えている会話: 直近 {self._talk.history_lines} 発言・{self._talk.history_chars} 字まで"
            f"（今は {view.visible} 発言が見えている）。それより前は見えていない"
        )
        if view.dropped:
            window += f"（さっき {view.dropped} 発言が外へ流れた）"
        out["window"] = window
        streams = [c for c in st.calls() if c.kind == "stream" and c.ok]
        if streams:
            last = streams[-1]
            who = f"（{last.tag}）" if last.tag else ""
            avg = sum(c.sec for c in streams) / len(streams)
            line = f"直前のセリフ{who}の生成: {last.sec:.1f} 秒"
            if len(streams) >= 3:
                line += f"。ここ {len(streams)} 回の平均は {avg:.1f} 秒"
            if last.tps:
                line += f"。1秒あたり、およそ {last.tps:.0f} トークン"
            out["gen"] = line
        if st.failures:
            line = f"この起動で、返事が止まった（タイムアウトや失敗）のは {st.failures} 回"
            if st.timeouts:
                line += f"。そのうちタイムアウトは {st.timeouts} 回"
            out["fail"] = line
        if view.searches_left is not None:
            out["search"] = (
                f"調べものは、この話し込みであと {view.searches_left} 回"
                if view.searches_left > 0
                else "調べものは、この話し込みではもう使い切った"
            )
        minutes = int((time.time() - st.started) / 60)
        out["uptime"] = f"起動してから: {minutes} 分" if minutes >= 1 else "起動してから: まだ1分たっていない"
        out["requests"] = f"この起動で、モデルへの問い合わせは {st.total} 回"
        n_events = getattr(self.town, "generated", 0) if self.town is not None else 0
        if n_events:
            out["town"] = f"町のできごとは、この起動のときにまとめて {n_events} 件作られた（中身はあなたには見えない）"
        if self.arrival:
            out["arrival"] = self.arrival
        return out

    def _lines_en(self, view: View) -> dict[str, str]:
        st = self._stats
        out: dict[str, str] = {}
        size = model_size(self.model)
        out["model"] = (
            f"The model running you: {self.model} ({self.engine}). The five girls' lines come from the same model"
            + (f". Going by the name, its size is {_fmt_size(size)}" if size else "")
            + (f'. When you say it out loud, keep it short: "{self.spoken_model}" '
               "(don't read out the quantization or uploader parts)"
               if self.spoken_model != self.model else "")
        )
        out["temp"] = f"How much randomness goes into picking each word (temperature): {self._llm_cfg.temperature:g}"
        window = (
            f"The conversation you can see at once: the last {self._talk.history_lines} turns, up to "
            f"{self._talk.history_chars} characters (right now {view.visible} turns are visible). "
            "Anything before that is out of sight"
        )
        if view.dropped:
            window += f" ({view.dropped} turns just slipped out of view)"
        out["window"] = window
        streams = [c for c in st.calls() if c.kind == "stream" and c.ok]
        if streams:
            last = streams[-1]
            who = f" ({last.tag})" if last.tag else ""
            avg = sum(c.sec for c in streams) / len(streams)
            line = f"Generating the last line{who} took {last.sec:.1f} seconds"
            if len(streams) >= 3:
                line += f". The average over the last {len(streams)} is {avg:.1f} seconds"
            if last.tps:
                line += f". About {last.tps:.0f} tokens per second"
            out["gen"] = line
        if st.failures:
            line = f"Since startup, replies have stalled (timed out or failed) {st.failures} times"
            if st.timeouts:
                line += f". {st.timeouts} of those were timeouts"
            out["fail"] = line
        if view.searches_left is not None:
            out["search"] = (
                f"Searches left in this conversation: {view.searches_left}"
                if view.searches_left > 0
                else "You have used up all the searches for this conversation"
            )
        minutes = int((time.time() - st.started) / 60)
        out["uptime"] = f"Time since startup: {minutes} minutes" if minutes >= 1 else "Time since startup: under a minute"
        out["requests"] = f"Requests to the model since startup: {st.total}"
        n_events = getattr(self.town, "generated", 0) if self.town is not None else 0
        if n_events:
            out["town"] = f"{n_events} town events were generated in one go at startup (you can't see what they were)"
        if self.arrival:
            out["arrival"] = self.arrival
        return out

    def _stats_text(self, view: View, cues: list[str]) -> str:
        lines = self._lines(view)
        need: list[str] = []
        for cue in cues:
            need += {
                "slow": ["gen"],
                "fail": ["fail"],
                "forgot": ["window"],
                "asked": ["model"],
                "switched": ["arrival"],
                "search_out": ["search"],
            }[cue]
        if "asked" in cues:
            # 話題に近い行を1つ足す（記憶の話なら見えている範囲、それ以外は生成の速さ）
            memory_re = language.pick(ja=_MEMORY_TALK_RE, en=_MEMORY_TALK_RE_EN)
            need.append("window" if memory_re.search(view.user_text) else "gen")
        # 実行状況は最大4行（きっかけの行3つ＋ほか1行）。長いと、それだけで発言が埋まる
        need = [k for k in dict.fromkeys(need) if k in lines][:3]
        # きっかけ以外の行は1〜2行まで（全部入れると、毎回同じことを言うため）
        rest = [k for k in lines if k not in need]
        random.shuffle(rest)
        extra = 1 if need else random.choice((1, 2))
        keys = need + rest[:extra]
        if not keys:
            return ""
        order = list(lines)
        keys.sort(key=order.index)
        heading = language.pick(
            ja="# いまの実行状況（あなただけに見えている。ここに書かれた数字だけを言ってよい）",
            en="# Current run status (only you can see this; these are the only numbers you may mention)",
        )
        return heading + "\n" + "\n".join(f"- {lines[k]}" for k in keys)

    # --- 気分 ----------------------------------------------------------------------------

    def _update_mood(self, tps: float | None = None) -> None:
        if self._drama is None:
            return
        if self._stats.failures:
            label = "ぼんやり"
        elif self._mood_final and tps and self._prev_tps and tps > self._prev_tps * 1.1:
            label = "調子がいい"
        elif self.switched:
            label = "そわそわ"
        else:
            label = "ふつう"
        if label != self._mood_label:
            self._mood_label = label
            self._drama.set_own_mood(self.id, language.pick(ja=_MOODS, en=_MOODS_EN)[label])
