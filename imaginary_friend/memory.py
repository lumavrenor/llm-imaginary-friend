"""キャラごとの個人記憶。

- 事実（fact）: 自分で調べた検索結果の全文。調べた本人だけが覚えている。
  ほかの分身は、本人が口で言った範囲（共有ログ）しか知らない ＝ 聞きかじり。
- 記憶メモ（note）: 会話の区切りで、各キャラが自分の視点で書く短いメモ（何の話だったか・誰が何を言ったか・どう感じたか）。
  自分で調べた事実は具体的に、他人の話はざっくり書かせるので、キャラごとに覚え方が違ってくる。

話すときは、その場の話題に近い自分の記憶だけを、キャラ固有部分のプロンプトに入れる。
近さは「漢字・カタカナ・英数字を含む2文字の重なり」で測る（埋め込みモデル不要。日本語の固有名詞に強い）。
"""

from __future__ import annotations

import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor

from . import language
from .config import CastMember, MemoryConfig
from .llm import LLMClient, LLMError
from .search import Hit, body_mark
from .store import Store

logger = logging.getLogger(__name__)

# 1件の事実として保存する最大字数（検索結果は長いので、プロンプトを食いすぎないよう切る）
_FACT_CHARS = 700
# 関連度を計算する候補の件数（新しい順）
_CANDIDATES = 200
# ひらがなだけの2文字（「って」「から」など）は話題を表さないので数えない
_CONTENT_CHAR = re.compile(r"[一-龠々ァ-ヶーA-Za-z0-9]")
_SPACE = re.compile(r"[\s、。，．,.！？!?…・〜「」『』（）()\-:：]+")
# 発音記号や「( 音声ファイル)」など。冒頭の1文を短くするために消す
_BRACKETS = re.compile(r"\[[^\]]*\]|\(\s*音声ファイル\s*\)")


_WORD_EN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
# 英語の、話題を表さない語（2文字の重なりの代わりに語の重なりで測るので、これを数えない）
_STOP_EN = frozenset(
    """
    the and for are but not you your yours with this that these those have has had was were been being
    she her hers him his they them their what when where which who whom why how all any can could
    would should will just like really very so too also about from into onto over under then than
    there here out our ours its it's i'm you're she's we're they're i've you've don't didn't doesn't
    isn't wasn't aren't can't won't let's yeah yes okay well maybe some more most much many one two
    get got going gonna want wanna know think thing things kind sort right now today did does say said
    """.split()
)


def _bigrams(text: str) -> set[str]:
    """話題の近さを測る単位。日本語は2文字、英語は語（小文字・語尾の s を落とす）。"""
    if language.is_en():
        out = set()
        for w in _WORD_EN.findall(text.lower()):
            w = w.removesuffix("'s")
            if len(w) < 3 or w in _STOP_EN:
                continue
            if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
                w = w[:-1]
            out.add(w)
        return out
    s = _SPACE.sub("", text)
    return {s[i : i + 2] for i in range(len(s) - 1) if _CONTENT_CHAR.search(s[i : i + 2])}


def _score(query: set[str], text: str) -> float:
    b = _bigrams(text)
    if not query or not b:
        return 0.0
    return len(query & b) / math.sqrt(len(query) * len(b))


def _compact(heading: str, hits: list[Hit]) -> str:
    """記憶にする検索結果。要約の冒頭は発音記号や別名で長いので1文だけにし、検索語を含む本文の文を優先して残す。"""
    lines = [heading]
    en = language.is_en()
    stop = ". " if en else "。"
    for h in hits:
        summary, _, body = h.snippet.partition(body_mark())
        first = _BRACKETS.sub("", summary).split(stop)[0].strip()[: 220 if en else 100].rstrip(".")
        line = f"- {h.title}: {first}" + ("." if en else "。")
        # 本文の文が要約の冒頭と同じことがある（記事の書き出しが要約と一致する場合）
        extra = [
            s.strip().rstrip(".") + ("." if en else "。")
            for s in body.split(stop)
            if s.strip() and s.strip() not in summary
        ]
        if extra:
            line += " " + " ".join(extra)
        lines.append(line)
    return "\n".join(lines)


def _date(ts: float) -> str:
    t = time.localtime(ts)
    if language.is_en():
        return time.strftime("%B ", t) + str(t.tm_mday)
    return f"{t.tm_mon}月{t.tm_mday}日"


class Memory:
    def __init__(self, store: Store, llm: LLMClient, config: MemoryConfig, user_ref: str):
        self._store = store
        self._llm = llm
        self._cfg = config
        self._uref = user_ref
        # 記憶メモ書きは会話の裏で回す。挙手の並列リクエストとはスレッドを分ける
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="memory")
        # キャラ id → 記憶メモを書くときの名乗り（6人目のように、ユウの心から生まれていない子だけ）
        self.intros: dict[str, str] = {}

    # --- 書く ----------------------------------------------------------------

    def add_fact(self, owner: str, query: str, hits: list[Hit], heading: str = "") -> None:
        """heading を省くと「「query」を調べた結果:」。"""
        default = language.pick(ja=f"「{query}」を調べた結果:", en=f'What you found when you looked up "{query}":')
        text = _compact(heading or default, hits)
        limit = _FACT_CHARS * (2 if language.is_en() else 1)  # 英語は同じ中身で字数が倍ほどになる
        if len(text) > limit:
            text = text[:limit] + "…"
        self._store.add_memory(owner, "fact", text)
        logger.info("記憶（事実）: %s が「%s」の結果を覚えた（%d字）", owner, query, len(text))

    def reflect_async(self, members: list[CastMember], lines: list[str], descs: str, since: float) -> None:
        """会話の区切りで、各キャラに自分の視点の記憶メモを書かせる（裏で）。since 以降に調べたことだけ見せる。"""
        for m in members:
            self._pool.submit(self._reflect, m, lines, descs, since)

    def _reflect(self, member: CastMember, lines: list[str], descs: str, since: float) -> None:
        # この会話のあいだに自分で調べたものだけ。前の話題の調べ物を渡すと、無関係な話がメモに混ざる
        facts = [text for ts, text in self._store.memories(member.id, "fact", 5) if ts >= since][:2]
        if language.is_en():
            system, content = self._reflect_prompt_en(member, lines, descs, facts)
        else:
            system, content = self._reflect_prompt_ja(member, lines, descs, facts)
        try:
            note = self._llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": content}],
                max_tokens=240 if not language.is_en() else 200,
                temperature=0.5,
            ).strip()
        except LLMError as e:
            logger.warning("%s: 記憶メモを書けなかった: %s", member.name, e)
            return
        if note:
            self._store.add_memory(member.id, "note", note)
            logger.info("記憶（メモ）: %s: %s", member.name, note)

    def _reflect_prompt_en(
        self, member: CastMember, lines: list[str], descs: str, facts: list[str]
    ) -> tuple[str, str]:
        user = self._uref
        intro = self.intros.get(member.id) or (
            f'You are "{member.name}", one of the imaginary friend girls who were born from a side of {user}\'s heart. '
            f"{member.desc}"
        )
        system = f"{intro}\n\n# The girls\n{descs}"
        content = (
            "# The conversation just now\n" + "\n".join(lines) + "\n\n"
            + ("# Things you looked up yourself\n" + "\n\n".join(facts) + "\n\n" if facts else "")
            + "# Instructions\n"
            f"Turn the conversation just now into a short memory note that {member.name} can look back on later.\n"
            "- In 1 to 3 sentences (under 60 words), write what it was about, who said what, and how you felt.\n"
            "- For things you looked up yourself that matter to the conversation, keep the key facts specific, down to names and years.\n"
            "- For what the others looked up or said, write it roughly, only as much as you heard. Leave out details you don't remember.\n"
            "- Also keep anything where feelings or relationships moved: a fight, making up, being happy, being sad.\n"
            "- Do not write numbers or detailed facts you didn't look up yourself as if they were certain. "
            'If someone only said it, write "she said ..." or leave it out.\n'
            f"- Write in the first person, in {member.name}'s own voice, and write only the note. No headings or bullet points."
        )
        return system, content

    def _reflect_prompt_ja(
        self, member: CastMember, lines: list[str], descs: str, facts: list[str]
    ) -> tuple[str, str]:
        user = self._uref
        intro = self.intros.get(member.id) or (
            f"あなたは「{member.name}」。{user}の心の一面から生まれた、イマジナリーフレンドの女の子の1人です。"
            f"{member.desc}"
        )
        system = f"{intro}\n\n# 女の子たち\n{descs}"
        content = (
            "# さっきの会話\n" + "\n".join(lines) + "\n\n"
            + ("# あなたが自分で調べたこと\n" + "\n\n".join(facts) + "\n\n" if facts else "")
            + "# 指示\n"
            f"さっきの会話を、{member.name}があとで思い出すための短い記憶メモにしてください。\n"
            "- 何の話だったか、誰が何を言ったか、自分がどう感じたかを、1〜3文・120字以内で書く。\n"
            "- 自分で調べたことのうち会話に関係する大事な事実は、名前や年まで具体的に残す。\n"
            "- ほかの子が調べたことや話したことは、聞いた範囲でざっくり書く。覚えていない細部は書かない。\n"
            "- ケンカした、仲直りした、うれしかった、悲しかったなど、気持ちや関係が動いたことも残す。\n"
            # 会話中の根拠のない数字が、記憶として定着しないようにする
            "- 自分で調べていない数字や細かい事実は、確かなこととして書かない。誰かが言っただけなら「〜と言っていた」と書くか、省く。\n"
            f"- 一人称と口調は{member.name}のもので、メモだけを書く。見出しや箇条書きにしない。"
        )
        return system, content

    # --- 思い出す --------------------------------------------------------------

    def recall(self, member: CastMember, context: str) -> str:
        """context（話題と直近の会話）に近い、member 本人の記憶。プロンプトに入れる文字列（無ければ空）。"""
        q = _bigrams(context)
        facts = self._pick(member.id, "fact", q, self._cfg.facts_in_prompt)
        notes = self._pick(member.id, "note", q, self._cfg.notes_in_prompt)
        if not facts and not notes:
            return ""
        en = language.is_en()
        out = [
            "# Your memories (only you remember these; the others may not know them)"
            if en
            else "# あなたの記憶（あなただけが覚えていること。ほかの子は知らないこともある）"
        ]
        if facts:
            out.append(("Things you looked up yourself:\n" if en else "自分で調べたこと:\n") + "\n\n".join(text for _, text in facts))
        if notes:
            out.append(
                ("Memories of earlier conversations:\n" if en else "前の会話の記憶:\n")
                + "\n".join(f"- {_date(ts)}: {text}" for ts, text in notes)
            )
        return "\n".join(out) + "\n"

    def _pick(self, owner: str, kind: str, q: set[str], n: int) -> list[tuple[float, str]]:
        """いちばん新しい1件（今の話の続きのことが多い）＋話題に近いもの。古い順に返す。"""
        if n <= 0:
            return []
        rows = self._store.memories(owner, kind, _CANDIDATES)
        if not rows:
            return []
        picked = [rows[0]]
        scored = sorted(((_score(q, text), i) for i, (_, text) in enumerate(rows[1:], 1)), reverse=True)
        for s, i in scored:
            if len(picked) >= n or s < self._cfg.min_score:
                break
            picked.append(rows[i])
        return sorted(picked)
