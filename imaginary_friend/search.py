"""ネット検索（ツールコーリング用）。

backend（複数書くと並列に引き、結果を「事典」「ニュース」などに分けて合わせる）:
  wikipedia … Wikipedia の検索 API。キー不要。事典にある事柄に強く、ニュースや天気には弱い
  news      … Google ニュースの RSS 検索。キー不要。最近の出来事・話題の作品や人に強い（見出しと日付と出典だけ）
  brave     … Brave Search API。api_key（または環境変数 BRAVE_API_KEY）が要る
  searxng   … 自分で立てた SearXNG。host を指定し、settings.yml で json 出力を有効にしておく
"""

from __future__ import annotations

import html
import logging
import os
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests

from . import language
from .config import SearchConfig
from .news import parse_feed

logger = logging.getLogger(__name__)

_USER_AGENT = "llm-imaginary-friend/0.1 (https://github.com/lumavrenor/llm-imaginary-friend; desktop chat app)"
_TAG_RE = re.compile(r"<[^>]+>")
# 本文から抜き出す文の最短の長さ（これより短いのは一覧の項目などの断片）
_MIN_SENTENCE = 12
# 題名どうしを比べるときに無視する記号（「＃題名」と「題名」を同じとみなす）
_TITLE_IGNORE = re.compile(r"[\s#・!?「」『』:\-~〜、。]")

TOOL_NAME = "web_search"
TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "ネットで調べる。最新の出来事、固有名詞の詳しい情報、自信のない事実を確かめるときだけ使う。"
            "雑談や気持ちの話、自分の意見を言うだけなら使わない。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    # 推測の年や細部を足すと、肝心の記事の要約から外れた結果になる
                    "description": (
                        "検索語。文ではなく名詞のキーワードを1〜3語、空白区切りで（例: VOICEVOX 開発者）。"
                        "年号や自分の推測した細かい条件は入れない"
                    ),
                }
            },
            "required": ["query"],
        },
    },
}
TOOL_EN = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Search the web. Use it only for recent events, details about specific names, or facts you are not sure of. "
            "Do not use it for small talk, feelings, or just giving your own opinion."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search terms: 1 to 3 keywords (nouns, not a sentence), separated by spaces "
                        "(e.g. Kokoro TTS developer). Do not add years or details you are only guessing."
                    ),
                }
            },
            "required": ["query"],
        },
    },
}


def body_mark() -> str:
    """Wikipedia の要約のあとに足す、本文から拾った文の目印（memory.py がここで分ける）。"""
    return language.pick(ja=" 本文より: ", en=" From the article: ")


def tool() -> dict:
    """今の言語の検索ツールの定義。"""
    return language.pick(ja=TOOL, en=TOOL_EN)


class SearchError(RuntimeError):
    pass


@dataclass
class Hit:
    title: str
    snippet: str
    url: str
    kind: str = ""  # 結果の種類（事典 / ニュース / ウェブ）。検索先を複数使うときの見出し


# 検索先 → 結果の種類
_KINDS = {"wikipedia": "事典", "news": "ニュース", "brave": "ウェブ", "searxng": "ウェブ"}


class WebSearch:
    def __init__(self, config: SearchConfig):
        self._cfg = config
        self._session = requests.Session()
        self._session.headers["User-Agent"] = _USER_AGENT
        self._api_key = config.api_key or os.environ.get("BRAVE_API_KEY", "")
        if "brave" in config.backend and not self._api_key:
            raise SearchError("[search] backend の \"brave\" には api_key か環境変数 BRAVE_API_KEY が要ります")
        if "searxng" in config.backend and not config.host:
            raise SearchError("[search] backend の \"searxng\" には host が要ります")
        self._backends = {
            "wikipedia": self._wikipedia_retry,
            "news": self._news,
            "brave": self._brave,
            "searxng": self._searxng,
        }
        self._pool = ThreadPoolExecutor(max_workers=max(1, len(config.backend)), thread_name_prefix="search")

    @property
    def label(self) -> str:
        return "+".join(self._cfg.backend)

    def search(self, query: str) -> list[Hit]:
        """全部の検索先を並列に引き、設定に書いた順に並べて返す。1つでも答えれば、失敗した先は飛ばす。"""
        query = query.strip()
        if not query:
            return []
        names = self._cfg.backend
        futures = {name: self._pool.submit(self._backends[name], query) for name in names}
        hits: list[Hit] = []
        errors: list[str] = []
        for name in names:
            try:
                found = futures[name].result()
            except (requests.RequestException, ValueError, KeyError, TypeError, ET.ParseError) as e:
                logger.warning("検索に失敗（%s）: %s: %s", name, type(e).__name__, e)
                errors.append(f"{name}: {type(e).__name__}: {e}")
                continue
            for h in found[: self._cfg.max_results]:
                h.kind = h.kind or _KINDS[name]
                hits.append(h)
        if errors and len(errors) == len(names):
            raise SearchError(
                language.pick(ja="検索に失敗しました（", en="Search failed (") + " / ".join(errors) + language.pick(ja="）", en=")")
            )
        return hits

    def _wikipedia_retry(self, query: str) -> list[Hit]:
        hits = self._wikipedia(query)
        # Wikipedia は全語を含むページしか返さないので、ヒットしなければ語を減らして引き直す
        words = query.split()
        while not hits and len(words) > 1:
            words = words[:-1]
            hits = self._wikipedia(" ".join(words))
        return hits

    def _news(self, query: str) -> list[Hit]:
        lang = self._cfg.lang
        region = {"ja": "JP", "en": "US"}.get(lang, lang.upper())
        hl = "en-US" if lang == "en" else lang
        resp = self._session.get(
            "https://news.google.com/rss/search",
            params={"q": query, "hl": hl, "gl": region, "ceid": f"{region}:{lang}"},
            timeout=self._cfg.timeout_sec,
        )
        resp.raise_for_status()
        _, items = parse_feed(resp.content)
        # 動画や SNS の投稿は見出しだけでは中身がわからないので後ろへ回す
        items.sort(key=lambda it: it.source in ("YouTube", "TikTok", "X", "Instagram"))
        return [
            Hit(it.title, " ".join(x for x in (it.date_label, it.source) if x), it.url, "ニュース")
            for it in items
        ]

    def _get(self, url: str, params: dict, headers: dict | None = None) -> dict:
        for attempt in range(2):
            resp = self._session.get(url, params=params, headers=headers, timeout=self._cfg.timeout_sec)
            if resp.status_code == 429 and attempt == 0:
                # 混み合っているときは少しだけ待って1回やり直す（会話が止まるので長くは待たない）
                try:
                    wait = float(resp.headers.get("Retry-After", 1))
                except ValueError:
                    wait = 1.0
                time.sleep(min(max(wait, 0.5), 2.0))
                continue
            break
        resp.raise_for_status()
        return resp.json()

    def _wikipedia(self, query: str) -> list[Hit]:
        hits = self._wikipedia_search(query, self._cfg.max_results)
        words = query.split()
        # 全文検索は語に分けて探すので、作品名などは一般的な語のページに負けて上位に来ないことがある
        # （先頭に「＃」が付く作品名を「＃」なしで探したら、題名の一部の一般的な語のページが先に並び、本物は5位だった）。
        # 題名がほぼ一致するページがあれば、それを1位にする
        exact = self._wikipedia_exact(words, hits)
        if exact is not None:
            hits = [exact] + [h for h in hits if h.title != exact.title]
        # 「VOICEVOX 開発者 作成」のように語が多いと、主題のページが上位に来ないことがある。
        # 検索語と同じ名前のページが無ければ、先頭の語（たいてい主題）だけでも引いて先頭に足す
        elif len(words) > 1 and not any(h.title.casefold() == w.casefold() for h in hits for w in words):
            subject = self._wikipedia_search(words[0], 1)
            if subject and subject[0].title not in {h.title for h in hits}:
                hits = subject + hits
        # 冒頭の要約には「高さ」のような細かい事実が無いことが多いので、
        # 1位のページだけ本文も取り、検索語に多く当たる文を添える
        if hits:
            rest = [w for w in words if w not in hits[0].title]
            if rest:
                try:
                    body = self._wikipedia_body(hits[0].title)
                except (requests.RequestException, ValueError) as e:
                    logger.warning("Wikipedia の本文を取れなかった（要約だけで続ける）: %s", e)
                    body = ""
                found = _best_sentences(body, rest, limit=3)
                if found:
                    hits[0].snippet += body_mark() + " ".join(found)
        return hits

    def _wikipedia_search(self, query: str, limit: int) -> list[Hit]:
        lang = self._cfg.lang
        data = self._get(
            f"https://{lang}.wikipedia.org/w/api.php",
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": limit,
                "prop": "extracts|info",
                "inprop": "url",
                "exintro": 1,
                "explaintext": 1,
                "exchars": 400,
                "format": "json",
                "formatversion": 2,
            },
        )
        pages = sorted(data.get("query", {}).get("pages", []), key=lambda p: p.get("index", 0))
        return [Hit(p["title"], p.get("extract", ""), p.get("fullurl", "")) for p in pages]

    def _wikipedia_exact(self, words: list[str], known: list[Hit]) -> Hit | None:
        """題名が検索語とほぼ一致するページ（記号・空白・全角半角の違いは無視、転送ページも追う）。

        語をつないだもの（「青い 自転車」→「青い自転車」）、次に先頭の語の順に試す。
        known（全文検索の結果）に一致するページがあれば、問い合わせずにそれを使う（Wikipedia の回数制限対策）。
        見つからない・失敗したときは None（全文検索の結果だけで続ける）。
        """
        candidates = ["".join(words)] + ([words[0]] if len(words) > 1 else [])
        api = f"https://{self._cfg.lang}.wikipedia.org/w/api.php"
        try:
            for word in candidates:
                key = _title_key(word)
                if not key:
                    continue
                found = next((h for h in known if _title_key(h.title) == key), None)
                if found is not None:
                    return found
                # opensearch は題名の前方一致。転送ページ（別名）の題名も返る
                data = self._get(
                    api, {"action": "opensearch", "search": word, "limit": 5, "namespace": 0, "format": "json"}
                )
                titles = data[1] if isinstance(data, list) and len(data) > 1 else []
                title = next((t for t in titles if _title_key(t) == key), None)
                if title is None:
                    continue
                page = self._get(
                    api,
                    {
                        "action": "query",
                        "titles": title,
                        "redirects": 1,
                        "prop": "extracts|info",
                        "inprop": "url",
                        "exintro": 1,
                        "explaintext": 1,
                        "exchars": 400,
                        "format": "json",
                        "formatversion": 2,
                    },
                )
                pages = [p for p in page.get("query", {}).get("pages", []) if not p.get("missing")]
                if pages:
                    p = pages[0]
                    return Hit(p["title"], p.get("extract", ""), p.get("fullurl", ""))
        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            logger.warning("Wikipedia の題名検索に失敗（全文検索だけで続ける）: %s", e)
        return None

    def _wikipedia_body(self, title: str) -> str:
        data = self._get(
            f"https://{self._cfg.lang}.wikipedia.org/w/api.php",
            {
                "action": "query",
                "titles": title,
                "prop": "extracts",
                "explaintext": 1,
                "format": "json",
                "formatversion": 2,
            },
        )
        pages = data.get("query", {}).get("pages", [])
        return pages[0].get("extract", "") if pages else ""

    def _brave(self, query: str) -> list[Hit]:
        data = self._get(
            "https://api.search.brave.com/res/v1/web/search",
            {"q": query, "count": self._cfg.max_results, "search_lang": self._cfg.lang},
            headers={"X-Subscription-Token": self._api_key, "Accept": "application/json"},
        )
        return [
            Hit(_plain(r.get("title", "")), _plain(r.get("description", "")), r.get("url", ""))
            for r in data.get("web", {}).get("results", [])
        ]

    def _searxng(self, query: str) -> list[Hit]:
        data = self._get(
            f"{self._cfg.host}/search",
            {"q": query, "format": "json", "language": self._cfg.lang},
        )
        return [
            Hit(_plain(r.get("title", "")), _plain(r.get("content", "")), r.get("url", ""))
            for r in data.get("results", [])
        ]


_KIND_EN = {"事典": "Encyclopedia", "ニュース": "News", "ウェブ": "Web"}


def format_hits(query: str, hits: list[Hit]) -> str:
    """LLM に渡す検索結果。"""
    if language.is_en():
        return _format_hits_en(query, hits)
    if not hits:
        return f"「{query}」の検索結果: 見つかりませんでした。"
    lines = [f"「{query}」の検索結果:"]
    kinds = list(dict.fromkeys(h.kind for h in hits))
    for kind in kinds:
        if kind == "ニュース":
            # 見出しだけで記事の中身まで話を作らないように
            lines.append("[ニュース]（見出しと日付と出典だけ。記事の中身はわからない）")
        elif len(kinds) > 1:
            lines.append(f"[{kind}]")
        for i, h in enumerate((h for h in hits if h.kind == kind), 1):
            snippet = re.sub(r"\s+", " ", h.snippet).strip()
            if kind == "ニュース":
                lines.append(f"{i}. {h.title}" + (f"（{snippet}）" if snippet else ""))
            else:
                lines.append(f"{i}. {h.title}: {snippet}")
    return "\n".join(lines)


def _format_hits_en(query: str, hits: list[Hit]) -> str:
    if not hits:
        return f'Search results for "{query}": nothing found.'
    lines = [f'Search results for "{query}":']
    kinds = list(dict.fromkeys(h.kind for h in hits))
    for kind in kinds:
        label = _KIND_EN.get(kind, kind)
        if kind == "ニュース":
            lines.append(f"[{label}] (headline, date and source only; the article itself is unknown)")
        elif len(kinds) > 1:
            lines.append(f"[{label}]")
        for i, h in enumerate((h for h in hits if h.kind == kind), 1):
            snippet = re.sub(r"\s+", " ", h.snippet).strip()
            if kind == "ニュース":
                lines.append(f"{i}. {h.title}" + (f" ({snippet})" if snippet else ""))
            else:
                lines.append(f"{i}. {h.title}: {snippet}")
    return "\n".join(lines)


# 英語の文の切れ目（Mr. などで切れることはあるが、本文から数文を拾うだけなので気にしない）
_SENTENCE_SPLIT = re.compile(r"(?<=。)|\n+")
_SENTENCE_SPLIT_EN = re.compile(r"(?<=[.!?])\s+|\n+")


def _best_sentences(body: str, words: list[str], limit: int) -> list[str]:
    """本文から検索語を多く含む文を選び、本文の順に並べて返す（同点なら前にある文を優先）。

    登場人物一覧などの短い断片（「アキ アキの父。」）は文脈が無く、読み違えの元になるので選ばない。
    """
    en = language.is_en()
    sentences = [
        s.strip()
        for s in language.pick(ja=_SENTENCE_SPLIT, en=_SENTENCE_SPLIT_EN).split(body)
        if len(s.strip()) >= _MIN_SENTENCE * (2 if en else 1) and not s.strip().startswith("=")
    ]
    scored = []
    for i, s in enumerate(sentences):
        hay = s.lower() if en else s
        score = sum(1 for w in words if (w.lower() if en else w) in hay)
        if score:
            scored.append((-score, i, s[: 300 if en else 150]))
    best = sorted(scored)[:limit]
    return [s for _, _, s in sorted(best, key=lambda t: t[1])]


def _title_key(text: str) -> str:
    """題名の比較用。全角半角をそろえ（NFKC で「＃」→「#」）、記号と空白を除き、大文字小文字を無視する。"""
    return _TITLE_IGNORE.sub("", unicodedata.normalize("NFKC", text)).casefold()


def _plain(text: str) -> str:
    return html.unescape(_TAG_RE.sub("", text))
