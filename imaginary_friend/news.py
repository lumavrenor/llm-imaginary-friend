"""ニュースの RSS（キー不要）。

- parse_feed: RSS 2.0 / RSS 1.0（RDF）/ Atom を読む。search.py のニュース検索（Google ニュースの RSS）でも使う
- NewsFeed: 設定した RSS を裏で定期的に読み、分身が「さっきネットで見かけた」話題として持ち込むネタを貯める。
  一度出したネタは DB（state テーブル）に記録し、再起動しても同じ話をくり返さない。
  説明文の無い記事（Yahoo! ニュースのトピックスなど）は、話題に出す直前に記事ページの概要（meta description）を読みに行く。
"""

from __future__ import annotations

import html
import json
import logging
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from datetime import datetime
from urllib.parse import urlparse

import requests

from . import language
from .config import NewsConfig
from .store import Store

logger = logging.getLogger(__name__)

_USER_AGENT = "Mozilla/5.0 (compatible; llm-imaginary-friend/0.1; desktop chat app)"
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_META_RE = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_META_NAME_RE = re.compile(r"""(?:property|name)\s*=\s*["'](og:description|description)["']""", re.IGNORECASE)
_META_CONTENT_RE = re.compile(r"""content\s*=\s*(["'])(.*?)\1""", re.IGNORECASE | re.DOTALL)
# 話題に渡す説明文の最大字数
_SUMMARY_CHARS = 220
# 記事ページは先頭だけ読めば meta が取れる
_PAGE_BYTES = 300_000
# 出したネタとして覚えておく件数
_USED_KEEP = 500
_USED_KEY = "news_used"
# ユウの発言がこれに触れたら、見かけたニュースを1人に渡す（「最近なにかニュースあった？」に答えられるように）
# 「いいニュースがあるの」「Good news!」のような自分の報告は拾わない
_ASK_RE = re.compile(r"(?<!いい)(?<!良い)(?<!悪い)(?<!嬉しい)(?<!うれしい)(?<!ビッグ)ニュース|時事")
_ASK_RE_EN = re.compile(
    r"\b(any|latest|interesting|recent|today's|the|world)\s+news\b|\bnews\s+(today|lately|recently)\b"
    r"|\bheadlines?\b|\bcurrent events\b",
    re.IGNORECASE,
)


def asks_news(text: str) -> bool:
    """ユウの発言が、最近のニュースを聞いているか。"""
    return bool(language.pick(ja=_ASK_RE, en=_ASK_RE_EN).search(text or ""))


@dataclass
class NewsItem:
    title: str
    summary: str
    source: str  # 配信元（フィード名や Google ニュースの出典）
    url: str
    published: float | None = None  # epoch 秒。不明なら None

    @property
    def date_label(self) -> str:
        if self.published is None:
            return ""
        t = time.localtime(self.published)
        if language.is_en():
            return time.strftime("%B ", t) + str(t.tm_mday)
        return f"{t.tm_mon}月{t.tm_mday}日"


def _local(tag: str) -> str:
    """名前空間を外したタグ名（{http://www.w3.org/2005/Atom}entry → entry）。"""
    return tag.rsplit("}", 1)[-1]


def _child(el: ET.Element, name: str) -> ET.Element | None:
    return next((c for c in el if _local(c.tag) == name), None)


def _text(el: ET.Element, *names: str) -> str:
    for name in names:
        c = _child(el, name)
        if c is not None and (c.text or "").strip():
            return c.text.strip()
    return ""


def plain(text: str) -> str:
    """HTML のタグと実体参照を外し、空白を1つにまとめる。"""
    return _SPACE_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", html.unescape(text or "")))).strip()


def _parse_date(text: str) -> float | None:
    if not text:
        return None
    try:
        return parsedate_to_datetime(text).timestamp()  # RSS 2.0（RFC 822）
    except (TypeError, ValueError, IndexError):
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()  # Atom / RDF（ISO 8601）
    except ValueError:
        return None


def parse_feed(content: bytes) -> tuple[str, list[NewsItem]]:
    """フィードを読み、(フィード名, 記事) を返す。壊れていれば ET.ParseError。"""
    root = ET.fromstring(content)
    channel = next((e for e in root.iter() if _local(e.tag) == "channel"), None)
    feed_title = plain(_text(channel if channel is not None else root, "title"))
    items: list[NewsItem] = []
    for el in root.iter():
        if _local(el.tag) not in ("item", "entry"):
            continue
        title = plain(_text(el, "title"))
        if not title:
            continue
        link = _text(el, "link")
        if not link:
            # Atom は <link href="..."/>
            link_el = _child(el, "link")
            link = link_el.get("href", "") if link_el is not None else ""
        summary = plain(_text(el, "description", "summary", "content"))
        source = plain(_text(el, "source")) or feed_title
        # Google ニュースは説明文が記事へのリンクの一覧（＝見出しのくり返し）なので捨てる。
        # 見出しの末尾の「 - 出典」も外す
        if "news.google." in link:
            summary = ""
            if source and title.endswith(f" - {source}"):
                title = title[: -len(source) - 3].rstrip()
        if summary.startswith(title):
            summary = summary[len(title) :].lstrip(" 　:：-")
        published = _parse_date(_text(el, "pubDate", "published", "updated", "date"))
        items.append(NewsItem(title, summary, source, link, published))
    return feed_title, items


def fetch_page_summary(session: requests.Session, url: str, timeout: float) -> str:
    """記事ページの概要（og:description / description）。取れなければ空。"""
    host = urlparse(url).hostname or ""
    if not url or "news.google." in host:
        # Google ニュースのリンクは JavaScript で転送するので、概要は Google 自身のものになる
        return ""
    try:
        with session.get(url, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            if "html" not in resp.headers.get("Content-Type", "html"):
                return ""
            raw = resp.raw.read(_PAGE_BYTES, decode_content=True)
            text = raw.decode(resp.encoding or "utf-8", errors="replace")
    except (requests.RequestException, OSError, LookupError) as e:
        logger.info("記事の概要を取れなかった: %s: %s", url, e)
        return ""
    found: dict[str, str] = {}
    for tag in _META_RE.findall(text):
        name, content = _META_NAME_RE.search(tag), _META_CONTENT_RE.search(tag)
        if name and content:
            found.setdefault(name.group(1).lower(), plain(content.group(2)))
    return found.get("og:description") or found.get("description") or ""


def title_key(title: str) -> str:
    return re.sub(r"[\s【】\[\]「」『』（）()、。・!！?？]", "", title).casefold()[:60]


class NewsFeed:
    """設定した RSS を裏で読み、分身が持ち込むネタを1つずつ渡す。"""

    def __init__(self, config: NewsConfig, store: Store):
        self._cfg = config
        self._store = store
        self._session = requests.Session()
        self._session.headers["User-Agent"] = _USER_AGENT
        self._lock = threading.Lock()
        self._feeds: dict[str, list[NewsItem]] = {}  # フィードの URL → 記事
        self._loaded = threading.Event()  # 1回目の読み込みが終わった
        try:
            self._used: list[str] = json.loads(store.get_state(_USED_KEY) or "[]")
        except ValueError:
            self._used = []
        self._skip = [w for w in config.skip_words if w]

    def start(self) -> None:
        threading.Thread(target=self._loop, name="news", daemon=True).start()

    def _loop(self) -> None:
        while True:
            self.refresh()
            time.sleep(max(5.0, self._cfg.refresh_min) * 60)

    def refresh(self) -> None:
        total = 0
        for url in self._cfg.feeds:
            try:
                resp = self._session.get(url, timeout=self._cfg.timeout_sec)
                resp.raise_for_status()
                _, items = parse_feed(resp.content)
            except (requests.RequestException, ET.ParseError) as e:
                logger.warning("ニュースの RSS を読めなかった: %s: %s", url, e)
                continue
            with self._lock:
                self._feeds[url] = items
            total += len(items)
        self._loaded.set()
        logger.info("ニュース: %d 件のフィードから %d 件を読んだ", len(self._feeds), total)

    def _ok(self, item: NewsItem, used: set[str], now: float) -> bool:
        if title_key(item.title) in used:
            return False
        if item.published is not None and now - item.published > self._cfg.max_age_hours * 3600:
            return False
        return not self._skipped(item.title + " " + item.summary)

    def _skipped(self, text: str) -> bool:
        if language.is_en():
            low = text.lower()
            # 語として当てる（"war" で "warm" を落とさない）。複数形・過去形などの語尾は許す
            return any(
                re.search(rf"\b{re.escape(w.lower())}(?:s|es|d|ed|ing|er|ers)?\b", low) for w in self._skip
            )
        return any(w in text for w in self._skip)

    def pick(self) -> NewsItem | None:
        """まだ話していないネタを1つ（フィードを無作為に選び、その中の新しいものから）。無ければ None。

        説明文が無ければ記事ページの概要を読みに行く（ネットを待つので、裏スレッドから呼ぶこと）。
        """
        # 起動直後の挨拶にも間に合うよう、1回目の読み込みだけは少し待つ
        self._loaded.wait(self._cfg.timeout_sec + 2)
        now = time.time()
        with self._lock:
            used = set(self._used)
            pools = [
                [it for it in items if self._ok(it, used, now)] for items in self._feeds.values()
            ]
        pools = [p for p in pools if p]
        random.shuffle(pools)
        for pool in pools:
            # 新しい順の上位から少しばらつかせて選ぶ（いつも最新の1件だと、フィードの先頭の速報に偏る）
            pool.sort(key=lambda it: it.published or 0.0, reverse=True)
            for item in random.sample(pool[:5], k=min(5, len(pool))):
                self._mark_used(item)
                if not item.summary:
                    item.summary = fetch_page_summary(self._session, item.url, self._cfg.timeout_sec)
                    if item.summary and self._skipped(item.summary):
                        continue
                item.summary = item.summary[:_SUMMARY_CHARS]
                return item
        return None

    def _mark_used(self, item: NewsItem) -> None:
        with self._lock:
            self._used.append(title_key(item.title))
            self._used = self._used[-_USED_KEEP:]
            data = json.dumps(self._used, ensure_ascii=False)
        self._store.set_state(_USED_KEY, data)
