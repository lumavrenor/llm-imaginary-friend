"""会話の言語（config.toml の [locale] lang）。

日本語版のプロンプトは実際に話させながら詰めてきたものなので、共通の英語プロンプトへ
一本化せず、「日本語の本文はそのまま残し、英語版を隣に足して実行時に選ぶ」形にする:

    text = language.pick(
        ja=f"（従来どおりの日本語）",
        en=f"(the English version)",
    )

起動時に main.py が set_language() を1回呼ぶ。呼ばなければ日本語のまま
（未知の言語コードも日本語に倒す。黙って英語にするより、今までどおり動くほうが事故が小さい）。
"""

from __future__ import annotations

SUPPORTED = ("ja", "en")

_current = "ja"


def set_language(lang: str) -> None:
    global _current
    key = (lang or "").strip().lower()
    _current = key if key in SUPPORTED else "ja"


def current() -> str:
    return _current


def is_en() -> bool:
    return _current == "en"


def pick(*, ja, en=None):
    """今の言語の値（文字列でも、正規表現や dict でもよい）。英語版が無ければ日本語版。"""
    if _current == "en" and en is not None:
        return en
    return ja
