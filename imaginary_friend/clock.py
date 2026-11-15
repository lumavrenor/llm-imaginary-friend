"""時刻の言い方。プロンプトに「今は11月14日（土）の夜、21時ごろ」「きのうの夕方」のように入れる。

英語版（[locale] lang = "en"）では "It's Saturday, November 14, in the evening, around 9 pm." のように言う。
"""

from __future__ import annotations

import datetime as dt

from . import language

WEEKDAYS = "月火水木金土日"
_WEEKDAYS_EN = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_MONTHS_EN = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
# 町の出来事の時間帯と、その時間帯の代表の時刻
BANDS = {"朝": 8, "昼": 12, "夕方": 17, "夜": 20}
_BANDS_EN = {"morning": 8, "afternoon": 12, "evening": 17, "night": 20}
_LATE_EN = "late night"


def bands() -> dict[str, int]:
    """今の言語の時間帯（町の出来事の schema の enum に使う）。"""
    return language.pick(ja=BANDS, en=_BANDS_EN)


def band(t: dt.datetime) -> str:
    h = t.hour
    names = list(bands()) + [language.pick(ja="深夜", en=_LATE_EN)]
    if 5 <= h < 10:
        return names[0]
    if 10 <= h < 15:
        return names[1]
    if 15 <= h < 18:
        return names[2]
    if 18 <= h < 23:
        return names[3]
    return names[4]


def date_label(d: dt.date) -> str:
    if language.is_en():
        return f"{_WEEKDAYS_EN[d.weekday()]}, {_MONTHS_EN[d.month - 1]} {d.day}"
    return f"{d.month}月{d.day}日（{WEEKDAYS[d.weekday()]}）"


def _hour_en(h: int) -> str:
    if h == 0:
        return "midnight"
    if h == 12:
        return "noon"
    return f"{h % 12} {'am' if h < 12 else 'pm'}"


def now_text(now: dt.datetime | None = None) -> str:
    """「今は11月14日（土）の夜、21時ごろ。」"""
    now = now or dt.datetime.now()
    if language.is_en():
        return f"It's {date_label(now.date())}, {_band_phrase_en(band(now))}, around {_hour_en(now.hour)}. "
    return f"今は{date_label(now.date())}の{band(now)}、{now.hour}時ごろ。"


def _band_phrase_en(b: str) -> str:
    return "at night" if b in ("night", _LATE_EN) else f"in the {b}"


def when_text(ts: float, now: dt.datetime | None = None) -> str:
    """出来事の時刻を、今から見た言い方にする（「けさ」「きのうの夜」「11月10日の昼」）。"""
    now = now or dt.datetime.now()
    t = dt.datetime.fromtimestamp(ts)
    days = (now.date() - t.date()).days
    b = band(t)
    if language.is_en():
        return _when_en(days, b, t)
    if days <= 0:
        return {"朝": "けさ", "深夜": "ゆうべ遅く"}.get(b, f"今日の{b}")
    if days == 1:
        return {"夜": "ゆうべ", "深夜": "ゆうべ遅く"}.get(b, f"きのうの{b}")
    if days == 2:
        return f"おとといの{b}"
    if days < 7:
        return f"{days}日前の{b}"
    return f"{t.month}月{t.day}日の{b}"


def _when_en(days: int, b: str, t: dt.datetime) -> str:
    if days <= 0:
        return {"morning": "this morning", "evening": "this evening", "night": "tonight", _LATE_EN: "late last night"}.get(
            b, f"this {b}"
        )
    if days == 1:
        return {"night": "last night", _LATE_EN: "late last night"}.get(b, f"yesterday {b}")
    if days < 7:
        return f"{_WEEKDAYS_EN[t.weekday()]} {b}" if days < 6 else f"{days} days ago, {_band_phrase_en(b)}"
    return f"{_MONTHS_EN[t.month - 1]} {t.day}, {_band_phrase_en(b)}"


def gap_text(since: float | None, now: dt.datetime | None = None) -> str:
    """前にユウが来てからの間隔（「3日ぶり」）。2時間未満や初めてのときは空。"""
    if since is None:
        return ""
    now = now or dt.datetime.now()
    t = dt.datetime.fromtimestamp(since)
    if (now - t).total_seconds() < 2 * 3600:
        return ""
    days = (now.date() - t.date()).days
    if language.is_en():
        if days <= 0:
            return "the first time in a few hours"
        if days == 1:
            return "the first time since yesterday"
        if days < 14:
            return f"the first time in {days} days"
        return "the first time in over two weeks, a long while"
    if days <= 0:
        return "数時間ぶり"
    if days == 1:
        return "きのう以来"
    if days < 14:
        return f"{days}日ぶり"
    return "2週間以上ぶりの、ひさしぶり"
