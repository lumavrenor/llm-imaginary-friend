"""天気予報（config.toml の ``[weather]``）。

取得先は Open-Meteo（https://open-meteo.com/ ）。API キーが要らず、緯度経度だけで
今の天気・気温と、今日・明日の天気・最高最低・降水確率が1リクエストで揃う。

場所は ``[weather]`` に緯度経度で明示する（IP や OS の位置情報から推測しない）。
これは「ユウが住んでいる町」という設定で、書かなければ天気には一切触れない。

使い方は2つ（brain.py）:
  - 今の空模様を1行、全員共通の「今の状況」に入れる（「おはよう、寒いね」と言えるように）。
    ユウの発言が天気・傘・明日などに触れたときは、予報も全員に渡す
  - ときどき1人が「さっき天気予報を見たら」と、自分から話題に出す（ニュースと同じ演出）

取得に失敗したとき・値が古すぎるときは :meth:`WeatherProvider.current` が ``None`` を返し、
天気には触れさせない（「たぶん晴れ」と言わせるより黙らせるほうが事故が小さい）。
取得は裏のスレッドで行い、会話のスレッドは待たせない。
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import threading
import time
from dataclasses import dataclass

import requests

from . import language
from .config import WeatherConfig

logger = logging.getLogger(__name__)

_USER_AGENT = "llm-imaginary-friend/0.1 (https://github.com/lumavrenor/llm-imaginary-friend; desktop chat app)"

# 取得に失敗してから次に試すまで
_RETRY_AFTER_SEC = 120.0
# キャッシュがここまで古くなったら捨てる（= 天気に触れさせない）。ネットが落ちたまま
# 半日動かしたとき、朝の値で「今の天気」を語らせないため
_STALE_LIMIT_SEC = 3 * 3600.0

# WMO weather code → 日本語の天気名。https://open-meteo.com/en/docs のコード表を、話す粒度に丸めた
_WMO_CODES: dict[int, str] = {
    0: "快晴",
    1: "晴れ",
    2: "薄曇り",
    3: "曇り",
    45: "霧",
    48: "霧",
    51: "霧雨",
    53: "霧雨",
    55: "強い霧雨",
    56: "凍える霧雨",
    57: "凍える霧雨",
    61: "小雨",
    63: "雨",
    65: "強い雨",
    66: "凍雨",
    67: "強い凍雨",
    71: "小雪",
    73: "雪",
    75: "大雪",
    77: "霧雪",
    80: "にわか雨",
    81: "にわか雨",
    82: "激しいにわか雨",
    85: "にわか雪",
    86: "にわか雪",
    95: "雷雨",
    96: "ひょうをともなう雷雨",
    99: "ひょうをともなう激しい雷雨",
}

_WMO_CODES_EN: dict[int, str] = {
    0: "clear skies",
    1: "sunny",
    2: "partly cloudy",
    3: "cloudy",
    45: "foggy",
    48: "foggy",
    51: "drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "showers",
    81: "showers",
    82: "heavy showers",
    85: "snow showers",
    86: "snow showers",
    95: "thunderstorms",
    96: "thunderstorms with hail",
    99: "heavy thunderstorms with hail",
}

# ユウの発言がこれに触れたら、予報も全員に渡す（「明日、傘いるかな」に答えられるように）
_ASK_RE = re.compile(
    r"天気|天候|予報|晴れ|曇|くもり|雨|雪|傘|かさ|気温|寒|暑|あつい|さむい|台風|風|洗濯|明日|あした|あす|週末"
)
_ASK_RE_EN = re.compile(
    r"\b(weather|forecast|sunny|sun|cloud\w*|rain\w*|snow\w*|umbrella|temperature|cold|chilly|hot|warm|"
    r"freezing|storm\w*|typhoon|hurricane|wind\w*|laundry|tomorrow|weekend)\b",
    re.IGNORECASE,
)


def describe_code(code: int | None) -> str:
    if code is None:
        return ""
    return language.pick(ja=_WMO_CODES, en=_WMO_CODES_EN).get(int(code), "")


def _round(value: object) -> int | None:
    return int(round(value)) if isinstance(value, (int, float)) else None


def _at(values: object, i: int) -> object:
    """Open-Meteo の daily は「日ごとの配列」で返る。"""
    if isinstance(values, list) and len(values) > i:
        return values[i]
    return None


@dataclass(frozen=True)
class Day:
    """1日ぶんの予報。"""

    description: str
    high_c: int | None
    low_c: int | None
    precip_prob: int | None  # 降水確率（％）

    def text(self) -> str:
        if language.is_en():
            # 数字は語に開かない（Kokoro の G2P が "18" を読める）。度は記号でなく語で書く
            parts = [self.description]
            if self.high_c is not None and self.low_c is not None:
                parts.append(f"high of {self.high_c} degrees Celsius, low of {self.low_c}")
            if self.precip_prob is not None:
                parts.append(f"{self.precip_prob} percent chance of rain")
            return ", ".join(p for p in parts if p)
        parts = [f"「{self.description}」" if self.description else ""]
        if self.high_c is not None and self.low_c is not None:
            parts.append(f"最高{self.high_c}度・最低{self.low_c}度")
        if self.precip_prob is not None:
            parts.append(f"降水確率{self.precip_prob}パーセント")
        return "、".join(p for p in parts if p)


@dataclass(frozen=True)
class Weather:
    """ある時点の天気。プロンプトへ渡すのはこの中身だけで、推測は足させない。"""

    location: str
    description: str  # 今の天気
    temp_c: int | None
    feels_like_c: int | None
    wind_kph: int | None
    today: Day | None
    tomorrow: Day | None
    fetched_at: float

    def now_line(self) -> str:
        """今の空模様を1文で（「東京は今「曇り」、気温14度。」）。"""
        if language.is_en():
            line = f"Right now in {self.location} it's {self.description}"
            if self.temp_c is not None:
                line += f", {self.temp_c} degrees Celsius"
                if self.feels_like_c is not None and abs(self.feels_like_c - self.temp_c) >= 2:
                    line += f" (feels like {self.feels_like_c})"
            return line + ". "
        parts = [f"{self.location}は今「{self.description}」"]
        if self.temp_c is not None:
            felt = ""
            if self.feels_like_c is not None and abs(self.feels_like_c - self.temp_c) >= 2:
                felt = f"（体感{self.feels_like_c}度）"
            parts.append(f"気温{self.temp_c}度{felt}")
        return "、".join(parts) + "。"

    def forecast_text(self) -> str:
        """今日と明日の予報を1文で（「天気予報では、今日は「晴れ」、最高18度・最低9度。明日は…」）。"""
        labels = language.pick(ja=("今日", "明日"), en=("Today", "Tomorrow"))
        days = [(label, d.text()) for label, d in zip(labels, (self.today, self.tomorrow)) if d is not None]
        days = [(label, t) for label, t in days if t]
        if not days:
            return ""
        if language.is_en():
            return "The forecast says: " + " ".join(f"{label}, {t}." for label, t in days) + " "
        return "天気予報では、" + "。".join(f"{label}は{t}" for label, t in days) + "。"

    def forecast_lines(self) -> list[str]:
        """天気予報の中身。1行1項目で、ここに無い数字は言わせない。"""
        if language.is_en():
            return self._forecast_lines_en()
        out = [f"今の天気: {self.description}"]
        if self.temp_c is not None:
            out.append(f"今の気温: {self.temp_c}度")
        if self.feels_like_c is not None and self.temp_c is not None and abs(self.feels_like_c - self.temp_c) >= 2:
            out.append(f"体感温度: {self.feels_like_c}度")
        if self.wind_kph is not None and self.wind_kph >= 20:
            out.append(f"風: 時速{self.wind_kph}キロ（強め）")
        if self.today is not None and self.today.text():
            out.append(f"今日の予報: {self.today.text()}")
        if self.tomorrow is not None and self.tomorrow.text():
            out.append(f"明日の予報: {self.tomorrow.text()}")
        return out

    def _forecast_lines_en(self) -> list[str]:
        out = [f"Weather right now: {self.description}"]
        if self.temp_c is not None:
            out.append(f"Temperature right now: {self.temp_c} degrees Celsius")
        if self.feels_like_c is not None and self.temp_c is not None and abs(self.feels_like_c - self.temp_c) >= 2:
            out.append(f"Feels like: {self.feels_like_c} degrees")
        if self.wind_kph is not None and self.wind_kph >= 20:
            out.append(f"Wind: {self.wind_kph} kilometers per hour (on the strong side)")
        if self.today is not None and self.today.text():
            out.append(f"Today's forecast: {self.today.text()}")
        if self.tomorrow is not None and self.tomorrow.text():
            out.append(f"Tomorrow's forecast: {self.tomorrow.text()}")
        return out


def asks_weather(text: str) -> bool:
    """ユウの発言が、天気や明日の空模様に触れているか。"""
    return bool(language.pick(ja=_ASK_RE, en=_ASK_RE_EN).search(text or ""))


class WeatherProvider:
    """Open-Meteo から天気を取ってきて、TTL つきで持っておく。

    :meth:`current` は待たない。古くなっていれば裏で取り直しを始め、手元の値（無ければ None）を返す。
    """

    def __init__(self, config: WeatherConfig):
        self._cfg = config
        self._refresh_sec = max(float(config.refresh_sec), 60.0)
        self._lock = threading.Lock()
        self._cache: Weather | None = None
        self._last_attempt = 0.0
        self._fetching = False
        self._session = requests.Session()
        self._session.headers["User-Agent"] = _USER_AGENT

    @property
    def location_name(self) -> str:
        return self._cfg.location_name

    def start(self) -> None:
        """起動時に1回取っておく（起動の挨拶に間に合うように）。"""
        self._kick(time.time())

    def current(self) -> Weather | None:
        """今の天気。取れていない・古すぎるときは ``None``（＝天気に触れさせない）。"""
        now = time.time()
        with self._lock:
            cache = self._cache
        if cache is None or now - cache.fetched_at >= self._refresh_sec:
            self._kick(now)
        if cache is None or now - cache.fetched_at > _STALE_LIMIT_SEC:
            return None
        return cache

    def _kick(self, now: float) -> None:
        with self._lock:
            if self._fetching or now - self._last_attempt < _RETRY_AFTER_SEC:
                return
            self._fetching = True
            self._last_attempt = now
        threading.Thread(target=self._run, name="weather", daemon=True).start()

    def _run(self) -> None:
        try:
            got = self._fetch()
        finally:
            with self._lock:
                self._fetching = False
        if got is not None:
            with self._lock:
                self._cache = got

    def _fetch(self) -> Weather | None:
        cfg = self._cfg
        params = {
            "latitude": cfg.latitude,
            "longitude": cfg.longitude,
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": 2,
        }
        try:
            resp = self._session.get(cfg.forecast_url, params=params, timeout=cfg.timeout_sec)
            resp.raise_for_status()
            data = resp.json()
            current = data["current"]
            daily = data.get("daily", {})
            days = []
            for i in range(2):
                code = _at(daily.get("weather_code"), i)
                days.append(
                    Day(
                        description=describe_code(code if isinstance(code, (int, float)) else None),
                        high_c=_round(_at(daily.get("temperature_2m_max"), i)),
                        low_c=_round(_at(daily.get("temperature_2m_min"), i)),
                        precip_prob=_round(_at(daily.get("precipitation_probability_max"), i)),
                    )
                )
            description = describe_code(current["weather_code"]) or language.pick(
                ja="はっきりしない空", en="hard-to-read skies"
            )
            weather = Weather(
                location=cfg.location_name,
                description=description,
                temp_c=_round(current.get("temperature_2m")),
                feels_like_c=_round(current.get("apparent_temperature")),
                wind_kph=_round(current.get("wind_speed_10m")),
                today=days[0],
                tomorrow=days[1],
                fetched_at=time.time(),
            )
        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            logger.warning("天気を取れませんでした（%s）: %s", cfg.location_name, e)
            return None
        logger.info("天気: %s", " / ".join(weather.forecast_lines()))
        return weather


def focus_day(now: dt.datetime | None = None) -> str:
    """予報のどちらの日を話すとよいか。夕方からは明日、それまで（深夜を含む）は今日。"""
    now = now or dt.datetime.now()
    if language.is_en():
        return "tomorrow" if now.hour >= 16 else "today"
    return "明日" if now.hour >= 16 else "今日"
