"""config.toml の読み込み。"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .display.models import known_accessories, known_hair_styles, known_models
from .drama import EMOTIONS


class ConfigError(ValueError):
    pass


SEARCH_BACKENDS = ("wikipedia", "news", "brave", "searxng")
TTS_BACKENDS = ("voicevox", "kokoro")
LANGUAGES = ("ja", "en")


@dataclass
class LLMProfile:
    name: str
    engine: str
    host: str
    model: str = ""
    api_key: str = ""
    display_model: str = ""  # 6人目に渡すモデル名（別名しか取れない接続先用。空なら接続先から取る）


@dataclass
class LLMConfig:
    profile: str = "auto"
    temperature: float = 0.9
    timeout_sec: int = 120
    profiles: list[LLMProfile] = field(default_factory=list)


@dataclass
class VoicevoxConfig:
    host: str = "http://127.0.0.1:50021"
    speed_scale: float = 1.15
    volume: float = 0.9
    gap_sec: float = 0.25
    emotion_prosody: bool = True  # 感情に合わせて声の高さ・抑揚・速さを変える


@dataclass
class KokoroConfig:
    """英語版の音声合成（kokoro.py。[tts] backend = "kokoro"）。"""

    model_dir: str = "kokoro_models"  # モデル一式の置き場所（無ければ初回起動時に取ってくる。約340MB）
    threads: int = 2  # onnxruntime のスレッド数（0 なら任せる）。全コアを渡すと LLM と画面に響く
    lexicon_file: str = "lexicon.en.txt"  # 読みを固定したい語（1行 "word<TAB>IPA"）
    speed: float = 1.0  # 全員の話速に掛ける倍率
    volume: float = 0.9
    gap_sec: float = 0.25
    emotion_prosody: bool = True  # 気持ちに合わせて話速を少し変える


@dataclass
class TalkConfig:
    speakers_per_turn: int = 2
    max_chars: int = 150
    max_sentences: int = 4
    history_lines: int = 30
    history_chars: int = 3000
    session_min_sec: float = 180.0
    session_max_sec: float = 300.0
    lookahead_sec: float = 20.0
    idle_chat: bool = True
    idle_min_sec: float = 12.0
    idle_max_sec: float = 28.0
    max_auto_rounds: int = 4
    greet_on_start: bool = True


@dataclass
class SearchConfig:
    enabled: bool = True
    # 調べるときに引く先。複数書くと並列に引いて結果を合わせる（文字列1つでもよい）
    backend: list[str] = field(default_factory=lambda: ["wikipedia", "news"])  # wikipedia / news / brave / searxng
    lang: str = "ja"
    max_results: int = 3
    timeout_sec: float = 8.0
    api_key: str = ""  # brave 用（空なら環境変数 BRAVE_API_KEY）
    host: str = ""  # searxng 用
    max_per_session: int = 3  # 話し込み中に分身が自分から検索してよい回数（ユーザーへの返答の検索は数えない）
    # 話していない子が、ほかの子の発言中に裏で調べておく（挙手の JSON で調べたいことを聞くので LLM の呼び出しは増えない）
    background: bool = True
    background_per_round: int = 1  # 1ラウンドで裏で調べてよい人数
    background_per_session: int = 4  # 1回の話し込み（またはユーザーの発言から次の発言まで）で裏で調べてよい回数
    show_background: bool = True  # 「○○がこっそり「…」を調べた」を画面に出す


@dataclass
class NewsConfig:
    enabled: bool = True  # 分身が自分から「ネットで見かけたニュース」を話題に持ち込む
    feeds: list[str] = field(
        default_factory=lambda: [
            "https://www.nhk.or.jp/rss/news/cat0.xml",
            "https://news.yahoo.co.jp/rss/topics/entertainment.xml",
            "https://news.yahoo.co.jp/rss/topics/science.xml",
            "https://gigazine.net/news/rss_2.0/",
        ]
    )
    refresh_min: float = 30.0  # RSS を読み直す間隔（分）
    timeout_sec: float = 8.0
    max_age_hours: float = 48.0  # これより古い記事は話題にしない
    start_chance: float = 0.5  # 起動時の挨拶で、見かけたニュースの話を出す確率
    idle_chance: float = 0.4  # 自発的な雑談のラウンドで、誰かが見かけたニュースの話を出す確率
    talk_chance: float = 0.15  # 話し込みの後半のラウンドで、話題を足すためにニュースの話を出す確率（1回の話し込みで1回まで）
    # 見出しか説明にこの語を含む記事は話題にしない（明るい4コマの子たちが【喜】で話すと気まずい話題）
    skip_words: list[str] = field(
        default_factory=lambda: [
            "死亡", "死去", "死者", "遺体", "殺", "事故", "事件", "逮捕", "容疑", "被害", "火災",
            "地震", "津波", "戦争", "攻撃", "爆発", "自殺", "虐待", "性的", "不倫", "訃報",
            "病気", "闘病", "難病", "診断", "障害", "いじめ", "感染", "ウイルス",
        ]
    )


@dataclass
class WeatherConfig:
    """天気予報（weather.py）。場所は緯度経度で明示する（推測しない）。書かなければ天気には触れない。"""

    enabled: bool = False
    location_name: str = ""  # 会話で呼ぶ地名（緯度経度は読み上げられないので必須）
    latitude: float | None = None
    longitude: float | None = None
    refresh_sec: float = 1800.0  # 取り直す間隔（天気は変化が遅いので取りに行きすぎない）
    timeout_sec: float = 5.0
    in_situation: bool = True  # 今の空模様を1行、全員共通の「今の状況」に入れる
    start_chance: float = 0.3  # 起動時の挨拶で、天気予報の話を出す確率
    idle_chance: float = 0.15  # 自発的な雑談のラウンドで、誰かが天気予報の話を出す確率
    talk_chance: float = 0.1  # 話し込みの後半のラウンドで出す確率（1回の話し込みで1回まで）
    min_gap_min: float = 90.0  # 天気予報の話を自分から出したあと、次に出すまでの間隔（分。再起動をまたぐ）
    forecast_url: str = "https://api.open-meteo.com/v1/forecast"  # 差し替え用（ふつうは書かない）


@dataclass
class MemoryConfig:
    enabled: bool = True
    reflect_min_turns: int = 6  # まだ記憶にしていない発言がこの数以上たまったら、区切りで記憶メモを書く
    facts_in_prompt: int = 2  # プロンプトに入れる「自分で調べたこと」の件数
    notes_in_prompt: int = 3  # プロンプトに入れる「前の会話の記憶」の件数
    min_score: float = 0.08  # 話題との近さがこれ未満の記憶は入れない（いちばん新しい1件は除く）


@dataclass
class DramaConfig:
    enabled: bool = True  # 喜怒哀楽のタグ・今日の気分・関係の変化
    beats: bool = True  # 話し込みに起承転結の演出（口ゲンカ・打ち明け話・仲直りなど）を仕掛ける
    idle_beat_chance: float = 0.6  # 自発的な雑談のラウンドで演出を仕掛ける確率
    light_beat_chance: float = 0.2  # 話し込みの「転」「結」以外のラウンドで、小さな笑い（ボケ・ほめ・からかい）を入れる確率
    show_mood: bool = True  # 今日の気分が決まったら、ログへ出す（町の出来事は出さない）


@dataclass
class Place:
    id: str
    name: str
    desc: str = ""


@dataclass
class TownConfig:
    enabled: bool = True  # 町（イマジナリーランド）での5人の暮らし。比較試験では false にする
    name: str = "イマジナリーランド"
    events_per_day: float = 3.0  # 1日に先に作っておく出来事の数（5人合計）
    max_events_per_launch: int = 6  # 起動時にまとめて作る上限（何日も空いたとき用）
    life_in_prompt: int = 2  # 発話のプロンプトに入れる、まだ話していない出来事の数
    secret_ratio: float = 0.2  # ないしょの出来事の割合の上限（モデルに任せず、コードで切る）
    fresh_days: int = 14  # これより前の、まだ話していない出来事はプロンプトに入れない
    told_overlap: float = 0.35  # セリフが出来事の文の2文字をこの割合以上含んだら「話した」とみなす
    places: list[Place] = field(default_factory=list)


@dataclass
class MetaConfig:
    """6人目「モデルの子」（meta.py、docs/2026-11-15-llmイマジナリーフレンド.md）。"""

    enabled: bool = False
    show_model_name: bool = True  # 画面の名前欄にモデル名を添える
    model_label: str = ""  # 名前欄に出すモデル名（空なら接続先の名前を自動で短くする）
    appear_weight: float = 0.4  # 発話者に選ばれやすさの倍率（ほかの5人は 1.0）
    cue_weight: float = 1.5  # きっかけがあるラウンドの倍率
    stats_chance: float = 0.25  # きっかけの無いラウンドで実行状況を渡す確率
    slow_sec: float = 8.0  # 1回の生成がこの秒数を超えたら「遅かった」
    stats_window: int = 20  # 実行状況の平均を取る直近のリクエスト数
    max_streak: int = 2  # 続けて話してよいラウンドの数
    absent_note: bool = True  # 「◯日ぶり」「前回は別のモデル」を渡す
    place: str = "denkishitsu"  # 居場所（[[town.places]] の id）
    self_desc: str = ""  # 本人だけが知っていること（本人のプロンプトにだけ入る）
    member_id: str = ""  # 組み立てた6人目の id（設定には書かない）


@dataclass
class DisplayConfig:
    font: str = "C:/Windows/Fonts/meiryo.ttc"
    width: int = 1280
    height: int = 720
    fps_limit: int = 30
    ime_aware: bool = True
    layout_jitter: float = 1.0  # 立ち位置の乱数のずれ（0 で整列、1 で既定。起動ごとに変わる）
    expressions: bool = False  # True で喜怒哀楽の表情（眉・ほお・涙・漫符）を頭に重ねる。既定は表情なし
    hand_raise: bool = False  # True で喋りたい度合いが高い子が、指名されるまで片手を挙げる（見た目だけ。挙手の仕組みは変わらない）


@dataclass
class Accessory:
    type: str
    color: str | None = None
    side: str | None = None


@dataclass
class CastMember:
    id: str
    name: str
    desc: str
    model: str = "girl"
    hair_style: str | None = None
    hair_color: str | None = None
    eye_color: str | None = None
    clothing_color: str | None = None
    accessory: list[Accessory] = field(default_factory=list)
    voicevox_speaker_name: str = ""
    voicevox_style: str = "ノーマル"
    search_line: str = "ちょっと調べてみるね。"  # 検索前に何も言わなかったときのつなぎの一言
    examples: list[str] = field(default_factory=list)  # 話し方の例（口調の手本。内容は真似させない）
    voicevox_emotions: dict[str, str] = field(default_factory=dict)  # 感情 → VOICEVOX のスタイル名（例: 哀 = "泣き"）
    voice_id: int = 0  # 起動時に /speakers と突き合わせて解決する
    emotion_voice_ids: dict[str, int] = field(default_factory=dict)  # 同上（voicevox_emotions の解決結果）
    # 英語版（Kokoro）の声。"af_heart" か、配合 [{ voice = "af_sky", weight = 0.6 }, ...]
    kokoro_voice: str | list = ""
    kokoro_speed: float = 1.0
    # 気持ちごとの声（キーは happy / cheerful / angry / sad / surprised / shy。値は kokoro_voice と同じ書き方）
    kokoro_emotions: dict[str, object] = field(default_factory=dict)


@dataclass
class Config:
    user_name: str
    llm: LLMConfig
    voicevox: VoicevoxConfig
    talk: TalkConfig
    display: DisplayConfig
    search: SearchConfig
    news: NewsConfig
    weather: WeatherConfig
    memory: MemoryConfig
    drama: DramaConfig
    town: TownConfig
    meta: MetaConfig
    store_path: Path
    cast: list[CastMember]  # 6人目がいるときは最後に入る
    affinity: dict[str, dict[str, int]]
    lang: str = "ja"  # [locale] lang。会話の言語（language.py）
    tts_backend: str = "voicevox"  # [tts] backend
    kokoro: KokoroConfig = field(default_factory=KokoroConfig)
    # 画面の名前欄だけで使う表示名（main.py が接続後に入れる。無ければ name）
    labels: dict[str, str] = field(default_factory=dict)

    def cast_by_id(self) -> dict[str, CastMember]:
        return {m.id: m for m in self.cast}

    @property
    def town_cast(self) -> list[CastMember]:
        """町で暮らす5人（6人目を除く）。"""
        return [m for m in self.cast if m.id != self.meta.member_id]


def _pick(cls, table: dict, section: str):
    known = cls.__dataclass_fields__
    unknown = set(table) - set(known)
    if unknown:
        raise ConfigError(f"[{section}] に未知のキーがあります: {', '.join(sorted(unknown))}")
    return cls(**table)


def _member(c: dict, section: str) -> CastMember:
    c = dict(c)
    acc = [_pick(Accessory, a, "cast.accessory") for a in c.pop("accessory", [])]
    m = _pick(CastMember, c, section[1:-1])
    m.accessory = acc
    if m.model not in known_models():
        raise ConfigError(f"{section} {m.id}: model={m.model!r} は未知です（{known_models()}）")
    if m.hair_style and m.hair_style not in known_hair_styles():
        raise ConfigError(f"{section} {m.id}: hair_style={m.hair_style!r} は未知です")
    for a in acc:
        if a.type not in known_accessories():
            raise ConfigError(f"{section} {m.id}: accessory type={a.type!r} は未知です")
    for e in m.voicevox_emotions:
        if e not in EMOTIONS:
            raise ConfigError(f"{section} {m.id}: voicevox_emotions の {e!r} は未知です（{' / '.join(EMOTIONS)}）")
    return m


def load(path: str | Path) -> Config:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    lang = str(raw.get("locale", {}).get("lang", "ja")).strip().lower()
    if lang not in LANGUAGES:
        raise ConfigError(f"[locale] lang = {lang!r} は未知です（{' / '.join(LANGUAGES)}）")
    tts_raw = dict(raw.get("tts", {}))
    tts_backend = str(tts_raw.pop("backend", "kokoro" if lang == "en" else "voicevox"))
    if tts_raw:
        raise ConfigError(f"[tts] に未知のキーがあります: {', '.join(sorted(tts_raw))}")
    if tts_backend not in TTS_BACKENDS:
        raise ConfigError(f"[tts] backend = {tts_backend!r} は未知です（{' / '.join(TTS_BACKENDS)}）")

    llm_raw = dict(raw.get("llm", {}))
    profiles = [
        LLMProfile(
            name=name,
            engine=tbl.get("engine", name),
            host=str(tbl.get("host", "")).rstrip("/"),
            model=tbl.get("model", ""),
            api_key=tbl.get("api_key", ""),
            display_model=tbl.get("display_model", ""),
        )
        for name, tbl in llm_raw.items()
        if isinstance(tbl, dict)
    ]
    for p in profiles:
        if not p.host:
            raise ConfigError(f"[llm.{p.name}] に host がありません")
    if not profiles:
        raise ConfigError("[llm.<名前>] の接続先が1つもありません")
    llm = LLMConfig(
        profiles=profiles,
        **{k: v for k, v in llm_raw.items() if not isinstance(v, dict)},
    )
    if llm.profile != "auto" and llm.profile not in {p.name for p in profiles}:
        raise ConfigError(f"[llm] profile = {llm.profile!r} に対応する [llm.{llm.profile}] がありません")

    cast = [_member(c, "[[cast]]") for c in raw.get("cast", [])]
    if not cast:
        raise ConfigError("[[cast]] が1人もいません")

    # 6人目「モデルの子」。[[cast]] には書かず、[meta] から組み立てて最後に足す
    meta_raw = dict(raw.get("meta", {}))
    meta_cast = meta_raw.pop("cast", None)
    meta = _pick(MetaConfig, meta_raw, "meta")
    meta.member_id = ""
    meta_id = str((meta_cast or {}).get("id", "llm"))
    if meta.enabled:
        if not meta_cast:
            raise ConfigError("[meta] enabled = true なのに [meta.cast] がありません")
        m = _member({"id": "llm", **meta_cast}, "[meta.cast]")
        meta.member_id = m.id
        meta.stats_window = max(1, meta.stats_window)
        meta.max_streak = max(1, meta.max_streak)
        cast.append(m)
    ids = [m.id for m in cast]
    if len(set(ids)) != len(ids):
        raise ConfigError("[[cast]] と [meta.cast] の id が重複しています")

    affinity = {a: dict(row) for a, row in raw.get("affinity", {}).items()}
    if not meta.enabled:
        # 6人目を切っているときは、6人目の行と列を無視する（config を書き換えずに切り替えられるように）
        affinity = {a: {b: v for b, v in row.items() if b != meta_id} for a, row in affinity.items() if a != meta_id}
    for a, row in affinity.items():
        for b in (a, *row):
            if b not in ids:
                raise ConfigError(f"[affinity] に未知のキャラ id があります: {b!r}")

    talk = _pick(TalkConfig, raw.get("talk", {}), "talk")
    talk.speakers_per_turn = max(1, min(talk.speakers_per_turn, len(cast)))
    talk.max_sentences = max(1, talk.max_sentences)
    if talk.session_max_sec < talk.session_min_sec:
        talk.session_max_sec = talk.session_min_sec

    search = _pick(SearchConfig, raw.get("search", {}), "search")
    search.host = search.host.rstrip("/")
    search.background_per_round = max(0, search.background_per_round)
    search.background_per_session = max(0, search.background_per_session)
    if isinstance(search.backend, str):
        search.backend = [b.strip() for b in search.backend.split(",")]
    search.backend = [b for b in dict.fromkeys(search.backend) if b]
    for b in search.backend:
        if b not in SEARCH_BACKENDS:
            raise ConfigError(f"[search] backend の {b!r} は未知です（{' / '.join(SEARCH_BACKENDS)}）")
    if not search.backend:
        raise ConfigError("[search] backend が空です")

    news = _pick(NewsConfig, raw.get("news", {}), "news")

    weather = _pick(WeatherConfig, raw.get("weather", {}), "weather")
    if weather.enabled:
        if not weather.location_name or weather.latitude is None or weather.longitude is None:
            raise ConfigError("[weather] enabled = true なら location_name・latitude・longitude が必要です")
        if not (-90 <= weather.latitude <= 90 and -180 <= weather.longitude <= 180):
            raise ConfigError("[weather] latitude は -90〜90、longitude は -180〜180 で書いてください")

    town_raw = dict(raw.get("town", {}))
    places = [_pick(Place, p, "town.places") for p in town_raw.pop("places", [])]
    town = _pick(TownConfig, town_raw, "town")
    town.places = places
    if town.enabled and not places:
        raise ConfigError("[town] enabled = true なのに [[town.places]] が1つもありません")
    if len({p.id for p in places}) != len(places):
        raise ConfigError("[[town.places]] の id が重複しています")
    if meta.enabled and town.enabled and meta.place not in {p.id for p in places}:
        raise ConfigError(f"[meta] place = {meta.place!r} に対応する [[town.places]] がありません")

    if tts_backend == "kokoro":
        missing = [m.name for m in cast if not m.kokoro_voice]
        if missing:
            raise ConfigError(f"[tts] backend = \"kokoro\" なのに kokoro_voice が無いキャラがいます: {', '.join(missing)}")

    return Config(
        user_name=raw.get("user", {}).get("name", "you" if lang == "en" else "あなた"),
        llm=llm,
        voicevox=_pick(VoicevoxConfig, raw.get("voicevox", {}), "voicevox"),
        talk=talk,
        display=_pick(DisplayConfig, raw.get("display", {}), "display"),
        search=search,
        news=news,
        weather=weather,
        memory=_pick(MemoryConfig, raw.get("memory", {}), "memory"),
        drama=_pick(DramaConfig, raw.get("drama", {}), "drama"),
        town=town,
        meta=meta,
        store_path=Path(raw.get("store", {}).get("path", "data/imaginary_friend.db")),
        cast=cast,
        affinity=affinity,
        lang=lang,
        tts_backend=tts_backend,
        kokoro=_pick(KokoroConfig, raw.get("kokoro", {}), "kokoro"),
    )
