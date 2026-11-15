"""llm-imaginary-friend エントリポイント。

    python -m imaginary_friend.main                 3D ウィンドウで起動
    python -m imaginary_friend.main --cli           ウィンドウなし（標準入力で会話。動作確認用）
"""

from __future__ import annotations

import argparse
import atexit
import logging
import queue
import sys

from . import config as config_mod
from . import language
from .brain import USER, Brain
from .drama import NEUTRAL, emotion_label
from .llm import LLMClient, LLMError, short_model_name
from .news import NewsFeed
from .search import SearchError, WebSearch
from .store import Store
from .voice import Player, Utterance, VoiceError, Voicevox
from .weather import WeatherProvider

logger = logging.getLogger(__name__)


def _resolve_voices(cfg: config_mod.Config, voicevox: Voicevox) -> str | None:
    """[[cast]] の声の名前を話者 ID へ解決する。VOICEVOX が居なければ警告文を返す。"""
    try:
        version = voicevox.version()
        speakers = voicevox.speakers()
    except Exception as e:  # noqa: BLE001
        return f"VOICEVOX に接続できません（{cfg.voicevox.host}）。音声なしで続けます: {type(e).__name__}"
    logger.info("VOICEVOX %s", version)
    problems = []
    for m in cfg.cast:
        styles = speakers.get(m.voicevox_speaker_name)
        if styles and m.voicevox_style in styles:
            m.voice_id = styles[m.voicevox_style]
        elif styles:
            m.voice_id = next(iter(styles.values()))
            problems.append(f"{m.name}: スタイル「{m.voicevox_style}」が無いので既定スタイルを使います")
        else:
            m.voice_id = next(iter(next(iter(speakers.values())).values()))
            problems.append(f"{m.name}: 「{m.voicevox_speaker_name}」が VOICEVOX に無いので別の声を使います")
        # 感情専用のスタイル（無ければ、ふだんの声の高さと抑揚を変えて表す）
        for emotion, style in m.voicevox_emotions.items():
            if styles and style in styles:
                m.emotion_voice_ids[emotion] = styles[style]
            else:
                problems.append(f"{m.name}: {emotion}のスタイル「{style}」が無いのでふだんの声を使います")
    return " / ".join(problems) or None


def _start_kokoro(cfg: config_mod.Config):
    """英語版の声（Kokoro）を読み込む。戻り値は (合成器, 画面に出す警告か None)。

    追加の依存が無い・声の名前が違うときは止める（英語版は声が出ないと成り立たないので、先に直してもらう）。
    モデルを取ってこられなかったときは、音声なしで続ける（VOICEVOX が居ないときと同じ）。
    """
    from .kokoro import INSTALL_HINT, Kokoro

    tts = Kokoro(cfg.kokoro)
    try:
        label = tts.load()
    except ImportError as e:
        sys.exit(f"{INSTALL_HINT}\n  ({type(e).__name__}: {e})")
    except (VoiceError, OSError) as e:
        return tts, f"Could not start Kokoro TTS; continuing without voice: {e}"
    logger.info("TTS: %s", label)
    try:
        tts.resolve(cfg.cast)
    except VoiceError as e:
        sys.exit(f"config error: {e}")
    return tts, None


def _voice_test(cfg: config_mod.Config, out_dir: str) -> None:
    """各キャラの examples（と気持ちごとの声）を合成して WAV に書く。LLM には繋がない。"""
    import time
    import wave
    from pathlib import Path

    import numpy as np

    from .voice import SAMPLE_RATE

    if cfg.tts_backend == "kokoro":
        tts, warning = _start_kokoro(cfg)
    else:
        tts = Voicevox(cfg.voicevox.host, cfg.voicevox.speed_scale, emotion_prosody=cfg.voicevox.emotion_prosody)
        warning = _resolve_voices(cfg, tts)
    if warning:
        print(warning)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for m in cfg.cast:
        lines = [(NEUTRAL, s) for s in m.examples] + [(NEUTRAL, m.search_line)]
        emotions = list(m.voicevox_emotions) if cfg.tts_backend == "voicevox" else []
        if cfg.tts_backend == "kokoro":
            from .kokoro import emotion_key

            emotions = [k for k in (emotion_key(e) for e in m.kokoro_emotions) if k]
        lines += [(e, m.examples[0] if m.examples else m.search_line) for e in emotions]
        for i, (emotion, text) in enumerate(lines):
            t0 = time.perf_counter()
            try:
                pcm, _, _ = tts.synth_for(m, text, emotion)
            except VoiceError as e:
                print(f"{m.name}: {e}")
                continue
            took = time.perf_counter() - t0
            tag = "" if emotion == NEUTRAL else f"_{emotion_label(emotion)}"
            path = out / f"{m.id}_{i}{tag}.wav"
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
            sec = len(pcm) / SAMPLE_RATE
            print(f"{path}  {sec:.1f}s (synth {took:.2f}s, RTF {took / max(sec, 0.01):.2f})  {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="5人の心の分身（と、6人目のモデルの子）と話すチャット")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--cli", action="store_true", help="ウィンドウを開かず標準入出力で会話する")
    parser.add_argument("--no-audio", action="store_true", help="音を出さない（ログと口パクは進む）")
    parser.add_argument("--no-idle", action="store_true", help="自発的な雑談と起動時の挨拶を止める")
    parser.add_argument("--no-search", action="store_true", help="ネット検索ツールを使わない")
    parser.add_argument("--no-news", action="store_true", help="分身が自分からニュースの話題を持ち込まない")
    parser.add_argument("--no-weather", action="store_true", help="天気予報を取らない（天気に触れない）")
    parser.add_argument("--log-level", default="INFO")
    # 開発用: 指定秒後にスクリーンショットを保存して終了する
    parser.add_argument("--screenshot", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--exit-after", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--say", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--voice-test",
        default=None,
        metavar="DIR",
        help="LLM を使わず、各キャラの examples を合成して DIR に WAV で保存して終わる（声の調整用）",
    )
    args = parser.parse_args()

    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    try:
        cfg = config_mod.load(args.config)
    except (config_mod.ConfigError, OSError) as e:
        sys.exit(f"設定エラー / config error: {e}")
    # プロンプト・時刻・天気などの言い方を、ここで [locale] lang に切り替える（ほかのモジュールを使う前に）
    language.set_language(cfg.lang)
    if args.no_idle:
        cfg.talk.idle_chat = False
        cfg.talk.greet_on_start = False
    if args.voice_test:
        _voice_test(cfg, args.voice_test)
        return

    try:
        llm = LLMClient(cfg.llm)
    except LLMError as e:
        sys.exit(str(e))
    meta_id = cfg.meta.member_id
    if meta_id and cfg.meta.show_model_name:
        # 6人目の名前欄には、呼び名の下に接続中のモデル名を添える（プロンプトの呼び名は変えない）
        m = cfg.cast_by_id()[meta_id]
        # 長い名前（量子化・配布元つき）は短くする。本人に渡すモデル名は元のまま
        label = cfg.meta.model_label or short_model_name(llm.model_name or llm.profile.model)
        cfg.labels[meta_id] = f"{m.name}\n{label}"

    search = None
    if cfg.search.enabled and not args.no_search:
        try:
            search = WebSearch(cfg.search)
            logger.info("検索: %s", search.label)
        except SearchError as e:
            sys.exit(f"設定エラー: {e}")

    if cfg.tts_backend == "kokoro":
        tts, voice_warning = _start_kokoro(cfg)
        volume, gap_sec = cfg.kokoro.volume, cfg.kokoro.gap_sec
    else:
        tts = Voicevox(cfg.voicevox.host, cfg.voicevox.speed_scale, emotion_prosody=cfg.voicevox.emotion_prosody)
        voice_warning = _resolve_voices(cfg, tts)
        volume, gap_sec = cfg.voicevox.volume, cfg.voicevox.gap_sec

    events: queue.Queue[tuple] = queue.Queue()
    store = Store(cfg.store_path)
    brain_ref: list[Brain] = []

    def on_player_event(kind: str, utt: Utterance) -> None:
        if kind == "start":
            events.put(("say", utt.speaker_id, utt.text, utt.turn, utt.emotion))
        if brain_ref:
            brain_ref[0].on_player_event(kind, utt)

    player = Player(
        on_player_event,
        volume=volume,
        gap_sec=gap_sec,
        audio=not args.no_audio,
    )
    atexit.register(player.close)
    news = None
    if cfg.news.enabled and cfg.news.feeds and not args.no_news:
        news = NewsFeed(cfg.news, store)
        news.start()
        logger.info("ニュース: %d 件のフィード", len(cfg.news.feeds))
    weather = None
    if cfg.weather.enabled and not args.no_weather:
        weather = WeatherProvider(cfg.weather)
        weather.start()
        logger.info("天気: %s（%.4f, %.4f）", cfg.weather.location_name, cfg.weather.latitude, cfg.weather.longitude)
    brain = Brain(cfg, llm, tts, player, store, events.put, search=search, news=news, weather=weather)
    brain_ref.append(brain)
    if voice_warning:
        events.put(("system", voice_warning))

    if args.cli:
        _run_cli(cfg, brain, events)
        return

    from .display.app import run_display

    run_display(
        cfg,
        brain,
        player,
        events,
        llm_label=llm.label,
        screenshot=args.screenshot,
        exit_after=args.exit_after,
        auto_say=args.say,
    )


def _run_cli(cfg: config_mod.Config, brain: Brain, events: queue.Queue) -> None:
    import threading

    names = {m.id: m.name for m in cfg.cast} | {USER: cfg.user_name}

    def printer() -> None:
        while True:
            ev = events.get()
            if ev[0] == "say":
                emo = ""
                if ev[4] != NEUTRAL:
                    emo = language.pick(ja=f"【{ev[4]}】", en=f" [{emotion_label(ev[4])}]")
                print(f"{names[ev[1]]}{emo}: {ev[2]}", flush=True)
            elif ev[0] == "system":
                print(f"[{ev[1]}]", flush=True)
            elif ev[0] == "hands" and ev[1]:
                label = language.pick(ja="  挙手 ", en="  hands ")
                print(label + " ".join(f"{names[k]}={v}" for k, v in ev[1].items()), flush=True)

    threading.Thread(target=printer, daemon=True).start()
    brain.start()
    for line in sys.stdin:
        line = line.strip()
        if line:
            print(f"{cfg.user_name}: {line}", flush=True)
            brain.submit(line)
    # 標準入力が閉じたら、話し終わるまで少し待って終了する
    import time

    time.sleep(1.0)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        time.sleep(0.5)
        if not brain._player.busy and brain._inbox.empty() and not _busy(brain):
            break


def _busy(brain: Brain) -> bool:
    return bool(getattr(brain, "_in_round", False))


if __name__ == "__main__":
    main()
