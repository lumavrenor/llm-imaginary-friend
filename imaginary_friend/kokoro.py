"""英語版の音声合成: Kokoro-82M（ONNX）を CPU で直接叩く（[tts] backend = "kokoro"）。

作り:

- ``kokoro-onnx`` パッケージは使わない（phonemizer-fork(GPL-3.0) と espeakng-loader を必須依存で引くため）。
  モデルの入力は tokens / style / speed の3つだけなので、onnxruntime を直接呼ぶ
- G2P は misaki（espeak 抜き）。辞書外語は g2p_fallback.py（g2p_en）で読む。
  fallback を差さないと、辞書外語で TypeError が飛んで、その文が丸ごと無音になる
- モデル一式（kokoro-v1.0.onnx 310MB / voices-v1.0.bin 27MB / config.json）は、
  無ければ初回起動時に [kokoro] model_dir へ取ってくる
- 量子化モデルは使わない（AVX-512 VNNI の無い CPU では fp32 より遅い）

声の作り分け
------------
Kokoro には VOICEVOX のような感情スタイルが無い。代わりに英語ボイスのスタイルベクトルを
**加重平均で配合**できるので、[[cast]] kokoro_voice に配合レシピを書いてキャラ固有の声を作る
（既定の af_heart 自体が Bella+Sarah の 50:50）。気持ちごとの声は kokoro_emotions に配合を書き、
書いていない気持ちは話速だけを変える（[kokoro] emotion_prosody）。

口パクは VOICEVOX のモーラ長の代わりに、合成した音の大きさから作る（voice.envelope_timeline）。
GPU は LLM が使っているので、ここは常に CPU で動かす。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .voice import _PROSODY, SAMPLE_RATE, VoiceError, envelope_timeline

if TYPE_CHECKING:
    from .config import CastMember, KokoroConfig

logger = logging.getLogger(__name__)

# Kokoro の出力は 24kHz。VOICEVOX と同じなので Player はそのまま使える
assert SAMPLE_RATE == 24000

# style 配列の第0次元。系列長で引くので、これがトークン数の上限でもある
_MAX_STYLE_LEN = 510
_MAX_TOKENS = _MAX_STYLE_LEN - 2  # 開始・終了の 0 を入れるぶん

_MODEL_NAME = "kokoro-v1.0.onnx"
_VOICES_NAME = "voices-v1.0.bin"
_CONFIG_NAME = "config.json"
_MODEL_URLS = {
    _MODEL_NAME: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    _VOICES_NAME: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
    _CONFIG_NAME: "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json",
}
_USER_AGENT = "llm-imaginary-friend/0.1 (desktop chat app; kokoro model fetch)"
_TIMEOUT = (10, 600)  # モデルが 310MB あるので読み取りは長めに

# 気持ちの名前（config の kokoro_emotions のキー）。英語でも日本語の1字でも書ける
_EMOTION_KEYS = {
    "happy": "喜", "cheerful": "楽", "angry": "怒", "sad": "哀", "surprised": "驚", "shy": "照",
    "喜": "喜", "楽": "楽", "怒": "怒", "哀": "哀", "驚": "驚", "照": "照",
}

INSTALL_HINT = (
    "The English edition needs extra packages: "
    ".venv\\Scripts\\python -m pip install -r requirements-en.txt "
    "&& .venv\\Scripts\\python -m spacy download en_core_web_sm "
    "(go.bat en / ./go.sh en installs them on first run)"
)


def emotion_key(name: str) -> str | None:
    """kokoro_emotions のキーを、内部の気持ちの名前（喜・怒…）にする。知らない名前なら None。"""
    return _EMOTION_KEYS.get(str(name).strip().lower()) or _EMOTION_KEYS.get(str(name).strip())


def _recipe(spec) -> list[tuple[str, float]]:
    """kokoro_voice の書き方（"af_heart" か [{voice=..., weight=...}, ...]）を (声, 重み) の列にする。"""
    if isinstance(spec, str):
        return [(spec, 1.0)] if spec else []
    out = []
    for part in spec or []:
        if isinstance(part, str):
            out.append((part, 1.0))
        elif isinstance(part, dict) and part.get("voice"):
            out.append((str(part["voice"]), float(part.get("weight", 1.0))))
        else:
            raise VoiceError(f"kokoro voice recipe not understood: {part!r} (write {{ voice = \"af_heart\", weight = 0.5 }})")
    return out


class _Voice:
    __slots__ = ("label", "style", "speed")

    def __init__(self, label: str, style: np.ndarray, speed: float):
        self.label = label
        self.style = style  # (510, 1, 256)。合成時に系列長で引く
        self.speed = speed


class Kokoro:
    def __init__(self, config: KokoroConfig):
        self._cfg = config
        self._model_dir = Path(config.model_dir)
        self._threads = max(0, int(config.threads))
        self._session = None
        self._voices = None
        self._vocab: dict[str, int] = {}
        self._g2p = None
        self._token_input = "tokens"
        # onnxruntime のセッションはスレッド安全だが、G2P（spaCy）は安全でないので合成全体を直列にする
        self._lock = threading.Lock()
        # キャラ id → {気持ち（"" は ふだん）: 声}
        self._cast: dict[str, dict[str, _Voice]] = {}

    # --- 起動 -----------------------------------------------------------------

    def load(self) -> str:
        """モデルを（無ければ取ってきて）読み込み、1回だけ試しに合成する。戻り値はログ用の名前。

        追加の依存が入っていなければ ImportError をそのまま投げる（main.py が入れ方を出して止める）。
        """
        self._ensure_loaded()
        with self._lock:
            pcm = self._synth_chunk("Ready.", self._voices["af_heart"], 1.0)
        if pcm.size == 0:
            raise VoiceError("Kokoro's test synthesis came out empty")
        import onnxruntime as ort

        return f"Kokoro-82M (onnxruntime {ort.__version__}, threads={self._threads or 'auto'})"

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        import json

        import onnxruntime as ort

        _fetch_models(self._model_dir)
        self._vocab = json.loads((self._model_dir / _CONFIG_NAME).read_text(encoding="utf-8"))["vocab"]
        self._voices = np.load(self._model_dir / _VOICES_NAME, allow_pickle=True)
        so = ort.SessionOptions()
        if self._threads > 0:
            so.intra_op_num_threads = self._threads
            so.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(self._model_dir / _MODEL_NAME), sess_options=so, providers=["CPUExecutionProvider"]
        )
        # 配布元によって入力名が "tokens" だったり "input_ids" だったりする
        self._token_input = self._session.get_inputs()[0].name
        self._g2p = _build_g2p(self._cfg.lexicon_file)
        logger.info("Kokoro: %s（声 %d 種）", self._model_dir, len(self.voice_names()))

    def voice_names(self) -> list[str]:
        v = self._voices
        if v is None:
            return []
        return list(v.keys()) if hasattr(v, "keys") else list(v.files)

    # --- 声の解決 ---------------------------------------------------------------

    def resolve(self, cast: list[CastMember]) -> None:
        """[[cast]] の kokoro_voice / kokoro_emotions を声にする。知らない声の名前は VoiceError（起動時に止める）。"""
        self._ensure_loaded()
        available = set(self.voice_names())
        for m in cast:
            base = _recipe(m.kokoro_voice)
            if not base:
                raise VoiceError(f'{m.name}: kokoro_voice is empty (e.g. kokoro_voice = "af_heart")')
            voices = {"": self._blend(m, "", base, available, m.kokoro_speed)}
            for name, spec in m.kokoro_emotions.items():
                key = emotion_key(name)
                if key is None:
                    raise VoiceError(
                        f"{m.name}: kokoro_emotions key {name!r} is unknown "
                        "(happy / cheerful / angry / sad / surprised / shy)"
                    )
                speed = m.kokoro_speed
                if isinstance(spec, dict) and "blend" in spec:
                    speed = float(spec.get("speed", speed))
                    spec = spec["blend"]
                voices[key] = self._blend(m, name, _recipe(spec), available, speed)
            self._cast[m.id] = voices
            logger.info("Kokoro の声: %s = %s", m.name, {k or "ふだん": v.label for k, v in voices.items()})

    def _blend(self, m: CastMember, what: str, recipe: list[tuple[str, float]], available: set[str], speed: float) -> _Voice:
        total = sum(w for _, w in recipe)
        if total <= 0:
            raise VoiceError(f"{m.name}: the weights of kokoro voice {what or 'kokoro_voice'} add up to {total}")
        blended = None
        for voice, weight in recipe:
            if voice not in available:
                raise VoiceError(
                    f"{m.name}: Kokoro has no voice named {voice!r}. Available: {', '.join(sorted(available))}"
                )
            part = self._voices[voice].astype(np.float32) * (weight / total)
            blended = part if blended is None else blended + part
        label = "+".join(f"{v}:{w:g}" for v, w in recipe)
        return _Voice(label, blended, max(0.5, min(2.0, speed * self._cfg.speed)))

    # --- 合成 -----------------------------------------------------------------

    def synth_for(self, member: CastMember, text: str, emotion: str) -> tuple[np.ndarray, list[float], list[float]]:
        """キャラの声で合成する（brain から。voice.Voicevox.synth_for と同じ呼び方）。"""
        voices = self._cast.get(member.id)
        if self._session is None or not voices:
            raise VoiceError("Kokoro is not ready")
        voice = voices.get(emotion) or voices[""]
        speed = voice.speed
        if self._cfg.emotion_prosody and emotion in _PROSODY:
            # 専用の声を配合していれば、話速の変化は控えめにする（VOICEVOX のスタイルと同じ扱い）
            k = 0.3 if emotion in voices else 1.0
            speed *= 1 + (_PROSODY[emotion][2] - 1) * k
        try:
            with self._lock:
                chunks = [self._synth_chunk(part, voice.style, speed) for part in self._split(text)]
        except Exception as e:  # noqa: BLE001 - 1文の失敗で会話を止めない
            raise VoiceError(f"Kokoro synthesis failed: {type(e).__name__}: {e}") from e
        chunks = [c for c in chunks if c.size]
        if not chunks:
            raise VoiceError(f"Kokoro produced no audio for: {text!r}")
        pcm = np.concatenate(chunks).astype(np.float32)
        times, opens = envelope_timeline(pcm, SAMPLE_RATE)
        return pcm, times, opens

    def _to_ids(self, text: str) -> list[int]:
        phonemes, _ = self._g2p(text)
        return [self._vocab[c] for c in phonemes if c in self._vocab]

    def _split(self, text: str) -> list[str]:
        """音素にすると 510 トークンを超える文を、語で詰めて収める（ふつうの1文はそのまま）。"""
        if len(self._to_ids(text)) <= _MAX_TOKENS:
            return [text]
        out: list[str] = []
        buf: list[str] = []
        for word in text.split():
            trial = " ".join(buf + [word])
            if buf and len(self._to_ids(trial)) > _MAX_TOKENS:
                out.append(" ".join(buf))
                buf = [word]
            else:
                buf.append(word)
        if buf:
            out.append(" ".join(buf))
        return out

    def _synth_chunk(self, text: str, style: np.ndarray, speed: float) -> np.ndarray:
        ids = self._to_ids(text)
        if not ids:
            return np.zeros(0, dtype=np.float32)
        row = style[len(ids) + 2]  # style は系列長で引く（ここを誤ると声が変わる）
        audio = self._session.run(
            None,
            {
                self._token_input: np.array([[0, *ids, 0]], dtype=np.int64),
                "style": np.asarray(row, dtype=np.float32).reshape(1, -1),
                "speed": np.array([speed], dtype=np.float32),
            },
        )[0]
        return np.asarray(audio, dtype=np.float32).flatten()


def _build_g2p(lexicon_file: str):
    """misaki の英語 G2P を espeak 抜きで組む（辞書外語は g2p_fallback）。"""
    from misaki import en

    from .g2p_fallback import ArpabetFallback

    g2p = en.G2P(trf=False, british=False, fallback=ArpabetFallback())
    _apply_lexicon(g2p, lexicon_file)
    return g2p


def _apply_lexicon(g2p, lexicon_file: str) -> None:
    """[kokoro] lexicon_file（1行 "word<TAB>IPA"）を misaki の辞書へ流し込む。無ければ何もしない。"""
    if not lexicon_file:
        return
    path = Path(lexicon_file)
    if not path.exists():
        logger.info("[kokoro] lexicon_file が無いので、読みの固定はしない: %s", path)
        return
    golds = getattr(getattr(g2p, "lexicon", None), "golds", None)
    if golds is None:
        logger.warning("misaki の辞書の形が想定と違うので、lexicon_file を使えない")
        return
    count = 0
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t") if "\t" in line else line.split(None, 1)
        if len(parts) != 2:
            logger.warning("%s:%d: 'word<TAB>IPA' の形ではない: %r", path, lineno, raw)
            continue
        word, ipa = parts[0].strip(), parts[1].strip()
        if word and ipa:
            # misaki は大文字で始まる語（名前）を元の綴りでも引くので、両方に入れる
            golds[word.lower()] = ipa
            golds[word] = ipa
            golds[word.capitalize()] = ipa
            count += 1
    logger.info("[kokoro] lexicon_file から %d 語の読みを入れた: %s", count, path)


def _fetch_models(model_dir: Path) -> None:
    """モデル一式を（無ければ）取ってくる。"""
    import requests

    model_dir.mkdir(parents=True, exist_ok=True)
    for name, url in _MODEL_URLS.items():
        dest = model_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        logger.info("Kokoro のモデルを取ってくる（初回だけ）: %s", name)
        print(f"[first-time setup] downloading Kokoro model file {name} ...", flush=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with requests.get(url, stream=True, timeout=_TIMEOUT, headers={"User-Agent": _USER_AGENT}) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as f:
                    for block in resp.iter_content(chunk_size=1 << 20):
                        f.write(block)
            tmp.replace(dest)  # 途中で落ちた不完全なファイルを掴まないように
        except Exception as e:
            tmp.unlink(missing_ok=True)
            raise VoiceError(f"Could not download {name} ({url}): {e}") from e
        logger.info("取ってきた: %s（%.1f MB）", dest, dest.stat().st_size / (1 << 20))
