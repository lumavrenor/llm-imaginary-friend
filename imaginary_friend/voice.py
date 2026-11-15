"""VOICEVOX 合成と、1本のキューで直列に再生するプレイヤー。

- Voicevox.synth(): audio_query → synthesis。モーラ長から口の開きのタイムラインも作る
- Player: 合成済みのセリフを順番に再生する。再生開始・終了は on_event で通知する
  （UI のログは「再生が始まった瞬間」に出す）
"""

from __future__ import annotations

import bisect
import io
import logging
import threading
import time
import wave
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import requests

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000  # VOICEVOX の既定出力

# 感情ごとの声の変え方: (pitchScale に足す量, intonationScale, 話速の倍率)。
# 専用のスタイル（「泣き」など）が無い声でも、高さと抑揚で気持ちが伝わるようにする
_PROSODY = {
    "喜": (0.03, 1.3, 1.05),
    "楽": (0.015, 1.15, 1.02),
    "怒": (-0.01, 1.4, 1.08),
    "哀": (-0.035, 0.75, 0.9),
    "驚": (0.045, 1.45, 1.06),
    "照": (0.02, 1.1, 0.96),
}

# 母音ごとの口の開き（0〜1）
_VOWEL_OPEN = {"a": 1.0, "o": 0.8, "e": 0.6, "i": 0.4, "u": 0.35, "N": 0.12, "cl": 0.0, "pau": 0.0}


class VoiceError(RuntimeError):
    pass


@dataclass
class Utterance:
    speaker_id: str            # キャスト id
    text: str
    pcm: np.ndarray            # float32 mono, SAMPLE_RATE
    # 口の開きのタイムライン: times[i] 秒から opens[i]（次の times まで）
    times: list[float] = field(default_factory=list)
    opens: list[float] = field(default_factory=list)
    uid: int = 0               # brain が履歴と対応づける番号（文ごと）
    turn: int = 0              # 同じ発言ターンの文は同じ番号（UI が1行にまとめる）
    emotion: str = "普通"      # 喜怒哀楽（drama.EMOTIONS）。UI の表情に使う


class Voicevox:
    def __init__(self, host: str, speed_scale: float = 1.0, timeout_sec: int = 30, emotion_prosody: bool = True):
        self._host = host.rstrip("/")
        self._speed = speed_scale
        self._prosody = emotion_prosody
        self._timeout = timeout_sec
        self._session = requests.Session()

    def version(self) -> str:
        resp = requests.get(f"{self._host}/version", timeout=5)
        resp.raise_for_status()
        return resp.text.strip().strip('"')

    def speakers(self) -> dict[str, dict[str, int]]:
        """キャラ名 → {スタイル名: 話者ID}"""
        resp = requests.get(f"{self._host}/speakers", timeout=10)
        resp.raise_for_status()
        return {s["name"]: {st["name"]: st["id"] for st in s["styles"]} for s in resp.json()}

    def synth_for(self, member, text: str, emotion: str) -> tuple[np.ndarray, list[float], list[float]]:
        """キャラの声で合成する（brain から。Kokoro と同じ呼び方）。感情専用のスタイルがあればその声で。"""
        voice_id = member.emotion_voice_ids.get(emotion, member.voice_id)
        return self.synth(text, voice_id, emotion, emotion in member.emotion_voice_ids)

    def synth(
        self, text: str, voice_id: int, emotion: str = "", styled: bool = False
    ) -> tuple[np.ndarray, list[float], list[float]]:
        """styled = 感情専用のスタイルの声を使っているか（そのときは高さと抑揚を控えめに変える）。"""
        try:
            q = self._session.post(
                f"{self._host}/audio_query",
                params={"text": text, "speaker": voice_id},
                timeout=self._timeout,
            )
            q.raise_for_status()
            query = q.json()
            query["speedScale"] = self._speed
            if self._prosody and emotion in _PROSODY:
                pitch, inton, speed = _PROSODY[emotion]
                k = 0.3 if styled else 1.0
                query["pitchScale"] = query.get("pitchScale", 0.0) + pitch * k
                query["intonationScale"] = query.get("intonationScale", 1.0) * (1 + (inton - 1) * k)
                query["speedScale"] = self._speed * (1 + (speed - 1) * k)
            query["outputSamplingRate"] = SAMPLE_RATE
            query["outputStereo"] = False
            s = self._session.post(
                f"{self._host}/synthesis",
                params={"speaker": voice_id},
                json=query,
                timeout=self._timeout,
            )
            s.raise_for_status()
        except requests.RequestException as e:
            raise VoiceError(f"VOICEVOX 合成に失敗: {e}") from e
        pcm = _wav_to_float32(s.content)
        times, opens = _mouth_timeline(query)
        return pcm, times, opens


def _mouth_timeline(query: dict) -> tuple[list[float], list[float]]:
    """audio_query のモーラ長から (開始秒, 口の開き) の列を作る。"""
    speed = query.get("speedScale") or 1.0
    t = (query.get("prePhonemeLength") or 0.0) / speed
    times: list[float] = [0.0]
    opens: list[float] = [0.0]

    def add(length: float | None, value: float) -> None:
        nonlocal t
        if not length:
            return
        times.append(t)
        opens.append(value)
        t += length / speed

    for phrase in query.get("accent_phrases", []):
        for mora in phrase.get("moras", []):
            add(mora.get("consonant_length"), 0.15)
            add(mora.get("vowel_length"), _VOWEL_OPEN.get(mora.get("vowel", ""), 0.5))
        pause = phrase.get("pause_mora")
        if pause:
            add(pause.get("vowel_length"), 0.0)
    times.append(t)
    opens.append(0.0)
    return times, opens


def envelope_timeline(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE, step_sec: float = 0.04) -> tuple[list[float], list[float]]:
    """音の大きさから口の開きのタイムラインを作る（モーラ長の取れない Kokoro 用）。

    step_sec ごとの RMS を、その発言のいちばん大きいところで割って 0〜1 にする。
    """
    step = max(1, int(sample_rate * step_sec))
    n = len(pcm) // step
    if n == 0:
        return [0.0], [0.0]
    frames = pcm[: n * step].reshape(n, step)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    peak = float(np.percentile(rms, 95)) or 1.0
    opens = np.clip(rms / peak, 0.0, 1.0)
    opens[opens < 0.12] = 0.0  # 息つぎや小さな残響では口を閉じる
    times = [i * step / sample_rate for i in range(n)] + [n * step / sample_rate]
    return times, [round(float(v), 3) for v in opens] + [0.0]


def _wav_to_float32(wav_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise VoiceError(f"unsupported WAV sample width: {wf.getsampwidth()}")
        n_channels = wf.getnchannels()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1)
    return pcm


class Player:
    """セリフを1本ずつ順番に再生する。

    音は sounddevice のコールバックが _pull() で引き取る。オーディオデバイスが
    開けないとき（または audio=False）は、同じ _pull() を実時間で空回しする
    スレッドで代用するので、ログと口パクは音なしでも同じタイミングで進む。
    """

    def __init__(
        self,
        on_event: Callable[[str, Utterance], None],
        volume: float = 0.9,
        gap_sec: float = 0.25,
        audio: bool = True,
    ):
        self._on_event = on_event
        self._volume = volume
        self._gap_frames = int(gap_sec * SAMPLE_RATE)
        self._lock = threading.Lock()
        self._queue: deque[Utterance] = deque()
        self._current: Utterance | None = None
        self._pos = 0
        self._gap_left = 0
        self._stream = None
        if audio:
            try:
                import sounddevice as sd

                self._stream = sd.OutputStream(
                    samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=self._callback
                )
                self._stream.start()
            except Exception as e:  # noqa: BLE001 - デバイス無しでも会話は続けたい
                logger.warning("オーディオ出力を開けません（音なしで続行）: %s", e)
                self._stream = None
        if self._stream is None:
            threading.Thread(target=self._silent_loop, name="player-silent", daemon=True).start()

    # --- brain 側から -------------------------------------------------------

    def enqueue(self, utt: Utterance) -> None:
        with self._lock:
            self._queue.append(utt)

    def clear(self) -> list[Utterance]:
        """まだ再生を始めていないセリフを捨てて返す（再生中の1本は最後まで言わせる）。"""
        with self._lock:
            dropped = list(self._queue)
            self._queue.clear()
        return dropped

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._current is not None or bool(self._queue)

    def pending_sec(self) -> float:
        """再生し終わるまでの残り秒数（再生中の残り＋キュー＋間の無音）。"""
        with self._lock:
            frames = sum(len(u.pcm) for u in self._queue) + len(self._queue) * self._gap_frames
            if self._current is not None:
                frames += len(self._current.pcm) - self._pos
            frames += self._gap_left
        return frames / SAMPLE_RATE

    # --- UI 側から ----------------------------------------------------------

    def speaking(self) -> tuple[str | None, float]:
        """(今喋っているキャスト id, 口の開き 0〜1)"""
        with self._lock:
            cur, pos = self._current, self._pos
        if cur is None:
            return None, 0.0
        i = bisect.bisect_right(cur.times, pos / SAMPLE_RATE) - 1
        return cur.speaker_id, cur.opens[i] if 0 <= i < len(cur.opens) else 0.0

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass

    # --- 再生 ---------------------------------------------------------------

    def _pull(self, out: np.ndarray) -> None:
        """out（float32 の1次元）を埋める。イベントはロックの外で通知する。"""
        out.fill(0.0)
        events: list[tuple[str, Utterance]] = []
        n = len(out)
        filled = 0
        with self._lock:
            while filled < n:
                if self._current is None:
                    if self._gap_left > 0:
                        step = min(self._gap_left, n - filled)
                        self._gap_left -= step
                        filled += step
                        continue
                    if not self._queue:
                        break
                    self._current = self._queue.popleft()
                    self._pos = 0
                    events.append(("start", self._current))
                cur = self._current
                step = min(len(cur.pcm) - self._pos, n - filled)
                out[filled : filled + step] = cur.pcm[self._pos : self._pos + step] * self._volume
                self._pos += step
                filled += step
                if self._pos >= len(cur.pcm):
                    events.append(("end", cur))
                    self._current = None
                    self._gap_left = self._gap_frames
        for kind, utt in events:
            self._on_event(kind, utt)

    def _callback(self, outdata, frames, time_info, status) -> None:  # noqa: ARG002
        self._pull(outdata[:, 0])

    def _silent_loop(self) -> None:
        block = SAMPLE_RATE // 50
        buf = np.zeros(block, dtype=np.float32)
        next_t = time.perf_counter()
        while True:
            self._pull(buf)
            next_t += block / SAMPLE_RATE
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()
