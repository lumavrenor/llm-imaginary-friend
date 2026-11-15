"""OpenAI 互換 /v1/chat/completions クライアント（SGLang / vLLM / Ollama 共通）。

- chat():   1往復。schema を渡すと structured output（挙手フェーズ用）
- stream(): 本文をチャンクで返す（発話用。文単位で先行合成するため）。
            tools を渡すと、ツール呼び出しを最後に ToolCall として返す

thinking 対応モデルは思考でトークンを使い切るので、どのエンジンでも抑止を送る。
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

import requests

from .config import LLMConfig, LLMProfile

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)
# サーバ側のツールパーサが効かず、呼び出しが本文に生で出たとき用（Qwen3 の XML 形式と JSON 形式）
_RAW_CALL_START = "<tool_call>"
_XML_FUNC_RE = re.compile(r"<function=([\w.-]+)>(.*?)(?:</function>|$)", re.DOTALL)
_XML_PARAM_RE = re.compile(r"<parameter=([\w.-]+)>\s*(.*?)\s*(?:</parameter>|(?=<parameter=)|$)", re.DOTALL)


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

    def as_message(self) -> dict:
        """assistant メッセージの tool_calls 要素。"""
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments, ensure_ascii=False)},
        }


@dataclass
class Call:
    """1回のリクエストの記録（6人目の「いまの実行状況」の材料）。"""

    seq: int
    kind: str  # "chat" / "stream"
    tag: str  # だれの・何のための呼び出しか（キャラの名前、「挙手」など）
    sec: float
    tokens: int  # 出力トークン数（usage が無ければ、届いたチャンクの数。0 は不明）
    ok: bool
    error: str = ""  # "timeout" / "error"

    @property
    def tps(self) -> float | None:
        return self.tokens / self.sec if self.ok and self.tokens > 0 and self.sec > 0.05 else None


class RunStats:
    """LLM へのリクエストの記録。直近 window 件だけメモリに持つ（DB には保存しない）。"""

    def __init__(self, window: int = 20):
        self._lock = threading.Lock()
        self._calls: deque[Call] = deque(maxlen=max(1, window))
        self._seq = 0
        self.started = time.time()
        self.total = 0
        self.failures = 0
        self.timeouts = 0

    def set_window(self, window: int) -> None:
        with self._lock:
            self._calls = deque(self._calls, maxlen=max(1, window))

    def record(self, kind: str, tag: str, sec: float, tokens: int, ok: bool, error: str = "") -> None:
        with self._lock:
            self._seq += 1
            self.total += 1
            if not ok:
                self.failures += 1
                if error == "timeout":
                    self.timeouts += 1
            self._calls.append(Call(self._seq, kind, tag, sec, tokens, ok, error))

    def calls(self) -> list[Call]:
        with self._lock:
            return list(self._calls)


_SIZE_RE = re.compile(r"\d+(?:\.\d+)?[bBmM]")
_VERSION_RE = re.compile(r"\d+(?:\.\d+)*")


def short_model_name(name: str) -> str:
    """画面の名前欄に出す、短いモデル名。

    パラメータ数（31B など）までを残し、その後ろ（-it / -Instruct / 量子化 / 配布元）を落とす。
    先頭の「名前-版」はつなげる。パラメータ数が見つからなければそのまま返す。
      gemma-4-31B-it-NVFP4-RedHat -> gemma4-31B
      gemma4:31b                  -> gemma4-31B
      Qwen3.8-27B                 -> Qwen3.8-27B
      Llama-3.1-8B-Instruct       -> Llama3.1-8B
    """
    parts = [t for t in re.split(r"[-_:]", name) if t]
    size = next((i for i, t in enumerate(parts) if _SIZE_RE.fullmatch(t)), None)
    if size is None or size == 0:
        return name
    parts = parts[: size + 1]
    parts[-1] = parts[-1].upper()
    if len(parts) > 2 and parts[0].isalpha() and _VERSION_RE.fullmatch(parts[1]):
        parts[:2] = [parts[0] + parts[1]]
    return "-".join(parts)


class LLMClient:
    def __init__(self, config: LLMConfig):
        self._config = config
        self.stats = RunStats()
        self.profile = self._resolve_profile()
        # 本当のモデル名（SGLang の --served-model-name などで別名が付いていても、元の重みの名前）
        self.model_name = self.profile.display_model or self._real_model_name(self.profile)
        if self.model_name != self.profile.model:
            logger.info("LLM: %s の実体は %s", self.profile.model, self.model_name)
        # 5人ぶんの並列リクエストで接続を使い回す
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=16)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    @property
    def label(self) -> str:
        p = self.profile
        if self.model_name and self.model_name != p.model:
            return f"{self.model_name} [{p.model}] ({p.engine})"
        return f"{p.model} ({p.engine})"

    # --- 接続先の決定 -------------------------------------------------------

    def _resolve_profile(self) -> LLMProfile:
        cfg = self._config
        candidates = (
            cfg.profiles if cfg.profile == "auto" else [p for p in cfg.profiles if p.name == cfg.profile]
        )
        tried = []
        for p in candidates:
            try:
                models = self._list_models(p)
            except requests.RequestException as e:
                tried.append(f"{p.name} ({p.host}): {type(e).__name__}")
                continue
            if not p.model:
                if not models:
                    tried.append(f"{p.name} ({p.host}): /v1/models が空")
                    continue
                p.model = models[0]
            elif models and p.model not in models:
                logger.warning("[llm.%s] model=%r は /v1/models に無い（%s）", p.name, p.model, models[:5])
            logger.info("LLM: %s -> %s model=%s", p.name, p.host, p.model)
            return p
        raise LLMError("LLM サーバに接続できません。\n  " + "\n  ".join(tried))

    @staticmethod
    def _headers(p: LLMProfile) -> dict:
        return {"Authorization": f"Bearer {p.api_key}"} if p.api_key else {}

    def _list_models(self, p: LLMProfile) -> list[str]:
        resp = requests.get(f"{p.host}/v1/models", headers=self._headers(p), timeout=3)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]

    def _real_model_name(self, p: LLMProfile) -> str:
        """別名ではない、重みそのものの名前。分からなければ p.model のまま。

        - SGLang: /get_model_info の model_path（--served-model-name を付けても元のパスが入っている）
        - vLLM:   /v1/models の root（同上）
        - Ollama: model 名がそのまま本当の名前
        """
        path = ""
        if p.engine == "sglang":
            try:
                resp = requests.get(f"{p.host}/get_model_info", headers=self._headers(p), timeout=3)
                resp.raise_for_status()
                path = resp.json().get("model_path") or ""
            except (requests.RequestException, ValueError) as e:
                logger.debug("get_model_info に失敗: %s", e)
        if not path:
            try:
                resp = requests.get(f"{p.host}/v1/models", headers=self._headers(p), timeout=3)
                resp.raise_for_status()
                for m in resp.json().get("data", []):
                    if m.get("id") == p.model and m.get("root"):
                        path = m["root"]
            except (requests.RequestException, ValueError):
                pass
        name = re.split(r"[/\\]", path.rstrip("/\\"))[-1] if path else ""
        return name or p.model

    # --- リクエスト ---------------------------------------------------------

    def _payload(self, messages: list[dict], max_tokens: int, temperature: float | None) -> dict:
        p = self.profile
        payload: dict = {
            "model": p.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self._config.temperature if temperature is None else temperature,
            "reasoning_effort": "none",
        }
        if p.engine in ("sglang", "vllm", "llamacpp"):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    def _post(self, payload: dict, stream: bool) -> requests.Response:
        p = self.profile
        resp = self._session.post(
            f"{p.host}/v1/chat/completions",
            json=payload,
            headers=self._headers(p),
            timeout=self._config.timeout_sec,
            stream=stream,
        )
        if resp.status_code >= 400:
            raise LLMError(f"LLM HTTP {resp.status_code}: {resp.text[:300]}")
        return resp

    def chat(
        self,
        messages: list[dict],
        schema: dict | None = None,
        max_tokens: int = 256,
        temperature: float | None = None,
        tag: str = "",
    ) -> str:
        payload = self._payload(messages, max_tokens, temperature)
        payload["stream"] = False
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema},
            }
        t0 = time.monotonic()
        try:
            body = self._post(payload, stream=False).json()
            content = body["choices"][0]["message"]["content"] or ""
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError, LLMError) as e:
            self.stats.record("chat", tag, time.monotonic() - t0, 0, False, _error_kind(e))
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"LLM 呼び出しに失敗: {e}") from e
        self.stats.record("chat", tag, time.monotonic() - t0, _usage_tokens(body), True)
        return _THINK_RE.sub("", content).strip()

    def stream(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float | None = None,
        tools: list[dict] | None = None,
        tag: str = "",
    ) -> Iterator[str | ToolCall]:
        """本文のチャンク（str）を順に返し、ツール呼び出しがあれば最後に ToolCall を返す。

        途中で break すると接続を閉じて生成を打ち切る。
        記録する秒数からは、呼び出し元がチャンクを受け取って止まっていた時間（音声合成など）を除く。
        """
        meter = {"usage": 0, "chunks": 0, "ok": True, "err": ""}
        t0 = time.monotonic()
        paused = 0.0
        inner = self._stream(messages, max_tokens, temperature, tools, meter)
        try:
            for item in inner:
                p = time.monotonic()
                yield item
                paused += time.monotonic() - p
        except LLMError as e:
            meter["ok"], meter["err"] = False, _error_kind(e)
            raise
        finally:
            inner.close()
            sec = max(0.0, time.monotonic() - t0 - paused)
            self.stats.record("stream", tag, sec, meter["usage"] or meter["chunks"], meter["ok"], meter["err"])

    def _stream(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float | None,
        tools: list[dict] | None,
        meter: dict,
    ) -> Iterator[str | ToolCall]:
        payload = self._payload(messages, max_tokens, temperature)
        payload["stream"] = True
        # 最後のチャンクに usage を付けてもらう（6人目の実行状況用。付かないサーバではチャンクの数で数える）
        payload["stream_options"] = {"include_usage": True}
        if tools:
            payload["tools"] = tools
        try:
            resp = self._post(payload, stream=True)
        except requests.RequestException as e:
            raise LLMError(f"LLM 呼び出しに失敗: {e}") from e
        calls: dict[int, dict] = {}
        held = ""  # 生のツール呼び出しかもしれない、まだ出していない本文
        raw_call = False
        try:
            for raw in resp.iter_lines():
                if not raw.startswith(b"data:"):
                    continue
                data = raw[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                meter["usage"] = _usage_tokens(obj) or meter["usage"]
                try:
                    delta = obj["choices"][0]["delta"]
                except (KeyError, IndexError, TypeError):
                    continue
                meter["chunks"] += 1
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                    fn = tc.get("function") or {}
                    slot["id"] = slot["id"] or tc.get("id") or ""
                    slot["name"] = slot["name"] or fn.get("name") or ""
                    slot["args"] += fn.get("arguments") or ""
                chunk = delta.get("content")
                if not chunk:
                    continue
                # tools を渡していなくても、モデルが前の流れから呼び出しを書くことがある。
                # 読み上げないように、生の呼び出しは常に本文から外す
                held += chunk
                if raw_call:
                    continue
                if _RAW_CALL_START in held:
                    raw_call = True
                    head, held = held.split(_RAW_CALL_START, 1)
                    if head:
                        yield head
                    continue
                # 「<tool_c」のように開始タグの途中で切れている可能性がある部分だけ手元に残す
                cut = held.rfind("<")
                if cut < 0 or not _RAW_CALL_START.startswith(held[cut:]):
                    cut = len(held)
                if cut:
                    yield held[:cut]
                    held = held[cut:]
        except requests.RequestException as e:
            raise LLMError(f"LLM ストリームが途切れました: {e}") from e
        finally:
            resp.close()
        if raw_call:
            yield from _parse_raw_calls(held)
        elif held:
            yield held
        for i, slot in sorted(calls.items()):
            try:
                args = json.loads(slot["args"]) if slot["args"].strip() else {}
            except ValueError:
                logger.warning("tool call の引数が JSON でない: %r", slot["args"][:200])
                args = {}
            yield ToolCall(slot["id"] or f"call_{i}", slot["name"], args if isinstance(args, dict) else {})


def _usage_tokens(body: object) -> int:
    try:
        return int((body.get("usage") or {}).get("completion_tokens") or 0)  # type: ignore[union-attr]
    except (AttributeError, TypeError, ValueError):
        return 0


def _error_kind(e: BaseException) -> str:
    if isinstance(e, requests.Timeout) or "timed out" in str(e).lower():
        return "timeout"
    return "error"


def _parse_raw_calls(text: str) -> list[ToolCall]:
    """本文に生で出たツール呼び出しを読む。"""
    out: list[ToolCall] = []
    for body in text.split(_RAW_CALL_START):
        body = body.replace("</tool_call>", "").strip()
        if not body:
            continue
        if m := _XML_FUNC_RE.search(body):
            args = {k: v for k, v in _XML_PARAM_RE.findall(m.group(2))}
            out.append(ToolCall(f"call_raw{len(out)}", m.group(1), args))
            continue
        try:
            obj = json.loads(body)
            out.append(ToolCall(f"call_raw{len(out)}", obj["name"], dict(obj.get("arguments") or {})))
        except (ValueError, KeyError, TypeError):
            logger.warning("読めないツール呼び出し: %r", body[:200])
    return out
