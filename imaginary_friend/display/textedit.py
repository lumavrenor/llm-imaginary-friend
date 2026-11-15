"""入力欄（Panda3D の DirectEntry）に、ふつうのテキスト欄らしい編集操作を足す。

DirectEntry にあるのは文字入力・IME・BackSpace/Delete・カーソルの左右移動だけで、範囲選択が無い。
ここで足すもの:

- 範囲選択: Shift+←→ / Shift+Home/End / Ctrl+A / マウスのドラッグ / Shift+クリック
  / ダブルクリックで単語、トリプルクリックで全体
- 単語単位: Ctrl+←→ で移動、Ctrl+BackSpace / Ctrl+Delete で削除
- 選択範囲の上で打つ・貼り付ける・BackSpace/Delete を押すと、選択範囲が置き換わる
- Ctrl+C / Ctrl+X / Ctrl+V（選択が無いときのコピー・切り取りは全文）
- Ctrl+Z で元に戻す、Ctrl+Y / Ctrl+Shift+Z でやり直す

文字の入力と BackSpace/Delete は DirectEntry（C++ 側）が先に処理するので、毎フレーム
「前のフレームからどう変わったか」を見て、選択範囲があれば置き換えの形に直す（reconcile）。
カーソル移動のキーは DirectEntry 側を止めて、こちらで全部処理する。
"""

from __future__ import annotations

import time
import unicodedata

from panda3d.core import CardMaker, ClockObject, Point3

from . import clipboard

_SEL_COLOR = (0.17, 0.36, 0.62, 1)  # 選択範囲の色（不透明にすると文字の後ろに描かれる）
_MULTI_CLICK = 0.4                   # ダブル・トリプルクリックとみなす間隔（秒）
_UNDO_MERGE = 1.0                    # この秒数以内の続けての入力は、元に戻すとき1回にまとめる
_UNDO_MAX = 200


# --- 文字列だけで決まる処理（画面に依存しない） ----------------------------------

def char_class(ch: str) -> int:
    """単語の区切りを決めるための文字の種類。同じ種類が続く範囲を1単語とみなす。"""
    if ch.isspace():
        return 0
    o = ord(ch)
    if 0x3040 <= o <= 0x309F:
        return 2  # ひらがな
    if 0x30A0 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F:
        return 3  # カタカナ（長音符を含む）
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF or ch in "々〆ヵヶ":
        return 4  # 漢字
    if ch == "_" or unicodedata.category(ch)[0] in "LN":
        return 1  # 英数字（全角を含む）
    return 5      # 記号・句読点


def word_left(text: str, pos: int) -> int:
    """Ctrl+← の行き先: 左の空白を飛ばし、その単語の先頭へ。"""
    i = max(0, min(pos, len(text)))
    while i > 0 and char_class(text[i - 1]) == 0:
        i -= 1
    if i > 0:
        cls = char_class(text[i - 1])
        while i > 0 and char_class(text[i - 1]) == cls:
            i -= 1
    return i


def word_right(text: str, pos: int) -> int:
    """Ctrl+→ の行き先: 今の単語の終わりまで進み、続く空白も飛ばす。"""
    n = len(text)
    i = max(0, min(pos, n))
    if i < n:
        cls = char_class(text[i])
        if cls != 0:
            while i < n and char_class(text[i]) == cls:
                i += 1
    while i < n and char_class(text[i]) == 0:
        i += 1
    return i


def word_at(text: str, pos: int) -> tuple[int, int]:
    """ダブルクリックで選ぶ範囲（pos の文字と同じ種類が続く範囲）。"""
    if not text:
        return 0, 0
    i = max(0, min(pos, len(text) - 1))
    cls = char_class(text[i])
    a, b = i, i + 1
    while a > 0 and char_class(text[a - 1]) == cls:
        a -= 1
    while b < len(text) and char_class(text[b]) == cls:
        b += 1
    return a, b


def reconcile(
    old: str, old_cur: int, sel: tuple[int, int] | None, new: str, new_cur: int
) -> tuple[str, int, str]:
    """DirectEntry が勝手に変えた文字列を、選択範囲を考えた結果に直す。

    戻り値は (文字列, カーソル位置, 種類)。種類は "insert"（文字が入った）/ "backspace" /
    "delete"（1文字消えた）/ "other"（それ以外。そのまま受け入れる）。
    sel があるときの insert / backspace / delete は、選択範囲を置き換えた結果になる。
    """
    grown = len(new) - len(old)
    if grown > 0 and new_cur - old_cur == grown and new[:old_cur] + new[new_cur:] == old:
        ins = new[old_cur:new_cur]
        if sel:
            a, b = sel
            return old[:a] + ins + old[b:], a + len(ins), "insert"
        return new, new_cur, "insert"
    if grown == -1 and old_cur > 0 and new == old[: old_cur - 1] + old[old_cur:]:
        kind = "backspace"
    elif grown == -1 and new == old[:old_cur] + old[old_cur + 1 :]:
        kind = "delete"
    else:
        return new, new_cur, "other"
    if sel:
        a, b = sel
        return old[:a] + old[b:], a, kind
    return new, new_cur, kind


def _strip_controls(text: str) -> str:
    # Ctrl+文字キーで制御文字（Ctrl+A → \x01 など）が入ってきたときに捨てる
    return "".join(ch for ch in text if ch >= " " and ch != "\x7f")


# --- 入力欄への取り付け --------------------------------------------------------

class EntryEditor:
    """DirectEntry に範囲選択・単語移動・元に戻すを足す。

    on_input(key) を Ursina の input から呼ぶこと（処理したキーなら True を返す）。
    毎フレームの更新（変化の検出・ドラッグ・選択範囲の描画）は自分で登録するタスクが行う。
    """

    def __init__(self, entry, base, held_keys) -> None:
        self.entry = entry
        self.item = entry.guiItem
        self.base = base
        self.held = held_keys
        # ←→ Home End は自分で処理する（DirectEntry に任せると選択範囲と食い違う）
        self.item.setCursorKeysActive(False)
        self.anchor: int | None = None  # 選択の起点。None または カーソルと同じなら選択なし
        self._text = self._get()
        self._cur = self.item.getCursorPosition()
        self._undo: list[tuple[str, int]] = []
        self._redo: list[tuple[str, int]] = []
        self._last_edit = ("", 0.0)  # (種類, 時刻) 続けての入力をまとめるため
        self._last_kind = ("", -1)   # (種類, フレーム番号) Ctrl+BackSpace の判定用
        self._drag = False
        self._clicks = (0, 0.0, -1)  # (回数, 時刻, 位置)
        self._widths: tuple[str, list[float]] = ("", [0.0])
        # 前に描いたときの文字列とカーソル（入力欄の横スクロール量を逆算するのに使う）
        self._drawn = (self._text, self._cur)

        cm = CardMaker("entry-selection")
        cm.setFrame(0, 1, 0, 1)
        self.card = entry.stateNodePath[0].attachNewNode(cm.generate(), 10)
        self.card.setColor(*_SEL_COLOR)
        for state in entry.stateNodePath[1:]:
            self.card.instanceTo(state, 10)
        self.card.hide()
        base.taskMgr.add(self._task, "entry-editor", sort=49)  # 描画（igLoop, 50）の直前

    # --- 外から使う操作 ---------------------------------------------------------

    def clear(self) -> None:
        """送信したあとに入力欄を空にする（元に戻す履歴も捨てる）。"""
        self._undo.clear()
        self._redo.clear()
        self._last_edit = ("", 0.0)
        self._apply("", 0, record=False)

    def selection(self) -> tuple[int, int] | None:
        if self.anchor is None or self.anchor == self._cur:
            return None
        return min(self.anchor, self._cur), max(self.anchor, self._cur)

    def on_input(self, key: str) -> bool:
        ctrl = bool(self.held["control"])
        shift = bool(self.held["shift"])
        base_key = key[:-5] if key.endswith(" hold") else key

        if key == "left mouse down":
            return self._mouse_down(shift)
        if key == "left mouse up":
            self._drag = False
            return False

        moves = {"left arrow", "right arrow", "home", "end"}
        if base_key in moves:
            self.sync()
            self._move(base_key, ctrl, shift)
            return True
        if not ctrl:
            if key == "insert" and shift:
                self.paste()
                return True
            return False

        if key == "a":
            self.sync()
            self.anchor = 0
            self._set_cursor(len(self._text))
        elif key in ("c", "x"):
            self.sync()
            self.copy(cut=key == "x")
        elif key == "v":
            self.paste()
        elif key == "z" and not shift:
            self.undo()
        elif key == "y" or (key == "z" and shift):
            self.redo()
        elif base_key in ("backspace", "delete"):
            self._delete_word(forward=base_key == "delete")
        else:
            return False
        return True

    def copy(self, cut: bool) -> None:
        sel = self.selection()
        a, b = sel if sel else (0, len(self._text))  # 選択が無ければ全文
        if a == b or not clipboard.set_text(self._text[a:b]):
            return
        if cut:
            self._edit(self._text[:a] + self._text[b:], a, "cut")

    def paste(self) -> None:
        self.sync()
        text = clipboard.clean_for_entry(clipboard.get_text())
        if text:
            self._replace_selection(text, "paste")

    def undo(self) -> None:
        self.sync()
        if self._undo:
            self._redo.append((self._text, self._cur))
            text, cur = self._undo.pop()
            self._last_edit = ("", 0.0)
            self._apply(text, cur, record=False)

    def redo(self) -> None:
        self.sync()
        if self._redo:
            self._undo.append((self._text, self._cur))
            text, cur = self._redo.pop()
            self._last_edit = ("", 0.0)
            self._apply(text, cur, record=False)

    # --- 変化の検出 -------------------------------------------------------------

    def sync(self) -> None:
        """DirectEntry が打鍵で変えた文字列を取り込み、選択範囲があれば置き換えに直す。"""
        new = self._get()
        new_cur = self.item.getCursorPosition()
        # Ctrl+A なら \x01 のように、Ctrl+文字キーで制御文字が入ることがある。
        # 選択範囲を「打って置き換えた」と取り違えないよう、比べる前に取り除く
        clean = _strip_controls(new)
        if clean != new:
            new_cur -= len(new[:new_cur]) - len(_strip_controls(new[:new_cur]))
            new = clean
            self.item.setWtext(new)
            self.item.setCursorPosition(new_cur)
            if new == self._text:
                return
        if new == self._text:
            if new_cur != self._cur:  # IME などがカーソルだけ動かしたとき
                self._cur = new_cur
                self.anchor = None
            return
        old, old_cur = self._text, self._cur
        text, cur, kind = reconcile(old, old_cur, self.selection(), new, new_cur)
        self._push_undo(old, old_cur, kind)
        self._last_kind = (kind, ClockObject.getGlobalClock().getFrameCount())
        self._apply(text, cur, record=False)

    # --- 内部 ---------------------------------------------------------------

    def _get(self) -> str:
        return self.item.getPlainWtext()

    def _apply(self, text: str, cur: int, *, record: bool = True, kind: str = "other") -> None:
        if record and text != self._text:
            self._push_undo(self._text, self._cur, kind)
        if text != self._get():
            self.item.setWtext(text)
        self._text = text
        self.anchor = None
        self._set_cursor(cur)

    def _edit(self, text: str, cur: int, kind: str) -> None:
        self._apply(text, cur, record=True, kind=kind)

    def _replace_selection(self, ins: str, kind: str) -> None:
        sel = self.selection()
        a, b = sel if sel else (self._cur, self._cur)
        self._edit(self._text[:a] + ins + self._text[b:], a + len(ins), kind)

    def _push_undo(self, text: str, cur: int, kind: str) -> None:
        now = time.monotonic()
        last_kind, last_t = self._last_edit
        # 続けて打った文字や続けての BackSpace は、元に戻すとき1回分にまとめる
        merge = kind in ("insert", "backspace", "delete") and kind == last_kind and now - last_t < _UNDO_MERGE
        self._last_edit = (kind, now)
        if merge and self._undo:
            return
        self._undo.append((text, cur))
        del self._undo[:-_UNDO_MAX]
        self._redo.clear()

    def _set_cursor(self, pos: int) -> None:
        pos = max(0, min(pos, len(self._text)))
        self._cur = pos
        self.item.setCursorPosition(pos)
        # カーソルが見えている範囲の外へ出たら、横スクロールを計算し直させる
        x = self._padding() + self._prefix_width(pos)
        if x < 0 or x > self.item.getMaxWidth():
            self.item.setWtext(self.item.getWtext())

    def _move(self, key: str, ctrl: bool, shift: bool) -> None:
        sel = self.selection()
        cur, text = self._cur, self._text
        if key == "home":
            target = 0
        elif key == "end":
            target = len(text)
        elif sel and not shift and not ctrl:
            # 選択中に ←→ だけを押したら、選択範囲の端へ寄せて選択を解く
            target = sel[0] if key == "left arrow" else sel[1]
        elif key == "left arrow":
            target = word_left(text, cur) if ctrl else cur - 1
        else:
            target = word_right(text, cur) if ctrl else cur + 1
        if shift:
            if self.anchor is None:
                self.anchor = cur
        else:
            self.anchor = None
        self._last_edit = ("", 0.0)  # カーソルを動かしたら、そこからの入力は別の1回にする
        self._set_cursor(target)

    def _delete_word(self, forward: bool) -> None:
        # DirectEntry はもう1文字消している（この同じフレームの "backspace" / "delete"）
        self.sync()
        kind, frame = self._last_kind
        if frame != ClockObject.getGlobalClock().getFrameCount() or kind != ("delete" if forward else "backspace"):
            return  # 選択範囲を消した・消すものが無かったなど
        text, cur = self._text, self._cur
        if forward:
            end = word_right(text, cur)
            new, pos = text[:cur] + text[end:], cur
        else:
            start = word_left(text, cur)
            new, pos = text[:start] + text[cur:], start
        self._last_kind = ("", -1)
        self._apply(new, pos, record=False)  # 1文字分の履歴にまとめて入れる

    # --- マウス -------------------------------------------------------------

    def _mouse_local(self):
        mw = self.base.mouseWatcherNode
        if not mw.hasMouse():
            return None
        m = mw.getMouse()
        return self.entry.getRelativePoint(self.base.render2d, Point3(m.x, m.y, 0))

    def _index_at(self, x: float) -> int:
        x -= self._padding()
        widths = self._cum_widths()
        best, best_d = 0, abs(x)
        for i, w in enumerate(widths):
            d = abs(x - w)
            if d < best_d:
                best, best_d = i, d
        return best

    def _mouse_down(self, shift: bool) -> bool:
        p = self._mouse_local()
        if p is None:
            return False
        left, right, bottom, top = self.item.getFrame()
        if not (left <= p.x <= right and bottom <= p.y <= top):
            return False
        self.sync()
        self._last_edit = ("", 0.0)
        idx = self._index_at(p.x)
        now = time.monotonic()
        count, last_t, last_idx = self._clicks
        count = count + 1 if now - last_t < _MULTI_CLICK and abs(idx - last_idx) <= 1 else 1
        self._clicks = (count, now, idx)
        if shift and count == 1:
            if self.anchor is None:
                self.anchor = self._cur
            self._set_cursor(idx)
            self._drag = True
        elif count == 2:
            a, b = word_at(self._text, idx)
            self.anchor = a
            self._set_cursor(b)
        elif count >= 3:
            self.anchor = 0
            self._set_cursor(len(self._text))
        else:
            self.anchor = idx
            self._set_cursor(idx)
            self._drag = True
        return True

    # --- 位置の計算 -----------------------------------------------------------

    def _cum_widths(self) -> list[float]:
        """先頭から i 文字目までの幅（入力欄の文字単位）。文字列が変わったときだけ測り直す。"""
        if self._widths[0] != self._text:
            tn = self.entry.onscreenText.textNode
            acc, out = 0.0, [0.0]
            for ch in self._text:
                acc += tn.calcWidth(ch)
                out.append(acc)
            self._widths = (self._text, out)
        return self._widths[1]

    def _prefix_width(self, pos: int) -> float:
        widths = self._cum_widths()
        return widths[max(0, min(pos, len(widths) - 1))]

    def _padding(self) -> float:
        """横スクロールの量。DirectEntry は教えてくれないので、前に描いたときのカーソル位置から逆算する。"""
        text, cur = self._drawn
        tn = self.entry.onscreenText.textNode
        return self.item.getCursorX() - tn.calcWidth(text[:cur])

    # --- 毎フレーム -----------------------------------------------------------

    def _task(self, task):
        self.sync()
        if self._drag:
            if not self.held["left mouse"]:
                self._drag = False
            else:
                p = self._mouse_local()
                if p is not None:
                    idx = self._index_at(p.x)
                    if idx != self._cur:
                        if self.anchor is None:
                            self.anchor = self._cur
                        self._set_cursor(idx)
        self._draw_selection()
        self._drawn = (self._text, self._cur)
        return task.cont

    def _draw_selection(self) -> None:
        sel = self.selection()
        if not sel:
            self.card.hide()
            return
        pad = self._padding()
        max_w = self.item.getMaxWidth()
        x1 = max(0.0, pad + self._prefix_width(sel[0]))
        x2 = min(max_w, pad + self._prefix_width(sel[1]))
        if x2 <= x1:
            self.card.hide()
            return
        _left, _right, bottom, top = self.item.getFrame()
        y1, y2 = bottom + 0.12, top - 0.12
        self.card.setPos(x1, y1, 0)
        self.card.setScale(x2 - x1, y2 - y1, 1)
        self.card.show()
