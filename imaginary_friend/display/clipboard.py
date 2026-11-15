"""クリップボードの文字の読み書きと、入力欄へ貼り付けるときの整形。

Panda3D の DirectEntry にはコピー・貼り付けの機能が無いので、Windows の API を ctypes で直接呼ぶ
（追加の依存なし）。Windows 以外では tkinter を使う。
"""

from __future__ import annotations

import ctypes
import re
import sys
import time

_CF_UNICODETEXT = 13
_GMEM_MOVEABLE = 0x0002
# 入力欄は1行なので、改行やタブは空白にする。そのほかの制御文字は捨てる
_LINEBREAKS = re.compile(r"[\r\n\t]+")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def clean_for_entry(text: str) -> str:
    """貼り付ける文字を1行の入力欄向けに整える。"""
    return _CONTROL.sub("", _LINEBREAKS.sub(" ", text)).strip()


def get_text() -> str:
    """クリップボードの文字（無い・文字以外なら空）。"""
    if sys.platform != "win32":
        return _tk_get()
    user32, kernel32 = _win()
    if not _open(user32):
        return ""
    try:
        handle = user32.GetClipboardData(_CF_UNICODETEXT)
        if not handle:
            return ""
        ptr = kernel32.GlobalLock(handle)
        if not ptr:
            return ""
        try:
            return ctypes.wstring_at(ptr)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def set_text(text: str) -> bool:
    """クリップボードに文字を入れる。成功したら True。"""
    if sys.platform != "win32":
        return _tk_set(text)
    user32, kernel32 = _win()
    buf = ctypes.create_unicode_buffer(text)  # UTF-16、末尾の NUL つき
    size = ctypes.sizeof(buf)
    if not _open(user32):
        return False
    try:
        user32.EmptyClipboard()
        handle = kernel32.GlobalAlloc(_GMEM_MOVEABLE, size)
        if not handle:
            return False
        ptr = kernel32.GlobalLock(handle)
        if not ptr:
            kernel32.GlobalFree(handle)
            return False
        ctypes.memmove(ptr, buf, size)
        kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(_CF_UNICODETEXT, handle):
            kernel32.GlobalFree(handle)  # 受け取られなかったときだけ自分で解放する
            return False
        return True
    finally:
        user32.CloseClipboard()


# --- Windows -----------------------------------------------------------------

_dlls: tuple | None = None


def _win():
    """型を宣言した user32 / kernel32。64bit でポインタが切り詰められないよう restype を決めておく。
    ほかのライブラリと宣言がぶつからないよう、ctypes.windll とは別のインスタンスを使う。"""
    global _dlls
    if _dlls is None:
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        user32.OpenClipboard.argtypes = [wintypes.HWND]
        user32.OpenClipboard.restype = wintypes.BOOL
        user32.CloseClipboard.restype = wintypes.BOOL
        user32.EmptyClipboard.restype = wintypes.BOOL
        user32.GetClipboardData.argtypes = [wintypes.UINT]
        user32.GetClipboardData.restype = wintypes.HANDLE
        user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
        user32.SetClipboardData.restype = wintypes.HANDLE
        kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
        kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalUnlock.restype = wintypes.BOOL
        kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalFree.restype = wintypes.HGLOBAL
        _dlls = (user32, kernel32)
    return _dlls


def _open(user32) -> bool:
    # ほかのアプリが一瞬つかんでいることがあるので、少しだけやり直す
    for _ in range(5):
        if user32.OpenClipboard(None):
            return True
        time.sleep(0.02)
    return False


# --- Windows 以外 ------------------------------------------------------------

def _tk_get() -> str:
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        try:
            return root.clipboard_get()
        finally:
            root.destroy()
    except Exception:  # noqa: BLE001 - クリップボードが空・tkinter が無いなど
        return ""


def _tk_set(text: str) -> bool:
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        root.destroy()
        return True
    except Exception:  # noqa: BLE001
        return False
