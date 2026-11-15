"""Ursina の画面（描画と UI だけ。LLM と音声は裏スレッド）。

上: 5人の 3D キャラ（poly_character）
下: 会話ログと入力欄

入力欄は Panda3D の DirectEntry。Ursina 標準の InputField はキー名ベースで
IME の確定文字を受け取れないが、DirectEntry は OS の文字入力（WM_CHAR / IME）を
そのまま受け取れる。DirectEntry には範囲選択・コピー・貼り付け・元に戻すが無いので、
textedit.py で足している（Shift+矢印・Ctrl+A・マウスでの選択、Ctrl+C/X/V、Ctrl+Z/Y など）。

会話ログはメッセージ単位で選んでコピーできる: クリックで1件、ドラッグや Shift+クリックで
続けて何件か選び、Ctrl+C でクリップボードへ。右クリックはその1件をすぐコピーする。
Esc か、ログの空いたところをクリックすると選択を解く。
"""

from __future__ import annotations

import math
import queue
import random
import sys
import time
import unicodedata
from pathlib import Path

from panda3d.core import (
    ClockObject,
    Filename,
    Point2,
    Point3,
    TextProperties,
    TextPropertiesManager,
    loadPrcFileData,
)

from .. import language
from ..brain import USER, Brain
from ..config import Config
from ..voice import Player
from . import clipboard
from .textedit import EntryEditor

_BG = (0.09, 0.10, 0.13)
_LOG_HEIGHT = 0.32       # 画面の高さに対するログ領域の割合
_LOG_LINES = 12          # 行の Text の数（実際に見せる行数はウィンドウから決める）
_LOG_SCALE = 0.85        # ログ文字の大きさ
_ENTRY_SCALE = 0.052     # 入力欄の文字の大きさ（aspect2d 単位）
_HAND_URGE = 7           # これ以上の挙手で実際に手を挙げる
_SEL_COLOR = (0.17, 0.36, 0.62, 0.55)  # ログで選んだメッセージの背景
_NOTICE_SECONDS = 1.5    # 「コピーしました」を出しておく秒数


def _hex(value: str | None, default):
    from ursina import color

    if not value:
        return default
    r, g, b = (int(value[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return color.rgb(r, g, b)


_ARC_RADIUS = 4.5         # 立ち位置の円の半径（中心は真ん中の子よりカメラ側。小さいほど強く囲む）
_ARC_STEP_DEG = 21.5      # 隣どうしの角度（5人までの間隔。半径 4.5 で約 1.7）
_ARC_MAX_DEG = 43.0       # 両端の角度（6人以上はこの中に詰める。画面からはみ出さないように）
_LOOK_Z = -9.0            # みんなが顔を向ける点（円の中心とカメラのあいだ。見ている側を向く）


def _arc_layout(n: int, jitter: float) -> list[tuple[float, float, float]]:
    """立ち位置（x, z, rotation_y）を左から順に返す。

    真ん中の子を z=0 に置き、カメラ側に中心がある円の上に並べる（端の子ほど手前へ回り込む）。
    jitter（0 で整列、1 で既定）の分だけ、角度・前後・向きを乱数で少しずらす（起動ごとに変わる）。
    """
    step = _ARC_STEP_DEG if n <= 5 else 2 * _ARC_MAX_DEG / max(1, n - 1)
    rng = random.Random()
    out = []
    for i in range(n):
        deg = (i - (n - 1) / 2) * step + rng.uniform(-0.12, 0.12) * step * jitter
        r = _ARC_RADIUS + rng.uniform(-0.3, 0.3) * jitter
        a = math.radians(deg)
        x = r * math.sin(a)
        z = r * math.cos(a) - _ARC_RADIUS
        # 正の rotation_y で -x 側（内側）を向く。少しだけそっぽを向く子もいる
        rot = math.degrees(math.atan2(x, z - _LOOK_Z)) + rng.uniform(-5.0, 5.0) * jitter
        out.append((x, z, rot))
    return out


def _width(ch: str) -> int:
    return 2 if unicodedata.east_asian_width(ch) in "FWA" else 1


def _wrap(text: str, cols: int) -> list[str]:
    """半角換算 cols 桁で折り返す（日本語はどこで切ってもよい）。英語版は語の切れ目で折り返す。"""
    if language.is_en():
        return _wrap_words(text, cols)
    lines, cur, w = [], "", 0
    for ch in text:
        cw = _width(ch)
        if w + cw > cols:
            lines.append(cur)
            cur, w = "", 0
        cur += ch
        w += cw
    if cur:
        lines.append(cur)
    return lines or [""]


def _wrap_words(text: str, cols: int) -> list[str]:
    lines: list[str] = []
    cur = ""
    for word in text.split(" "):
        while sum(_width(c) for c in word) > cols:  # 1語で1行を超える（URL など）は切る
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(word[:cols])
            word = word[cols:]
        trial = f"{cur} {word}" if cur else word
        if sum(_width(c) for c in trial) > cols and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines or [""]


def run_display(
    cfg: Config,
    brain: Brain,
    player: Player,
    events: queue.Queue,
    llm_label: str = "",
    screenshot: str | None = None,
    exit_after: float | None = None,
    auto_say: str | None = None,
) -> None:
    # ウィンドウを開く前に決める必要がある設定
    loadPrcFileData("", f"ime-aware {'true' if cfg.display.ime_aware else 'false'}")

    from direct.gui.DirectGui import DirectEntry
    from ursina import (
        AmbientLight,
        DirectionalLight,
        Entity,
        Text,
        Ursina,
        application,
        camera,
        color,
        held_keys,
        mouse,
        scene,
        window,
    )
    from ursina.shaders import lit_with_shadows_shader

    from .expression import Face
    from .poly_character import Character

    app = Ursina(
        title="Imaginary Friend",
        fullscreen=False,
        borderless=False,
        size=(cfg.display.width, cfg.display.height),
        vsync=True,
        development_mode=False,
    )
    clock = ClockObject.get_global_clock()
    clock.set_mode(ClockObject.M_limited)
    clock.set_frame_rate(cfg.display.fps_limit)
    window.color = color.rgb(*_BG)

    font_path = Path(cfg.display.font)
    application.fonts_folder = font_path.parent
    Text.default_font = font_path.name

    # ライト（rotation は個別キーワードで渡さないと効かない）
    DirectionalLight(rotation_x=40, rotation_y=-28, color=color.rgb(1.0, 0.98, 0.95))
    AmbientLight(color=color.rgba(1, 1, 1, 0.38))

    # --- 3D: 5人（6人目がいれば6人）を、見ている側を囲む円弧に並べる -------------------------------
    cast = cfg.cast
    n = len(cast)
    characters: dict[str, Character] = {}
    camera_z = -15.5
    for m, (x, z, rot) in zip(cast, _arc_layout(n, cfg.display.layout_jitter)):
        characters[m.id] = Character(
            m.id,
            name=m.name,
            model=m.model,
            hair_style=m.hair_style,
            hair_color=_hex(m.hair_color, color.rgb(0.35, 0.25, 0.2)),
            eye_color=_hex(m.eye_color, color.black),
            dress_color=_hex(m.clothing_color, color.gray),
            accessories=[(a.type, _hex(a.color, None), a.side) for a in m.accessory],
            x=x,
            z=z,
            rotation_y=rot,
        )
    # 喜怒哀楽の表情（眉・ほお・涙・漫符）
    faces = (
        {m.id: Face(characters[m.id], _hex(m.hair_color, color.rgb(0.35, 0.25, 0.2))) for m in cast}
        if cfg.display.expressions
        else {}
    )
    Entity(
        model="plane",
        shader=lit_with_shadows_shader,
        color=color.rgb(0.16, 0.17, 0.22),
        scale=(40, 1, 24),
        y=-0.01,
    )
    # ログが画面の下 1/3 を覆うので、カメラは足元より下を狙ってキャラを上側へ寄せる
    camera.position = (0, 1.2, camera_z)
    camera.rotation_x = 2.8
    camera.fov = 34

    lens = application.base.camLens

    def project(world_pos) -> tuple[float, float] | None:
        cam_space = camera.getRelativePoint(scene, world_pos)
        out = Point2()
        if not lens.project(Point3(cam_space[0], cam_space[1], cam_space[2]), out):
            return None
        return (out[0] * 0.5 * camera.aspect_ratio, out[1] * 0.5)

    plate_colors = {
        m.id: _hex(m.clothing_color, color.white).tint(0.25) for m in cast
    }
    nameplates = {}
    for m in cast:
        plate = Text(text=cfg.labels.get(m.id, m.name), parent=camera.ui, origin=(0, 0), scale=0.9, color=color.light_gray)
        plate.create_background(color=color.rgba(0, 0, 0, 0.55), padding=(0.012, 0.008))
        nameplates[m.id] = plate

    # --- 2D: ログ・入力欄・ステータス -------------------------------------------
    log_panel = Entity(parent=camera.ui, model="quad", color=color.rgba(0, 0, 0, 0.72), z=1)
    log_texts = [
        Text(text=" ", parent=camera.ui, origin=(-0.5, 0.5), scale=_LOG_SCALE) for _ in range(_LOG_LINES)
    ]
    # 選んだメッセージの行の背景（パネルと文字のあいだに描く）
    sel_bars = [
        Entity(parent=camera.ui, model="quad", origin=(-0.5, 0.5), color=color.rgba(*_SEL_COLOR), z=0.5,
               enabled=False)
        for _ in range(_LOG_LINES)
    ]
    notice = Text(text=" ", parent=camera.ui, origin=(0.5, -0.5), scale=0.8, color=color.azure)
    notice.create_background(color=color.rgba(0, 0, 0, 0.55), padding=(0.012, 0.006))
    notice.enabled = False
    status = Text(text=" ", parent=camera.ui, origin=(-0.5, 0.5), scale=0.8, color=color.light_gray)
    header = Text(
        text=f"LLM: {llm_label}", parent=camera.ui, origin=(-0.5, 0.5), scale=0.7, color=color.gray
    )

    # 全角1文字の幅を実測して、折り返し桁数を決める
    en = language.is_en()
    # 英語版は日本語グリフの無いフォント（Segoe UI など）でもよいように、小文字の平均幅を半角1桁として測る
    probe = Text(
        text="abcdefghijklmnopqrst" if en else "あ" * 20, parent=camera.ui, scale=_LOG_SCALE, enabled=False
    )
    zen_width = probe.width / 20 * (2 if en else 1)
    line_height = probe.height * 1.25

    base = application.base
    entry_font = base.loader.loadFont(Filename.fromOsSpecific(str(font_path)).getFullpath())
    # IME 変換中の文字列の見た目（ime_aware = true のとき入力欄の中に出る）
    tpm = TextPropertiesManager.getGlobalPtr()
    for prop_name, fg in (("candidate_active", (1, 0.85, 0.3, 1)), ("candidate_inactive", (0.7, 0.8, 1, 1))):
        tp = TextProperties()
        tp.setTextColor(*fg)
        tpm.setProperties(prop_name, tp)

    names = {m.id: m.name for m in cast} | {USER: cfg.user_name}
    line_colors = plate_colors | {USER: color.white, "system": color.gray}
    # ログの元データ: (話者, 本文, ターン番号)。描画用の行は layout/redraw で作り直す
    entries: list[tuple[str, str, int]] = [(s, t, -1) for s, t in brain.past_lines()]
    if entries:
        entries.append(("system", language.pick(ja="── ここまで前回の会話 ──", en="--- end of last conversation ---"), -1))
    # row_entry: いま見えている各行が entries の何番目か（クリック位置からメッセージを引くため）
    view = {"cols": 60, "lines": 8, "scroll": 0, "aspect": 0.0, "dirty": True, "top": 0.0, "row_entry": []}
    # ログの選択: anchor〜end の entries の番号（entries は後ろに足すだけなので番号はずれない）
    sel: dict = {"anchor": None, "end": None, "drag": False, "notice_until": 0.0}

    def selected_range() -> tuple[int, int] | None:
        if sel["anchor"] is None:
            return None
        return min(sel["anchor"], sel["end"]), max(sel["anchor"], sel["end"])

    def clear_selection() -> None:
        if sel["anchor"] is not None:
            sel["anchor"] = sel["end"] = None
            view["dirty"] = True
        sel["drag"] = False

    def entry_at_mouse() -> int | None:
        """マウスの下にあるログの行のメッセージ番号（ログの外・空いた行なら None）。"""
        if not (view["top"] - view["lines"] * line_height < mouse.y <= view["top"]):
            return None
        row = int((view["top"] - mouse.y) / line_height)
        rows = view["row_entry"]
        return rows[row] if 0 <= row < len(rows) else None

    def copy_entries(a: int, b: int) -> None:
        picked = entries[a : b + 1]
        if len(picked) == 1:
            text = picked[0][1]  # 1件だけなら本文そのまま
        else:
            text = "\n".join(f"{names[s]}: {t}" for s, t, _ in picked if s != "system")
        if text and clipboard.set_text(text):
            n_msgs = sum(1 for s, _, _ in picked if s != "system") or 1
            if language.is_en():
                notice.text = "Copied" if n_msgs == 1 else f"Copied {n_msgs} messages"
            else:
                notice.text = "コピーしました" if n_msgs == 1 else f"{n_msgs}件コピーしました"
            notice.create_background(color=color.rgba(0, 0, 0, 0.55), padding=(0.012, 0.006))
            notice.enabled = True
            sel["notice_until"] = time.time() + _NOTICE_SECONDS

    def on_submit(text: str) -> None:
        editor.clear()
        entry["focus"] = 1
        text = text.strip()
        if not text:
            return
        entries.append((USER, text, -1))
        view["scroll"] = 0
        view["dirty"] = True
        brain.submit(text)

    entry = DirectEntry(
        parent=base.aspect2d,
        scale=_ENTRY_SCALE,
        width=10,
        numLines=1,
        focus=1,
        entryFont=entry_font,
        text_fg=(1, 1, 1, 1),
        frameColor=(0.2, 0.22, 0.28, 1),
        command=on_submit,
        suppressMouse=0,
        # 幅を超えたら横にスクロールする（既定の 0 だと、幅いっぱいで入力や貼り付けが切れる）
        overflow=1,
    )

    editor = EntryEditor(entry, base, held_keys)
    prompt = Text(text=">", parent=camera.ui, origin=(-0.5, 0), scale=1.0, color=color.azure)

    def layout() -> None:
        """ウィンドウの縦横比が変わったら 2D 部品を置き直す。"""
        aspect = camera.aspect_ratio
        view["aspect"] = aspect
        left, bottom = -aspect / 2, -0.5
        margin = 0.02
        log_panel.scale = (aspect, _LOG_HEIGHT)
        log_panel.position = (0, bottom + _LOG_HEIGHT / 2)
        top = bottom + _LOG_HEIGHT - 0.012
        view["top"] = top
        for i, t in enumerate(log_texts):
            t.position = (left + margin, top - i * line_height)
            sel_bars[i].position = (left + margin * 0.5, top - i * line_height)
            sel_bars[i].scale = (aspect - margin, line_height)
        notice.position = (-left - margin, bottom + _LOG_HEIGHT + 0.012)  # ログのすぐ上、右寄せ
        status.position = (left + margin, 0.5 - 0.045)
        header.position = (left + margin, 0.5 - 0.012)
        view["lines"] = max(3, min(_LOG_LINES, int((_LOG_HEIGHT - 0.012 - 0.075) / line_height)))
        view["cols"] = max(20, int((aspect - margin * 2) / zen_width * 2) - 2)
        # 入力欄（aspect2d は x:±aspect, z:±1 なので camera.ui の2倍の座標）
        entry_y = bottom + 0.032
        prompt.position = (left + margin, entry_y + 0.008)
        # ursina は座標系を y-up にしているので、2D の縦は z ではなく y
        entry.setPos((left + margin + 0.03) * 2, entry_y * 2 - _ENTRY_SCALE * 0.1, 0)
        entry["width"] = (aspect - margin * 2 - 0.03) * 2 / _ENTRY_SCALE
        entry.resetFrameSize()
        view["dirty"] = True

    def redraw() -> None:
        rows: list[tuple[str, object, int]] = []
        first = max(0, len(entries) - 200)
        for idx, (speaker, text, _turn) in enumerate(entries[first:], start=first):
            # ursina のカラータグ対策（英語版のフォントには全角の＜＞が無いことがある）
            text = text.replace("<", "‹").replace(">", "›") if en else text.replace("<", "＜").replace(">", "＞")
            if speaker == "system":
                wrapped = _wrap(text, view["cols"])
            else:
                if en:
                    label = f"{names[speaker]}: "
                    pad = " " * len(label)
                else:
                    label = f"{names[speaker]}｜"
                    pad = "　" * (sum(_width(c) for c in label) // 2)
                body = _wrap(text, view["cols"] - len(pad) * (1 if en else 2))
                wrapped = [label + body[0]] + [pad + b for b in body[1:]]
            rows += [(line, line_colors[speaker], idx) for line in wrapped]
        n_lines = view["lines"]
        max_scroll = max(0, len(rows) - n_lines)
        view["scroll"] = min(max(view["scroll"], 0), max_scroll)
        end = len(rows) - view["scroll"]
        visible = rows[max(0, end - n_lines) : end]
        view["row_entry"] = [r[2] for r in visible]
        picked = selected_range()
        for i, t in enumerate(log_texts):
            if i < len(visible):
                t.text = visible[i][0] or " "
                t.color = visible[i][1]
            else:
                t.text = " "
            sel_bars[i].enabled = bool(
                picked and i < len(visible) and picked[0] <= visible[i][2] <= picked[1]
            )
        view["dirty"] = False

    hands: dict[str, int] = {}
    arm = {m.id: 0.0 for m in cast}  # 挙手の上がり具合 0〜1
    hand_raise = cfg.display.hand_raise
    started = time.time()
    flags = {"shot": False, "said": False}

    def update() -> None:
        while True:
            try:
                ev = events.get_nowait()
            except queue.Empty:
                break
            kind = ev[0]
            if kind == "say":
                _, speaker, text, turn, emotion = ev
                if speaker in faces:
                    faces[speaker].feel(emotion, time.time())
                if entries and entries[-1][0] == speaker and entries[-1][2] == turn:
                    joiner = " " if en else ""  # 英語は文と文のあいだに空白を入れる
                    entries[-1] = (speaker, entries[-1][1] + joiner + text, turn)
                else:
                    entries.append((speaker, text, turn))
                view["dirty"] = True
            elif kind == "system":
                entries.append(("system", ev[1], -1))
                view["dirty"] = True
            elif kind == "status":
                status.text = ev[1] or " "
            elif kind == "hands":
                hands.clear()
                hands.update(ev[1])
            elif kind == "moods":
                for cid, (valence, label) in ev[1].items():
                    if cid in faces:
                        faces[cid].set_mood(valence, label)

        # ログをドラッグして、続けて何件か選ぶ
        if sel["drag"]:
            if not held_keys["left mouse"]:
                sel["drag"] = False
            else:
                idx = entry_at_mouse()
                if idx is not None and idx != sel["end"]:
                    sel["end"] = idx
                    view["dirty"] = True
        if sel["notice_until"] and time.time() > sel["notice_until"]:
            sel["notice_until"] = 0.0
            notice.enabled = False

        if abs(camera.aspect_ratio - view["aspect"]) > 1e-4:
            layout()
        if view["dirty"]:
            redraw()
        if not entry.guiItem.getFocus():
            entry["focus"] = 1

        t = time.time()
        speaker, mouth = player.speaking()
        dt = min(clock.get_dt(), 0.1)
        for cid, ch in characters.items():
            talking = cid == speaker
            ch.animate(t, is_speaking=talking, rms=mouth * 0.17)
            # 挙手: 喋りたい度合いが高いキャラは、指名されるまで片手を挙げる（[display] hand_raise）
            want = 1.0 if (hand_raise and hands.get(cid, 0) >= _HAND_URGE and not talking) else 0.0
            arm[cid] += (want - arm[cid]) * min(1.0, dt * 8)
            k = arm[cid]
            if k > 0.01:
                ch.arm_r.rotation_x = 165 * k
                ch.arm_r.rotation_z = -22 * k
            plate = nameplates[cid]
            pos = project(ch.label_anchor())
            if pos is None:
                plate.enabled = False
            else:
                plate.enabled = True
                plate.position = pos
                plate.color = plate_colors[cid] if talking else color.light_gray
            if cid in faces:
                faces[cid].update(t, dt, talking, pos)

        elapsed = t - started
        if auto_say and not flags["said"] and elapsed > 1.5:
            flags["said"] = True
            on_submit(auto_say)
        if screenshot and not flags["shot"] and exit_after and elapsed > exit_after - 0.5:
            flags["shot"] = True
            base.win.saveScreenshot(Filename.fromOsSpecific(str(Path(screenshot).resolve())))
        if exit_after and elapsed > exit_after:
            sys.exit(0)

    def on_log_input(key: str) -> bool:
        """会話ログのメッセージを選ぶ・コピーする。処理したキーなら True。"""
        if key == "left mouse down":
            idx = entry_at_mouse()
            if idx is None:
                clear_selection()
                return False
            if held_keys["shift"] and sel["anchor"] is not None:
                sel["end"] = idx
            else:
                sel["anchor"] = sel["end"] = idx
            sel["drag"] = True
            view["dirty"] = True
            return True
        if key == "right mouse down":
            idx = entry_at_mouse()
            if idx is None:
                return False
            sel["anchor"] = sel["end"] = idx
            view["dirty"] = True
            copy_entries(idx, idx)
            return True
        if key == "escape" and sel["anchor"] is not None:
            clear_selection()
            return True
        # 入力欄で範囲を選んでいるときは、Ctrl+C は入力欄のほうに任せる
        if key == "c" and held_keys["control"] and selected_range() and not editor.selection():
            copy_entries(*selected_range())
            return True
        return False

    def on_input(key: str) -> None:
        if key == "left mouse down":
            if editor.on_input(key):
                clear_selection()  # 入力欄をクリックしたらログの選択は解く
            else:
                on_log_input(key)
            return
        if on_log_input(key) or editor.on_input(key):
            return
        if key == "scroll up":
            view["scroll"] += 2
            view["dirty"] = True
        elif key == "scroll down":
            view["scroll"] -= 2
            view["dirty"] = True

    controller = Entity()
    controller.update = update
    controller.input = on_input

    layout()
    brain.start()
    app.run()
