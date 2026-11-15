"""喜怒哀楽の表情。

poly_character.Character には手を入れないので、表情のパーツ（眉・ほおの赤み・涙）は
ここで頭の子として足し、毎フレーム Character.animate() のあとに目・口・頭の向きを上書きする。
頭上の漫符（♪ や ！？）は画面の 2D に出す。

感情はセリフごとに届き（brain が【喜】などのタグから読む）、最後のセリフから EMOTION_SEC 秒続く。
そのあとは今日の気分からくる「ふだんの顔」（眠い子は半目、落ちこんでいる子は八の字眉）に戻る。
"""

from __future__ import annotations

import math

from ursina import Circle, Entity, Text, camera, color

NEUTRAL = "普通"
SLEEPY = "眠"  # 今日の気分が「ねむい」の子のふだんの顔（感情タグには無い）

# 感情ごとの見た目
#   eye / eye_w: 目の高さと幅（ふだん 0.15 / 0.12）
#   brow: 眉の角度（正 = 眉間が下がる怒り眉、負 = 八の字の困り眉）  brow_y: 眉の高さのずれ
#   blush: ほおの赤みの濃さ  mouth_w / mouth_h: 口の幅と、閉じているときの高さ
#   nod: うつむく角度（度。負は上を向く）
_LOOK: dict[str, dict[str, float]] = {
    NEUTRAL: dict(eye=0.15, eye_w=0.12, brow=0.0, brow_y=0.0, blush=0.15, mouth_w=0.26, mouth_h=0.04, nod=0.0),
    "喜": dict(eye=0.06, eye_w=0.13, brow=-6.0, brow_y=0.03, blush=0.5, mouth_w=0.32, mouth_h=0.09, nod=-4.0),
    "楽": dict(eye=0.09, eye_w=0.12, brow=-4.0, brow_y=0.02, blush=0.3, mouth_w=0.3, mouth_h=0.05, nod=-2.0),
    "怒": dict(eye=0.11, eye_w=0.12, brow=24.0, brow_y=-0.03, blush=0.12, mouth_w=0.17, mouth_h=0.04, nod=4.0),
    "哀": dict(eye=0.1, eye_w=0.11, brow=-22.0, brow_y=0.01, blush=0.1, mouth_w=0.15, mouth_h=0.03, nod=12.0),
    "驚": dict(eye=0.21, eye_w=0.15, brow=-4.0, brow_y=0.07, blush=0.12, mouth_w=0.12, mouth_h=0.13, nod=-5.0),
    "照": dict(eye=0.08, eye_w=0.12, brow=-12.0, brow_y=0.0, blush=0.95, mouth_w=0.15, mouth_h=0.035, nod=7.0),
    SLEEPY: dict(eye=0.035, eye_w=0.12, brow=-5.0, brow_y=-0.01, blush=0.12, mouth_w=0.2, mouth_h=0.04, nod=6.0),
}
# 頭上の漫符（文字と色）。フォントに入っている字だけを使う
_MARKS = {
    "喜": ("♪", (1.0, 0.82, 0.3)),
    "楽": ("♪", (0.6, 0.9, 0.5)),
    "怒": ("ムカッ", (1.0, 0.4, 0.35)),
    "哀": ("しゅん…", (0.55, 0.72, 1.0)),
    "驚": ("！？", (1.0, 0.9, 0.3)),
    "照": ("てれっ", (1.0, 0.55, 0.72)),
    SLEEPY: ("zzz", (0.7, 0.72, 0.95)),
}
_FACE_Z = -0.735  # 目・口（-0.72）より少し手前


class Face:
    EMOTION_SEC = 14.0  # 最後のセリフから、この秒数はその感情の顔を続ける

    def __init__(self, character, hair_color) -> None:
        self._ch = character
        head = character.head
        brow_color = color.rgb(hair_color[0] * 0.45, hair_color[1] * 0.45, hair_color[2] * 0.45)
        self._brows = [
            Entity(parent=head, model="quad", color=brow_color, scale=(0.16, 0.03), x=sx * 0.17, y=0.2, z=_FACE_Z)
            for sx in (-1, 1)
        ]
        self._blush = [
            Entity(
                parent=head,
                model=Circle(16),
                color=color.rgba(1.0, 0.45, 0.55, 0.0),
                scale=(0.16, 0.07),
                x=sx * 0.29,
                y=-0.1,
                z=-0.715,
            )
            for sx in (-1, 1)
        ]
        self._tears = [
            Entity(
                parent=head,
                model="quad",
                color=color.rgba(0.55, 0.82, 1.0, 0.9),
                scale=(0.04, 0.1),
                x=sx * 0.19,
                y=-0.07,
                z=-0.728,
                enabled=False,
            )
            for sx in (-1, 1)
        ]
        self._mark = Text(text=" ", parent=camera.ui, origin=(0, 0), scale=1.1, enabled=False)
        self._cur = dict(_LOOK[NEUTRAL])
        self.emotion = NEUTRAL
        self._since = -1e9  # 今の感情になった時刻
        self._until = 0.0
        self._base = NEUTRAL  # 今日の気分からくる、ふだんの顔
        self._base_k = 0.0

    def feel(self, emotion: str, t: float) -> None:
        """セリフの再生が始まったとき。"""
        if emotion not in _LOOK:
            emotion = NEUTRAL
        if emotion != self.emotion or t > self._until:
            self._since = t
        self.emotion = emotion
        self._until = t + self.EMOTION_SEC

    def set_mood(self, valence: float, label: str) -> None:
        """今日の気分（と、そのときどきの機嫌）から、ふだんの顔を決める。"""
        if label == "ねむい":
            self._base, self._base_k = SLEEPY, 0.7
        elif valence <= -1.0:
            self._base, self._base_k = "哀", 0.4
        elif valence >= 1.2:
            self._base, self._base_k = "楽", 0.5
        else:
            self._base, self._base_k = NEUTRAL, 0.0

    def update(self, t: float, dt: float, speaking: bool, plate_pos) -> None:
        active = self.emotion != NEUTRAL and t < self._until
        emo, k = (self.emotion, 1.0) if active else (self._base, self._base_k)
        goal, neutral = _LOOK[emo], _LOOK[NEUTRAL]
        a = min(1.0, dt * 7.0)
        c = self._cur
        for key in c:
            want = neutral[key] + (goal[key] - neutral[key]) * k
            c[key] += (want - c[key]) * a
        ch = self._ch

        # 目（まばたき中は animate() が閉じた値にしているので触らない）
        if ch.eye_l.scale_y > 0.03:
            ch.eye_l.scale_y = ch.eye_r.scale_y = c["eye"]
        ch.eye_l.scale_x = ch.eye_r.scale_x = c["eye_w"]

        # 眉。画面の左の眉は、時計回り（rotation_z 正）で眉間側が下がる
        for sx, brow in zip((-1, 1), self._brows):
            brow.rotation_z = -sx * c["brow"]
            brow.y = 0.2 + c["brow_y"]

        for blush in self._blush:
            blush.color = color.rgba(1.0, 0.45, 0.55, min(1.0, c["blush"]))

        # 涙は、悲しいセリフのあいだだけ流す
        crying = active and emo == "哀"
        phase = (t * 1.4) % 1.0
        for tear in self._tears:
            tear.enabled = crying
            if crying:
                tear.scale_y = 0.05 + 0.09 * phase
                tear.y = -0.06 - 0.05 * phase

        # 口。喋っている間は開き具合（scale_y）を animate() に任せ、幅だけ変える
        ch.mouth.scale_x = c["mouth_w"]
        if not speaking:
            ch.mouth.scale_y = c["mouth_h"]

        # 頭の向き（animate() が毎フレーム rotation_x を書き戻すので、ここで上書きする。負がうつむき）
        ch.head.rotation_x = -c["nod"]
        onset = t - self._since
        if active and emo == "怒" and onset < 0.8:
            ch.head.rotation_y = math.sin(t * 40.0) * 4.0 * (1.0 - onset / 0.8)
        else:
            ch.head.rotation_y = 0.0
        if active and emo == "喜" and speaking:
            ch.head.y += abs(math.sin(t * 9.0)) * 0.03
        if active and emo == "驚" and onset < 0.4:
            ch.head.y += 0.06 * (1.0 - onset / 0.4)

        # 漫符
        mark = _MARKS.get(emo) if (active or emo == SLEEPY) else None
        if mark is None or plate_pos is None:
            self._mark.enabled = False
            return
        text, rgb = mark
        if self._mark.text != text:
            self._mark.text = text
        self._mark.color = color.rgb(*rgb)
        self._mark.enabled = True
        self._mark.position = (plate_pos[0] + 0.06, plate_pos[1] + 0.045 + math.sin(t * 4.0) * 0.006)
