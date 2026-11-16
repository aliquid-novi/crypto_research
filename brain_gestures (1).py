#!/usr/bin/env python3
"""
brain_gestures.py — one-hand webcam control for the Brain graph (works on any app).

It drives your real OS cursor, so it controls the Brain artifact in the browser
exactly like a mouse would: hover highlights, click opens a note, drag on a node
moves it, drag on empty canvas pans, scroll zooms at the cursor.

GESTURE GRAMMAR (one hand, five poses, no overlapping modes)
------------------------------------------------------------
  POINT   index finger out (others curled)   -> move cursor (trackpad-style:
                                                 slow hand = fine, fast = far)
  PINCH   thumb touches index tip            -> mouse down. Release = click.
          (from POINT)                          Move while pinched = drag
                                                 (on a node: move it, on empty
                                                 canvas: PAN).
  V       index + middle out                 -> ZOOM at the frozen cursor:
                                                 raise hand = zoom in, lower = out
  FIST    all fingers curled                 -> CLUTCH: cursor holds still, so
                                                 you can re-centre your hand
  PALM    open hand, held ~0.7 s             -> Escape (closes the open note)
  BOTH PALMS held ~1 s (two hands)           -> ARM / DISARM the whole system

Starts DISARMED. Show both open palms for a second to arm it.
The HUD is a small panel in the bottom-right corner (click it first for keys):
  a = arm/disarm, h = show/hide camera, t = dark/light theme, q = quit.

Design choices (why it works this way):
  * Cursor follows the PALM CENTRE, not the fingertip. Pinching moves the
    index tip; the palm barely moves, so clicks land where you aimed.
  * As thumb and index approach, cursor gain drops (PinchLens-style
    "semi-pinch"), so the cursor settles onto the node before the click.
  * Relative (trackpad) mapping with pointer acceleration + One Euro filter:
    precise when slow, fast when you sweep. Physical mouse keeps working.
  * A pinch must travel a dead-zone before it becomes a drag, so a pinch-tap
    on a node opens it instead of nudging it.
  * Clicks are spaced >= 0.45 s apart: the Brain treats two taps on empty
    canvas within 320 ms as "create a new note".
  * Every mode change needs the pose held for a few frames (debounce), and
    pinch can only start from POINT, so fists and transitions don't fire clicks.
  * Thresholds use MediaPipe's 3D world landmarks (metres), so they don't
    change with distance from the camera.

Tune everything in the CONFIG block below; the HUD shows the live pinch
distance in mm so you can calibrate PINCH_ON/OFF to your own hand.

macOS: grant your terminal (or IDE) Camera + Accessibility permission.
Python 3.10+ (tested with mediapipe 0.10.32).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

# ============================== CONFIG ==================================== #


@dataclass
class Config:
    camera: int = 0
    cam_w: int = 640
    cam_h: int = 480
    hand: str = "Right"            # preferred controlling hand ("Right"/"Left")
    any_hand_if_alone: bool = True  # use whichever hand if only one is visible

    # Cursor gain, in multiples of screen width per full camera-width of hand
    # travel. Blended by hand speed (pointer acceleration).
    gain_slow: float = 0.55        # precise end
    gain_fast: float = 2.6         # sweeping end
    speed_slow: float = 0.06       # hand speed (camera-widths/s) where slow gain applies
    speed_fast: float = 0.70       # ... where fast gain applies

    # One Euro filter on palm position (normalised units)
    euro_min_cutoff: float = 1.2   # lower = steadier when still, more lag
    euro_beta: float = 18.0        # higher = less lag when moving fast

    # Pinch (thumb tip <-> index tip, millimetres, 3D world landmarks)
    pinch_on_mm: float = 28.0      # closes below this -> mouse down
    pinch_off_mm: float = 42.0     # opens above this  -> mouse up (hysteresis)
    pinch_semi_mm: float = 65.0    # start slowing the cursor below this
    semi_min_gain: float = 0.18    # cursor gain multiplier right before the click
    drag_deadzone_px: float = 16.0 # pinch must move this far to become a drag
    min_click_gap_s: float = 0.45  # avoids the Brain's double-tap = new note

    # Pose detection
    ext_ratio: float = 1.22        # finger "extended" if wrist->tip > ratio * wrist->pip
    stable_frames: int = 3         # frames a pose must hold before mode switches
    lost_grace_s: float = 0.25     # hand missing this long -> release + idle

    # Zoom (V pose): natural-log zoom per camera-width of vertical hand travel
    zoom_speed: float = 7.0        # ~0.1 of frame height up ≈ 1.7x zoom in
    zoom_invert: bool = False
    zoom_settle_s: float = 0.12    # ignore motion right after entering zoom
    wheel_zoom_const: float = 0.0015  # Brain's zoom = exp(-deltaY * this)
    scroll_px_per_step: float = field(
        default_factory=lambda: 10.0 if sys.platform == "darwin" else 100.0
    )  # browser deltaY per scroll step (macOS pynput: 10 px; Windows notch ≈ 100)

    # Holds
    palm_hold_s: float = 0.7       # open palm -> Escape
    arm_hold_s: float = 1.0        # both palms -> arm/disarm

    model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    )


# ============================ HAND MODEL ================================== #

WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20
PALM_IDS = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


@dataclass
class HandObs:
    label: str                 # "Left"/"Right"
    palm: np.ndarray           # palm centre, isotropic image units (camera-widths)
    pinch_mm: float            # thumb-index tip distance
    index_reach: float         # wrist->index tip / wrist->index mcp (curled < ~1.1)
    ext: tuple                 # (index, middle, ring, pinky) extended flags
    pose: str                  # POINT / V / FIST / PALM / OTHER
    px: np.ndarray             # 21x2 pixel coords for drawing


def _d(a, b):
    return float(np.linalg.norm(a - b))


def analyse_hand(lm, wlm, label, frame_w, frame_h, cfg: Config) -> HandObs:
    img = np.array([[p.x, p.y] for p in lm], dtype=np.float64)
    world = np.array([[p.x, p.y, p.z] for p in wlm], dtype=np.float64)

    def extended(tip, pip):
        return _d(world[WRIST], world[tip]) > cfg.ext_ratio * _d(world[WRIST], world[pip])

    ext = (
        extended(INDEX_TIP, INDEX_PIP),
        extended(MIDDLE_TIP, MIDDLE_PIP),
        extended(RING_TIP, RING_PIP),
        extended(PINKY_TIP, PINKY_PIP),
    )
    i, m, r, p = ext
    if i and m and r and p:
        pose = "PALM"
    elif not (i or m or r or p):
        pose = "FIST"
    elif i and not m and not r and not p:
        pose = "POINT"
    elif i and m and not r and not p:
        pose = "V"
    else:
        pose = "OTHER"

    palm = img[PALM_IDS].mean(axis=0)
    palm_iso = np.array([palm[0], palm[1] * frame_h / frame_w])  # both in camera-widths

    return HandObs(
        label=label,
        palm=palm_iso,
        pinch_mm=_d(world[THUMB_TIP], world[INDEX_TIP]) * 1000.0,
        index_reach=_d(world[WRIST], world[INDEX_TIP]) / max(_d(world[WRIST], world[INDEX_MCP]), 1e-6),
        ext=ext,
        pose=pose,
        px=img * np.array([frame_w, frame_h]),
    )


# ============================== FILTERS =================================== #


class OneEuro:
    """One Euro filter (Casiez et al. 2012), works element-wise on numpy arrays."""

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(dt, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t_prev, 1e-4)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


def smoothstep(x, lo, hi):
    if hi <= lo:
        return 1.0
    u = min(max((x - lo) / (hi - lo), 0.0), 1.0)
    return u * u * (3 - 2 * u)


# ============================== OUTPUT ==================================== #


class Output:
    """OS mouse/keyboard via pynput. All actions are no-ops while disarmed."""

    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        self.armed = False
        self.pressed = False
        self._last_cmd = None
        self._fx = self._fy = None
        if not dry_run:
            from pynput.mouse import Controller as Mouse, Button
            from pynput.keyboard import Controller as Keys, Key
            self._mouse, self._btn = Mouse(), Button.left
            self._keys, self._esc = Keys(), Key.esc

    @property
    def live(self):
        return self.armed and not self.dry_run

    def move_by(self, dx, dy):
        if not self.live or (dx == 0 and dy == 0):
            return
        ox, oy = self._mouse.position
        # If the physical mouse moved since our last command, resync to it.
        if self._last_cmd is None or abs(ox - self._last_cmd[0]) > 2 or abs(oy - self._last_cmd[1]) > 2:
            self._fx, self._fy = float(ox), float(oy)
        self._fx += dx
        self._fy += dy
        target = (int(round(self._fx)), int(round(self._fy)))
        self._mouse.position = target
        self._last_cmd = self._mouse.position  # OS may clamp at screen edges
        self._fx = min(max(self._fx, self._last_cmd[0] - 1), self._last_cmd[0] + 1)
        self._fy = min(max(self._fy, self._last_cmd[1] - 1), self._last_cmd[1] + 1)

    def press(self):
        if self.live and not self.pressed:
            self._mouse.press(self._btn)
        self.pressed = self.live or self.pressed

    def release(self):
        if self.pressed and not self.dry_run:
            self._mouse.release(self._btn)
        self.pressed = False

    def scroll(self, steps: int):
        if self.live and steps:
            self._mouse.scroll(0, steps)

    def escape(self):
        if self.live:
            self._keys.press(self._esc)
            self._keys.release(self._esc)


# ============================== ENGINE ==================================== #

POSE_TO_MODE = {"POINT": "POINT", "V": "ZOOM", "FIST": "CLUTCH", "PALM": "PALM"}


class GestureEngine:
    def __init__(self, cfg: Config, out: Output, screen_w: int):
        self.cfg, self.out, self.screen_w = cfg, out, screen_w
        self.filt = OneEuro(cfg.euro_min_cutoff, cfg.euro_beta)
        self.mode = "IDLE"
        self.pending, self.pending_n = None, 0
        self.prev, self.prev_t = None, None
        self.lost_t0 = None
        self.arm_t0, self.arm_fired = None, False
        self.palm_t0, self.palm_fired = None, False
        self.zoom_t0, self.zoom_acc = None, 0.0
        self.pinch_acc = np.zeros(2)
        self.dragging = False
        self.press_pending = False
        self.release_n = 0
        self.down = False          # logical button state (also tracked in dry-run)
        self.last_release_t = -1e9
        self.event = ""            # last discrete event, for the HUD
        self.event_t = 0.0
        self.primary: HandObs | None = None

    # ---- helpers ----
    def _flash(self, msg, t):
        self.event, self.event_t = msg, t

    def _gain_px(self, speed):
        c = self.cfg
        g = c.gain_slow + (c.gain_fast - c.gain_slow) * smoothstep(speed, c.speed_slow, c.speed_fast)
        return g * self.screen_w

    def _pick_primary(self, hands):
        pref = [h for h in hands if h.label == self.cfg.hand]
        if pref:
            return pref[0]
        if len(hands) == 1 and self.cfg.any_hand_if_alone:
            return hands[0]
        return None

    def _hard_reset(self):
        self._pinch_end(time.monotonic(), silent=True)
        self.mode = "IDLE"
        self.pending, self.pending_n = None, 0
        self.prev = None
        self.filt.reset()

    def toggle_arm(self, t):
        self._pinch_end(t, silent=True)
        self.out.armed = not self.out.armed
        self._flash("ARMED" if self.out.armed else "DISARMED", t)

    # ---- pinch ----
    def _pinch_start(self, t):
        self.mode = "PINCH"
        self.pinch_acc[:] = 0
        self.dragging = False
        self.release_n = 0
        if t - self.last_release_t >= self.cfg.min_click_gap_s:
            self._press()
        else:
            self.press_pending = True

    def _press(self):
        self.out.press()
        self.down = True
        self.press_pending = False

    def _pinch_move(self, delta_px, t):
        if self.press_pending and t - self.last_release_t >= self.cfg.min_click_gap_s:
            self._press()
        if self.press_pending:
            return
        if not self.dragging:
            self.pinch_acc += delta_px
            if np.linalg.norm(self.pinch_acc) > self.cfg.drag_deadzone_px:
                self.dragging = True
                self.out.move_by(*self.pinch_acc)
        else:
            self.out.move_by(*delta_px)

    def _pinch_end(self, t, silent=False):
        was = self.down
        self.out.release()
        self.down = False
        self.press_pending = False
        if was:
            self.last_release_t = t
            if not silent:
                self._flash("DRAG END" if self.dragging else "CLICK", t)
        self.dragging = False

    # ---- main update ----
    def update(self, hands: list[HandObs], t: float):
        c = self.cfg

        # Two-hand arm/disarm: both palms open and held.
        if len(hands) >= 2 and hands[0].pose == "PALM" and hands[1].pose == "PALM":
            if self.arm_t0 is None:
                self.arm_t0, self.arm_fired = t, False
                self._pinch_end(t, silent=True)
            elif not self.arm_fired and t - self.arm_t0 >= c.arm_hold_s:
                self.toggle_arm(t)
                self.arm_fired = True
            self.mode = "BOTH PALMS"
            self.prev = None
            self.filt.reset()
            self.primary = None
            return
        self.arm_t0 = None

        h = self._pick_primary(hands)
        self.primary = h
        if h is None:
            if self.lost_t0 is None:
                self.lost_t0 = t
            if t - self.lost_t0 > c.lost_grace_s and self.mode != "IDLE":
                self._hard_reset()
            return
        self.lost_t0 = None

        # Palm motion -> per-frame delta (camera-widths) and speed.
        p = self.filt(h.palm, t)
        if self.prev is None:
            d, dt = np.zeros(2), 1 / 30
        else:
            d, dt = p - self.prev, max(t - self.prev_t, 1e-3)
        self.prev, self.prev_t = p, t
        speed = float(np.linalg.norm(d)) / dt
        delta_px = d * self._gain_px(speed)

        # ---- PINCH has priority and ignores other fingers until released ----
        if self.mode == "PINCH":
            self.release_n = self.release_n + 1 if h.pinch_mm > c.pinch_off_mm else 0
            if self.release_n >= 2:
                self._pinch_end(t)
                self.mode = "POINT"
                self.pending, self.pending_n = "POINT", c.stable_frames
            else:
                self._pinch_move(delta_px, t)
            return

        can_pinch = self.mode in ("POINT", "HOLD") and h.index_reach > 1.25
        if can_pinch and h.pinch_mm < c.pinch_on_mm:
            self._pinch_start(t)
            return

        # ---- debounced pose -> mode ----
        target = POSE_TO_MODE.get(h.pose, "HOLD")
        if target == self.pending:
            self.pending_n += 1
        else:
            self.pending, self.pending_n = target, 1
        if target != self.mode and self.pending_n >= c.stable_frames:
            self.mode = target
            if target == "PALM":
                self.palm_t0, self.palm_fired = t, False
            elif target == "ZOOM":
                self.zoom_t0, self.zoom_acc = t, 0.0

        # ---- act ----
        if self.mode == "POINT":
            semi = smoothstep(h.pinch_mm, c.pinch_on_mm, c.pinch_semi_mm)
            k = c.semi_min_gain + (1 - c.semi_min_gain) * semi
            self.out.move_by(*(delta_px * k))

        elif self.mode == "ZOOM":
            if t - self.zoom_t0 >= c.zoom_settle_s:
                ln_zoom = -d[1] * c.zoom_speed * (-1 if c.zoom_invert else 1)
                self.zoom_acc += ln_zoom / (c.wheel_zoom_const * c.scroll_px_per_step)
                n = int(self.zoom_acc)
                if n:
                    self.zoom_acc -= n
                    self.out.scroll(n)

        elif self.mode == "PALM":
            if not self.palm_fired and t - self.palm_t0 >= c.palm_hold_s:
                self.out.escape()
                self.palm_fired = True
                self._flash("ESC", t)

        # CLUTCH / HOLD: deliberately nothing; prev keeps updating so no jump on resume.


# ================================ HUD ===================================== #
# A small fixed-size panel styled like the Brain artifact (same tokens, IBM Plex,
# eyebrow labels, pills, sidebar-style rows). Rendered with Pillow, shown with
# OpenCV in an AUTOSIZE window pinned to the bottom-right corner of the screen.

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

THEMES = {
    "dark": dict(
        bg="#101317", panel="#171B20", panel2="#20252B", ink="#E2E5E0", muted="#8E979F",
        line="#2A3037", edge="#3B434C", accent="#95A3FF", accent_soft="#252C4C", danger="#FF8A80",
        g_ink="#8F9EFF", g_teal="#4FC7C2", g_moss="#93C76A", g_ochre="#E3B34F",
        g_rust="#F08A68", g_plum="#C394E6", g_rose="#F07BA8", g_slate="#9AA8B5",
    ),
    "light": dict(
        bg="#ECEEE9", panel="#F8F9F6", panel2="#E4E7E1", ink="#1C2126", muted="#646C73",
        line="#D2D7CF", edge="#B2BAB2", accent="#2F43B8", accent_soft="#DCE0F5", danger="#B3261E",
        g_ink="#2F43B8", g_teal="#0F7C7A", g_moss="#4F7A28", g_ochre="#A87410",
        g_rust="#B4472A", g_plum="#7B3FA0", g_rose="#B23A6B", g_slate="#56636F",
    ),
}

# (mode, label, gesture hint, group colour)
MODE_ROWS = [
    ("POINT", "Move", "index", "g_teal"),
    ("PINCH", "Click / drag", "pinch", "g_ink"),
    ("ZOOM", "Zoom", "V · up = in", "g_moss"),
    ("CLUTCH", "Clutch", "fist", "g_slate"),
    ("PALM", "Close note", "palm hold", "g_ochre"),
    ("BOTH PALMS", "Arm / disarm", "both palms", "g_rose"),
]

FONT_CDN = {
    "sans": "https://cdn.jsdelivr.net/npm/@ibm/plex-sans@1.1.0/fonts/complete/woff/IBMPlexSans-Regular.woff",
    "sans_med": "https://cdn.jsdelivr.net/npm/@ibm/plex-sans@1.1.0/fonts/complete/woff/IBMPlexSans-Medium.woff",
    "mono": "https://cdn.jsdelivr.net/npm/@ibm/plex-mono@2.5.0/fonts/complete/woff/IBMPlexMono-Regular.woff",
}
FONT_FALLBACKS = {
    "sans": ["/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc",
             "C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "mono": ["/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc",
             "C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
}


def _rgb(hexs):
    hexs = hexs.lstrip("#")
    return tuple(int(hexs[i:i + 2], 16) for i in (0, 2, 4))


def _bgr(hexs):
    r, g, b = _rgb(hexs)
    return (b, g, r)


def _mix(a, b, t):
    a, b = _rgb(a), _rgb(b)
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _font_path(key):
    """IBM Plex (cached next to the script) -> system font -> None."""
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    path = os.path.join(folder, os.path.basename(FONT_CDN[key]))
    if not os.path.exists(path):
        try:
            os.makedirs(folder, exist_ok=True)
            with urllib.request.urlopen(FONT_CDN[key], timeout=5) as r, open(path + ".part", "wb") as f:
                f.write(r.read())
            os.replace(path + ".part", path)
        except Exception:  # noqa: BLE001
            path = None
    if path:
        return path
    for p in FONT_FALLBACKS["mono" if key == "mono" else "sans"]:
        if os.path.exists(p):
            return p
    return None


def _load_font(key, size):
    p = _font_path(key)
    try:
        return ImageFont.truetype(p, size) if p else ImageFont.load_default(size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


class Hud:
    def __init__(self, cfg: Config, theme="dark", scale=1.0, show_camera=True):
        self.cfg, self.theme, self.s, self.show_camera = cfg, theme, scale, show_camera
        S = self._px
        self.W = S(320)
        self.pad = S(14)
        self.cam_w = self.W - 2 * self.pad
        self.cam_h = int(self.cam_w * 3 / 4)
        self.f_title = _load_font("sans_med", S(15))
        self.f_ui = _load_font("sans", S(13))
        self.f_ui_med = _load_font("sans_med", S(13))
        self.f_mono = _load_font("mono", S(11))
        self.f_toast = _load_font("sans_med", S(12))
        m = Image.new("L", (self.cam_w, self.cam_h), 0)
        ImageDraw.Draw(m).rounded_rectangle((0, 0, self.cam_w - 1, self.cam_h - 1), radius=S(8), fill=255)
        self.cam_mask = m

    def _px(self, v):
        return max(1, int(round(v * self.s)))

    @property
    def T(self):
        return THEMES[self.theme]

    def toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"

    # ---- small drawing helpers ----
    def _eyebrow(self, d, x, y, text, fill):
        """Mono, uppercase, letter-spaced — the Brain's .eyebrow."""
        track = self._px(11) * 0.08
        for ch in text.upper():
            d.text((x, y), ch, font=self.f_mono, fill=fill, anchor="lm")
            x += self.f_mono.getlength(ch) + track
        return x

    def _pill(self, d, x_right, y_mid, text, dot):
        S, T = self._px, self.T
        tw = self.f_mono.getlength(text)
        w, h = tw + S(26), S(20)
        x0 = x_right - w
        d.rounded_rectangle((x0, y_mid - h / 2, x_right, y_mid + h / 2), radius=h / 2, fill=_rgb(T["panel2"]))
        d.ellipse((x0 + S(9), y_mid - S(3), x0 + S(15), y_mid + S(3)), fill=_rgb(dot))
        d.text((x0 + S(20), y_mid), text, font=self.f_mono, fill=_rgb(T["ink"]), anchor="lm")

    def _kbd(self, d, x, y_mid, key, label):
        S, T = self._px, self.T
        kw = self.f_mono.getlength(key) + S(9)
        d.rounded_rectangle((x, y_mid - S(8), x + kw, y_mid + S(8)), radius=S(3),
                            fill=_rgb(T["panel"]), outline=_rgb(T["line"]))
        d.text((x + kw / 2, y_mid), key, font=self.f_mono, fill=_rgb(T["ink"]), anchor="mm")
        x += kw + S(5)
        d.text((x, y_mid), label, font=self.f_ui, fill=_rgb(T["muted"]), anchor="lm")
        return x + self.f_ui.getlength(label) + S(12)

    # ---- camera thumbnail with landmarks ----
    def _camera(self, frame, hands, eng, now):
        T, S = self.T, self._px
        fh, fw = frame.shape[:2]
        # centre-crop to 4:3 so any camera aspect fits without distortion
        tw = min(fw, int(fh * 4 / 3))
        th = int(tw * 3 / 4)
        x0, y0 = (fw - tw) // 2, (fh - th) // 2
        img = cv2.resize(frame[y0:y0 + th, x0:x0 + tw], (self.cam_w, self.cam_h), interpolation=cv2.INTER_AREA)
        k = self.cam_w / tw
        teal, muted, ochre, accent = _bgr(T["g_teal"]), _bgr(T["muted"]), _bgr(T["g_ochre"]), _bgr(T["accent"])
        lw = max(1, S(1.4))
        for hand in hands:
            primary = eng.primary is hand
            col = teal if primary else muted
            pts = ((hand.px - (x0, y0)) * k).astype(int)
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, tuple(pts[a]), tuple(pts[b]), col, lw, cv2.LINE_AA)
            for q in pts:
                cv2.circle(img, tuple(q), S(2), col, -1, cv2.LINE_AA)
            if primary:
                pc = tuple(pts[PALM_IDS].mean(axis=0).astype(int))
                rc = accent if eng.mode == "PINCH" else teal
                cv2.circle(img, pc, S(11), rc, lw, cv2.LINE_AA)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    cv2.line(img, (pc[0] + dx * S(6), pc[1] + dy * S(6)),
                             (pc[0] + dx * S(17), pc[1] + dy * S(17)), rc, lw, cv2.LINE_AA)
                close = 1 - smoothstep(hand.pinch_mm, self.cfg.pinch_on_mm, self.cfg.pinch_semi_mm)
                if close > 0.02:
                    cv2.line(img, tuple(pts[THUMB_TIP]), tuple(pts[INDEX_TIP]), ochre,
                             lw + int(S(3) * close), cv2.LINE_AA)
        if eng.arm_t0 is not None and not eng.arm_fired:
            frac = min((now - eng.arm_t0) / self.cfg.arm_hold_s, 1.0)
            cv2.ellipse(img, (self.cam_w // 2, self.cam_h // 2), (S(30), S(30)), -90, 0, int(360 * frac),
                        _bgr(T["g_rose"]), S(3), cv2.LINE_AA)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # ---- full panel ----
    def render(self, frame, hands, eng: "GestureEngine", fps: float):
        T, S, pad, W = self.T, self._px, self.pad, self.W
        now = time.monotonic()
        row_h = S(30)
        H = (S(48) + (self.cam_h + S(12) if self.show_camera else 0) + S(26)
             + row_h * len(MODE_ROWS) + S(14) + S(46) + S(40))
        im = Image.new("RGB", (W, H), _rgb(T["panel"]))
        d = ImageDraw.Draw(im)

        # header: eyebrow + title + status pill
        y = S(16)
        self._eyebrow(d, pad, y, "Brain · gestures", _rgb(T["muted"]))
        d.text((pad, y + S(17)), "Hand control", font=self.f_title, fill=_rgb(T["ink"]), anchor="lm")
        if eng.out.dry_run:
            tag, dot = "DRY RUN", T["g_ochre"]
        elif eng.out.armed:
            tag, dot = "ARMED", T["g_moss"]
        else:
            tag, dot = "DISARMED", T["danger"]
        self._pill(d, W - pad, y + S(9), tag, dot)
        y = S(48)

        # camera card
        if self.show_camera:
            cam = self._camera(frame, hands, eng, now)
            im.paste(cam, (pad, y), self.cam_mask)
            d.rounded_rectangle((pad, y, pad + self.cam_w - 1, y + self.cam_h - 1), radius=S(8),
                                outline=_rgb(T["line"]))
            if eng.event and now - eng.event_t < 0.7:
                tw = self.f_toast.getlength(eng.event)
                bx, by = pad + S(8), y + self.cam_h - S(30)
                d.rounded_rectangle((bx, by, bx + tw + S(20), by + S(22)), radius=S(6), fill=_rgb(T["accent"]))
                d.text((bx + S(10), by + S(11)), eng.event, font=self.f_toast, fill=_rgb(T["panel"]), anchor="lm")
            y += self.cam_h + S(12)
        elif eng.event and now - eng.event_t < 0.7:
            self._pill(d, W - pad, S(16) + S(9), eng.event, T["accent"])

        # mode list (sidebar .item rows)
        hand_txt = (f"{eng.primary.label.lower()} hand" if eng.primary else "no hand") + f" · {fps:.0f} fps"
        y += S(10)
        self._eyebrow(d, pad, y, "Mode", _rgb(T["muted"]))
        d.text((W - pad, y), hand_txt, font=self.f_mono, fill=_rgb(T["muted"]), anchor="rm")
        y += S(16)
        for mode, label, hint, gcol in MODE_ROWS:
            on = eng.mode == mode
            if on:
                d.rounded_rectangle((pad - S(6), y, W - pad + S(6), y + row_h - S(2)), radius=S(6),
                                    fill=_rgb(T["accent_soft"]))
            cy = y + (row_h - S(2)) / 2
            dot = _rgb(T[gcol]) if on else _mix(T[gcol], T["panel"], 0.45)
            d.ellipse((pad + S(2), cy - S(4.5), pad + S(11), cy + S(4.5)), fill=dot)
            d.text((pad + S(20), cy), label, font=self.f_ui_med if on else self.f_ui,
                   fill=_rgb(T["ink"] if on else T["muted"]), anchor="lm")
            d.text((W - pad, cy), hint, font=self.f_mono, fill=_rgb(T["accent"] if on else T["muted"]), anchor="rm")
            # hold-progress underline (palm -> Esc, both palms -> arm)
            t0, dur, fired = None, 1.0, True
            if mode == "PALM" and on:
                t0, dur, fired = eng.palm_t0, self.cfg.palm_hold_s, eng.palm_fired
            if mode == "BOTH PALMS" and on:
                t0, dur, fired = eng.arm_t0, self.cfg.arm_hold_s, eng.arm_fired
            if t0 is not None and not fired:
                frac = min((now - t0) / dur, 1.0)
                x0, x1 = pad + S(20), W - pad
                d.rounded_rectangle((x0, y + row_h - S(5), x0 + (x1 - x0) * frac, y + row_h - S(3)),
                                    radius=S(1), fill=_rgb(T[gcol]))
            y += row_h

        # pinch meter
        y += S(14)
        self._eyebrow(d, pad, y, "Pinch", _rgb(T["muted"]))
        if eng.primary:
            mm = eng.primary.pinch_mm
            state = "drag" if eng.dragging else ("down" if eng.down else "")
            val = f"{mm:3.0f} mm" + (f" · {state}" if state else "")
        else:
            mm, val = None, "--"
        d.text((W - pad, y), val, font=self.f_mono, fill=_rgb(T["ink"] if eng.down else T["muted"]), anchor="rm")
        y += S(16)
        span = 90.0
        x0, x1 = pad, W - pad
        d.rounded_rectangle((x0, y, x1, y + S(6)), radius=S(3), fill=_rgb(T["panel2"]))
        if mm is not None:
            fill = min(max(1 - mm / span, 0.0), 1.0)
            col = T["accent"] if eng.down else T["g_teal"]
            if fill > 0:
                d.rounded_rectangle((x0, y, x0 + max((x1 - x0) * fill, S(6)), y + S(6)), radius=S(3), fill=_rgb(col))
        for thr in (self.cfg.pinch_on_mm, self.cfg.pinch_off_mm):
            tx = x0 + (x1 - x0) * (1 - thr / span)
            d.line((tx, y - S(3), tx, y + S(9)), fill=_rgb(T["edge"]), width=1)
        y += S(30)

        # footer: hint keys (the Brain's .hint with <kbd>)
        d.line((0, y - S(6), W, y - S(6)), fill=_rgb(T["line"]), width=1)
        cy = y + S(14)
        x = pad
        for key, label in (("a", "arm"), ("h", "camera"), ("t", "theme"), ("q", "quit")):
            x = self._kbd(d, x, cy, key, label)

        # outer hairline, like the Brain's panels
        d.rectangle((0, 0, W - 1, H - 1), outline=_rgb(T["line"]))
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


# ================================ MAIN ==================================== #


def ensure_model(cfg: Config) -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
    if not os.path.exists(path):
        print("Downloading MediaPipe hand model (~8 MB)…")
        try:
            urllib.request.urlretrieve(cfg.model_url, path)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"Could not download the model ({e}).\nGet it manually from:\n  {cfg.model_url}\n"
                     f"and save it as:\n  {path}")
    return path


def screen_size(override_w: int | None) -> tuple[int, int]:
    w, h = 1920, 1080
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        w, h = int(root.winfo_screenwidth()), int(root.winfo_screenheight())
        root.destroy()
    except Exception:  # noqa: BLE001
        pass
    if override_w:
        h = int(h * override_w / w)
        w = override_w
    return w, h


def main():
    ap = argparse.ArgumentParser(description="One-hand webcam control for the Brain graph.")
    ap.add_argument("--camera", type=int, default=None, help="camera index (default 0)")
    ap.add_argument("--hand", choices=["Right", "Left"], default=None, help="controlling hand")
    ap.add_argument("--armed", action="store_true", help="start armed")
    ap.add_argument("--dry-run", action="store_true", help="never touch mouse/keyboard (calibration)")
    ap.add_argument("--no-preview", action="store_true", help="no HUD window at all")
    ap.add_argument("--no-camera", action="store_true", help="HUD without the camera thumbnail (toggle: h)")
    ap.add_argument("--theme", choices=["dark", "light"], default="dark", help="HUD theme (toggle: t)")
    ap.add_argument("--hud-scale", type=float, default=1.0, help="HUD size multiplier, e.g. 1.25")
    ap.add_argument("--screen-width", type=int, default=None, help="override detected screen width (px)")
    args = ap.parse_args()

    cfg = Config()
    if args.camera is not None:
        cfg.camera = args.camera
    if args.hand:
        cfg.hand = args.hand

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model(cfg)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
    )

    cap = cv2.VideoCapture(cfg.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {cfg.camera}. Try --camera 1, and check camera permissions.")

    out = Output(dry_run=args.dry_run)
    sw, sh = screen_size(args.screen_width)
    eng = GestureEngine(cfg, out, sw)
    if args.armed and not args.dry_run:
        out.armed = True

    win = "Brain gestures"
    hud = None
    placed = False
    if not args.no_preview:
        hud = Hud(cfg, theme=args.theme, scale=args.hud_scale, show_camera=not args.no_camera)
        # AUTOSIZE = window is exactly the panel's size and can't be stretched to full screen.
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    print(__doc__.split("Design choices")[0])
    t0 = time.monotonic()
    last_ts = -1
    fps, fps_t, fps_n = 0.0, time.monotonic(), 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)  # mirror: move right -> cursor right
            fh, fw = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            now = time.monotonic()
            ts = max(int((now - t0) * 1000), last_ts + 1)
            last_ts = ts
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            hands = []
            for i, lm in enumerate(res.hand_landmarks or []):
                label = res.handedness[i][0].category_name if res.handedness else "Right"
                hands.append(analyse_hand(lm, res.hand_world_landmarks[i], label, fw, fh, cfg))

            eng.update(hands, now)

            fps_n += 1
            if now - fps_t >= 0.5:
                fps, fps_t, fps_n = fps_n / (now - fps_t), now, 0

            if hud is not None:
                panel = hud.render(frame, hands, eng, fps)
                cv2.imshow(win, panel)
                if not placed:
                    ph, pw = panel.shape[:2]
                    cv2.moveWindow(win, max(sw - pw - 24, 0), max(sh - ph - 90, 0))  # bottom-right
                    try:
                        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
                    except Exception:  # noqa: BLE001
                        pass
                    placed = True
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("a"), ord("A")) and not args.dry_run:
                    eng.toggle_arm(now)
                if key in (ord("h"), ord("H")):
                    hud.show_camera = not hud.show_camera
                if key in (ord("t"), ord("T")):
                    hud.toggle_theme()
                try:
                    if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                        break  # window closed with the close button
                except Exception:  # noqa: BLE001
                    pass
    except KeyboardInterrupt:
        pass
    finally:
        out.release()
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
