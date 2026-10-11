"""Render a photo/video montage reel from a timeline JSON (see edits/papa_reel.json).

Unlike render.py (which only cuts source clips), this composites every frame itself:
Ken Burns moves on photos, slow-mo clips, old prints on a blurred backdrop, a collage,
per-shot colour grades, dissolve/blur/flash/dip transitions, film grain, vignette,
light leaks and animated text, then muxes the voice-over with ffmpeg.

    .venv/bin/python tools/make_reel.py edits/papa_reel.json
    .venv/bin/python tools/make_reel.py edits/papa_reel.json --stills 3.0,10.9,36   # PNG frames only
    .venv/bin/python tools/make_reel.py edits/papa_reel.json --preview              # half-res, fast
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONTS = ROOT / "assets" / "fonts"

# ---------------------------------------------------------------- colour grades
GRADES = {
    #            sat   contrast warmth lift  teal  orange bright
    "golden":    (0.92, 1.08,   0.75,  0.03, 0.035, 0.05, 1.00),
    "present":   (0.90, 1.08,   0.45,  0.03, 0.04,  0.04, 1.00),
    "papa":      (0.95, 1.06,   0.60,  0.03, 0.03,  0.05, 1.02),
    "lost":      (0.60, 1.06,   0.10,  0.04, 0.06,  0.01, 0.97),
    "bright":    (0.98, 1.06,   0.55,  0.03, 0.03,  0.05, 1.04),
    "nostalgic": (0.78, 1.02,   0.85,  0.06, 0.03,  0.06, 1.00),
    "coach":     (0.75, 1.08,   0.30,  0.04, 0.05,  0.03, 0.98),
    "vintage":   (0.80, 1.00,   0.80,  0.07, 0.02,  0.06, 1.00),
}


def grade(img, name):
    sat, con, warm, lift, teal, orange, bright = GRADES[name]
    lum = img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722
    lum = lum[..., None]
    img = lum + (img - lum) * sat
    img = img * bright
    # gentle S-curve: blend linear contrast with smoothstep
    img = np.clip(img, 0, 1)
    s = img * img * (3 - 2 * img)
    img = img + (s - img) * (con - 1) * 2.0
    # warmth
    img[..., 0] *= 1 + 0.06 * warm
    img[..., 2] *= 1 - 0.09 * warm
    # split tone: teal shadows, orange highlights
    sh = np.clip(1 - lum * 2.2, 0, 1)
    hi = np.clip(lum * 1.6 - 0.6, 0, 1)
    img[..., 0] += -teal * sh[..., 0] + orange * hi[..., 0]
    img[..., 1] += teal * 0.4 * sh[..., 0] + orange * 0.45 * hi[..., 0]
    img[..., 2] += teal * sh[..., 0] - orange * 0.6 * hi[..., 0]
    # faded blacks
    img = lift + img * (1 - lift)
    return np.clip(img, 0, 1)


# ---------------------------------------------------------------- helpers
def ease(p):  # easeInOutSine
    p = min(max(p, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * p)


def ease_out(p):
    p = min(max(p, 0.0), 1.0)
    return 1 - (1 - p) ** 3


def load_rgb(path):
    im = Image.open(ROOT / path).convert("RGB")
    return np.asarray(im).astype(np.float32) / 255.0


def interp_keys(keys, x):
    """keys: [[x, cx, cy, w], ...] sorted by x. Eased piecewise interpolation."""
    if x <= keys[0][0]:
        return keys[0][1:]
    for a, b in zip(keys, keys[1:]):
        if x <= b[0]:
            p = ease((x - a[0]) / (b[0] - a[0]))
            return [a[i] + (b[i] - a[i]) * p for i in (1, 2, 3)]
    return keys[-1][1:]


def warp_rect(src, cx, cy, w, W, H, interp=cv2.INTER_CUBIC):
    """Sample a W:H window centred on (cx, cy) with width w (source px) into W x H."""
    s = w / W
    h = s * H
    M = np.float32([[s, 0, cx - w / 2], [0, s, cy - h / 2]])
    return cv2.warpAffine(src, M, (W, H), flags=interp | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT)


def neutralise(img, amount=0.75):
    """Partial gray-world white balance (old faded prints have a strong pink cast)."""
    h, w = img.shape[:2]
    c = img[h // 6: -h // 6, w // 6: -w // 6].reshape(-1, 3)
    means = c.mean(0)
    gains = means.mean() / means
    gains = 1 + (gains - 1) * amount
    out = np.clip(img * gains, 0, 1)
    # old prints are washed out - restore a bit of punch
    lo, hi = np.percentile(out, 1), np.percentile(out, 99.5)
    return np.clip((out - lo) / (hi - lo), 0, 1)


def make_card(img, width, border=0.035):
    """Photo print with a cream border and soft drop shadow -> premultiplied RGBA float."""
    h, w = img.shape[:2]
    height = int(round(width * h / w))
    photo = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    b = int(width * border)
    pad = int(width * 0.08)
    cw, ch = width + 2 * b, height + 2 * b
    W, H = cw + 2 * pad, ch + 2 * pad
    rgb = np.zeros((H, W, 3), np.float32)
    a = np.zeros((H, W), np.float32)
    # shadow
    sh = np.zeros((H, W), np.float32)
    sh[pad + int(pad * 0.25): pad + ch + int(pad * 0.25), pad + int(pad * 0.1): pad + cw + int(pad * 0.1)] = 0.6
    sh = cv2.GaussianBlur(sh, (0, 0), pad * 0.35)
    a = sh.copy()
    # cream border + photo
    rgb[pad:pad + ch, pad:pad + cw] = (0.96, 0.93, 0.86)
    rgb[pad + b:pad + b + height, pad + b:pad + b + width] = photo
    a[pad:pad + ch, pad:pad + cw] = 1.0
    return rgb * a[..., None], a


def place_rgba(frame, rgb, a, cx, cy, scale, rot_deg, alpha=1.0):
    """Composite premultiplied rgb/a onto frame, centred at (cx, cy), scaled and rotated."""
    if alpha <= 0.001:
        return frame
    H, W = frame.shape[:2]
    h, w = a.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot_deg, scale)
    M[0, 2] += cx - w / 2
    M[1, 2] += cy - h / 2
    stack = np.dstack([rgb, a])
    out = cv2.warpAffine(stack, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    pa = out[..., 3:4] * alpha
    return frame * (1 - pa) + out[..., :3] * alpha


# ---------------------------------------------------------------- sources
class ImageSeg:
    def __init__(self, spec, W, H):
        img = load_rgb(spec["source"])
        if spec.get("inpaint_green"):
            img = self._inpaint_green(img)
        self.img = img
        self.spec = spec
        self.W, self.H = W, H
        self.scale = 1.0

    @staticmethod
    def _inpaint_green(img):
        u8 = (img * 255).astype(np.uint8)
        r, g, b = [u8[..., i].astype(int) for i in range(3)]
        mask = ((g > 140) & (g > r + 50) & (g > b + 30)).astype(np.uint8) * 255
        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        fixed = cv2.inpaint(bgr, mask, 9, cv2.INPAINT_TELEA)
        return cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def render(self, t, u):
        s = self.spec
        if "keys_abs" in s:
            cx, cy, w = interp_keys(s["keys_abs"], t)
        else:
            cx, cy, w = interp_keys(s["keys"], u)
        return warp_rect(self.img, cx * self.scale, cy * self.scale, w * self.scale, self.W, self.H)


class VideoSeg:
    def __init__(self, spec, W, H, fps):
        self.spec, self.W, self.H, self.fps = spec, W, H, fps
        path = str(ROOT / spec["source"])
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=width,height", "-of", "csv=p=0", path],
                               capture_output=True, text=True, check=True).stdout.strip().split(",")
        self.sw, self.sh = int(probe[0]), int(probe[1])
        self.path = path
        self.proc = None
        self.idx = -1
        self.cur = None

    def _open(self):
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", self.path, "-f", "rawvideo",
                                      "-pix_fmt", "rgb24", "-r", str(self.fps), "-"],
                                     stdout=subprocess.PIPE)

    def frame_at(self, tau):
        if self.proc is None:
            self._open()
        want = int(round((tau + self.spec.get("src_offset", 0.0)) * self.fps))
        n = self.sw * self.sh * 3
        while self.idx < want:
            buf = self.proc.stdout.read(n)
            if len(buf) < n:  # end of clip: hold last frame
                break
            self.cur = np.frombuffer(buf, np.uint8).reshape(self.sh, self.sw, 3)
            self.idx += 1
        return self.cur.astype(np.float32) / 255.0

    def render(self, t, u, tau):
        src = self.frame_at(tau)
        cx, cy, w = interp_keys(self.spec["keys"], u)
        out = warp_rect(src, cx, cy, w, self.W, self.H, cv2.INTER_LANCZOS4)
        if "content" in self.spec:
            # source is letterboxed: fill the bars with a blurred, darkened copy of the picture
            x, y, cw, ch = self.spec["content"]
            mask = np.zeros(src.shape[:2], np.float32)
            mask[y + 2:y + ch - 2, x:x + cw] = 1
            m = warp_rect(mask, cx, cy, w, self.W, self.H, cv2.INTER_LINEAR)
            m = cv2.GaussianBlur(m, (0, 0), 3)[..., None]
            bg = blurred_backdrop(src[y:y + ch, x:x + cw], self.W, self.H, darken=0.6)
            out = out * m + bg * (1 - m)
        if self.spec.get("sharpen"):
            blur = cv2.GaussianBlur(out, (0, 0), 2.2)
            out = np.clip(out + (out - blur) * 0.6, 0, 1)
        return out


def blurred_backdrop(img, W, H, darken=0.55):
    h, w = img.shape[:2]
    s = max(W / w, H / h) * 1.15
    big = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
    y0, x0 = (big.shape[0] - H) // 2, (big.shape[1] - W) // 2
    bg = big[y0:y0 + H, x0:x0 + W]
    bg = cv2.GaussianBlur(bg, (0, 0), 40)
    return np.clip(bg * darken, 0, 1)


class PrintSeg:
    def __init__(self, spec, W, H, card_w=None):
        x0, y0, x1, y1 = spec["crop"]
        img = neutralise(load_rgb(spec["source"])[y0:y1, x0:x1])
        self.spec, self.W, self.H = spec, W, H
        self.bg = blurred_backdrop(img, W, H, darken=0.7)
        self.card = make_card(img, card_w or int(W * 0.86))

    def render(self, t, u):
        r0, r1 = self.spec.get("rot", [0, 0])
        z = 1.0 + 0.04 * ease(u)
        bg = warp_rect(self.bg, self.W / 2, self.H / 2, self.W / (1 + 0.05 * u), self.W, self.H, cv2.INTER_LINEAR)
        return place_rgba(bg, *self.card, self.W / 2, self.H * 0.42, z, r0 + (r1 - r0) * ease(u))


class CollageSeg:
    def __init__(self, spec, W, H):
        self.spec, self.W, self.H = spec, W, H
        bgimg = neutralise(load_rgb(spec["background"]))
        self.bg = blurred_backdrop(bgimg, W, H, darken=0.42)
        self.cards = []
        for c in spec["cards"]:
            x0, y0, x1, y1 = c["crop"]
            img = neutralise(load_rgb(c["source"])[y0:y1, x0:x1])
            self.cards.append((c, make_card(img, c["width"])))

    def render(self, t, u, tau):
        push = 1.0 + 0.05 * ease(u)
        frame = warp_rect(self.bg, self.W / 2, self.H / 2, self.W / push, self.W, self.H, cv2.INTER_LINEAR)
        for c, (rgb, a) in self.cards:
            p = ease_out((tau - 0.35 - c["delay"]) / 0.7)
            if p <= 0:
                continue
            cx = self.W / 2 + (c["center"][0] - self.W / 2) * push
            cy = self.H / 2 + (c["center"][1] - self.H / 2) * push
            sc = (1.25 - 0.25 * p) * push
            frame = place_rgba(frame, rgb, a, cx, cy, sc, c["rot"] + 6 * (1 - p), alpha=min(1.0, p * 1.6))
        return frame


# ---------------------------------------------------------------- text
class TextLayer:
    STYLES = {
        "q":     dict(font="CormorantGaramond[wght].ttf", wght=600, size=96, color=(255, 248, 236), y=1300),
        "a":     dict(font="GreatVibes-Regular.ttf", size=118, color=(243, 209, 156), y=1418),
        "label": dict(font="CormorantGaramond[wght].ttf", wght=500, size=40, color=(255, 244, 228), y=330, track=14),
        "note":  dict(font="PlayfairDisplay-Italic[wght].ttf", wght=500, size=62, color=(255, 246, 232), y=330),
        "end":   dict(font="GreatVibes-Regular.ttf", size=150, color=(246, 216, 166), y=1440),
    }

    def __init__(self, item, W, H, scale=1.0):
        st = self.STYLES[item["style"]]
        self.item, self.scale = item, scale
        size = int(st["size"] * scale)
        font = ImageFont.truetype(str(FONTS / st["font"]), size)
        if "wght" in st:
            try:
                font.set_variation_by_axes([st["wght"]])
            except Exception:
                pass
        text = item["text"]
        track = int(st.get("track", 0) * scale)
        # measure
        widths = [font.getbbox(ch)[2] if ch != " " else size * 0.28 for ch in text]
        tw = int(sum(widths) + track * (len(text) - 1)) if track else int(font.getbbox(text)[2])
        asc, desc = font.getmetrics()
        pad = int(40 * scale)
        cw, ch = tw + 2 * pad, asc + desc + 2 * pad
        img = Image.new("L", (cw, ch), 0)
        d = ImageDraw.Draw(img)
        if track:
            x = pad
            for c, wdt in zip(text, widths):
                d.text((x, pad), c, font=font, fill=255)
                x += wdt + track
        else:
            d.text((pad, pad), text, font=font, fill=255)
        glyph = np.asarray(img).astype(np.float32) / 255.0
        shadow = np.asarray(img.filter(ImageFilter.GaussianBlur(14 * scale))).astype(np.float32) / 255.0
        shadow = np.clip(shadow * 1.6, 0, 1) * 0.65
        col = np.array(st["color"], np.float32) / 255.0
        # premultiplied: shadow (black) under glyph
        self.a = np.clip(glyph + shadow * (1 - glyph), 0, 1)
        self.rgb = glyph[..., None] * col
        # per-line "cx"/"y" overrides move a line off baked-in text or faces
        self.x = int(item.get("cx", W / scale / 2) * scale) - cw // 2
        self.y = int(item.get("y", st["y"]) * scale) - ch // 2

    def apply(self, frame, t):
        it = self.item
        if t < it["in"] or t > it["out"]:
            return frame
        fin = ease_out((t - it["in"]) / 0.45)
        fout = 1.0 if it["style"] == "end" else ease((it["out"] - t) / 0.3)
        alpha = min(fin, fout)
        if alpha <= 0:
            return frame
        dy = int(round((1 - fin) * 26 * self.scale))
        y = self.y + dy
        h, w = self.a.shape
        H, W = frame.shape[:2]
        y0, y1 = max(y, 0), min(y + h, H)
        x0, x1 = max(self.x, 0), min(self.x + w, W)
        a = self.a[y0 - y:y1 - y, x0 - self.x:x1 - self.x, None] * alpha
        rgb = self.rgb[y0 - y:y1 - y, x0 - self.x:x1 - self.x] * alpha
        frame[y0:y1, x0:x1] = frame[y0:y1, x0:x1] * (1 - a) + rgb
        return frame


# ---------------------------------------------------------------- reel
class Reel:
    def __init__(self, spec, scale=1.0):
        self.spec = spec
        W0, H0 = spec["size"]
        self.W, self.H = int(W0 * scale) // 2 * 2, int(H0 * scale) // 2 * 2
        self.scale = scale
        self.fps = spec["fps"]
        self.trans = {round(tr["at"], 3): tr for tr in spec["transitions"]}
        self.segs = []
        for s in spec["segments"]:
            s = self._scaled(s, scale)
            kind = s["type"]
            if kind == "image":
                obj = ImageSeg(s, self.W, self.H)
                obj.scale = 1.0
            elif kind == "video":
                obj = VideoSeg(s, self.W, self.H, self.fps)
            elif kind == "print":
                obj = PrintSeg(s, self.W, self.H)
            elif kind == "collage":
                obj = CollageSeg(s, self.W, self.H)
            else:
                raise ValueError(kind)
            tin = self.trans.get(round(s["start"], 3), {"dur": 0})["dur"]
            tout = self.trans.get(round(s["end"], 3), {"dur": 0})["dur"]
            vs, ve = s["start"] - tin / 2, s["end"] + tout / 2
            self.segs.append((s, obj, vs, ve))
        self.texts = [TextLayer(it, self.W, self.H, scale) for it in spec["texts"]]
        self._prep_finishing()

    @staticmethod
    def _scaled(s, k):
        if k == 1.0:
            return s
        s = json.loads(json.dumps(s))
        if s["type"] == "image":
            pass  # image keys are in source pixels; output size handles the scale
        if s["type"] == "collage":
            for c in s["cards"]:
                c["center"] = [v * k for v in c["center"]]
                c["width"] = int(c["width"] * k)
        return s

    def _prep_finishing(self):
        W, H = self.W, self.H
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / math.sqrt(2)
        self.vignette = (1 - 0.42 * np.clip(r * 1.15, 0, 1) ** 2.2)[..., None]
        # soft darkening behind the lower-third text
        yn = yy[:, :1] / H
        self.text_shade = (1 - 0.38 * np.clip((yn - 0.55) / 0.2, 0, 1) * np.clip((0.92 - yn) / 0.12, 0.35, 1))[..., None]
        rng = np.random.default_rng(7)
        self.grain = []
        for _ in range(12):
            n = rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32)
            self.grain.append(cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)[..., None])
        # light leak: soft warm blobs hugging the right/top edge
        leak = np.zeros((H, W, 3), np.float32)
        for (cx, cy, rad, col) in [(1.05, 0.15, 0.55, (1.0, 0.55, 0.2)), (0.95, 0.55, 0.35, (1.0, 0.35, 0.15)),
                                   (-0.05, 0.85, 0.45, (1.0, 0.75, 0.4))]:
            d = ((xx / W - cx) ** 2 + ((yy / H - cy) * H / W) ** 2) ** 0.5
            leak += np.exp(-(d / (rad * 0.6)) ** 2)[..., None] * np.array(col, np.float32)
        self.leak = np.clip(leak, 0, 1)

    def _seg_frame(self, s, obj, vs, ve, t):
        tau = t - vs
        u = min(max(tau / (ve - vs), 0), 1)
        if isinstance(obj, (VideoSeg, CollageSeg)):
            img = obj.render(t, u, tau)
        else:
            img = obj.render(t, u)
        return grade(img, s["grade"])

    def frame(self, t):
        active = [(s, o, vs, ve) for (s, o, vs, ve) in self.segs if vs <= t < ve]
        if not active:
            active = [self.segs[-1]]
        if len(active) == 1:
            s, o, vs, ve = active[0]
            out = self._seg_frame(s, o, vs, ve, t)
        else:
            a, b = active[0], active[1]
            tr = self.trans[round(b[0]["start"], 3)]
            d = tr["dur"]
            p = ease((t - (tr["at"] - d / 2)) / d)
            fa = self._seg_frame(*a, t)
            fb = self._seg_frame(*b, t)
            kind = tr["type"]
            if kind == "blur":
                R = 18 * self.scale
                if p > 0.02:
                    fa = cv2.GaussianBlur(fa, (0, 0), R * p)
                if p < 0.98:
                    fb = cv2.GaussianBlur(fb, (0, 0), R * (1 - p))
            out = fa * (1 - p) + fb * p
            bump = 4 * p * (1 - p)
            if kind == "flash":
                warm = np.array([1.0, 0.95, 0.86], np.float32)
                k = bump * (0.9 if d > 0.5 else 0.55)
                out = out * (1 - k) + warm * k
            elif kind == "dip":
                out = out * (1 - 0.85 * bump)
        return self._finish(out, t)

    def _finish(self, img, t):
        dur = self.spec["duration"]
        # light leaks over the old-memories section and around flashes
        leak_k = 0.0
        if t >= 34.4:
            leak_k = 0.18 + 0.10 * math.sin(t * 1.3) ** 2
        for tr in self.spec["transitions"]:
            if tr["type"] == "flash":
                leak_k = max(leak_k, 0.45 * max(0.0, 1 - abs(t - tr["at"]) / (tr["dur"] * 0.9)))
        if leak_k > 0:
            shift = int((math.sin(t * 0.7) * 0.08) * self.W)
            leak = np.roll(self.leak, shift, axis=1)
            img = 1 - (1 - img) * (1 - leak * leak_k)
        img = img * self.vignette * self.text_shade
        # film grain, a touch stronger on the old prints
        g = 0.028 if t < 34.6 else 0.045
        n = self.grain[int(t * self.fps) % len(self.grain)]
        mid = 4 * img * (1 - img)
        img = img + n * g * (0.35 + mid)
        for tl in self.texts:
            img = tl.apply(img, t)
        # fades
        if t < self.spec["fade_in"]:
            img = img * ease(t / self.spec["fade_in"])
        if t > dur - self.spec["fade_out"]:
            img = img * ease((dur - t) / self.spec["fade_out"])
        return np.clip(img, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("timeline")
    ap.add_argument("--preview", action="store_true", help="half resolution, fast encode")
    ap.add_argument("--stills", help="comma-separated times; write PNGs to build/stills instead of a video")
    ap.add_argument("-o", "--output")
    args = ap.parse_args()

    spec = json.loads(Path(args.timeline).read_text())
    scale = 0.5 if args.preview else 1.0
    reel = Reel(spec, scale)
    fps, dur = spec["fps"], spec["duration"]

    if args.stills:
        outdir = ROOT / "build" / "stills"
        outdir.mkdir(parents=True, exist_ok=True)
        times = [float(x) for x in args.stills.split(",")]
        # video readers only move forward, so render stills in time order from fresh readers
        for t in sorted(times):
            img = reel.frame(t)
            p = outdir / f"t{t:05.2f}.png"
            Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(p)
            print(p)
        return

    out = Path(args.output or spec["output"])
    if args.preview:
        out = out.with_name(out.stem + "_preview" + out.suffix)
    out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{reel.W}x{reel.H}", "-r", str(fps), "-i", "-",
           "-i", str(ROOT / spec["audio"]), "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", "veryfast" if args.preview else "slow",
           "-crf", "23" if args.preview else "16", "-pix_fmt", "yuv420p",
           "-profile:v", "high", "-c:a", "aac", "-b:a", "256k",
           "-af", f"afade=t=out:st={dur - 0.4}:d=0.4", "-t", f"{dur}",
           "-movflags", "+faststart", str(out)]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(math.ceil(dur * fps))
    for i in range(n):
        t = i / fps
        img = reel.frame(t)
        enc.stdin.write((img * 255 + 0.5).astype(np.uint8).tobytes())
        if i % 60 == 0:
            print(f"\r{i}/{n} frames", end="", file=sys.stderr, flush=True)
    enc.stdin.close()
    enc.wait()
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
