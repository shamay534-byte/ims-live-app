# Reels renderer: forecast 28.11 - coats and umbrellas - a turnaround next week (alert-style title, side day card, face smoothing)
import sys, json, math, subprocess, random, functools, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
A = lambda n: os.path.join(HERE, "assets", n)
W, H, FPS = 1080, 1920, 30
VIDEO = os.path.join(HERE, "r17", "video.mp4")
NFR = 2240
KEEP = [tuple(k) for k in json.load(open(os.path.join(HERE, "r17", "keep.json"))) if k[1] - k[0] > 5]
CUTS = [(a1 / 30, b0 / 30) for (a0, a1), (b0, b1) in zip(KEEP, KEEP[1:])]
DUR = NFR / FPS

PERSON = np.load(os.path.join(HERE, "r17", "person.npy"), mmap_mode="r")
HEAD = np.load(os.path.join(HERE, "r17", "head.npy"), mmap_mode="r")
FACE = np.load(os.path.join(HERE, "r17", "face.npy"), mmap_mode="r")
from smooth13 import smooth_face

# ---------------------------------------------------------------- helpers
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ease_out_cubic(x): x = clamp(x); return 1 - (1 - x) ** 3
def ease_in_out(x): x = clamp(x); return x * x * (3 - 2 * x)
def ease_out_back(x, s=1.9):
    x = clamp(x); x -= 1; return x * x * ((s + 1) * x + s) + 1

def window(t, t0, t1, fin=0.25, fout=0.2):
    """returns (in_progress 0..1, alpha 0..1) or None if not visible"""
    if t < t0 or t > t1 + fout: return None
    pin = clamp((t - t0) / fin)
    a = ease_out_cubic(pin)
    if t > t1: a *= 1 - clamp((t - t1) / fout)
    return pin, a

@functools.lru_cache(None)
def font(size, weight=700):
    f = ImageFont.truetype(A("Assistant.ttf"), size)
    f.set_variation_by_axes([weight])
    return f

@functools.lru_cache(None)
def icon(name, size):
    im = Image.open(A(name)).convert("RGBA").resize((size, size), Image.LANCZOS)
    return np.asarray(im).astype(np.float32)

@functools.lru_cache(None)
def icon_shadowed(name, size, blur=10, off=10, op=0.45):
    im = Image.open(A(name)).convert("RGBA").resize((size, size), Image.LANCZOS)
    pad = blur * 3 + off
    can = Image.new("RGBA", (size + 2 * pad, size + 2 * pad), (0, 0, 0, 0))
    sh = Image.new("RGBA", can.size, (0, 0, 0, 0))
    a = im.split()[3].point(lambda v: int(v * op))
    blk = Image.new("RGBA", im.size, (10, 20, 40, 255)); blk.putalpha(a)
    sh.paste(blk, (pad + off // 2, pad + off), blk)
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    can = Image.alpha_composite(can, sh)
    can.alpha_composite(im, (pad, pad))
    return np.asarray(can).astype(np.float32), pad

def scaled(arr, s):
    if abs(s - 1) < 1e-3: return arr
    h, w = arr.shape[:2]
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    return cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA)

def rotated(arr, deg):
    if abs(deg) < 0.1: return arr
    h, w = arr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))

class Frame:
    def __init__(self, img, pm, hm):
        self.img = img          # float32 HxWx3
        self.pm = pm            # person mask float 0..1
        self.hm = hm            # dilated head mask (bool)
        self.head_overlap = 0

    def blit(self, spr, cx, cy, alpha=1.0, behind=False, scale=1.0, rot=0.0):
        if alpha <= 0.003: return
        spr = rotated(scaled(spr, scale), rot)
        h, w = spr.shape[:2]
        x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
        X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
        if X1 <= X0 or Y1 <= Y0: return
        s = spr[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        a = s[..., 3:4] * (alpha / 255.0)
        if behind:
            a = a * (1.0 - self.pm[Y0:Y1, X0:X1, None])
        else:
            ov = np.count_nonzero((a[..., 0] > 0.05) & self.hm[Y0:Y1, X0:X1])
            self.head_overlap += ov
        d = self.img[Y0:Y1, X0:X1]
        d *= (1 - a); d += s[..., :3] * a

# ---------------------------------------------------------------- text
def text_layer(text, size, weight=800, fill=(255, 255, 255), stroke=0, stroke_fill=(0, 0, 0),
               extrude=0, extrude_fill=(0, 0, 0), gradient=None, shadow=0, pad=40, fixed_line=False):
    f = font(size, weight)
    tmp = ImageDraw.Draw(Image.new("L", (1, 1)))
    bb = tmp.textbbox((0, 0), text, font=f, stroke_width=stroke, direction="rtl", language="he")
    if fixed_line:  # same vertical box for every word -> shared baseline
        asc, desc = f.getmetrics()
        bb = (bb[0], -stroke, bb[2], asc + desc + stroke)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    Wd, Hd = tw + 2 * pad + extrude, th + 2 * pad + extrude
    org = (pad - bb[0], pad - bb[1])
    out = Image.new("RGBA", (Wd, Hd), (0, 0, 0, 0))
    def mask(off=(0, 0), sw=stroke):
        m = Image.new("L", (Wd, Hd), 0)
        ImageDraw.Draw(m).text((org[0] + off[0], org[1] + off[1]), text, font=f, fill=255,
                               stroke_width=sw, stroke_fill=255, direction="rtl", language="he")
        return m
    if shadow:
        sm = mask((extrude + shadow // 2, extrude + shadow), stroke).filter(ImageFilter.GaussianBlur(shadow))
        sl = Image.new("RGBA", (Wd, Hd), (0, 0, 0, 255)); sl.putalpha(sm.point(lambda v: int(v * 0.55)))
        out = Image.alpha_composite(out, sl)
    for k in range(extrude, 0, -1):
        m = mask((k, k), stroke)
        c = tuple(int(v * (0.75 + 0.25 * (1 - k / max(1, extrude)))) for v in extrude_fill)
        l = Image.new("RGBA", (Wd, Hd), c + (255,)); l.putalpha(m)
        out = Image.alpha_composite(out, l)
    if stroke:
        l = Image.new("RGBA", (Wd, Hd), stroke_fill + (255,)); l.putalpha(mask((0, 0), stroke))
        out = Image.alpha_composite(out, l)
    m = mask((0, 0), 0)
    if gradient:
        c0, c1 = np.array(gradient[0], np.float32), np.array(gradient[1], np.float32)
        yy = np.linspace(0, 1, Hd)[:, None, None]
        g = (c0 * (1 - yy) + c1 * yy) * np.ones((1, Wd, 1))
        l = Image.fromarray(g.astype(np.uint8), "RGB").convert("RGBA")
    else:
        l = Image.new("RGBA", (Wd, Hd), tuple(fill) + (255,))
    l.putalpha(m)
    out = Image.alpha_composite(out, l)
    return out

def to_np(im): return np.asarray(im).astype(np.float32)

def rounded(w, h, r, fill, grad=None, border=None, shadow=0):
    pad = shadow * 2 + 4
    can = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    if shadow:
        sm = Image.new("L", can.size, 0)
        ImageDraw.Draw(sm).rounded_rectangle((pad, pad + shadow // 2, pad + w, pad + h + shadow // 2), r, fill=120)
        sm = sm.filter(ImageFilter.GaussianBlur(shadow))
        sl = Image.new("RGBA", can.size, (5, 15, 35, 255)); sl.putalpha(sm)
        can = Image.alpha_composite(can, sl)
    m = Image.new("L", can.size, 0)
    ImageDraw.Draw(m).rounded_rectangle((pad, pad, pad + w, pad + h), r, fill=255)
    if grad:
        c0, c1 = np.array(grad[0], np.float32), np.array(grad[1], np.float32)
        xx = np.linspace(0, 1, can.size[1])[:, None, None]
        g = (c0 * (1 - xx) + c1 * xx) * np.ones((1, can.size[0], 1))
        body = Image.fromarray(g.clip(0, 255).astype(np.uint8), "RGB").convert("RGBA")
        if len(grad[0]) == 3: pass
        body.putalpha(m.point(lambda v: int(v * (fill[3] / 255 if len(fill) == 4 else 1))))
    else:
        body = Image.new("RGBA", can.size, fill[:3] + (255,))
        body.putalpha(m.point(lambda v: int(v * (fill[3] / 255 if len(fill) == 4 else 1))))
    can = Image.alpha_composite(can, body)
    if border:
        bl = Image.new("RGBA", can.size, (0, 0, 0, 0))
        ImageDraw.Draw(bl).rounded_rectangle((pad, pad, pad + w, pad + h), r, outline=border[0], width=border[1])
        can = Image.alpha_composite(can, bl)
    # glossy highlight on top half
    gl = Image.new("L", can.size, 0)
    ImageDraw.Draw(gl).rounded_rectangle((pad + 3, pad + 3, pad + w - 3, pad + h // 2), max(1, r - 3), fill=38)
    gl = Image.composite(gl, Image.new("L", can.size, 0), m)
    wl = Image.new("RGBA", can.size, (255, 255, 255, 255)); wl.putalpha(gl)
    can = Image.alpha_composite(can, wl)
    return can

# ---------------------------------------------------------------- content
DAYS = [("ראשון", "29.11", "local_rain_3d.png", "rain"),
        ("שני", "30.11", "sun_behind_rain_cloud_3d.png", "rain"),
        ("שני בלילה", "30.11", "cloud_with_lightning_and_rain_3d.png", "storm"),
        ("שלישי", "1.12", "local_rain_3d.png", "rain"),
        ("שלישי בערב", "1.12", "cloud_with_rain_3d.png", "rain"),
        ("רביעי–שבת", "2-5.12", "cloud_with_lightning_and_rain_3d.png", "storm"),
        ("השבוע הבא", "7-9.12", "cloud_with_rain_3d.png", "cloud")]
DAYCARD = [(14.84, 23.45, 0), (23.52, 36.2, 1), (36.28, 45.0, 2), (45.2, 51.3, 3), (51.6, 54.8, 4), (54.86, 64.75, 5), (64.82, 70.45, 6)]
THEME = {"sun": ((255, 196, 40), (255, 112, 30)), "rain": ((64, 156, 255), (36, 90, 220)),
         "storm": ((90, 110, 255), (40, 50, 170)), "cloud": ((150, 170, 200), (90, 110, 150))}
SCREEN_WINDOWS = []
TS = 0
BEHIND = [  # t0, t1, icon, cx, cy, size   (right side, behind the presenter)
    (5.4, 6.85, "sun_3d.png", 885, 720, 210),
    (6.94, 14.7, "cloud_with_lightning_and_rain_3d.png", 885, 720, 230),
    (70.54, 99, "cloud_with_rain_3d.png", 885, 720, 210),
]
LIGHTNING = [40.5, 40.9]
CHIPS = [  # t0, t1, text, icon, side
    (5.32, 6.9, "חובבי החום והשמש", "sun_with_face_3d.png", "R"),
    (6.94, 8.8, "נגמרה החגיגה", "cross_mark_3d.png", "L"),
    (9.44, 12.5, "מערכת גשם חורפית וארוכה", "cloud_with_rain_3d.png", "R"),
    (12.62, 14.7, "בעיקר משני בערב", "calendar_3d.png", "L"),
    (15.94, 19.0, "מחר: גשם מקומי וחזק", "local_rain_3d.png", "R"),
    (19.1, 21.1, "בעיקר בצפון ובחוף", "round_pushpin_3d.png", "L"),
    (21.16, 23.45, "ברוב האזורים - יבש", "sun_behind_small_cloud_3d.png", "R"),
    (23.52, 27.3, "בבוקר שמשי וחמים יותר", "sun_3d.png", "L"),
    (27.34, 29.1, "אובך בדרום", "fog_3d.png", "R"),
    (29.12, 33.4, "אחה״צ: גשם בדרום ובמזרח", "cloud_with_rain_3d.png", "L"),
    (33.46, 36.2, "סכנה לשיטפונות", "water_wave_3d.png", "R"),
    (36.28, 40.0, "הגשם מתפשט", "umbrella_with_rain_drops_3d.png", "L"),
    (40.02, 41.4, "ברקים ורעמים", "high_voltage_3d.png", "R"),
    (41.42, 45.0, "הצפות בערי החוף", "water_wave_3d.png", "L"),
    (45.2, 48.3, "גשם בצפון ובחוף המרכזי", "local_rain_3d.png", "R"),
    (48.36, 51.3, "ברוב הארץ - יבש", "sun_behind_small_cloud_3d.png", "L"),
    (51.6, 54.8, "הגשם שוב מתפשט", "cloud_with_rain_3d.png", "R"),
    (57.42, 61.0, "חורף אמיתי!", "cold_face_3d.png", "L"),
    (61.24, 63.1, "מעילים, מטריות, מגפיים", "coat_3d.png", "R"),
    (63.22, 64.75, "שלג בחרמון", "snowflake_3d.png", "L"),
    (64.82, 70.4, "אולי עד אמצע השבוע הבא", "thinking_face_3d.png", "R"),
]
FOLLOW_T0, TAP_T = 70.9, 72.64
KW_Y, SUB_Y = 1392, 1494
SAFE_W = 720

def draw_screen(F, t):
    return

def strip_vis(t):
    return 1.0

MOOD = [(0, 0.0), (99, 0.0)]
RAIN = [(0, 0), (15.9, 0), (16.3, 0.3), (23.2, 0.3), (23.5, 0), (29.1, 0), (29.5, 0.45), (36.2, 0.45), (36.5, 0.95), (45.0, 0.95), (45.3, 0.35), (51.3, 0.35),
        (51.6, 0.6), (54.8, 0.6), (55.1, 0.9), (64.7, 0.9), (65.0, 0.4), (70.4, 0.4), (70.7, 0), (99, 0)]


def kf(t, pts):
    xs, ys = zip(*pts); return float(np.interp(t, xs, ys))
TS = KEEP[0][0] / FPS
ZOOM_CUTS = sorted(set([round(b / 30, 4) for a, b in KEEP[1:]] + [d[0] for d in DAYCARD]))

def zoom_at(t):
    tt = t - TS
    z0 = 1.0 + 0.10 * (1 - ease_out_cubic(tt / 0.55))  # intro punch-out
    if tt < 4.6:
        return z0 if tt < 0.55 else (1.0 + 0.035 * ease_in_out((tt - 0.6) / 3.8))
    prev, cur, tc = 1.035, 1.035, TS + 4.6
    vals = [1.045, 1.0]
    for i, c in enumerate([c for c in ZOOM_CUTS if c > TS + 4.6]):
        if t >= c: prev, cur, tc = cur, vals[i % 2], c
    if any(abs(tc - b) < 0.05 for _, b in CUTS): return cur  # hard cut -> instant punch
    return prev + (cur - prev) * ease_out_cubic((t - tc) / 0.16)

# rain streaks
rng = random.Random(7)
DROPS = [(rng.uniform(-200, W + 100), rng.uniform(0, H), rng.uniform(1500, 2300), rng.uniform(40, 95),
          rng.uniform(0.25, 0.75), rng.uniform(1.5, 3.2)) for _ in range(170)]
FDROPS = [(rng.choice([rng.uniform(40, 270), rng.uniform(800, 1040)]), rng.uniform(0, 1), rng.uniform(380, 560),
           rng.randint(38, 60)) for _ in range(9)]

# subtitles
FIX = {"התרברו": "יתרבו", "יקראנו": "היכונו"}
TFIX = {"התרברו": "יתרבו", "יקראנו": "היכונו"}
def load_words():
    ws = []
    for s in json.load(open(os.path.join(HERE, "r17", "trw.json"))):
        for w in s["words"]:
            txt = w["w"].strip()
            brk = txt.endswith((",", ".", "?", "!"))
            txt = txt.strip(",.?!")
            if any(a - 0.01 <= w["s"] < b - 0.05 for a, b in CUTS): continue
            k = round(w["s"], 2)
            if k in TFIX: txt = TFIX[k]
            if FIX.get(txt, txt) == "": continue
            ws.append({"t": FIX.get(txt, txt), "s": w["s"], "e": w["e"], "brk": brk})
    return ws

def chunks():
    ws = load_words(); out = []; cur = []
    for i, w in enumerate(ws):
        cur.append(w)
        nxt_gap = ws[i + 1]["s"] - w["e"] if i + 1 < len(ws) else 9
        L = sum(len(x["t"]) + 1 for x in cur)
        cut_next = i + 1 < len(ws) and any(w["s"] < a <= ws[i + 1]["s"] for a, b in CUTS)
        if cut_next or w["brk"] or len(cur) >= 4 or L >= 17 or nxt_gap > 0.35:
            out.append(cur); cur = []
    if cur: out.append(cur)
    res = []
    for i, c in enumerate(out):
        t0 = c[0]["s"]; t1 = out[i + 1][0]["s"] if i + 1 < len(out) else DUR
        t1 = min(t1, c[-1]["e"] + 0.6)
        for a, b in CUTS:
            if t0 < a < t1: t1 = a
        res.append((t0, t1, c))
    return res
CHUNKS = chunks()

# ---------------------------------------------------------------- cached sprites
@functools.lru_cache(None)
def sub_word(text, active):
    return to_np(text_layer(text, 70, 800, fill=(255, 222, 40) if active else (255, 255, 255),
                            stroke=7, stroke_fill=(0, 0, 0), shadow=8, pad=24, fixed_line=True))

@functools.lru_cache(None)
def sub_word2(text, state):
    f = font(76, 800); pad = 34; st = 6
    asc, desc = f.getmetrics()
    d0 = ImageDraw.Draw(Image.new("L", (1, 1)))
    bb = d0.textbbox((0, 0), text, font=f, direction="rtl", language="he", stroke_width=st)
    w = bb[2] - bb[0] + 2 * pad; h = asc + desc + 2 * st + 2 * pad
    org = (pad - bb[0], pad + st)
    def mask(sw):
        mm = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mm).text(org, text, font=f, fill=255, direction="rtl", language="he", stroke_width=sw, stroke_fill=255)
        return np.asarray(mm, np.float32) / 255
    body = mask(0); stroke = mask(st)
    out = np.zeros((h, w, 4), np.float32)
    out[..., 3] = np.roll(cv2.GaussianBlur(stroke, (0, 0), 7), 6, 0) * 200
    if state == 1:
        gl = cv2.GaussianBlur(stroke, (0, 0), 14); a = (gl * 0.85)[..., None]
        out[..., :3] = out[..., :3] * (1 - a) + np.array((255, 170, 40), np.float32) * a; out[..., 3:] = np.maximum(out[..., 3:], gl[..., None] * 230)
    sc_ = np.array((40, 22, 8) if state == 1 else (12, 14, 30), np.float32)
    a = stroke[..., None]; out[..., :3] = out[..., :3] * (1 - a) + sc_ * a; out[..., 3:] = np.maximum(out[..., 3:], stroke[..., None] * 255)
    ys = np.linspace(0, 1, h)[:, None, None]
    fill = (np.array((255, 246, 170), np.float32) * (1 - ys) + np.array((255, 176, 30), np.float32) * ys) if state == 1 else (np.array((255, 255, 255), np.float32) * (1 - ys) + np.array((226, 228, 255), np.float32) * ys)
    fill = fill * np.ones((1, w, 1), np.float32)
    a = body[..., None]; out[..., :3] = out[..., :3] * (1 - a) + fill * a
    if state == 0: out[..., 3] *= 0.62
    return out, org[1] + asc

@functools.lru_cache(None)
def chip_sprite(text, ic):
    tl = text_layer(text, 46, 800, fill=(20, 32, 62), pad=0)
    tw = tl.size[0]
    w, h = tw + 150, 92
    base = rounded(w, h, 46, (255, 255, 255, 240), border=((255, 255, 255, 255), 2), shadow=12)
    pad = (base.size[0] - w) // 2
    base.alpha_composite(tl, (pad + 32, pad + (h - tl.size[1]) // 2 - 3))
    icn = Image.open(A(ic)).convert("RGBA").resize((112, 112), Image.LANCZOS)
    can = Image.new("RGBA", (base.size[0] + 30, base.size[1] + 30), (0, 0, 0, 0))
    can.alpha_composite(base, (0, 15))
    can.alpha_composite(icn, (pad + w - 112 + 18, 15 + pad + (h - 112) // 2 - 4))
    return to_np(can)

def _ic(n, s): return Image.open(A(n)).convert("RGBA").resize((s, s), Image.LANCZOS)

@functools.lru_cache(None)
def day_icon(ic):
    can = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    if ic == "ICON_SUNRAINBOLT":   # sun + rain cloud + lightning
        can.alpha_composite(_ic("sun_behind_rain_cloud_3d.png", 92), (0, 0))
        can.alpha_composite(_ic("high_voltage_3d.png", 46), (40, 48))
    elif ic == "ICON_STORMSUN":    # storm cloud with a little sun peeking
        can.alpha_composite(_ic("sun_3d.png", 54), (42, 0))
        can.alpha_composite(_ic("cloud_with_lightning_and_rain_3d.png", 86), (0, 10))
    else:
        can.alpha_composite(_ic(ic, 96), (0, 0))
    return can

@functools.lru_cache(None)
def day_card(i, active):
    name, date, ic, th = DAYS[i]
    w, h = 140, 178
    if active:
        base = rounded(w, h, 26, (255, 255, 255, 255), grad=THEME[th], border=((255, 255, 255, 235), 4), shadow=7)
    else:
        base = rounded(w, h, 26, (14, 26, 52, 165), border=((255, 255, 255, 70), 2), shadow=8)
    pad = (base.size[0] - w) // 2
    fs = 34
    while True:
        t1 = text_layer(name, fs, 800, fill=(255, 255, 255), pad=0, shadow=0)
        if t1.size[0] <= w - 18 or fs <= 22: break
        fs -= 2
    fs2 = 30
    while True:
        t2 = text_layer(date, fs2, 700, fill=(255, 255, 255) if active else (205, 220, 245), pad=0)
        if t2.size[0] <= w - 14 or fs2 <= 20: break
        fs2 -= 2
    base.alpha_composite(t1, (pad + (w - t1.size[0]) // 2, pad + 10))
    icn = day_icon(ic)
    base.alpha_composite(icn, (pad + (w - 96) // 2, pad + 46))
    base.alpha_composite(t2, (pad + (w - t2.size[0]) // 2, pad + h - 22 - t2.size[1]))
    return to_np(base)

@functools.lru_cache(None)
def title_parts():
    # huge gold headline, thick dark outline + 3D extrusion
    l1 = text_layer("מהפך בשבוע הבא", 104, 800, gradient=((255, 247, 120), (255, 176, 0)),
                    stroke=7, stroke_fill=(20, 12, 40), extrude=11, extrude_fill=(120, 40, 0), shadow=16, pad=80)
    # red banner subline with a white date chip built in at its left end
    t2 = text_layer("גשם ממושך משני בערב", 52, 800, fill=(255, 255, 255), pad=0, shadow=0)
    t3 = text_layer("26.11", 40, 800, fill=(200, 0, 30), pad=0)
    cw, ch = t3.size[0] + 32, 56
    w2, h2 = t2.size[0] + 70 + cw + 16, 78
    ban = rounded(w2, h2, 22, (255, 255, 255, 255), grad=((235, 30, 45), (150, 0, 25)), border=((255, 225, 90, 255), 4), shadow=7)
    bp = (ban.size[0] - w2) // 2
    chip = rounded(cw, ch, 14, (255, 255, 255, 255), grad=((255, 255, 255), (232, 236, 244)), shadow=0)
    cp = (chip.size[0] - cw) // 2
    chip.alpha_composite(t3, (cp + 16, cp + (ch - t3.size[1]) // 2 - 4))
    ban.alpha_composite(chip, (bp + 12 - cp, bp + (h2 - ch) // 2 - cp))
    ban.alpha_composite(t2, (bp + 12 + cw + 16 + 24, bp + (h2 - t2.size[1]) // 2 - 6))
    st = ban
    def fit(im, maxw):
        a = to_np(im); ys, xs = np.where(a[..., 3] > 40)
        vis = xs.max() - xs.min() + 1
        sc = min(1.0, maxw / vis)
        if sc < 1: a = cv2.resize(a, (int(a.shape[1] * sc), int(a.shape[0] * sc)), interpolation=cv2.INTER_AREA)
        return a
    return None, fit(l1, 800), fit(ban, 740)


def l2_left():
    a = title_parts()[2]; xs = np.where(a[..., 3].max(0) > 40)[0]
    return 545 - a.shape[1] / 2 + xs.min()

TITLE_W = 800
@functools.lru_cache(None)
def title_img():
    im = Image.open(A("title_28-11.png")).convert("RGBA")
    a = np.asarray(im)[..., 3]; ys, xs = np.where(a > 24)
    im = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    w = TITLE_W; h = int(im.size[1] * w / im.size[0])
    arr = to_np(im.resize((w, h), Image.LANCZOS))
    al = arr[..., 3]; arr[..., 3] = np.clip((al - 40) * 255 / 215, 0, 255)   # drop the faint haze around the edges
    ys = np.where(arr[..., 3].max(1) > 12)[0]; arr = arr[:ys.max() + 1]
    return arr, 444 - arr.shape[0] / 2   # bottom edge at y=444 (above the head)

@functools.lru_cache(None)
def kw_sprite(text, theme):
    g0, g1, sf, ef = {"sun": ((255, 240, 120), (255, 140, 30), (90, 35, 0), (150, 65, 0)),
                      "storm": ((215, 225, 255), (110, 120, 255), (16, 18, 70), (30, 34, 120)),
                      "rain": ((200, 235, 255), (70, 150, 255), (8, 24, 70), (14, 42, 120)),
                      "cloud": ((240, 245, 255), (165, 185, 215), (25, 35, 60), (45, 60, 95))}[theme]
    t = to_np(text_layer(text, 74, 800, gradient=(g0, g1), stroke=5, stroke_fill=sf, extrude=7, extrude_fill=ef, shadow=12, pad=60))
    sc = min(1.0, 600 / t.shape[1])
    if sc < 1: t = cv2.resize(t, (int(t.shape[1] * sc), int(t.shape[0] * sc)), interpolation=cv2.INTER_AREA)
    return t

def theme_at(t):
    for (a0, a1, k) in DAYCARD:
        if a0 <= t < a1: return DAYS[k][3]
    return "rain" if t < 9 else "sun"

def draw_kw(F, t, t0, t1, text, ic, side, y=KW_Y):
    w = window(t, t0, t1, 0.22, 0.18)
    if not w: return
    pin, a = w
    th = theme_at(t0 + 0.01)
    spr = kw_sprite(text, th); tw = spr.shape[1]
    sgn = -1 if side == "L" else 1
    cx = 540 + sgn * 30 + (1 - ease_out_cubic(clamp(pin * 1.6))) * 260 * sgn
    rot = -4 * sgn * (1 - 0.6 * ease_out_cubic(clamp(pin * 2)))
    col = {"sun": (255, 150, 60), "rain": (80, 160, 255), "cloud": (180, 195, 225), "storm": (120, 130, 255)}[th]
    gs = min(420, max(300, int(tw * 0.9)))
    F.blit(flat_glow(gs, col), cx, y + 24, alpha=a * 0.35)
    F.blit(spr, cx, y, alpha=a, scale=0.55 + 0.45 * ease_out_back(pin, 2.6), rot=rot)
    sw = clamp((t - t0 - 0.25) / 0.45)
    if 0 < sw < 1:
        F.blit(flat_glow(170, (255, 255, 255)), cx - tw / 2 + tw * sw, y + 6, alpha=a * 0.5 * math.sin(math.pi * sw))
    vis = tw / 2 - 60
    ix = cx + vis + 80 if cx + vis + 80 < W - 110 else cx - vis - 80
    if 330 < ix < 750: ix = 330 if ix < 540 else 750   # never under the chin
    ix = min(max(ix, 110), W - 110)
    pi_ = clamp((t - t0 - 0.08) / 0.3)
    if pi_ > 0:
        F.blit(icon_shadowed(ic, 140)[0], ix, y - 12 + math.sin(t * 4) * 6, alpha=a * ease_out_cubic(pi_),
               scale=0.3 + 0.7 * ease_out_back(pi_, 3.0), rot=math.sin(t * 3) * 10)

@functools.lru_cache(None)
def follow_card(followed):
    w, h = 800, 196
    base = rounded(w, h, 44, (255, 255, 255, 246), shadow=18)
    pad = (base.size[0] - w) // 2
    # avatar with IG gradient ring (on the right side for RTL)
    src = Image.open("/root/.claude/uploads/96f0ae6d-dec7-55de-b937-8c9ce790e805/4f834ef5-image.jpg").convert("RGB")
    cx, cy, r = 156, 178, 95
    av = src.crop((cx - r, cy - r, cx + r, cy + r)).resize((132, 132), Image.LANCZOS).convert("RGBA")
    m = Image.new("L", (132, 132), 0); ImageDraw.Draw(m).ellipse((0, 0, 131, 131), fill=255); av.putalpha(m)
    ring = Image.new("RGBA", (152, 152), (0, 0, 0, 0))
    g = np.zeros((152, 152, 3), np.float32)
    yy, xx = np.mgrid[0:152, 0:152] / 151.0
    tt = ((xx + (1 - yy)) / 2)[..., None]
    c0, c1, c2 = np.array([254, 218, 117.]), np.array([214, 41, 118.]), np.array([79, 91, 213.])
    g = np.where(tt < 0.5, c0 * (1 - tt * 2) + c1 * tt * 2, c1 * (2 - tt * 2) + c2 * (tt * 2 - 1))
    ring = Image.fromarray(g.astype(np.uint8)).convert("RGBA")
    rm = Image.new("L", (152, 152), 0); d = ImageDraw.Draw(rm); d.ellipse((0, 0, 151, 151), fill=255)
    d.ellipse((6, 6, 145, 145), fill=0); ring.putalpha(rm)
    wr = Image.new("RGBA", (152, 152), (0, 0, 0, 0)); ImageDraw.Draw(wr).ellipse((5, 5, 146, 146), fill=(255, 255, 255, 255))
    ring = Image.alpha_composite(wr, ring); ring.alpha_composite(av, (10, 10))
    base.alpha_composite(ring, (pad + w - 152 - 24, pad + (h - 152) // 2))
    t1 = text_layer("עקבו אחריי", 48, 800, fill=(18, 24, 38), pad=0)
    num = text_layer("63.3K", 50, 800, fill=(18, 24, 38), pad=0)
    lab = text_layer("עוקבים", 36, 600, fill=(110, 118, 132), pad=0)
    rx = pad + w - 152 - 48
    base.alpha_composite(t1, (rx - t1.size[0], pad + 30))
    base.alpha_composite(num, (rx - num.size[0], pad + 98))
    base.alpha_composite(lab, (rx - num.size[0] - 14 - lab.size[0], pad + 108))
    bw, bh = 190, 84
    btn = rounded(bw, bh, 22, (239, 239, 239, 255) if followed else (0, 149, 246, 255), shadow=0)
    bp = (btn.size[0] - bw) // 2
    bt = text_layer("במעקב" if followed else "עקוב", 44 if not followed else 40, 800,
                    fill=(18, 24, 38) if followed else (255, 255, 255), pad=0)
    btn.alpha_composite(bt, (bp + (bw - bt.size[0]) // 2, bp + (bh - bt.size[1]) // 2 - 3))
    base.alpha_composite(btn, (pad + 28 - bp, pad + (h - bh) // 2 - bp))
    return to_np(base), (pad + 28 + bw / 2 - base.size[0] / 2, pad + h / 2 - base.size[1] / 2)

@functools.lru_cache(None)
def glow(size, color):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d = np.hypot(xx - size / 2, yy - size / 2) / (size / 2)
    a = np.clip(1 - d, 0, 1) ** 2.2 * 255
    arr = np.zeros((size, size, 4), np.float32); arr[..., :3] = color; arr[..., 3] = a
    return arr

@functools.lru_cache(None)
def flat_glow(size, color):
    g = glow(size, color)
    return cv2.resize(g, (size, int(size * 0.42)), interpolation=cv2.INTER_AREA)

@functools.lru_cache(None)
def rays(size):
    im = Image.new("L", (size, size), 0); d = ImageDraw.Draw(im); c = size / 2
    for k in range(14):
        a0 = k * 2 * math.pi / 14
        pts = [(c, c), (c + c * math.cos(a0 - 0.07), c + c * math.sin(a0 - 0.07)),
               (c + c * math.cos(a0 + 0.07), c + c * math.sin(a0 + 0.07))]
        d.polygon(pts, fill=255)
    im = im.filter(ImageFilter.GaussianBlur(6))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    fade = np.clip(1 - np.hypot(xx - c, yy - c) / c, 0, 1)
    arr = np.zeros((size, size, 4), np.float32); arr[..., :3] = (255, 225, 120)
    arr[..., 3] = np.asarray(im, np.float32) * fade * 0.55
    return arr

# ---------------------------------------------------------------- frame
def warp(img, z, interp=cv2.INTER_LINEAR):
    if abs(z - 1) < 1e-4: return img
    ax, ay = 540, 560
    M = np.array([[z, 0, ax - z * ax], [0, z, ay - z * ay]], np.float32)
    return cv2.warpAffine(img, M, (W, H), flags=interp, borderMode=cv2.BORDER_REFLECT)

DARK_HALO = None
def render(i, raw):
    global DARK_HALO
    if DARK_HALO is None:
        g = glow(1200, (6, 10, 24)); DARK_HALO = cv2.resize(g, (1200, 430), interpolation=cv2.INTER_AREA)
    t = i / FPS
    z = zoom_at(t)
    fm = cv2.resize(np.asarray(FACE[min(i, len(FACE) - 1)]).astype(np.float32) / 255, (W, H), interpolation=cv2.INTER_LINEAR)
    raw = raw.astype(np.float32)
    img = warp(raw, z).astype(np.float32)
    pm = cv2.resize(np.asarray(PERSON[min(i, len(PERSON) - 1)]), (W, H), interpolation=cv2.INTER_LINEAR)
    hm = cv2.resize(np.asarray(HEAD[min(i, len(HEAD) - 1)]), (W, H), interpolation=cv2.INTER_LINEAR)
    pm = warp(pm, z).astype(np.float32) / 255.0
    pm = cv2.GaussianBlur(pm, (0, 0), 1.6)
    hm = warp(hm, z)
    hm = cv2.dilate((hm > 90).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    F = Frame(img, pm, hm)
    bgw = (1 - pm)[..., None]

    # --- background mood grade (behind the person only)
    mood = kf(t, MOOD)
    if mood > 0.01:
        lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
        storm = lum[..., None] * np.array([0.70, 0.78, 0.92], np.float32) * 0.82
        k = 0.55 * mood
        img += (storm - img) * (k * bgw)
    elif mood < -0.01:
        warm = img * np.array([1.07, 1.02, 0.92], np.float32) + np.array([8, 4, 0], np.float32)
        img += (warm - img) * ((-mood) * 0.6 * bgw)

    # --- lightning flashes on the sky (behind)
    for lt in LIGHTNING:
        dt = t - lt
        if -0.05 < dt < 0.5:
            fl = max(0, 1 - abs(dt) / 0.08) * 0.55 + max(0, 1 - abs(dt - 0.2) / 0.07) * 0.4
            if fl > 0: img += (255 - img) * (fl * 0.6 * bgw)
            F.blit(icon("high_voltage_3d.png", 230), 175, 735, alpha=clamp(1 - dt / 0.5) * clamp((dt + 0.05) / 0.05),
                   behind=True, scale=0.9 + 0.2 * ease_out_back(clamp((dt + 0.05) / 0.2)), rot=-12)

    # --- rain streaks behind
    ri = kf(t, RAIN)
    if ri > 0.01:
        lay = np.zeros((H, W), np.uint8)
        n = int(len(DROPS) * ri)
        for (x0, y0, sp, ln, op, wd) in DROPS[:n]:
            y = (y0 + sp * t) % (H + 200) - 100
            x = x0 + 0.18 * y
            cv2.line(lay, (int(x), int(y)), (int(x + 0.18 * ln), int(y + ln)), int(255 * op), max(1, int(wd)), cv2.LINE_AA)
        a = (lay.astype(np.float32) / 255.0 * 0.55 * min(1, ri * 1.4))[..., None] * bgw
        img += (np.array([225, 236, 255], np.float32) - img) * a

    # --- behind weather sprites (3D: hidden by the person's body)
    for (t0, t1, ic, cx, cy, sz) in BEHIND:
        w = window(t, t0, t1, 0.45, 0.3)
        if not w: continue
        pin, a = w
        sc = 0.55 + 0.45 * ease_out_back(pin, 1.5)
        bob = math.sin((t - t0) * 2.2) * 10
        if ic == "sun_3d.png":
            gsz = int(sz * 2.0)
            F.blit(glow(gsz, (255, 214, 90)), cx, cy + bob, alpha=a * 0.75, behind=True, scale=sc)
            F.blit(rays(int(sz * 1.9)), cx, cy + bob, alpha=a, behind=True, scale=sc, rot=t * 14)
        if ic == "crescent_moon_3d.png":
            F.blit(glow(int(sz * 1.8), (170, 200, 255)), cx, cy + bob, alpha=a * 0.6, behind=True, scale=sc)
        spr, _ = icon_shadowed(ic, sz, 14, 14, 0.35)
        F.blit(spr, cx + math.sin((t - t0) * 1.3) * 6, cy + bob, alpha=a, behind=True, scale=sc,
               rot=math.sin((t - t0) * 1.7) * 3)
    # extra stars around moon
    if False:
        for k, (sx, sy, ss) in enumerate([(760, 620, 60), (1010, 880, 50), (300, 860, 46), (90, 610, 54)]):
            tw = 0.6 + 0.4 * math.sin(t * 5 + k * 1.7)
            F.blit(icon("star_3d.png", ss), sx, sy, alpha=clamp((t - 83.2 - k * 0.15) / 0.3) * tw, behind=True)

    draw_screen(F, t)

    # --- front droplets (sides only, never near the head)
    if ri > 0.6:
        for (fx, ph, sp, sz) in FDROPS:
            y = ((ph * 1400 + sp * t) % 1400) + 560
            if y > 1330: continue
            fade = clamp((1330 - y) / 150) * clamp((ri - 0.6) / 0.3)
            F.blit(icon("droplet_3d.png", sz), fx, y, alpha=fade * 0.9)

    # --- keywords (front, chest level): 3D gradient lettering + floating icon
    for (t0, t1, txt, ic, side) in CHIPS:
        draw_kw(F, t, t0, t1, txt, ic, side)

    # --- title (GPT Image 2.6 design): drop-in with a bounce + flash, then perfectly steady
    spr, tcy = title_img()
    tt = t - TS
    p = 1.0   # title fully visible from the first frame
    if p > 0:
        if p >= 1:
            F.blit(spr, 540, tcy)
        else:
            F.blit(spr, 540, tcy - (1 - ease_out_cubic(p)) * 90, alpha=ease_out_cubic(clamp(p * 2)),
                   scale=0.55 + 0.45 * ease_out_back(p, 1.8))

    # --- day card (right side, beside the head) — changes with the day being discussed
    for (a0, a1, k) in DAYCARD:
        w_ = window(t, a0, a1, 0.35, 0.25)
        if not w_: continue
        pin, a = w_
        sc = 0.6 + 0.4 * ease_out_back(pin, 2.0) + 0.012 * math.sin(t * 5)
        F.blit(day_card(k, True), 880 + (1 - ease_out_cubic(pin)) * 120, 700, alpha=a, scale=sc * 1.1)

    # --- follow card
    if t >= FOLLOW_T0:
        p = clamp((t - FOLLOW_T0) / 0.45)
        followed = t >= TAP_T + 0.12
        spr, (bx, by) = follow_card(followed)
        press = 1 - 0.06 * max(0, 1 - abs(t - TAP_T - 0.05) / 0.12)
        cy = 1440 + (1 - ease_out_back(p, 1.7)) * 260
        bx, by = bx * 0.84, by * 0.84
        F.blit(spr, 540, cy, alpha=ease_out_cubic(p), scale=press * 0.84)
        # tapping hand
        if TAP_T - 0.9 < t < TAP_T + 0.9:
            hp = clamp((t - (TAP_T - 0.9)) / 0.8)
            hx = 540 + bx + 30
            hy = cy + by + 70 + (1 - ease_out_cubic(hp)) * 220 + (8 if abs(t - TAP_T) < 0.1 else 0)
            ha = clamp((t - (TAP_T - 0.9)) / 0.2) * (1 - clamp((t - TAP_T - 0.6) / 0.3))
            spr2, _ = icon_shadowed("backhand_index_pointing_up_3d_default.png", 120, 8, 8, 0.4)
            F.blit(spr2, hx, hy, alpha=ha)
        if TAP_T < t < TAP_T + 1.2:
            q = (t - TAP_T) / 1.2
            for k in range(5):
                F.blit(icon("red_heart_3d.png", 56), 540 + bx - 70 + (k - 2) * 26 - q * 110, cy - 40 - q * (40 + k * 14),
                       alpha=(1 - q), scale=0.6 + 0.6 * q, rot=(k - 2) * 12)

    # --- subtitles (styled: active word glowing gold + pop, upcoming words translucent)
    for (t0, t1, ws) in CHUNKS:
        if t >= FOLLOW_T0 - 0.05: break
        if t0 <= t < t1:
            p = clamp((t - t0) / 0.12)
            items = []
            for j, w_ in enumerate(ws):
                end = ws[j + 1]["s"] if j + 1 < len(ws) else w_["e"] + 0.2
                st = 1 if w_["s"] <= t < end else (2 if t >= end else 0)
                spr, bl = sub_word2(w_["t"], st)
                pop = 1.0 + (0.10 * ease_out_back(clamp((t - w_["s"]) / 0.14), 2.4) if st == 1 else 0)
                items.append((spr, bl, pop))
            gap = -38; total = sum(q[0].shape[1] for q in items) + gap * (len(items) - 1)
            sc = min(1.0, SAFE_W / total) * (0.9 + 0.1 * ease_out_back(p, 2.8))
            x = 540 + total * sc / 2
            for spr, bl, pop in items:
                ww = spr.shape[1] * sc; s2 = sc * pop
                cy = SUB_Y + (spr.shape[0] / 2 - bl) * s2 + (bl - spr.shape[0] / 2) * sc
                F.blit(spr, x - ww / 2, cy, alpha=ease_out_cubic(p), scale=s2); x -= ww + gap * sc
            break

    np.clip(img, 0, 255, out=img)
    return img.astype(np.uint8), F.head_overlap

# ---------------------------------------------------------------- main
def run(start, end, outpath):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start / FPS:.4f}", "-i", VIDEO, "-vf", f"fps={FPS},scale={W}:{H}",
           "-frames:v", str(end - start), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    dec = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "13",
                            "-pix_fmt", "yuv420p", outpath], stdin=subprocess.PIPE)
    log = []
    i = start
    while i < end:
        b = dec.stdout.read(W * H * 3)
        if len(b) < W * H * 3: break
        fr = np.frombuffer(b, np.uint8).reshape(H, W, 3)
        out, ov = render(i, fr)
        if ov: log.append((i, ov))
        enc.stdin.write(out.tobytes())
        i += 1
    enc.stdin.close(); enc.wait()
    print(f"chunk {start}-{end}: frames {i - start}, head-overlap frames {len(log)} {log[:5]}", flush=True)

OUT = [f for a, b in KEEP for f in range(a, b)]

def run_out(o0, o1, outpath):
    src = OUT[o0:o1]; want = set(src)
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{src[0] / FPS:.4f}", "-i", VIDEO, "-vf", f"fps={FPS},scale={W}:{H}",
           "-frames:v", str(src[-1] - src[0] + 1), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    dec = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "13",
                            "-pix_fmt", "yuv420p", outpath], stdin=subprocess.PIPE)
    log = []; n = 0
    for i in range(src[0], src[-1] + 1):
        b = dec.stdout.read(W * H * 3)
        if len(b) < W * H * 3: break
        if i not in want: continue
        out, ov = render(i, np.frombuffer(b, np.uint8).reshape(H, W, 3))
        if ov: log.append((i, ov))
        enc.stdin.write(out.tobytes()); n += 1
    enc.stdin.close(); enc.wait()
    print(f"out {o0}-{o1}: frames {n}/{len(src)}, head-overlap frames {len(log)} {log[:4]}", flush=True)

if __name__ == "__main__":
    if sys.argv[1] == "out":
        run_out(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]); sys.exit()
    if sys.argv[1] == "still":
        tsec = float(sys.argv[2])
        i = int(tsec * FPS)
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{i / FPS:.4f}", "-i", VIDEO, "-vf", f"scale={W}:{H}",
                              "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        out, ov = render(i, np.frombuffer(raw, np.uint8).reshape(H, W, 3))
        Image.fromarray(out).resize((540, 960), Image.LANCZOS).save(sys.argv[3])
        print("overlap", ov)
    else:
        run(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
