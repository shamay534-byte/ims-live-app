# Reels renderer: weather update 4.11
import sys, json, math, subprocess, random, functools, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
A = lambda n: os.path.join(HERE, "assets", n)
W, H, FPS = 1080, 1920, 30
VIDEO = os.path.join(HERE, "r7", "dl", "video.mp4")
NFR = 1785
KEEP = [(0, 2), (5, 264), (270, 352), (359, 451), (454, 465), (473, 538), (542, 655), (661, 766), (774, 840), (848, 1006), (1013, 1097), (1103, 1147), (1158, 1245), (1252, 1404), (1410, 1489), (1585, 1726), (1735, 1774), (1784, 1785)]
CUTS = [(a1 / 30, b0 / 30) for (a0, a1), (b0, b1) in zip(KEEP, KEEP[1:])]
DUR = NFR / FPS

PERSON = np.load(os.path.join(HERE, "r7", "person.npy"), mmap_mode="r")
HEAD = np.load(os.path.join(HERE, "r7", "head.npy"), mmap_mode="r")

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
DAYS = [("שישי", "13.11", "sun_3d.png", "sun"),
        ("שבת", "14.11", "beach_with_umbrella_3d.png", "sun"),
        ("ראשון", "15.11", "sun_3d.png", "sun"),
        ("שני", "16.11", "sun_3d.png", "sun"),
        ("שלישי", "17.11", "sun_3d.png", "sun"),
        ("רביעי", "18.11", "sun_behind_small_cloud_3d.png", "sun")]
ACTIVE = [(8.86, 49.48, [1]), (49.48, 57.5, [1, 2, 3, 4, 5]), (57.5, 99, [1])]
DAY_CHANGES = [49.48]
THEME = {"sun": ((255, 196, 40), (255, 112, 30)), "rain": ((64, 156, 255), (36, 90, 220)),
         "storm": ((90, 110, 255), (40, 50, 170))}
STRIP_T0 = 8.86
TITLE_T1 = 5.5
SCREEN_WINDOWS = [(17.76, 30.2)]

BEHIND = [  # t0, t1, icon, cx, cy, size
    (0.2, 17.6, "sun_3d.png", 870, 690, 250),
    (0.5, 8.8, "umbrella_on_ground_3d.png", 165, 790, 220),
    (8.86, 17.6, "sailboat_3d.png", 150, 1080, 190),
    (30.3, 49.4, "sun_3d.png", 870, 690, 250),
    (30.3, 43.9, "sailboat_3d.png", 150, 1080, 190),
    (43.92, 49.4, "ring_buoy_3d.png", 165, 800, 200),
    (49.48, 99, "beach_with_umbrella_3d.png", 165, 790, 220),
]
LIGHTNING = []
CHIPS = [  # t0, t1, text, icon, side
    (0.98, 3.0, "ההזדמנות האחרונה לים", "beach_with_umbrella_3d.png", "L"),
    (3.12, 5.5, "לפני שהמים מתקררים", "thermometer_3d.png", "R"),
    (5.96, 8.8, "סופ״ש שמשי, אביבי אפילו", "sun_with_face_3d.png", "L"),
    (9.78, 12.7, "עוד עלייה בטמפרטורות", "chart_increasing_3d.png", "R"),
    (14.5, 17.7, "הים כמעט פלטה", "water_wave_3d.png", "L"),
    (19.1, 22.7, "רוח מזרחית: מהיבשה לים", "dashing_away_3d.png", "L"),
    (24.6, 30.0, "ים של תאילנד", "palm_tree_3d.png", "L"),
    (30.44, 33.3, "לא חם מדי, וזה היתרון", "sun_3d.png", "R"),
    (33.42, 37.9, "השמש חלשה מבקיץ", "sunglasses_3d.png", "L"),
    (38.2, 43.9, "ההזדמנות האחרונה של נובמבר", "beach_with_umbrella_3d.png", "R"),
    (43.92, 47.6, "רק לחופים מוכרזים עם מצילים", "ring_buoy_3d.png", "L"),
    (47.64, 49.4, "יש חופים שפועלים כל השנה", "umbrella_on_ground_3d.png", "R"),
    (49.48, 54.1, "התנאים יישארו איתנו", "check_mark_button_3d.png", "L"),
    (54.16, 57.4, "לפחות עד שלישי-רביעי", "alarm_clock_3d.png", "R"),
]
FOLLOW_T0, TAP_T = 57.5, 57.75

# ---- beach background (person + the ledge he sits on are kept)
BG_PLATE = np.load(os.path.join(HERE, "r7", "bg_plate.npy"))
_led = np.zeros((1920, 1080), np.uint8)
cv2.fillPoly(_led, [np.array([[0, 1680], [260, 1676], [760, 1592], [1080, 1590], [1080, 1920], [0, 1920]], np.int32)], 255)
LEDGE = cv2.GaussianBlur(_led.astype(np.float32), (0, 0), 2) / 255
_sr = random.Random(11)
GLINTS = [(_sr.uniform(0, 1080), _sr.uniform(1030, 1580), _sr.uniform(0, 6.28), _sr.uniform(1.5, 4)) for _ in range(140)]

def beachify(i, raw):
    t = i / FPS
    pm = cv2.resize(np.asarray(PERSON[min(i, len(PERSON) - 1)]), (W, H)).astype(np.float32) / 255
    pm = cv2.GaussianBlur(pm, (0, 0), 1.8)
    fg = np.clip(np.maximum(pm, LEDGE), 0, 1)[..., None]
    dx = int(round(math.sin(t * 0.25) * 6))
    bg = np.roll(BG_PLATE, dx, axis=1).astype(np.float32) * np.array([1.03, 1.0, 0.96], np.float32)
    # sun glints twinkling on the sea
    lay = np.zeros((H, W), np.float32)
    for (x, y, ph, sp) in GLINTS:
        v = max(0.0, math.sin(t * sp + ph)) ** 6
        if v > 0.05:
            cv2.circle(lay, (int(x + dx), int(y)), 2, float(v), -1, cv2.LINE_AA)
    lay = cv2.GaussianBlur(lay, (0, 0), 1.2)[..., None]
    bg = bg + (255 - bg) * np.clip(lay * 0.9, 0, 1)
    return raw.astype(np.float32) * fg + bg * (1 - fg)

# ---- east-wind simulation screen (behind the presenter)
SCX, SCY, SW_, SH_ = 842, 642, 372, 610
MAP_W, MAP_H = SW_ - 24, SH_ - 24 - 70 - 40
_sh = Image.open(os.path.join(HERE, "r6", "map", "shape_s.png")).convert("RGBA")
_crop = _sh.crop((0, 0, _sh.size[0], int(_sh.size[1] * 0.52)))
_s = (MAP_H - 10) / _crop.size[1]
_crop = _crop.resize((int(_crop.size[0] * _s), int(_crop.size[1] * _s)), Image.LANCZOS)
_ma = np.zeros((MAP_H, MAP_W), np.float32)
_ox = MAP_W - _crop.size[0] - 6
_ma[5:5 + _crop.size[1], _ox:_ox + _crop.size[0]] = np.asarray(_crop.split()[3], np.float32) / 255
COAST = np.zeros(MAP_H, np.int32)
for yy in range(MAP_H):
    xs_ = np.where(_ma[yy] > 0.5)[0]
    COAST[yy] = xs_.min() if len(xs_) else -1
_valid = np.where(COAST >= 0)[0]
COAST = np.interp(np.arange(MAP_H), _valid, COAST[_valid]).astype(np.int32)
SEA = (np.arange(MAP_W)[None, :] < COAST[:, None]).astype(np.float32)
_wr = random.Random(5)
WINDP = [(_wr.uniform(0, 1), _wr.uniform(15, MAP_H - 15), _wr.uniform(0.7, 1.3), _wr.uniform(0, 6.28)) for _ in range(26)]

@functools.lru_cache(None)
def screen_frame():
    pad = 90
    Wd, Hd = SW_ + 2 * pad, SH_ + 2 * pad
    can = Image.new("RGBA", (Wd, Hd), (0, 0, 0, 0))
    g = Image.new("L", (Wd, Hd), 0); ImageDraw.Draw(g).rounded_rectangle((pad, pad, pad + SW_, pad + SH_), 30, fill=200)
    gl = Image.new("RGBA", (Wd, Hd), (255, 210, 120, 255)); gl.putalpha(g.filter(ImageFilter.GaussianBlur(26)).point(lambda v: int(v * 0.55)))
    can = Image.alpha_composite(can, gl)
    body = Image.new("RGBA", (Wd, Hd), (0, 0, 0, 0)); bd = ImageDraw.Draw(body)
    bd.rounded_rectangle((pad, pad, pad + SW_, pad + SH_), 30, fill=(10, 30, 56, 245), outline=(255, 235, 190, 255), width=4)
    can = Image.alpha_composite(can, body)
    rnd = random.Random(4)
    cl = Image.open(A("cloud_3d.png")).convert("RGBA")
    spots = [(pad + k * SW_ / 8, pad - 8) for k in range(9)] + [(pad + k * SW_ / 8, pad + SH_ + 6) for k in range(9)]
    spots += [(pad - 6, pad + k * SH_ / 9) for k in range(1, 9)] + [(pad + SW_ + 6, pad + k * SH_ / 9) for k in range(1, 9)]
    for (x, y) in spots:
        sz = rnd.randint(70, 118)
        c = cl.resize((sz, sz), Image.LANCZOS).rotate(rnd.uniform(-20, 20))
        can.alpha_composite(c, (int(x - sz / 2), int(y - sz / 2)))
    return to_np(can), pad

@functools.lru_cache(None)
def screen_header(text):
    sz = 42
    while True:
        im = text_layer(text, sz, 800, fill=(255, 255, 255), pad=6)
        if im.size[0] <= SW_ - 40 or sz <= 24: return to_np(im)
        sz -= 2

@functools.lru_cache(None)
def small_lab(text, color):
    return to_np(text_layer(text, 30, 800, fill=color, stroke=3, stroke_fill=(10, 20, 40), pad=6))

def paste(dst, spr, x, y, alpha=1.0):
    h, w = spr.shape[:2]; x, y = int(x), int(y)
    X0, Y0, X1, Y1 = max(0, x), max(0, y), min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if X1 <= X0 or Y1 <= Y0: return
    s_ = spr[Y0 - y:Y1 - y, X0 - x:X1 - x]; a = s_[..., 3:4] / 255 * alpha
    d = dst[Y0:Y1, X0:X1, :3]; d *= (1 - a); d += s_[..., :3] * a

def wind_sim(t):
    img = np.zeros((MAP_H, MAP_W, 4), np.float32); img[..., 3] = 255
    xx = np.arange(MAP_W)[None, :].astype(np.float32); yy = np.arange(MAP_H)[:, None].astype(np.float32)
    dist = np.clip((COAST[:, None] - xx) / 120.0, 0, 1)            # 0 at the coast -> 1 offshore
    calm = clamp((t - 22.8) / 2.0)                                  # waves pushed west after "דוחקת"
    sea = np.zeros((MAP_H, MAP_W, 3), np.float32)
    deep, shallow = np.array([20, 90, 170.]), np.array([40, 200, 210.])
    k = (1 - dist)[..., None]
    sea[:] = deep * (1 - k) + shallow * k
    glass = clamp((t - 24.3) / 1.2)
    sea += (np.array([60, 235, 220.]) - sea) * (k * glass * 0.6)       # turquoise "Thailand" strip
    amp = np.clip(dist * (1 - calm) + dist ** 2 * calm, 0, 1)        # near the coast the sea flattens
    wav = np.sin(xx / 9.0 + t * 5.5 + np.sin(yy / 23.0) * 2) * np.sin(yy / 6.0 + xx / 40.0)
    crest = np.clip((wav - 0.55) / 0.45, 0, 1) * amp
    sea += (255 - sea) * crest[..., None] * 0.55
    land = np.array([196, 170, 110.]) * (1 - 0.15 * (yy / MAP_H))[..., None] * np.ones((1, MAP_W, 1))
    land = land * (1 - _ma[..., None] * 0.25) + np.array([120, 160, 90.]) * _ma[..., None] * 0.25
    img[..., :3] = sea * SEA[..., None] + land * (1 - SEA[..., None])
    edge = np.abs(np.diff(SEA, axis=1, prepend=0)) > 0
    img[edge, :3] = (255, 250, 230)
    # wind arrows: east -> west
    wa = clamp((t - 19.2) / 0.6)
    if wa > 0:
        lay = Image.new("RGBA", (MAP_W, MAP_H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
        for (u, y0, sp, ph) in WINDP:
            q = (u + t * 0.45 * sp) % 1.0
            x = MAP_W * (1.05 - q * 1.2)
            y = y0 + math.sin(q * 6 + ph) * 8
            L = 46
            col = (255, 190, 90) if x > COAST[int(min(max(y, 0), MAP_H - 1))] else (255, 255, 255)
            al = int(255 * wa * math.sin(min(1, q) * math.pi))
            d.line((x, y, x + L, y + math.sin(ph) * 3), fill=col + (al,), width=5)
            d.polygon([(x - 12, y), (x + 4, y - 9), (x + 4, y + 9)], fill=col + (al,))
        arr = np.asarray(lay.filter(ImageFilter.GaussianBlur(0.6))).astype(np.float32)
        a = arr[..., 3:4] / 255; img[..., :3] = img[..., :3] * (1 - a) + arr[..., :3] * a
    # labels
    la = clamp((t - 20.7) / 0.4)
    if la > 0:
        paste(img, small_lab("יבשה", (255, 225, 160)), MAP_W - 110, MAP_H * 0.55, la)
        paste(img, small_lab("ים", (190, 240, 255)), 20, MAP_H * 0.55, la)
    if t > 24.3:
        paste(img, small_lab("פלטה!", (120, 255, 235)), 30, MAP_H * 0.30, clamp((t - 24.3) / 0.4))
    return img

def draw_screen(F, t):
    st = None
    for a0, a1 in SCREEN_WINDOWS:
        if a0 - 0.05 <= t <= a1 + 0.4:
            st = (clamp((t - a0) / 0.45), clamp((t - a1) / 0.35), a0)
    if not st: return
    pin, pout, a0 = st
    frame, pad = screen_frame()
    spr = frame.copy()
    x0, y0 = pad + 12, pad + 12 + 70
    spr[y0:y0 + MAP_H, x0:x0 + MAP_W] = wind_sim(t)
    hd = screen_header("רוח מזרחית"); hh, hw = hd.shape[:2]
    paste(spr, hd, pad + SW_ // 2 - hw // 2, pad + 12 + (70 - hh) // 2)
    ic = icon("compass_3d.png", 52); paste(spr, ic, pad + SW_ - 66, pad + 20)
    ft = small_lab("מזרח ← מערב" if False else "מהיבשה אל הים", (255, 230, 170))
    paste(spr, ft, pad + SW_ // 2 - ft.shape[1] // 2, pad + SH_ - 48)
    h_, w_ = spr.shape[:2]
    src_pts = np.float32([[0, 0], [w_, 0], [w_, h_], [0, h_]]); k3 = 0.045
    dst_pts = np.float32([[0, h_ * k3], [w_, 0], [w_, h_], [0, h_ * (1 - k3)]])
    spr = cv2.warpPerspective(spr, cv2.getPerspectiveTransform(src_pts, dst_pts), (w_, h_), flags=cv2.INTER_LINEAR)
    sc = (0.35 + 0.65 * ease_out_back(pin, 1.5)) * (1 - 0.6 * ease_out_cubic(pout))
    F.blit(spr, SCX, SCY + math.sin(t * 1.3) * 5, alpha=clamp(pin * 2) * (1 - pout), behind=True, scale=sc)
    for k in range(7):
        for qq in (pin, pout):
            if 0 < qq < 1:
                ang = k * 2 * math.pi / 7
                F.blit(icon("cloud_3d.png", 120), SCX + math.cos(ang) * (120 + 160 * qq), SCY + math.sin(ang) * (200 + 160 * qq),
                       alpha=(1 - qq) * 0.9, behind=True, scale=0.6 + 0.6 * qq)
    # wind streaks across the real sky behind the presenter (east -> west)
    wa = clamp((t - 19.2) / 0.6) * (1 - clamp((t - 29.6) / 0.5))
    if wa > 0:
        lay = np.zeros((H, W), np.float32)
        for (u, y0_, sp, ph) in WINDP:
            q = (u + t * 0.6 * sp) % 1.0
            x = W * (1.1 - q * 1.3); y = 330 + y0_ * 1.1
            cv2.line(lay, (int(x), int(y)), (int(x + 140), int(y + math.sin(ph) * 6)), 0.55 * math.sin(q * math.pi), 3, cv2.LINE_AA)
        lay = cv2.GaussianBlur(lay, (0, 0), 1.5)[..., None] * wa * (1 - F.pm[..., None])
        F.img += (255 - F.img) * lay

def strip_vis(t):
    v = 1.0
    for a0, a1 in SCREEN_WINDOWS:
        if a0 - 0.3 <= t <= a1 + 0.4:
            v = min(v, 1 - clamp((t - (a0 - 0.3)) / 0.3) + clamp((t - a1) / 0.4))
    return clamp(v)

MOOD = [(0, -0.2), (99, -0.2)]
RAIN = [(0, 0), (99, 0)]


def kf(t, pts):
    xs, ys = zip(*pts); return float(np.interp(t, xs, ys))
ZOOM_CUTS = [0.1667, 9.0, 11.9667, 13.28, 15.1333, 15.7667, 18.0667, 22.0333, 22.8, 24.16, 25.8, 28.2667, 30.44, 33.7667, 36.7667, 38.6, 41.7333, 43.92, 47.0, 47.64, 52.8333, 54.16, 57.8333, 59.4667]

def zoom_at(t):
    z0 = 1.0 + 0.16 * (1 - ease_out_cubic(t / 0.55))  # intro punch-out
    if t < 4.6:
        return z0 if t < 0.55 else (1.0 + 0.035 * ease_in_out((t - 0.6) / 3.8))
    prev, cur, tc = 1.035, 1.035, 4.6
    vals = [1.075, 1.0]
    for i, c in enumerate(ZOOM_CUTS):
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
FIX = {"הצילים": "המצילים"}
def load_words():
    ws = []
    for s in json.load(open(os.path.join(HERE, "r7", "tr_fixed.json"))):
        for w in s["words"]:
            txt = w["w"].strip()
            brk = txt.endswith((",", ".", "?", "!"))
            txt = txt.strip(",.?!")
            if any(a - 0.01 <= w["s"] < b - 0.05 for a, b in CUTS): continue
            if w["s"] < 1 and txt == "כן": txt = "כאן"
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
    t1 = text_layer(name, 34, 800, fill=(255, 255, 255), pad=0, shadow=0)
    t2 = text_layer(date, 30, 700, fill=(255, 255, 255) if active else (205, 220, 245), pad=0)
    base.alpha_composite(t1, (pad + (w - t1.size[0]) // 2, pad + 10))
    icn = day_icon(ic)
    base.alpha_composite(icn, (pad + (w - 96) // 2, pad + 46))
    base.alpha_composite(t2, (pad + (w - t2.size[0]) // 2, pad + h - 22 - t2.size[1]))
    return to_np(base)

@functools.lru_cache(None)
def title_parts():
    l1 = text_layer("שבת לים!", 210, 800, gradient=((255, 238, 90), (255, 140, 25)),
                    stroke=6, stroke_fill=(110, 45, 0), extrude=12, extrude_fill=(170, 75, 0), shadow=16, pad=80)
    l2 = text_layer(" ", 10, 800, gradient=((255, 255, 255), (214, 232, 255)),
                    stroke=5, stroke_fill=(16, 38, 92), extrude=9, extrude_fill=(20, 52, 130), shadow=14, pad=80)
    return to_np(l1), to_np(l2)

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
    num = text_layer("62.1K", 50, 800, fill=(18, 24, 38), pad=0)
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

def render(i, raw):
    t = i / FPS
    z = zoom_at(t)
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

    # --- chips (front, chest level)
    for (t0, t1, txt, ic, side) in CHIPS:
        w = window(t, t0, t1, 0.3, 0.18)
        if not w: continue
        pin, a = w
        spr = chip_sprite(txt, ic)
        cw = spr.shape[1]
        cx = 62 + cw / 2 if side == "L" else 945 - cw / 2
        cx = min(max(cx, 40 + cw / 2), W - 40 - cw / 2)
        sc = 0.6 + 0.4 * ease_out_back(pin, 2.2)
        F.blit(spr, cx + (1 - ease_out_cubic(pin)) * (-60 if side == "L" else 60), 1182 + math.sin(t * 3) * 3,
               alpha=a, scale=sc)

    # --- title (intro)
    if t < TITLE_T1 + 0.35:
        l1, l2 = title_parts()
        out = clamp((t - TITLE_T1) / 0.3)
        oa = 1 - ease_in_out(out)
        oy = -ease_in_out(out) * 60
        def elem(spr, t0, cy):
            p = clamp((t - t0) / 0.38)
            if p <= 0: return
            F.blit(spr, 540, round(cy + oy - (1 - ease_out_cubic(p)) * 40), alpha=ease_out_cubic(p) * oa,
                   scale=(0.35 + 0.65 * ease_out_back(p, 1.2)))
        elem(l1, 0.0, 420)

    # --- forecast strip (top, above the head)
    svis = strip_vis(t)
    if t >= STRIP_T0 and svis > 0.01:
        act = []
        for (a0, a1, idx) in ACTIVE:
            if a0 <= t < a1: act = idx
        for k in range(6):
            p = clamp((t - STRIP_T0 - k * 0.06) / 0.35)
            if p <= 0: continue
            cx = 1012 - 70 - k * 154
            isact = k in act
            # pulse when it becomes active
            sc = 1.0
            if isact:
                t_on = max(a0 for (a0, a1, idx) in ACTIVE if k in idx and a0 <= t)
                sc = 1.0 + 0.04 * ease_out_back(clamp((t - t_on) / 0.3), 2.5) + 0.015 * math.sin(t * 6)
            cy = 368 + (1 - ease_out_back(p, 1.6)) * -60 - (1 - svis) * 80
            F.blit(day_card(k, isact), cx, cy, alpha=ease_out_cubic(p) * svis * (1.0 if isact or not act else 0.92), scale=sc)

    # --- follow card
    if t >= FOLLOW_T0:
        p = clamp((t - FOLLOW_T0) / 0.45)
        followed = t >= TAP_T + 0.12
        spr, (bx, by) = follow_card(followed)
        press = 1 - 0.06 * max(0, 1 - abs(t - TAP_T - 0.05) / 0.12)
        cy = 1282 + (1 - ease_out_back(p, 1.7)) * 260
        F.blit(spr, 540, cy, alpha=ease_out_cubic(p), scale=press)
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

    # --- subtitles
    for (t0, t1, ws) in CHUNKS:
        if t0 <= t < t1:
            p = clamp((t - t0) / 0.12)
            sprs = [sub_word(w["t"], w["s"] <= t < (ws[j + 1]["s"] if j + 1 < len(ws) else w["e"] + 0.2)) for j, w in enumerate(ws)]
            gap = -30
            total = sum(s.shape[1] for s in sprs) + gap * (len(sprs) - 1)
            sc = min(1.0, 860 / total) * (0.9 + 0.1 * ease_out_back(p, 2.5))
            x = 540 + total * sc / 2
            for s in sprs:  # RTL: first word at the right
                ww = s.shape[1] * sc
                F.blit(s, x - ww / 2, 1460, alpha=ease_out_cubic(p), scale=sc)
                x -= ww + gap * sc
            break

    np.clip(img, 0, 255, out=img)
    return img.astype(np.uint8), F.head_overlap

# ---------------------------------------------------------------- main
def run(start, end, outpath):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start / FPS:.4f}", "-i", VIDEO, "-vf", f"fps={FPS},scale={W}:{H}",
           "-frames:v", str(end - start), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    dec = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
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
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
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
