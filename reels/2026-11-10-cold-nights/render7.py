# Reels renderer: weather update 4.11
import sys, json, math, subprocess, random, functools, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
A = lambda n: os.path.join(HERE, "assets", n)
W, H, FPS = 1080, 1920, 30
VIDEO = os.path.join(HERE, "r6", "dl", "video.mp4")
NFR = 1711
KEEP = [(0, 233), (243, 379), (390, 465), (472, 583), (591, 683), (693, 846), (855, 913), (926, 979), (985, 1039), (1047, 1205), (1211, 1313), (1326, 1418), (1426, 1465), (1470, 1544), (1550, 1620), (1635, 1706), (1710, 1711)]
CUTS = [(a1 / 30, b0 / 30) for (a0, a1), (b0, b1) in zip(KEEP, KEEP[1:])]
DUR = NFR / FPS

PERSON = np.load(os.path.join(HERE, "r6", "person.npy"), mmap_mode="r")
HEAD = np.load(os.path.join(HERE, "r6", "head.npy"), mmap_mode="r")

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
DAYS = [("שלישי", "10.11", "crescent_moon_3d.png", "storm"),
        ("רביעי", "11.11", "sun_behind_rain_cloud_3d.png", "rain"),
        ("חמישי", "12.11", "sun_behind_small_cloud_3d.png", "sun"),
        ("שישי", "13.11", "sun_3d.png", "sun"),
        ("שבת", "14.11", "sun_3d.png", "sun"),
        ("ראשון", "15.11", "sun_3d.png", "sun"),
        ("שני", "16.11", "sun_3d.png", "sun")]
ACTIVE = [(6.7, 32.62, [0]), (32.62, 41.66, [1]), (41.66, 43.86, [2]), (43.86, 46.48, [2, 3, 4, 5, 6]),
          (46.48, 49.0, [3, 4]), (49.0, 99, [3, 4, 5, 6])]
DAY_CHANGES = [32.62, 41.66, 43.86, 46.48, 49.0]
THEME = {"sun": ((255, 196, 40), (255, 112, 30)), "rain": ((64, 156, 255), (36, 90, 220)),
         "storm": ((90, 110, 255), (40, 50, 170))}
STRIP_T0 = 6.7
TITLE_T1 = 6.6
SCREEN_WINDOWS = [(21.6, 29.45)]

BEHIND = [  # t0, t1, icon, cx, cy, size
    (0.2, 6.6, "crescent_moon_3d.png", 862, 740, 270),
    (0.5, 6.6, "snowflake_3d.png", 175, 735, 200),
    (6.7, 21.5, "crescent_moon_3d.png", 862, 740, 270),
    (7.02, 21.5, "cold_face_3d.png", 175, 730, 190),
    (29.5, 32.55, "sun_3d.png", 862, 750, 280),
    (32.62, 43.8, "sun_behind_rain_cloud_3d.png", 862, 760, 300),
    (43.86, 48.9, "sun_3d.png", 862, 750, 280),
    (46.48, 48.9, "backpack_3d.png", 175, 720, 200),
    (49.0, 54.3, "crescent_moon_3d.png", 862, 740, 260),
    (54.34, 99, "sun_3d.png", 862, 750, 280),
]
LIGHTNING = []
CHIPS = [  # t0, t1, text, icon, side
    (0.5, 2.6, "עוד לילה קר מאוד", "cold_face_3d.png", "L"),
    (2.66, 6.6, "סביב 10° ופחות", "thermometer_3d.png", "R"),
    (7.02, 9.9, "לא נפוץ בנובמבר", "warning_3d.png", "L"),
    (10.5, 13.3, "בד״כ רק בסוף החודש", "alarm_clock_3d.png", "R"),
    (15.32, 19.3, "הכי קר כבר 20 שנה", "chart_decreasing_3d.png", "L"),
    (19.32, 21.5, "גם הלילה הקרוב", "crescent_moon_3d.png", "R"),
    (29.48, 32.5, "ימים שמשיים ונוחים", "sun_3d.png", "R"),
    (32.62, 35.3, "מחר: יותר עננים", "cloud_3d.png", "L"),
    (35.32, 39.4, "סיכוי לטפטוף בצפון ובחוף", "cloud_with_rain_3d.png", "R"),
    (39.48, 42.6, "מחר בערב עד חמישי בבוקר", "alarm_clock_3d.png", "L"),
    (42.68, 43.8, "ממש זניח", "check_mark_button_3d.png", "R"),
    (43.86, 46.4, "שמשי בימים הקרובים", "sun_3d.png", "L"),
    (46.48, 48.8, "סופ״ש מושלם לטיולים", "backpack_3d.png", "R"),
    (49.0, 51.2, "לילות פחות קרים", "crescent_moon_3d.png", "L"),
    (51.28, 53.9, "עד אמצע השבוע הבא", "alarm_clock_3d.png", "R"),
    (54.34, 55.45, "אין צפי לגשם", "check_mark_button_3d.png", "L"),
]
FOLLOW_T0, TAP_T = 55.5, 55.75

# ---- night-temperature map in FRONT of the presenter (left column, never over the head)
MAPDIR = os.path.join(HERE, "r6", "map")
MB = [int(v) for v in open(os.path.join(MAPDIR, "bbox.txt")).read().split()]   # x0 y0 w h in source image
MS, MX, MY = 0.64, 20, 382
CITIES = [  # name, temp, source-image x, y, label side
    ("חרמון", 0, 838, 497, "R"), ("מרום גולן", 3, 830, 559, "R"), ("חיפה", 12, 580, 676, "C"),
    ("טבריה", 12, 749, 686, "C"), ("תל אביב", 10, 508, 954, "C"), ("ירושלים", 9, 657, 1064, "C"),
    ("עין גדי", 18, 707, 1181, "C"), ("באר שבע", 10, 513, 1262, "C"), ("מצפה רמון", 9, 522, 1492, "R"),
    ("אילת", 14, 567, 1884, "R")]
LABEL_Y = {"טבריה": 568}
COLD = {"חרמון", "מרום גולן"}
MAP_T0, MAP_T1 = 21.68, 29.4

def tcol(T):
    stops = [(0, (60, 70, 200)), (5, (50, 120, 230)), (10, (40, 200, 190)), (15, (240, 210, 60)), (18, (255, 140, 40))]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if T <= b:
            k = clamp((T - a) / (b - a)); return tuple(int(ca[i] + (cb[i] - ca[i]) * k) for i in range(3))
    return stops[-1][1]

def to_scr(x, y): return MX + (x - MB[0]) * MS, MY + (y - MB[1]) * MS

@functools.lru_cache(None)
def map_shape():
    im = Image.open(os.path.join(MAPDIR, "shape_s.png")).convert("RGBA")
    w, h = int(im.size[0] * MS), int(im.size[1] * MS)
    im = im.resize((w, h), Image.LANCZOS)
    a = im.split()[3]
    pad = 30
    can = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    gl = Image.new("RGBA", can.size, (150, 210, 255, 0)); big = Image.new("L", can.size, 0); big.paste(a, (pad, pad))
    gl.putalpha(big.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(12)).point(lambda v: int(v * 0.8)))
    can = Image.alpha_composite(can, gl)
    edge = Image.new("RGBA", can.size, (255, 255, 255, 0)); edge.putalpha(big.filter(ImageFilter.MaxFilter(5)))
    can = Image.alpha_composite(can, edge)
    can.alpha_composite(im, (pad, pad))
    return to_np(can), pad

@functools.lru_cache(None)
def city_pill(name, T):
    nm = text_layer(name, 30, 800, fill=(255, 255, 255), pad=0)
    num = Image.new("RGBA", (64, 52), (0, 0, 0, 0)); d = ImageDraw.Draw(num)
    d.ellipse((2, 0, 62, 50), fill=tcol(T) + (255,), outline=(255, 255, 255, 255), width=3)
    f = font(30, 800); txt = f"{T}"
    bb = d.textbbox((0, 0), txt, font=f, direction="ltr")
    d.text((32 - (bb[2] - bb[0]) / 2 - bb[0], 25 - (bb[3] - bb[1]) / 2 - bb[1]), txt, font=f, fill=(255, 255, 255, 255),
           direction="ltr", stroke_width=2, stroke_fill=(10, 20, 40, 255))
    w = nm.size[0] + 64 + 26; h = 54
    base = Image.new("RGBA", (w + 20, h + 20), (0, 0, 0, 0)); bd = ImageDraw.Draw(base)
    bd.rounded_rectangle((10, 10, 10 + w, 10 + h), 27, fill=(10, 22, 48, 225), outline=(170, 225, 255, 200), width=2)
    base.alpha_composite(num, (10 + w - 64 - 2, 10 + 1))          # number on the right (RTL)
    base.alpha_composite(nm, (10 + 14, 10 + (h - nm.size[1]) // 2 - 2))
    return to_np(base)

@functools.lru_cache(None)
def map_header():
    t1 = text_layer("טמפרטורות הלילה", 44, 800, fill=(255, 255, 255), pad=0)
    t2 = text_layer("שלישי 10.11", 30, 700, fill=(190, 225, 255), pad=0)
    w = max(t1.size[0], t2.size[0]) + 50; h = 104
    base = Image.new("RGBA", (w + 24, h + 24), (0, 0, 0, 0)); bd = ImageDraw.Draw(base)
    bd.rounded_rectangle((12, 12, 12 + w, 12 + h), 26, fill=(20, 70, 170, 235), outline=(255, 255, 255, 240), width=3)
    base.alpha_composite(t1, (12 + (w - t1.size[0]) // 2, 12 + 8))
    base.alpha_composite(t2, (12 + (w - t2.size[0]) // 2, 12 + 62))
    return to_np(base)

@functools.lru_cache(None)
def map_legend():
    w, h = 260, 18
    im = Image.new("RGBA", (w + 90, 70), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for x in range(w):
        T = 18 * x / (w - 1); d.line((45 + x, 10, 45 + x, 10 + h), fill=tcol(T) + (255,))
    d.rounded_rectangle((44, 9, 46 + w, 11 + h), 6, outline=(255, 255, 255, 220), width=2)
    f = font(24, 700)
    d.text((8, 4), "0°", font=f, fill=(220, 235, 255, 255), direction="ltr")
    d.text((w + 52, 4), "18°", font=f, fill=(255, 220, 160, 255), direction="ltr")
    lab = text_layer("קר", 24, 700, fill=(200, 220, 255), pad=0); im.alpha_composite(lab, (45, 36))
    lab2 = text_layer("חמים", 24, 700, fill=(255, 210, 150), pad=0); im.alpha_composite(lab2, (w + 40 - lab2.size[0], 36))
    return to_np(im)

FROST = [(random.Random(k).uniform(20, 300), random.Random(k + 99).uniform(380, 1300), random.Random(k + 7).uniform(0, 6.28)) for k in range(18)]

def draw_screen(F, t):
    if not (MAP_T0 - 0.05 <= t <= MAP_T1 + 0.45): return
    pin = clamp((t - MAP_T0) / 0.5); pout = clamp((t - MAP_T1) / 0.4)
    a = ease_out_cubic(pin) * (1 - ease_out_cubic(pout))
    dx = -(1 - ease_out_cubic(pin)) * 260 - ease_out_cubic(pout) * 260
    shp, pad = map_shape()
    # reveal north -> south
    rv = clamp((t - MAP_T0) / 1.1)
    s2 = shp.copy()
    cut = int(s2.shape[0] * ease_out_cubic(rv))
    if cut < s2.shape[0]: s2[cut:, :, 3] *= np.clip(1 - (np.arange(s2.shape[0] - cut)[:, None] / 40.0), 0, 1)
    F.blit(s2, MX + shp.shape[1] / 2 - pad + dx, MY + shp.shape[0] / 2 - pad + math.sin(t * 1.4) * 4, alpha=a)
    F.blit(map_header(), 222 + dx, 318, alpha=a * ease_out_cubic(clamp((t - MAP_T0 - 0.1) / 0.35)),
           scale=0.6 + 0.4 * ease_out_back(clamp((t - MAP_T0 - 0.1) / 0.4), 2))
    F.blit(map_legend(), 190 + dx, 1352, alpha=a * clamp((t - MAP_T0 - 0.8) / 0.4))
    # frost sparkles drifting over the map
    for k, (fx, fy, ph) in enumerate(FROST):
        q = (t * 0.25 + k / 18) % 1.0
        F.blit(icon("snowflake_3d.png", 34), fx + dx + math.sin(t + ph) * 10, fy + q * 60, alpha=a * 0.55 * math.sin(q * math.pi), rot=t * 60)
    # city pills pop north -> south
    for j, (name, T, sx, sy, side) in enumerate(CITIES):
        p = clamp((t - (MAP_T0 + 0.6 + j * 0.28)) / 0.3)
        if p <= 0: continue
        x, y = to_scr(sx, sy); x += dx
        ly = LABEL_Y.get(name, y)
        pill = city_pill(name, T); pw = pill.shape[1]
        if side == "R": px = x + 14 + pw / 2
        else: px = min(max(30 + pw / 2, x), 300 - pw / 2 + 10)
        dot_r = 9
        F.blit(soft_dot_(dot_r), x, y, alpha=a * p)
        if name in COLD and t > 25.4:
            q = ((t - 25.4) * 1.3) % 1.0
            F.blit(ring_(int(18 + 40 * q)), x, y, alpha=a * (1 - q))
        F.blit(pill, px, ly, alpha=a * ease_out_cubic(p), scale=0.5 + 0.5 * ease_out_back(p, 2.4) + (0.06 * math.sin((t - 25.4) * 8) if (name in COLD and t > 25.4) else 0))

@functools.lru_cache(None)
def soft_dot_(r):
    s = r * 2 + 6; im = Image.new("RGBA", (s, s), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.ellipse((3, 3, s - 3, s - 3), fill=(255, 255, 255, 255), outline=(20, 40, 80, 255), width=2)
    return to_np(im)

@functools.lru_cache(None)
def ring_(r):
    s = r * 2 + 10; im = Image.new("RGBA", (s, s), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.ellipse((5, 5, s - 5, s - 5), outline=(120, 180, 255, 255), width=4)
    return to_np(im)

def strip_vis(t):
    v = 1.0
    for a0, a1 in SCREEN_WINDOWS:
        if a0 - 0.3 <= t <= a1 + 0.4:
            v = min(v, 1 - clamp((t - (a0 - 0.3)) / 0.3) + clamp((t - a1) / 0.4))
    return clamp(v)

MOOD = [(0, 0), (99, 0)]
RAIN = [(0, 0), (35.3, 0), (35.8, 0.15), (43.0, 0.15), (43.6, 0), (99, 0)]


def kf(t, pts):
    xs, ys = zip(*pts); return float(np.interp(t, xs, ys))
ZOOM_CUTS = [7.02, 8.1, 13.0, 15.7333, 19.7, 21.68, 23.1, 28.5, 29.48, 30.8667, 32.8333, 34.9, 39.48, 40.3667, 44.2, 46.48, 47.5333, 49.0, 51.6667, 54.5, 55.56, 57.0]

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
FIX = {"-5": "מ־5", "מ": ""}
def load_words():
    ws = []
    for s in json.load(open(os.path.join(HERE, "r6", "tr.json"))):
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
    w, h = 128, 178
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
    l1 = text_layer("לילות קרים!", 124, 800, gradient=((235, 248, 255), (150, 200, 255)),
                    stroke=5, stroke_fill=(16, 38, 92), extrude=10, extrude_fill=(20, 52, 130), shadow=14, pad=80)
    l2 = text_layer("עדכון התחזית", 90, 800, gradient=((255, 255, 255), (214, 232, 255)),
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
    num = text_layer("61.3K", 50, 800, fill=(18, 24, 38), pad=0)
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
        elem(l1, 0.08, 326)
        elem(l2, 0.4, 418)

    # --- forecast strip (top, above the head)
    svis = strip_vis(t)
    if t >= STRIP_T0 and svis > 0.01:
        act = []
        for (a0, a1, idx) in ACTIVE:
            if a0 <= t < a1: act = idx
        for k in range(7):
            p = clamp((t - STRIP_T0 - k * 0.06) / 0.35)
            if p <= 0: continue
            cx = 1020 - 64 - k * 136
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
        cy = 1170 + (1 - ease_out_back(p, 1.7)) * 260
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
