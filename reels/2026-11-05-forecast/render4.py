# Reels renderer: weather update 4.11
import sys, json, math, subprocess, random, functools, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
A = lambda n: os.path.join(HERE, "assets", n)
W, H, FPS = 1080, 1920, 30
VIDEO = os.path.join(HERE, "r3", "dl", "video.mp4")
NFR = 2605
KEEP = [(0, 2605)]  # source frame ranges kept
CUTS = [(a1 / 30, b0 / 30) for (a0, a1), (b0, b1) in zip(KEEP, KEEP[1:])]
DUR = NFR / FPS

PERSON = np.load(os.path.join(HERE, "r3", "person.npy"), mmap_mode="r")
HEAD = np.load(os.path.join(HERE, "r3", "head.npy"), mmap_mode="r")

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
DAYS = [("חמישי", "5.11", "sun_behind_rain_cloud_3d.png", "rain"),
        ("שישי", "6.11", "ICON_SUNRAINBOLT", "storm"),
        ("שבת", "7.11", "ICON_STORMSUN", "storm"),
        ("ראשון", "8.11", "sun_behind_rain_cloud_3d.png", "rain")]
ACTIVE = [(13.4, 25.34, [0]), (25.34, 30.2, [1]), (81.3, 99, [1, 2, 3])]
DAY_CHANGES = [25.34, 81.3]
THEME = {"sun": ((255, 196, 40), (255, 112, 30)), "rain": ((64, 156, 255), (36, 90, 220)),
         "storm": ((140, 92, 255), (76, 40, 190))}
STRIP_T0 = 13.4
TITLE_T1 = 4.85
SCREEN_WINDOWS = [(5.0, 13.4), (30.2, 81.3)]

BEHIND = [  # t0, t1, icon, cx, cy, size
    (0.3, 4.9, "umbrella_with_rain_drops_3d.png", 862, 760, 300),
    (0.6, 4.9, "cloud_with_rain_3d.png", 178, 735, 240),
    (5.0, 13.4, "cloud_with_lightning_and_rain_3d.png", 175, 720, 250),
    (13.5, 17.3, "sun_behind_small_cloud_3d.png", 862, 760, 300),
    (17.5, 23.9, "sun_behind_rain_cloud_3d.png", 862, 760, 300),
    (23.94, 25.3, "leaf_fluttering_in_wind_3d.png", 862, 760, 240),
    (25.34, 30.1, "cloud_with_rain_3d.png", 862, 760, 300),
    (30.2, 44.2, "cloud_with_rain_3d.png", 175, 720, 240),
    (44.24, 48.8, "sun_behind_small_cloud_3d.png", 175, 720, 240),
    (48.86, 63.9, "cloud_with_lightning_and_rain_3d.png", 175, 720, 250),
    (64.0, 66.2, "sun_3d.png", 175, 720, 220),
    (66.3, 77.1, "cloud_with_rain_3d.png", 175, 720, 240),
    (68.0, 75.0, "water_wave_3d.png", 180, 1040, 210),
    (77.2, 81.2, "sun_behind_rain_cloud_3d.png", 175, 720, 240),
    (81.3, 99, "umbrella_with_rain_drops_3d.png", 862, 760, 300),
]
LIGHTNING = [12.5, 50.74, 63.04]
CHIPS = [  # t0, t1, text, icon, side
    (1.0, 4.8, "הוציאו מטריות!", "umbrella_3d.png", "L"),
    (5.4, 8.4, "גשם מקומי כבר עכשיו", "cloud_with_rain_3d.png", "L"),
    (8.46, 11.7, "צפון · נהריה · עכו", "round_pushpin_3d.png", "R"),
    (11.8, 13.4, "גשם חזק + ברקים", "high_voltage_3d.png", "L"),
    (14.0, 17.4, "רוב הארץ: חמים ונוח", "sun_3d.png", "L"),
    (17.6, 20.6, "גשם קצר, יכול להיות חזק", "umbrella_3d.png", "R"),
    (20.7, 22.3, "מהצהריים עד הערב", "alarm_clock_3d.png", "L"),
    (22.4, 23.9, "מרכז וחוף דרומי", "round_pushpin_3d.png", "R"),
    (23.94, 25.3, "יום מעבר", "leaf_fluttering_in_wind_3d.png", "L"),
    (26.5, 30.1, "סגרירי וחורפי יותר", "cloud_3d.png", "R"),
    (30.9, 36.3, "ECMWF · המודל האמין ביותר", "check_mark_button_3d.png", "L"),
    (36.6, 38.5, "בוקר רגוע", "sun_behind_small_cloud_3d.png", "R"),
    (38.7, 42.0, "הגשם מתרבה", "cloud_with_rain_3d.png", "L"),
    (42.0, 44.2, "נרחב ורציף", "umbrella_with_rain_drops_3d.png", "R"),
    (44.3, 46.3, "יהיו גם חורים", "sun_behind_small_cloud_3d.png", "L"),
    (46.4, 48.8, "שמש ושמיים כחולים", "sun_3d.png", "R"),
    (48.9, 51.4, "גשם חזק + ברקים", "high_voltage_3d.png", "L"),
    (51.5, 53.5, "שימו לב לשבת", "warning_3d.png", "R"),
    (55.3, 59.0, "יותר גשם, נרחב יותר", "cloud_with_rain_3d.png", "L"),
    (59.2, 61.1, "גם בשבת יהיו חורים", "sun_behind_small_cloud_3d.png", "R"),
    (61.8, 63.9, "גשם חזק + ברקים", "high_voltage_3d.png", "L"),
    (64.0, 66.2, "שעה של שמש", "sun_3d.png", "R"),
    (66.3, 67.9, "הרבה יותר קריר", "thermometer_3d.png", "L"),
    (68.0, 71.8, "סכנת הצפות בערי החוף", "warning_3d.png", "R"),
    (71.9, 75.1, "מערכות הניקוז סתומות", "water_wave_3d.png", "L"),
    (75.2, 77.1, "הרבה גשם בזמן קצר", "cloud_with_rain_3d.png", "R"),
    (77.3, 79.7, "ראשון: גשם עד הצהריים", "cloud_with_rain_3d.png", "L"),
    (79.8, 81.2, "ואז מתבהר", "sun_3d.png", "R"),
    (81.3, 83.9, "היכונו למערכת גשם!", "umbrella_with_rain_drops_3d.png", "L"),
]
FOLLOW_T0, TAP_T = 84.0, 85.6

# ---- virtual screen with a cloud frame (behind the presenter)
MAPS = os.path.join(HERE, "r3", "maps")
SCREEN_SEQ = [  # t, image, header, source
    (5.0, "radar.png", "מכ״ם גשם · חמישי 09:30", "weather2day · IMS"),
    (30.2, "fri08.png", "שישי · 08:00", "ECMWF · IMS"), (38.6, "fri14.png", "שישי · 14:00", "ECMWF · IMS"),
    (42.0, "fri20.png", "שישי · 20:00", "ECMWF · IMS"), (51.5, "sat02.png", "שבת · 02:00", "ECMWF · IMS"),
    (53.6, "sat08.png", "שבת · 08:00", "ECMWF · IMS"), (56.7, "sat14.png", "שבת · 14:00", "ECMWF · IMS"),
    (61.2, "sat20.png", "שבת · 20:00", "ECMWF · IMS"), (69.0, "sun02.png", "ראשון · 02:00", "ECMWF · IMS"),
    (77.22, "sun08.png", "ראשון · 08:00", "ECMWF · IMS"), (78.6, "sun14.png", "ראשון · 14:00", "ECMWF · IMS"),
    (80.0, "sun20.png", "ראשון · 20:00", "ECMWF · IMS"), (80.7, "mon02.png", "שני · 02:00", "ECMWF · IMS"),
]
SCX, SCY, SW_, SH_ = 842, 642, 372, 610          # screen centre and size
MAP_W, MAP_H = SW_ - 24, SH_ - 24 - 70 - 40       # inner map area

@functools.lru_cache(None)
def map_img(name):
    im = Image.open(os.path.join(MAPS, name)).convert("RGB")
    iw, ih = im.size; s = max(MAP_W / iw, MAP_H / ih)
    im = im.resize((int(iw * s) + 1, int(ih * s) + 1), Image.LANCZOS)
    x0 = (im.size[0] - MAP_W) // 2; y0 = 0 if name != "radar.png" else (im.size[1] - MAP_H) // 2
    return np.asarray(im.crop((x0, y0, x0 + MAP_W, y0 + MAP_H))).astype(np.float32)

@functools.lru_cache(None)
def screen_frame():
    pad = 90
    Wd, Hd = SW_ + 2 * pad, SH_ + 2 * pad
    can = Image.new("RGBA", (Wd, Hd), (0, 0, 0, 0))
    # glow + shadow
    g = Image.new("L", (Wd, Hd), 0); ImageDraw.Draw(g).rounded_rectangle((pad, pad, pad + SW_, pad + SH_), 30, fill=200)
    gl = Image.new("RGBA", (Wd, Hd), (120, 200, 255, 255)); gl.putalpha(g.filter(ImageFilter.GaussianBlur(26)).point(lambda v: int(v * 0.55)))
    can = Image.alpha_composite(can, gl)
    body = Image.new("RGBA", (Wd, Hd), (0, 0, 0, 0)); bd = ImageDraw.Draw(body)
    bd.rounded_rectangle((pad, pad, pad + SW_, pad + SH_), 30, fill=(10, 22, 44, 245), outline=(170, 225, 255, 255), width=4)
    can = Image.alpha_composite(can, body)
    # puffy cloud frame from real cloud icons around the border
    rnd = random.Random(4)
    cl = Image.open(A("cloud_3d.png")).convert("RGBA")
    spots = []
    for k in range(9): spots.append((pad + k * SW_ / 8, pad - 8))
    for k in range(9): spots.append((pad + k * SW_ / 8, pad + SH_ + 6))
    for k in range(1, 9): spots.append((pad - 6, pad + k * SH_ / 9))
    for k in range(1, 9): spots.append((pad + SW_ + 6, pad + k * SH_ / 9))
    for (x, y) in spots:
        sz = rnd.randint(70, 118)
        c = cl.resize((sz, sz), Image.LANCZOS).rotate(rnd.uniform(-20, 20))
        can.alpha_composite(c, (int(x - sz / 2), int(y - sz / 2)))
    return to_np(can), pad

@functools.lru_cache(None)
def screen_header(text):
    sz = 40
    while True:
        im = text_layer(text, sz, 800, fill=(255, 255, 255), pad=6)
        if im.size[0] <= SW_ - 40 or sz <= 24: return to_np(im)
        sz -= 2

@functools.lru_cache(None)
def screen_src(text):
    f = font(22, 700); d0 = ImageDraw.Draw(Image.new("L", (1, 1)))
    bb = d0.textbbox((0, 0), text, font=f, direction="ltr"); im = Image.new("RGBA", (bb[2] - bb[0] + 12, bb[3] - bb[1] + 12), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((6 - bb[0], 6 - bb[1]), text, font=f, fill=(190, 225, 255, 230), direction="ltr")
    return to_np(im)

def screen_alpha(t):
    for a0, a1 in SCREEN_WINDOWS:
        if a0 - 0.05 <= t <= a1 + 0.4:
            return clamp((t - a0) / 0.45), clamp((t - a1) / 0.35), a0
    return None

def draw_screen(F, t):
    st = screen_alpha(t)
    if not st: return
    pin, pout, a0 = st
    frame, pad = screen_frame()
    spr = frame.copy()
    # current & previous map
    seq = [s for s in SCREEN_SEQ if s[0] <= max(t, a0)]
    cur = seq[-1]; prev = seq[-2] if len(seq) > 1 and seq[-2][0] >= a0 else None
    x0, y0 = pad + 12, pad + 12 + 70
    m = map_img(cur[1])
    fa = clamp((t - cur[0]) / 0.3)
    if prev is not None and fa < 1:
        m = map_img(prev[1]) * (1 - fa) + m * fa
    spr[y0:y0 + MAP_H, x0:x0 + MAP_W, :3] = m
    spr[y0:y0 + MAP_H, x0:x0 + MAP_W, 3] = 255
    # header with day/time (ticks in on change)
    hd = screen_header(cur[2]); hh, hw = hd.shape[:2]
    hx, hy = pad + SW_ // 2 - hw // 2, pad + 12 + (70 - hh) // 2
    hk = ease_out_cubic(clamp((t - cur[0]) / 0.25)) if cur[0] > a0 else 1
    yy = hy + int((1 - hk) * 20)
    sub = spr[yy:yy + hh, hx:hx + hw]; al = hd[..., 3:4] / 255 * hk
    sub[..., :3] = sub[..., :3] * (1 - al) + hd[..., :3] * al
    # timeline dots for the model maps
    if cur[1] != "radar.png":
        maps = [s for s in SCREEN_SEQ if s[1] != "radar.png"]
        idx = maps.index(cur); n = len(maps)
        yb = pad + SH_ - 26
        for k in range(n):
            xk = int(pad + SW_ - 30 - k * (SW_ - 60) / (n - 1))   # RTL: time runs right-to-left
            r = 7 if k == idx else 4
            col = (255, 220, 90) if k == idx else ((200, 230, 255) if k < idx else (90, 120, 160))
            cv2.circle(spr, (xk, yb), r, col + (255,), -1, cv2.LINE_AA)
    src = screen_src(cur[3]); sh_, sw2 = src.shape[:2]
    sx, sy = pad + 18, pad + SH_ - 40 - sh_ // 2 if cur[1] == "radar.png" else pad + SH_ - 52
    sub = spr[sy:sy + sh_, sx:sx + sw2]; al = src[..., 3:4] / 255
    sub[..., :3] = sub[..., :3] * (1 - al) + src[..., :3] * al
    # radar: pulsing marker on the rain near Nahariya/Akko
    if cur[1] == "radar.png" and 8.4 < t < 11.9:
        q = ((t - 8.4) * 1.6) % 1.0
        cx_, cy_ = int(x0 + 0.40 * MAP_W), int(y0 + 0.47 * MAP_H)
        cv2.circle(spr, (cx_, cy_), int(16 + 46 * q), (255, 70, 70, int(255 * (1 - q))), 4, cv2.LINE_AA)
        cv2.circle(spr, (cx_, cy_), 9, (255, 70, 70, 255), -1, cv2.LINE_AA)
    # gentle scanline sheen
    sweep = ((t * 0.35) % 1.4) - 0.2
    yy0 = int(y0 + sweep * MAP_H)
    if y0 <= yy0 < y0 + MAP_H - 30:
        band = spr[yy0:yy0 + 30, x0:x0 + MAP_W, :3]
        band += (255 - band) * 0.12
    # slight 3D perspective (turned towards the presenter)
    h_, w_ = spr.shape[:2]
    src_pts = np.float32([[0, 0], [w_, 0], [w_, h_], [0, h_]])
    k3 = 0.045
    dst_pts = np.float32([[0, h_ * k3], [w_, 0], [w_, h_], [0, h_ * (1 - k3)]])
    spr = cv2.warpPerspective(spr, cv2.getPerspectiveTransform(src_pts, dst_pts), (w_, h_), flags=cv2.INTER_LINEAR)
    sc = (0.35 + 0.65 * ease_out_back(pin, 1.5)) * (1 - 0.6 * eoc_(pout))
    a = clamp(pin * 2) * (1 - pout)
    F.blit(spr, SCX, SCY + math.sin(t * 1.3) * 5, alpha=a, behind=True, scale=sc)
    # cloud puff burst on entry / exit
    for k in range(7):
        for qq in (pin, pout):
            if 0 < qq < 1:
                ang = k * 2 * math.pi / 7
                F.blit(icon("cloud_3d.png", 120), SCX + math.cos(ang) * (120 + 160 * qq), SCY + math.sin(ang) * (200 + 160 * qq),
                       alpha=(1 - qq) * 0.9, behind=True, scale=0.6 + 0.6 * qq)

def eoc_(x): return ease_out_cubic(x)

def strip_vis(t):
    v = 1.0
    for a0, a1 in SCREEN_WINDOWS:
        if a0 - 0.3 <= t <= a1 + 0.4:
            v = min(v, 1 - clamp((t - (a0 - 0.3)) / 0.3) + clamp((t - a1) / 0.4))
    return clamp(v)

def kf(t, pts):
    xs, ys = zip(*pts); return float(np.interp(t, xs, ys))
MOOD = [(0, 0.45), (4.8, 0.45), (5.5, 0.7), (13.3, 0.7), (14.0, -0.4), (17.4, -0.4), (18.0, 0.5), (24.0, 0.4), (25.4, 0.8),
        (36.4, 0.6), (38.6, 0.6), (39.2, 0.9), (44.2, 0.9), (44.8, 0.0), (48.7, 0.0), (49.2, 1.0), (59.0, 1.0), (59.6, 0.8),
        (63.9, 1.0), (64.2, 0.0), (66.2, 0.0), (66.6, 0.9), (77.1, 0.9), (77.6, 0.6), (79.7, 0.6), (80.2, 0.0), (81.2, 0.35), (99, 0.35)]
RAIN = [(0, 0.4), (4.9, 0.4), (5.5, 0.8), (13.4, 0.8), (14.0, 0), (17.5, 0), (18.0, 0.6), (24.0, 0.4), (25.4, 0.5), (36.4, 0.5),
        (37.0, 0.2), (38.6, 0.2), (39.5, 1.0), (44.2, 1.0), (44.8, 0.1), (48.7, 0.1), (49.2, 1), (63.9, 1), (64.2, 0.05),
        (66.2, 0.05), (66.8, 1), (77.1, 1), (79.7, 0.6), (80.2, 0), (81.2, 0.5), (99, 0.5)]
ZOOM_CUTS = [5.0, 11.12, 13.58, 17.54, 20.7, 23.94, 25.34, 30.2, 33.4, 36.36, 38.54, 42.04, 44.24, 46.3, 48.86, 51.5, 53.6,
             55.24, 59.12, 61.18, 64.0, 66.32, 67.98, 71.9, 75.18, 77.22, 79.72, 81.26, 84.0]

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
FIX = {"העבר": "מעבר", "קורפית": "חורפית", "ECNWF": "ECMWF", "והגבול": "תעקבו"}
def load_words():
    ws = []
    for s in json.load(open(os.path.join(HERE, "r3", "tr.json"))):
        for w in s["words"]:
            txt = w["w"].strip()
            brk = txt.endswith((",", ".", "?", "!"))
            txt = txt.strip(",.?!")
            if any(a - 0.01 <= w["s"] < b - 0.05 for a, b in CUTS): continue
            if w["s"] > 85 and txt == "עדכונים": txt = "לעדכונים"
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
    w, h = 210, 196
    if active:
        base = rounded(w, h, 26, (255, 255, 255, 255), grad=THEME[th], border=((255, 255, 255, 235), 4), shadow=14)
    else:
        base = rounded(w, h, 26, (14, 26, 52, 165), border=((255, 255, 255, 70), 2), shadow=8)
    pad = (base.size[0] - w) // 2
    t1 = text_layer(name, 42, 800, fill=(255, 255, 255), pad=0, shadow=0)
    t2 = text_layer(date, 30, 700, fill=(255, 255, 255) if active else (205, 220, 245), pad=0)
    base.alpha_composite(t1, (pad + (w - t1.size[0]) // 2, pad + 10))
    icn = day_icon(ic)
    base.alpha_composite(icn, (pad + (w - 96) // 2, pad + 52))
    base.alpha_composite(t2, (pad + (w - t2.size[0]) // 2, pad + h - 22 - t2.size[1]))
    return to_np(base)

@functools.lru_cache(None)
def title_parts():
    l1 = text_layer("רגע לפני מהפך", 116, 800, gradient=((255, 255, 255), (214, 232, 255)),
                    stroke=5, stroke_fill=(16, 38, 92), extrude=10, extrude_fill=(20, 52, 130), shadow=14, pad=80)
    l2 = text_layer("ומערכת גשם", 100, 800, gradient=((120, 230, 255), (40, 150, 255)),
                    stroke=5, stroke_fill=(10, 30, 80), extrude=9, extrude_fill=(14, 40, 110), shadow=14, pad=80)
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
    num = text_layer("60.9K", 50, 800, fill=(18, 24, 38), pad=0)
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
        elem(l1, 0.08, 336)
        elem(l2, 0.4, 432)

    # --- forecast strip (top, above the head)
    svis = strip_vis(t)
    if t >= STRIP_T0 and svis > 0.01:
        act = []
        for (a0, a1, idx) in ACTIVE:
            if a0 <= t < a1: act = idx
        for k in range(4):
            p = clamp((t - STRIP_T0 - k * 0.06) / 0.35)
            if p <= 0: continue
            cx = 1010 - 105 - k * 230
            isact = k in act
            # pulse when it becomes active
            sc = 1.0
            if isact:
                t_on = max(a0 for (a0, a1, idx) in ACTIVE if k in idx and a0 <= t)
                sc = 1.0 + 0.04 * ease_out_back(clamp((t - t_on) / 0.3), 2.5) + 0.015 * math.sin(t * 6)
            cy = 384 + (1 - ease_out_back(p, 1.6)) * -60 - (1 - svis) * 80
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
