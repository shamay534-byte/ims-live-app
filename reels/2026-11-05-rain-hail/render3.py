# Reel #2 v2: one continuous shot - camera journey through a storm
import sys, os, json, math, random, subprocess, functools, importlib.util
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("R", os.path.join(HERE, "render.py"))
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
clamp, eoc, eio, eob = R.clamp, R.ease_out_cubic, R.ease_in_out, R.ease_out_back
text_layer, rounded, icon, icon_shadowed, to_np, A = R.text_layer, R.rounded, R.icon, R.icon_shadowed, R.to_np, R.A
D = os.path.join(HERE, "r2")
W, H, FPS = 1080, 1920, 30
INTRO = 8.3
VO0 = 8.05                     # voiceover starts right after "פינת העשרה"
VO_DUR = 58.30
DUR = VO0 + VO_DUR
NFR = int(round(DUR * FPS))
FLASH1, FLASH2 = 6.933, 7.233
HEAD = np.load(os.path.join(D, "seed2_head.npy"), mmap_mode="r")
CYAN = (90, 230, 255)

# ------------------------------------------------------------------ world + camera
WORLD = np.asarray(Image.open(os.path.join(D, "bg", "world.jpg")).convert("RGB"))
WH_, WW_ = WORLD.shape[:2]
KEYS = [(-0.45, 5650, 1250), (0.45, 7140, 1080), (2.8, 7090, 1100), (5.0, 6150, 1250), (9.6, 5850, 1150),
        (12.2, 5000, 1400), (15.4, 4870, 1300), (18.6, 4720, 1300), (21.5, 2500, 1300), (23.3, 2380, 1300),
        (27.3, 6700, 1300), (28.6, 6900, 1420), (29.7, 6870, 1420), (31.2, 1567, 1500), (33.4, 1560, 1500),
        (35.0, 3900, 2048), (38.4, 4100, 2048), (39.9, 2600, 1500), (41.2, 4300, 1500), (42.5, 2500, 1500),
        (43.9, 2900, 1500), (47.8, 6300, 1400), (51.62, 7380, 1150), (52.4, 7300, 1300), (55.0, 4300, 2048),
        (58.5, 4150, 2048)]
_kt = [k[0] for k in KEYS]
CAM_Y = PchipInterpolator(_kt, [k[1] for k in KEYS])
CAM_W = PchipInterpolator(_kt, [k[2] for k in KEYS])

def cam(v):
    v = min(max(v, _kt[0]), _kt[-1])
    cy, ww = float(CAM_Y(v)), float(CAM_W(v))
    cx = 1024 + 120 * clamp((cy - 6600) / 700) + math.sin(v * 0.7) * 14
    cy += math.sin(v * 1.1) * 8
    return cx, cy, ww

def h_of_y(y):
    return float(np.interp(-y, [-7350, -6050, -4300, -2500, -1100, -300], [0, 1.5, 5, 8, 11, 13]))
def temp_of_h(h): return 30 - 8 * h
FREEZE_Y = 6050 - (3.75 - 1.5) / 3.5 * 1750

def world_to_screen(x, y, c):
    cx, cy, ww = c; s = W / ww
    return (x - cx) * s + W / 2, (y - cy) * s + H / 2

def world_frame(v, shake=(0, 0)):
    cx, cy, ww = cam(v); s = W / ww
    M = np.array([[s, 0, W / 2 - s * cx + shake[0]], [0, s, H / 2 - s * cy + shake[1]]], np.float32)
    img = cv2.warpAffine(WORLD, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # motion blur proportional to camera speed
    c2 = cam(v + 1 / FPS)
    sp = abs((c2[1] - cy) * s)
    if sp > 6:
        k = int(min(70, sp * 1.2)) | 1
        ker = np.zeros((k, 1), np.float32); ker[:, 0] = 1 / k
        img = cv2.filter2D(img, -1, ker)
    return img.astype(np.float32), sp

@functools.lru_cache(None)
def vignette():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = 1 - 0.32 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 1.6)) ** 2)
    g = np.clip(1 - 0.5 * np.clip((yy - 1300) / 620, 0, 1), 0, 1)
    return (np.clip(v, 0.6, 1) * g)[..., None]

# ------------------------------------------------------------------ sprites
@functools.lru_cache(None)
def soft_dot(r, color=(255, 255, 255), hard=0):
    s = int(r * 2 + 4)
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    d = np.hypot(xx - s / 2, yy - s / 2) / r
    a = np.clip(1 - d, 0, 1) ** (0.5 if hard else 1.7) * 255
    out = np.zeros((s, s, 4), np.float32); out[..., :3] = color; out[..., 3] = a
    return out

@functools.lru_cache(None)
def ring(r, width=5, color=CYAN, alpha=255, dash=0):
    s = int(r * 2 + width * 2 + 6); im = Image.new("RGBA", (s, s), (0, 0, 0, 0)); d = ImageDraw.Draw(im); c = s / 2
    if dash:
        for k in range(0, 360, dash * 2): d.arc((c - r, c - r, c + r, c + r), k, k + dash, fill=color + (alpha,), width=width)
    else:
        d.ellipse((c - r, c - r, c + r, c + r), outline=color + (alpha,), width=width)
    g = im.filter(ImageFilter.GaussianBlur(6)); out = Image.alpha_composite(g, im)
    return to_np(out)

@functools.lru_cache(None)
def crystal(size):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0)); d = ImageDraw.Draw(im); c = size / 2; L = size * 0.45
    w = max(2, size // 22)
    for k in range(6):
        a = k * math.pi / 3
        ex, ey = c + L * math.cos(a), c + L * math.sin(a)
        d.line((c, c, ex, ey), fill=(235, 248, 255, 255), width=w)
        for f in (0.4, 0.65):
            bx, by = c + L * f * math.cos(a), c + L * f * math.sin(a)
            for sgn in (-1, 1):
                b = a + sgn * math.pi / 4.5
                d.line((bx, by, bx + L * 0.28 * math.cos(b), by + L * 0.28 * math.sin(b)), fill=(210, 238, 255, 255), width=max(1, w - 1))
    d.regular_polygon((c, c, size * 0.08), 6, fill=(255, 255, 255, 255))
    g = im.filter(ImageFilter.GaussianBlur(size / 25))
    gl = Image.new("RGBA", im.size, (120, 210, 255, 0)); gl.putalpha(g.split()[3].point(lambda v: int(v * 0.9)))
    out = Image.alpha_composite(gl, im)
    return to_np(out)

@functools.lru_cache(None)
def droplet(r):
    s = int(r * 2 + 10); im = Image.new("RGBA", (s, s), (0, 0, 0, 0)); d = ImageDraw.Draw(im); c = s / 2
    for k in range(10, 0, -1):
        rr = r * k / 10
        col = (int(90 + 100 * (1 - k / 10)), int(170 + 60 * (1 - k / 10)), 255, int(120 + 80 * (1 - k / 10)))
        d.ellipse((c - rr, c - rr, c + rr, c + rr), fill=col)
    d.ellipse((c - r, c - r, c + r, c + r), outline=(225, 245, 255, 255), width=max(2, r // 12))
    d.ellipse((c - r * 0.55, c - r * 0.62, c - r * 0.12, c - r * 0.22), fill=(255, 255, 255, 220))
    return to_np(im.filter(ImageFilter.GaussianBlur(0.7)))

@functools.lru_cache(None)
def hailstone(r, layers):
    s = int(r * 2 + 30); im = Image.new("RGBA", (s, s), (0, 0, 0, 0)); d = ImageDraw.Draw(im); c = s / 2
    for k in range(max(1, layers), 0, -1):
        rr = r * k / max(1, layers)
        col = (238, 246, 255, 250) if k % 2 else (178, 210, 238, 250)
        d.ellipse((c - rr, c - rr, c + rr, c + rr), fill=col, outline=(120, 165, 215, 255), width=2)
    d.ellipse((c - r * 0.55, c - r * 0.65, c - r * 0.1, c - r * 0.25), fill=(255, 255, 255, 200))
    g = im.filter(ImageFilter.GaussianBlur(8))
    gl = Image.new("RGBA", im.size, (150, 220, 255, 0)); gl.putalpha(g.split()[3].point(lambda v: int(v * 0.6)))
    return to_np(Image.alpha_composite(gl, im))

@functools.lru_cache(None)
def glass(w, h, r=28, border=CYAN):
    pad = 24; im = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((pad, pad, pad + w, pad + h), r, fill=(8, 22, 48, 165))
    g = Image.new("RGBA", im.size, (0, 0, 0, 0)); ImageDraw.Draw(g).rounded_rectangle((pad, pad, pad + w, pad + h), r, outline=border + (255,), width=4)
    im = Image.alpha_composite(im, g.filter(ImageFilter.GaussianBlur(7)))
    im = Image.alpha_composite(im, g)
    return to_np(im)

@functools.lru_cache(None)
def txt(text, size, color=(255, 255, 255), weight=800, stroke=0, sf=(0, 0, 0), shadow=6):
    return to_np(text_layer(text, size, weight, fill=color, stroke=stroke, stroke_fill=sf, shadow=shadow, pad=30))

@functools.lru_cache(None)
def ltr(text, size, color=(255, 255, 255), weight=800, glow=True):
    f = R.font(size, weight)
    d0 = ImageDraw.Draw(Image.new("L", (1, 1)))
    bb = d0.textbbox((0, 0), text, font=f, direction="ltr", stroke_width=4)
    w, h = bb[2] - bb[0] + 40, bb[3] - bb[1] + 40
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.text((20 - bb[0], 20 - bb[1]), text, font=f, fill=color + (255,), direction="ltr", stroke_width=4, stroke_fill=(5, 15, 40, 255))
    if glow:
        g = Image.new("RGBA", im.size, color + (0,)); g.putalpha(im.split()[3].filter(ImageFilter.GaussianBlur(8)).point(lambda v: int(v * 0.6)))
        im = Image.alpha_composite(g, im)
    return to_np(im)

@functools.lru_cache(None)
def big(text, size, g0, g1, sf=(10, 26, 66), ef=(16, 40, 110), extrude=8):
    return to_np(text_layer(text, size, 800, gradient=(g0, g1), stroke=5, stroke_fill=sf, extrude=extrude,
                            extrude_fill=ef, shadow=16, pad=90))

@functools.lru_cache(None)
def label_sprite(text, ic=None, color=CYAN):
    t = text_layer(text, 46, 800, fill=(255, 255, 255), pad=0)
    w = t.size[0] + (120 if ic else 56); h = 84
    base = Image.fromarray(glass(w, h, 42, color).astype(np.uint8))
    pad = 24
    base.alpha_composite(t, (pad + 28, pad + (h - t.size[1]) // 2 - 3))
    if ic:
        i = Image.open(A(ic)).convert("RGBA").resize((92, 92), Image.LANCZOS)
        base.alpha_composite(i, (pad + w - 92 + 6, pad + (h - 92) // 2))
    return to_np(base)

def blit_layer(F, im, alpha=1.0):
    F.blit(np.asarray(im).astype(np.float32), W / 2, H / 2, alpha=alpha)

def popin(F, spr, x, y, v, t0, t1, s0=0.4, back=2.2, rot=0.0, alpha=1.0, fin=0.28, fout=0.22, bob=3):
    w = R.window(v, t0, t1, fin, fout)
    if not w: return 0
    pin, a = w
    F.blit(spr, x, y + math.sin(v * 3 + x) * bob, alpha=a * alpha, scale=s0 + (1 - s0) * eob(pin, back), rot=rot)
    return a

def callout(F, v, t0, t1, text, ic, tx, ty, lx, ly, color=CYAN):
    """pulsing target dot + leader line + glass label"""
    w = R.window(v, t0, t1, 0.3, 0.22)
    if not w: return
    pin, a = w
    pr = (v - t0) * 2.2 % 1.0
    F.blit(soft_dot(16, color, 1), tx, ty, alpha=a)
    F.blit(ring(int(22 + 40 * pr), 4, color), tx, ty, alpha=a * (1 - pr))
    q = eoc(clamp((v - t0) / 0.25))
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    ex, ey = tx + (lx - tx) * q, ty + (ly - ty) * q
    d.line((tx, ty, ex, ey), fill=color + (230,), width=4)
    F.blit(np.asarray(im).astype(np.float32), W / 2, H / 2, alpha=a)
    if q > 0.6:
        popin(F, label_sprite(text, ic, color), lx, ly, v, t0 + 0.15, t1, s0=0.5)

# ------------------------------------------------------------------ HUD (always alive)
@functools.lru_cache(None)
def hud_panel():
    w, h = 150, 860
    im = Image.fromarray(glass(w, h, 30).astype(np.uint8)); d = ImageDraw.Draw(im); pad = 24
    x0 = pad + 40
    for km in range(0, 13):
        y = pad + h - 40 - (h - 160) * km / 12
        d.line((x0, y, x0 + (34 if km % 2 == 0 else 20), y), fill=(200, 240, 255, 230), width=3)
    return to_np(im), (w, h, pad)

def hud(F, v, sp):
    a = clamp((v - 0.35) / 0.4) * (1 - clamp((v - 52.3) / 0.5))
    if a <= 0: return
    cx, cy, ww = cam(v)
    h = h_of_y(cy); T = temp_of_h(h)
    panel, (w, ph, pad) = hud_panel()
    px, py = 118, 760
    F.blit(panel, px, py, alpha=a)
    top = py - ph / 2 - pad + pad + 40
    for km in range(0, 13, 2):
        y = py - ph / 2 + ph - 40 - (ph - 160) * km / 12
        F.blit(txt(str(km), 30, (200, 240, 255), 700, 0, shadow=0), px + 40, y, alpha=a * 0.9)
    my = py - ph / 2 + ph - 40 - (ph - 160) * min(h, 12) / 12
    col = tuple(int(c) for c in np.interp(T, [-50, 0, 30], [0, 0, 0]) * 0 + np.array(
        [np.interp(T, [-50, 0, 30], [120, 200, 255]), np.interp(T, [-50, 0, 30], [200, 240, 150]), np.interp(T, [-50, 0, 30], [255, 255, 60])]))
    F.blit(soft_dot(22, col, 1), px - 30, my, alpha=a)
    F.blit(ring(26, 4, col), px - 30, my, alpha=a * (0.5 + 0.5 * math.sin(v * 8)))
    # digital readouts
    F.blit(glass(250, 150, 26), 330, 395, alpha=a)
    F.blit(ltr(f"{h:.1f}", 50), 365, 362, alpha=a)
    F.blit(txt("ק״מ", 34, (200, 240, 255), 700, shadow=0), 260, 366, alpha=a)
    tcol = (int(np.interp(T, [-50, 0, 30], [120, 200, 255])), int(np.interp(T, [-50, 0, 30], [200, 245, 170])), int(np.interp(T, [-50, 0, 30], [255, 255, 90])))
    Ti = int(round(T)); F.blit(ltr(("+" if Ti > 0 else "") + f"{Ti}°C", 52, tcol), 330, 430, alpha=a)
    # fall speed readout during the dive
    if 44.0 < v < 52.0:
        spd = int(np.interp(v, [44.2, 46.6, 51.5], [0, 72, 85]))
        b = clamp((v - 44.0) / 0.3) * (1 - clamp((v - 51.7) / 0.3))
        F.blit(glass(250, 110, 26, (255, 170, 80)), 330, 560, alpha=a * b)
        F.blit(ltr(f"{spd}", 52, (255, 210, 140)), 380, 558, alpha=a * b)
        F.blit(txt("קמ״ש", 36, (255, 210, 140), 700, shadow=0), 270, 562, alpha=a * b)

# ------------------------------------------------------------------ intro title
@functools.lru_cache(None)
def title_words():
    w1 = [("איך", big("איך", 104, (255, 255, 255), (210, 230, 255))),
          ("נוצר", big("נוצר", 104, (255, 255, 255), (210, 230, 255))),
          ("גשם", big("גשם", 140, (150, 230, 255), (30, 130, 255), (6, 30, 90), (8, 50, 140), 10))]
    w2 = [("ולמה", big("ולמה", 66, (255, 255, 255), (220, 235, 255), extrude=5)),
          ("ברד", big("ברד", 80, (220, 250, 255), (120, 200, 255), (10, 40, 90), (20, 70, 140), 6)),
          ("יכול", big("יכול", 66, (255, 255, 255), (220, 235, 255), extrude=5)),
          ("לרדת", big("לרדת", 66, (255, 255, 255), (220, 235, 255), extrude=5)),
          ("ביום", big("ביום", 66, (255, 255, 255), (220, 235, 255), extrude=5)),
          ("שרב?", big("שרב?", 84, (255, 230, 90), (255, 70, 30), (80, 20, 0), (130, 40, 0), 6))]
    return w1, w2

def layout_rtl(words, cx, gap=-150, maxw=950):
    pad = 180
    tot = sum(s.shape[1] for _, s in words) + gap * (len(words) - 1)
    k = min(1.0, maxw / (tot - 0))
    x = cx + tot * k / 2; out = []
    for _, s in words:
        out.append(x - s.shape[1] * k / 2); x -= (s.shape[1] + gap) * k
    return out, k

def intro_overlay(F, t):
    w1, w2 = title_words()
    if t < FLASH1 + 0.2:
        ex = clamp((t - FLASH1) / 0.18)
        oa = 1 - ex; osc = 1 + 0.6 * ex
        # dark gradient for legibility
        if True:
            g = (np.linspace(1, 0, 600) ** 1.5 * 0.55)[:, None, None]
            F.img[0:600] *= (1 - g * oa * clamp(t / 0.3))
        xs1, k1 = layout_rtl(w1, 540, gap=-162, maxw=1100); xs2, k2 = layout_rtl(w2, 540, gap=-162, maxw=1130)
        for k, ((wd, s), x) in enumerate(zip(w1, xs1)):
            p = clamp((t - 0.1 - k * 0.22) / 0.35)
            if p <= 0: continue
            y = 372 if wd != "גשם" else 362
            F.blit(s, 540 + (x - 540) * osc, y - (1 - eoc(p)) * 50, alpha=eoc(p) * oa, scale=(0.3 + 0.7 * eob(p, 2.4)) * osc * k1)
            if wd == "גשם" and p >= 1:
                for j in range(4):
                    q = ((t * 0.9 + j * 0.25) % 1.0)
                    F.blit(droplet(9), x - 60 + j * 38, 420 + q * 120, alpha=oa * (1 - q) * 0.9)
        for k, ((wd, s), x) in enumerate(zip(w2, xs2)):
            p = clamp((t - 0.9 - k * 0.13) / 0.3)
            if p <= 0: continue
            sc = (0.3 + 0.7 * eob(p, 2.6)) * osc * k2
            rot = 0
            if wd == "שרב?": sc *= 1 + 0.04 * math.sin(t * 9)
            if wd == "ברד": rot = math.sin(t * 6) * 2
            F.blit(s, 540 + (x - 540) * osc, 500, alpha=eoc(p) * oa, scale=sc, rot=rot)
        # shine sweep
        sw = (t - 1.6) / 0.9
        if 0 < sw < 1:
            band = np.zeros((H, W), np.float32)
            xx = np.arange(W)[None, :] + np.arange(H)[:, None] * 0.35
            cxp = -300 + sw * 1700
            band = np.exp(-((xx - cxp) / 60) ** 2)[..., None] * 0
    # "פינת העשרה" badge slams in with the lightning flash
    if t >= FLASH1:
        p = clamp((t - FLASH1) / 0.25)
        out = clamp((t - 7.95) / 0.3)
        F.blit(badge(), 540, 420 - 80 * eio(out), alpha=eoc(p) * (1 - out), scale=(2.0 - 1.0 * eob(p, 1.8)) * (1 + 0.5 * out))
        for k in range(10):
            q = clamp((t - FLASH1 - k * 0.02) / 0.7)
            if 0 < q < 1:
                ang = k * 2 * math.pi / 10
                F.blit(icon("sparkles_3d.png", 64), 540 + math.cos(ang) * (200 + 260 * q), 420 + math.sin(ang) * (60 + 120 * q),
                       alpha=(1 - q) * (1 - out), scale=0.5 + 0.6 * q)

@functools.lru_cache(None)
def badge():
    tl = text_layer("פינת העשרה", 80, 800, fill=(255, 255, 255), pad=0)
    w, h = tl.size[0] + 220, 134
    base = rounded(w, h, 67, (255, 255, 255), grad=((40, 210, 255), (120, 60, 255)), border=((255, 255, 255, 240), 5), shadow=18)
    pad = (base.size[0] - w) // 2
    base.alpha_composite(tl, (pad + 52, pad + (h - tl.size[1]) // 2 - 4))
    ic = Image.open(A("test_tube_3d.png")).convert("RGBA").resize((160, 160), Image.LANCZOS)
    can = Image.new("RGBA", (base.size[0] + 50, base.size[1] + 50), (0, 0, 0, 0))
    can.alpha_composite(base, (0, 25)); can.alpha_composite(ic, (pad + w - 160 + 40, 25 + pad + (h - 160) // 2))
    return to_np(can)

# ------------------------------------------------------------------ journey graphics
rng = random.Random(9)
VAPOR = [(rng.uniform(80, 1000), rng.uniform(0, 1), rng.uniform(0.6, 1.1), rng.uniform(8, 18)) for _ in range(80)]
MOLS = [(rng.uniform(0, 6.283), rng.uniform(260, 400), rng.uniform(0, 1.0)) for _ in range(50)]
BOKEH = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.6, 1.6), rng.uniform(0, 6.28), rng.randint(10, 34)) for _ in range(34)]
SWARM = [(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.5, 1.5)) for _ in range(300)]
CRY = [(rng.uniform(120, 960), rng.uniform(380, 1300), rng.uniform(0, 6.28), rng.randint(60, 150), rng.uniform(0, 1.2)) for _ in range(14)]
RAIN = [(rng.uniform(-100, W), rng.uniform(0, H), rng.uniform(1700, 2400), rng.uniform(50, 110)) for _ in range(170)]
STREAMS = [(x, rng.uniform(0, 1)) for x in np.linspace(180, 900, 7)]
SHARDS = [(rng.uniform(-1, 1), rng.uniform(-1.4, -0.4), rng.randint(14, 30)) for _ in range(16)]

def rain(F, v, amt, slant=0.15):
    if amt <= 0.01: return
    lay = np.zeros((H, W), np.uint8)
    for (x0, y0, sp, ln) in RAIN[: int(len(RAIN) * amt)]:
        y = (y0 + sp * v) % (H + 200) - 100; x = x0 + slant * y
        cv2.line(lay, (int(x), int(y)), (int(x + slant * ln), int(y + ln)), 210, 2, cv2.LINE_AA)
    a = (lay.astype(np.float32) / 255 * 0.55 * amt)[..., None]
    F.img += (np.array([230, 240, 255], np.float32) - F.img) * a

def journey(F, v, c):
    # --- evaporation: vapor rising off the sea
    w = R.window(v, -0.3, 5.2, 0.4, 0.5)
    if w:
        _, a = w
        sx, sy = world_to_screen(1000, 7330, c)
        for (x0, ph, spd, r) in VAPOR:
            q = (v * 0.28 * spd + ph) % 1.0
            y = sy - q * 900; x = x0 + math.sin(q * 8 + ph * 6) * 30
            F.blit(soft_dot(r * (1 + 1.2 * q)), x, y, alpha=a * math.sin(q * math.pi) * 0.7)
    callout(F, v, 1.1, 2.75, "אידוי", "droplet_3d.png", 640, world_to_screen(1000, 7330, c)[1] - 30, 700, 1090)
    # --- warm air parcel rising & cooling
    w = R.window(v, 2.75, 5.3, 0.3, 0.35)
    if w:
        _, a = w
        hh = h_of_y(c[1]); k = clamp(hh / 1.5)
        col = (int(255 - 135 * k), int(150 + 60 * k), int(70 + 185 * k))
        F.blit(soft_dot(150, col), 640, 980, alpha=a * 0.55)
        F.blit(ring(150, 5, col, dash=8), 640, 980, alpha=a, rot=v * 40)
        for j in range(4):
            q = (v * 1.5 + j / 4) % 1
            F.blit(txt("⬆" if False else "︿", 60, col, 800, shadow=0), 640, 1150 - q * 120, alpha=a * (1 - q))
        popin(F, label_sprite("אוויר חם עולה", "up_arrow_3d.png", col), 640, 780, v, 3.0, 4.35)
        popin(F, label_sprite("ומתקרר", "snowflake_3d.png"), 640, 780, v, 4.4, 5.25)
    # --- condensation lens
    w = R.window(v, 5.0, 10.3, 0.35, 0.35)
    if w:
        pin, a = w
        lx, ly, r = 640, 860, 250
        sc = 0.4 + 0.6 * eob(pin, 1.5)
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        d.ellipse((lx - r, ly - r, lx + r, ly + r), fill=(6, 20, 50, 200))
        for gx in range(lx - r, lx + r, 40): d.line((gx, ly - r, gx, ly + r), fill=(90, 230, 255, 28), width=1)
        for gy in range(ly - r, ly + r, 40): d.line((lx - r, gy, lx + r, gy), fill=(90, 230, 255, 28), width=1)
        mask = Image.new("L", (W, H), 0); ImageDraw.Draw(mask).ellipse((lx - r, ly - r, lx + r, ly + r), fill=255)
        im.putalpha(Image.fromarray(np.minimum(np.asarray(im.split()[3]), np.asarray(mask))))
        nuc = clamp((v - 5.85) / 0.4)
        if nuc > 0:
            pts = [(lx + 54 * nuc * math.cos(k * 0.9) * (1 + 0.28 * math.sin(k * 2.3)), ly + 48 * nuc * math.sin(k * 0.9) * (1 + 0.2 * math.cos(k * 1.7))) for k in range(7)]
            d.polygon(pts, fill=(170, 120, 75, 255))
            d.rectangle((lx + 30 * nuc, ly - 20 * nuc, lx + 66 * nuc, ly + 16 * nuc), fill=(245, 245, 255, 255))
        F.blit(np.asarray(im).astype(np.float32), W / 2, H / 2, alpha=a)
        F.blit(ring(r, 7, CYAN), lx, ly, alpha=a, scale=sc)
        F.blit(ring(r + 26, 3, CYAN, dash=6), lx, ly, alpha=a * 0.8, scale=sc, rot=v * 30)
        F.blit(ltr("×100,000", 32, CYAN), lx + r - 40, ly + r + 10, alpha=a)
        if v > 6.0:
            for (ang, rad, dl) in MOLS:
                q = clamp((v - 6.1 - dl) / 2.4)
                if q <= 0 or q >= 1: continue
                rr = rad * (1 - eio(q)) * 0.6 + 40
                x, y = lx + math.cos(ang + v * 0.4) * rr, ly + math.sin(ang + v * 0.4) * rr
                F.blit(soft_dot(14, (110, 200, 255), 1), x, y, alpha=a * min(1, q * 5) * (1 - clamp((v - 8.6) / 0.5)))
        g = clamp((v - 8.6) / 1.2)
        if g > 0: F.blit(droplet(130), lx, ly, alpha=a * eoc(g), scale=0.15 + 0.85 * eob(g, 1.4))
        popin(F, label_sprite("גרעין התעבות", None), lx, ly - r - 70, v, 5.6, 10.3)
        popin(F, label_sprite("אבק", "rock_3d.png", (255, 190, 120)), lx - 250, ly + r + 70, v, 7.4, 10.3)
        popin(F, label_sprite("מלח", "sparkles_3d.png", (230, 240, 255)), lx + 170, ly + r + 70, v, 7.95, 10.3)
    # --- billions -> cloud
    w = R.window(v, 10.25, 12.6, 0.2, 0.4)
    if w:
        pin, a = w
        q = eoc(clamp((v - 10.3) / 1.4))
        for (ux, uy, s) in SWARM:
            x = 640 + ux * 560 * q * s; y = 860 + uy * 900 * q * s
            F.blit(soft_dot(6, (230, 245, 255), 1), x, y, alpha=a * 0.9)
        popin(F, big("מיליארדי טיפונת", 92, (255, 255, 255), (200, 230, 255)), 540, 650, v, 10.35, 12.6, s0=0.3)
        popin(F, big("= ענן", 110, (255, 236, 120), (255, 160, 40), (70, 30, 0), (120, 55, 0)), 540, 1110, v, 11.9, 12.6, s0=0.2, back=3)
    # --- floating droplets with size bracket
    w = R.window(v, 12.4, 18.7, 0.4, 0.3)
    if w:
        _, a = w
        for (x, y, depth, ph, r) in BOKEH:
            yy = (y - (cam(v)[1] - 4900) * 0.18 * depth) % H
            F.blit(droplet(int(r * depth * 0.8) + 4), x + math.sin(v * 0.8 + ph) * 26, yy + math.cos(v * 0.6 + ph) * 18,
                   alpha=a * (0.55 + 0.3 * depth))
    callout(F, v, 12.6, 15.4, "קוטר ~0.02 מ״מ", "microscope_3d.png", 560, 900, 620, 640)
    w = R.window(v, 15.4, 17.5, 0.3, 0.25)
    if w:
        pin, a = w
        q = clamp((v - 15.6) / 1.5)
        n = 1000000 if q >= 1 else int(round(10 ** (6 * eio(q))))
        F.blit(ltr(f"×{n:,}", 120, (255, 220, 90)), 560, 600, alpha=a, scale=0.6 + 0.4 * eob(pin))
        F.blit(droplet(120), 560, 930, alpha=a, scale=0.08 + 0.92 * eio(q))
        F.blit(txt("כדי ליפול", 48, (255, 236, 160)), 560, 470, alpha=a)
    w = R.window(v, 17.45, 18.75, 0.15, 0.2)
    if w:
        pin, a = w
        q = clamp((v - 17.5) / 0.45)
        if q < 1:
            off = 340 * (1 - eio(q))
            F.blit(droplet(60), 560 - off, 900, alpha=a); F.blit(droplet(60), 560 + off, 900, alpha=a)
        else:
            F.blit(droplet(86), 560, 900, alpha=a, scale=0.8 + 0.2 * eob(clamp((v - 17.95) / 0.25), 3))
            b = clamp((v - 17.95) / 0.45)
            if b < 1: F.blit(icon("collision_3d.png", 320), 560, 880, alpha=1 - b, scale=0.5 + 0.7 * b)
        popin(F, big("מתנגשות ומתאחדות", 78, (255, 255, 255), (200, 230, 255)), 560, 600, v, 17.6, 18.75)
    # --- freezing level line (world-anchored)
    fy = world_to_screen(0, FREEZE_Y, c)[1]
    if -60 < fy < H + 60 and 18.0 < v < 27.0:
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        off = (v * 120) % 48
        for x in range(-48, W, 48): d.line((x + off, fy, x + off + 28, fy), fill=(150, 230, 255, 255), width=6)
        F.blit(np.asarray(im.filter(ImageFilter.GaussianBlur(1))).astype(np.float32), W / 2, H / 2, alpha=1.0)
        F.blit(label_sprite("קו הקיפאון 0°C", "snowflake_3d.png"), 700, fy - 70, alpha=1.0)
    # --- ice crystals at the top of the cloud
    w = R.window(v, 20.6, 24.4, 0.3, 0.35)
    if w:
        _, a = w
        for k, (x, y, ph, s, dl) in enumerate(CRY):
            g = clamp((v - 20.8 - dl) / 0.5)
            if g <= 0: continue
            grow = 1 + 0.5 * clamp((v - 22.5) / 1.0)
            F.blit(crystal(s), x + math.sin(v + ph) * 14, y + math.cos(v * 0.8 + ph) * 10, alpha=a * eoc(g),
                   scale=(0.2 + 0.8 * eob(g, 2)) * grow, rot=v * 30 * (1 if k % 2 else -1))
        popin(F, label_sprite("גבישי קרח", "ice_3d.png"), 600, 1240, v, 21.6, 24.4)
        popin(F, label_sprite("גדלים מהר!", "up_arrow_3d.png", (255, 220, 120)), 600, 1350, v, 22.55, 24.4)
    # --- falling ice -> melts at 0°C -> rain
    w = R.window(v, 23.3, 27.9, 0.2, 0.3)
    if w:
        _, a = w
        fy = world_to_screen(0, FREEZE_Y, c)[1]
        py = 900
        melt = clamp((py - fy + 40) / 120)
        if melt < 1: F.blit(crystal(150), 600, py, alpha=a * (1 - melt), rot=v * 120)
        if melt > 0: F.blit(droplet(52), 600, py, alpha=a * melt)
        for k in range(10):
            F.blit(soft_dot(16 - k, (220, 240, 255)), 600, py - 30 - k * 28, alpha=a * (0.45 - k * 0.04))
        popin(F, label_sprite("נופלות", "down_arrow_3d.png"), 640, 1180, v, 24.4, 25.4)
        popin(F, label_sprite("נמס!", "droplet_3d.png", (130, 210, 255)), 640, 1180, v, 25.45, 27.0, back=3)
    rain(F, v, clamp((v - 26.0) / 0.6) * (1 - clamp((v - 28.6) / 0.5)) * 0.9)
    popin(F, big("גשם!", 150, (255, 255, 255), (150, 210, 255)), 540, 640, v, 27.25, 28.0, s0=0.2, back=3)
    # --- hail section
    popin(F, big("ועכשיו: ברד", 112, (225, 245, 255), (120, 190, 255)), 540, 640, v, 27.85, 29.6, s0=0.2)
    popin(F, icon_shadowed("ice_3d.png", 220)[0], 540, 900, v, 27.95, 29.6, s0=0.2, rot=math.sin(v * 5) * 8)
    w = R.window(v, 28.9, 30.1, 0.25, 0.2)
    if w:
        _, a = w
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        for k in range(6):
            pts = [(x, 1330 - k * 30 + math.sin(x / 45 + v * 7 + k) * 12) for x in range(40, 1040, 12)]
            d.line(pts, fill=(255, 150, 60, 210 - k * 30), width=7)
        F.blit(np.asarray(im.filter(ImageFilter.GaussianBlur(2))).astype(np.float32), W / 2, H / 2, alpha=a)
        popin(F, label_sprite("+35°C ביום חם", "fire_3d.png", (255, 170, 80)), 600, 1180, v, 28.9, 30.1)
    popin(F, big("10 ק״מ למעלה", 96, (255, 255, 255), (200, 230, 255)), 560, 700, v, 30.1, 32.3, s0=0.3)
    popin(F, ltr("-50°C", 190, (170, 225, 255)), 560, 820, v, 32.4, 33.65, s0=0.2, back=3)
    w = R.window(v, 32.4, 33.65, 0.3, 0.2)
    if w:
        for k, (x, y, ph, s, dl) in enumerate(CRY[:8]):
            F.blit(crystal(s // 2 + 30), x, (y + v * 60) % 1300 + 300, alpha=w[1] * 0.8, rot=v * 50)
    # --- updraft streamlines
    w = R.window(v, 33.6, 38.6, 0.4, 0.3)
    if w:
        _, a = w
        heat = clamp((v - 36.45) / 0.6)
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        for (x0, ph) in STREAMS:
            col = (int(90 + 165 * heat), int(230 - 80 * heat), int(255 - 175 * heat))
            for j in range(5):
                q = (v * (0.45 + 0.35 * heat) + ph + j / 5) % 1.0
                y = 1380 - q * 1000
                xs = x0 + math.sin(q * 6 + ph * 5) * 40
                L = 90 + 60 * heat
                pts = [(xs + math.sin((q - 0.02 * i) * 6 + ph * 5) * 40, y + i * L / 6) for i in range(7)]
                d.line(pts, fill=col + (int(255 * math.sin(q * math.pi)),), width=int(8 + 6 * heat))
                d.polygon([(xs - 18 - 6 * heat, y + 16), (xs, y - 16 - 6 * heat), (xs + 18 + 6 * heat, y + 16)], fill=col + (int(255 * math.sin(q * math.pi)),))
        g = im.filter(ImageFilter.GaussianBlur(6))
        F.blit(np.asarray(Image.alpha_composite(g, im)).astype(np.float32), W / 2, H / 2, alpha=a)
        popin(F, label_sprite("ענן סערה", "cloud_with_lightning_3d.png"), 600, 580, v, 33.65, 34.7)
        popin(F, label_sprite("זרמי אוויר עולים חזקים", "dashing_away_3d.png"), 600, 580, v, 34.75, 36.4)
        popin(F, label_sprite("החום בקרקע מחזק אותם", "fire_3d.png", (255, 170, 80)), 600, 580, v, 36.45, 38.6)
    # --- hailstone loops (camera follows it)
    w = R.window(v, 38.5, 52.0, 0.3, 0.1)
    if w:
        _, a = w
        layers = 1 + int(clamp((v - 38.6) / 5.2) * 5)
        r = int(44 + 10 * layers)
        sx, sy = 560, 930
        if v > 51.62: return
        # direction trail
        c2 = cam(v + 0.05); dy = (c2[1] - c[1])
        for k in range(10):
            F.blit(soft_dot(r - k * 2 if r - k * 2 > 3 else 3, (220, 240, 255)), sx, sy - math.copysign(1, dy or 1) * (k + 1) * 18,
                   alpha=a * (0.4 - k * 0.035) * min(1, abs(dy) / 20))
        F.blit(hailstone(r, layers), sx, sy, alpha=a, rot=v * 90)
        popin(F, label_sprite("גרגר קרח", "ice_3d.png"), 560, 760, v, 38.6, 40.25)
        popin(F, label_sprite("שוב ושוב ⟳" if False else "שוב ושוב", "up_arrow_3d.png"), 560, 760, v, 40.3, 41.7)
        popin(F, label_sprite(f"עוד שכבת קרח", "ice_3d.png"), 560, 760, v, 41.75, 43.4)
        # cross-section inset
        wi = R.window(v, 41.75, 44.4, 0.3, 0.25)
        if wi:
            F.blit(glass(300, 330, 30), 840, 660, alpha=wi[1])
            F.blit(hailstone(110, layers), 840, 640, alpha=wi[1], scale=0.5 + 0.5 * eob(wi[0]))
            F.blit(txt(f"{layers} שכבות", 40, (255, 255, 255)), 840, 790, alpha=wi[1])
        popin(F, icon_shadowed("onion_3d.png", 150)[0], 840, 470, v, 43.4, 44.4, back=3)
        popin(F, label_sprite("כבדה מדי!", "down_arrow_3d.png", (255, 170, 80)), 560, 760, v, 44.5, 46.0)
        popin(F, big("עשרות קמ״ש!", 96, (255, 236, 120), (255, 150, 40), (70, 30, 0), (120, 55, 0)), 560, 640, v, 46.55, 48.0, s0=0.3)
        popin(F, label_sprite("אין זמן להפשיר", "stopwatch_3d.png"), 560, 700, v, 48.5, 51.6)
    # --- impact
    b = clamp((v - 51.62) / 0.6)
    if 0 < b < 1:
        F.blit(icon("collision_3d.png", 380), 560, 1000, alpha=1 - b, scale=0.4 + 0.9 * b)
        for (ux, uy, s) in SHARDS:
            F.blit(hailstone(s // 2, 2), 560 + ux * 420 * b, 1000 + uy * 380 * b + 500 * b * b, alpha=1 - b)
    # --- outro
    popin(F, big("עכשיו אתם יודעים!", 92, (255, 236, 120), (255, 160, 40), (70, 30, 0), (120, 55, 0)), 540, 560, v, 52.5, 55.3, s0=0.3)
    popin(F, icon("light_bulb_3d.png", 130), 930, 430, v, 52.6, 55.3, rot=12)
    if v > 55.2:
        rr = R
        follow(F, v, 55.25, 55.85)

@functools.lru_cache(None)
def follow_card_609(followed):
    old = R.text_layer
    def patched(text, *a, **k):
        if text == "60.8K": text = "60.9K"
        return old(text, *a, **k)
    R.text_layer = patched
    try:
        return R.follow_card.__wrapped__(followed)
    finally:
        R.text_layer = old

def follow(F, v, t0, tap):
    p = clamp((v - t0) / 0.4)
    spr, (bx, by) = follow_card_609(v >= tap + 0.1)
    press = 1 - 0.06 * max(0, 1 - abs(v - tap - 0.05) / 0.12)
    cy = 1150 + (1 - eob(p, 1.7)) * 300
    F.blit(spr, 540, cy, alpha=eoc(p), scale=press)
    if tap - 0.6 < v < tap + 0.9:
        hp = clamp((v - (tap - 0.6)) / 0.55)
        hy = cy + by + 70 + (1 - eoc(hp)) * 200 + (8 if abs(v - tap) < 0.1 else 0)
        ha = clamp((v - (tap - 0.6)) / 0.15) * (1 - clamp((v - tap - 0.6) / 0.3))
        F.blit(icon_shadowed("backhand_index_pointing_up_3d_default.png", 120, 8, 8, 0.4)[0], 540 + bx + 30, hy, alpha=ha)
    if tap < v < tap + 1.2:
        q = (v - tap) / 1.2
        for k in range(6):
            F.blit(icon("red_heart_3d.png", 56), 540 + bx + (k - 2.5) * 40, cy - 110 - q * (160 + k * 14), alpha=1 - q,
                   scale=0.6 + 0.6 * q, rot=(k - 2.5) * 10)

# ------------------------------------------------------------------ subtitles
FIXW = {"כשלוהד": "כשלוהט", "מתאבים": "מתעבים", "לכבישי": "לגבישי", "זהירים": "זעירים", "לברת": "לברד"}
def all_words():
    out = []
    for s in json.load(open(os.path.join(D, "tr_video.json"))):
        for w in s["words"]: out.append((w["w"], w["s"], w["e"]))
    for s in json.load(open(os.path.join(D, "tr_final.json"))):
        for w in s["words"]: out.append((w["w"], w["s"] + VO0, w["e"] + VO0))
    res = []
    for t, s, e in out:
        t = t.strip(); brk = t.endswith((",", ".", "?", "!")); core = t.strip(",.?!")
        res.append({"t": FIXW.get(core, core) + ("?" if t.endswith("?") else ""), "s": s, "e": min(e, s + 0.9), "brk": brk})
    return res

def make_chunks():
    ws = all_words(); res = []; cur = []
    for i, w in enumerate(ws):
        cur.append(w)
        L = sum(len(x["t"]) + 1 for x in cur)
        cross = i + 1 < len(ws) and w["s"] < VO0 <= ws[i + 1]["s"]
        if cross or w["brk"] or len(cur) >= 3 or L >= 15:
            res.append(cur); cur = []
    if cur: res.append(cur)
    out = []
    for i, c in enumerate(res):
        t0 = c[0]["s"]; t1 = res[i + 1][0]["s"] if i + 1 < len(res) else DUR
        out.append((t0, min(t1, c[-1]["e"] + 0.4), c))
    return out
CHUNKS = make_chunks()

def subtitles(F, t):
    for (t0, t1, ws) in CHUNKS:
        if t0 <= t < t1:
            p = clamp((t - t0) / 0.1)
            sprs = [R.sub_word(w["t"], w["s"] <= t < (ws[j + 1]["s"] if j + 1 < len(ws) else w["e"] + 0.2)) for j, w in enumerate(ws)]
            gap = -30
            total = sum(s.shape[1] for s in sprs) + gap * (len(sprs) - 1)
            sc = min(1.0, 860 / total) * (0.88 + 0.12 * eob(p, 2.8))
            x = 540 + total * sc / 2
            for s in sprs:
                ww = s.shape[1] * sc
                F.blit(s, x - ww / 2, 1480, alpha=eoc(p), scale=sc)
                x -= ww + gap * sc
            return

# ------------------------------------------------------------------ frame
def shake_at(t):
    sx = sy = 0.0
    for t0, amp in [(FLASH1, 22), (FLASH2, 12), (VO0 + 51.62, 30)]:
        d = t - t0
        if 0 <= d < 0.4:
            k = (1 - d / 0.4) ** 2 * amp
            sx += math.sin(d * 90) * k; sy += math.cos(d * 70) * k
    return sx, sy

def render(i, raw):
    t = i / FPS
    v = t - VO0
    shx, shy = shake_at(t)
    hm = np.zeros((H, W), bool)
    if t < 7.95:
        img = raw.astype(np.float32)
        if abs(shx) + abs(shy) > 0.5:
            M = np.array([[1.04, 0, -21.6 + shx], [0, 1.04, -38.4 + shy]], np.float32)
            img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        k = min(i, len(HEAD) - 1)
        hmm = cv2.resize(np.asarray(HEAD[k]), (W, H))
        hm = cv2.dilate((hmm > 90).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
        sp = 0
    else:
        img, sp = world_frame(v, (shx, shy))
        img *= vignette()
        if t < 8.45:   # camera push from the intro into the rain (no cut)
            a = eio((t - 7.95) / 0.5)
            z = 1 + 1.6 * eio((t - 7.95) / 0.5)
            M = np.array([[z, 0, 540 - z * 540], [0, z, 900 - z * 900]], np.float32)
            src = raw.astype(np.float32) if raw is not None else img
            pushed = cv2.warpAffine(src, M, (W, H), borderMode=cv2.BORDER_REFLECT)
            ker = np.ones((31, 1), np.float32) / 31
            pushed = cv2.filter2D(pushed, -1, ker)
            img = pushed * (1 - a) + img * a
    F = R.Frame(img, np.zeros((H, W), np.float32), hm)
    if t < 8.45:
        intro_overlay(F, t)
    if t >= 7.95:
        c = cam(v)
        journey(F, v, c)
        hud(F, v, sp)
    subtitles(F, t)
    np.clip(F.img, 0, 255, out=F.img)
    return F.img.astype(np.uint8), F.head_overlap

def run(o0, o1, path):
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{min(o0, 248) / FPS:.4f}", "-i", os.path.join(D, "seed2_1080.mp4"),
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE) if o0 < 254 else None
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    ov = []; last = None
    for i in range(o0, o1):
        raw = last
        if dec is not None and i < 249:
            b = dec.stdout.read(W * H * 3)
            if len(b) == W * H * 3: raw = np.frombuffer(b, np.uint8).reshape(H, W, 3); last = raw
        out, o = render(i, raw)
        if o: ov.append((i, o))
        enc.stdin.write(out.tobytes())
    enc.stdin.close(); enc.wait()
    print(f"{o0}-{o1} done, head-overlap frames {len(ov)} {ov[:4]}", flush=True)

def raw_at(tt):
    tt = min(tt, 8.25)
    b = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{tt:.3f}", "-i", os.path.join(D, "seed2_1080.mp4"), "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(b, np.uint8).reshape(H, W, 3)

if __name__ == "__main__":
    if sys.argv[1] == "still":
        outs = []
        for ts in sys.argv[3:]:
            tt = float(ts); i = int(round(tt * FPS))
            o, ov = render(i, raw_at(tt) if tt < 8.5 else None); print(ts, "overlap", ov)
            outs.append(Image.fromarray(o).resize((360, 640), Image.LANCZOS))
        c = Image.new("RGB", (360 * len(outs), 640))
        for k, im in enumerate(outs): c.paste(im, (360 * k, 0))
        c.save(sys.argv[2], quality=90)
    else:
        run(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
