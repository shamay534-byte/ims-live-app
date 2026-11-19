# Cirrus / fire-sunset reel: 7 Seedance scenes + 3D graphics, subtitles, follow card
import sys, os, json, math, random, subprocess, functools, importlib.util
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("R", os.path.join(HERE, "render.py"))
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
clamp, eoc, eio, eob = R.clamp, R.ease_out_cubic, R.ease_in_out, R.ease_out_back
text_layer, rounded, icon, icon_shadowed, to_np, A = R.text_layer, R.rounded, R.icon, R.icon_shadowed, R.to_np, R.A
D = os.path.join(HERE, "r8")
W, H, FPS = 1080, 1920, 30
END_HOLD = 3.4

# ------------------------------------------------------------------ timeline
DUR = {k: float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                os.path.join(D, "seg", f"s{k}.mp4")], capture_output=True, text=True).stdout) for k in range(1, 8)}
NF = {k: int(round(DUR[k] * FPS)) for k in DUR}
START = {}; acc = 0
for k in range(1, 8): START[k] = acc; acc += NF[k]
NFR = acc + int(END_HOLD * FPS)
TOTAL = NFR / FPS

def seg_of(i):
    for k in range(7, 0, -1):
        if i >= START[k]: return k, (i - START[k]) / FPS
    return 1, 0.0

_masks = {}
def masks(k):
    if k not in _masks:
        _masks[k] = (np.load(os.path.join(D, f"m{k}_person.npy"), mmap_mode="r"), np.load(os.path.join(D, f"m{k}_head.npy"), mmap_mode="r"))
    return _masks[k]

# ------------------------------------------------------------------ helpers
@functools.lru_cache(None)
def big(text, size, g0, g1, sf=(10, 26, 66), ef=(16, 40, 110), extrude=8):
    return to_np(text_layer(text, size, 800, gradient=(g0, g1), stroke=5, stroke_fill=sf, extrude=extrude, extrude_fill=ef, shadow=16, pad=80))
FIRE = ((255, 236, 110), (255, 90, 30), (90, 20, 0), (140, 40, 0))
ICE = ((235, 250, 255), (120, 200, 255), (10, 40, 90), (20, 70, 140))
WHITE = ((255, 255, 255), (210, 230, 255), (10, 26, 66), (16, 40, 110))

def bigt(text, size, pal): return big(text, size, pal[0], pal[1], pal[2], pal[3])

def popin(F, spr, x, y, u, t0, t1, s0=0.4, back=2.2, rot=0.0, alpha=1.0, fin=0.28, fout=0.22, bob=3, behind=False):
    w = R.window(u, t0, t1, fin, fout)
    if not w: return 0
    pin, a = w
    F.blit(spr, x, y + math.sin(u * 3 + x) * bob, alpha=a * alpha, scale=s0 + (1 - s0) * eob(pin, back), rot=rot, behind=behind)
    return a

def chip(F, u, t0, t1, text, ic, side="L", y=1230):
    spr = R.chip_sprite(text, ic); cw = spr.shape[1]
    x = 70 + cw / 2 if side == "L" else 1010 - cw / 2
    x = min(max(x, 40 + cw / 2), W - 40 - cw / 2)
    popin(F, spr, x, y, u, t0, t1, s0=0.6, back=2.2)

@functools.lru_cache(None)
def soft_dot(r, color=(255, 255, 255), hard=0):
    s = int(r * 2 + 4); yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    d = np.hypot(xx - s / 2, yy - s / 2) / r
    a = np.clip(1 - d, 0, 1) ** (0.5 if hard else 1.7) * 255
    out = np.zeros((s, s, 4), np.float32); out[..., :3] = color; out[..., 3] = a
    return out

@functools.lru_cache(None)
def crystal(size):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0)); d = ImageDraw.Draw(im); c = size / 2; L = size * 0.45; w = max(2, size // 22)
    for k in range(6):
        a = k * math.pi / 3; d.line((c, c, c + L * math.cos(a), c + L * math.sin(a)), fill=(235, 248, 255, 255), width=w)
        for f in (0.4, 0.65):
            bx, by = c + L * f * math.cos(a), c + L * f * math.sin(a)
            for sg in (-1, 1):
                b = a + sg * math.pi / 4.5
                d.line((bx, by, bx + L * 0.28 * math.cos(b), by + L * 0.28 * math.sin(b)), fill=(210, 238, 255, 255), width=max(1, w - 1))
    d.regular_polygon((c, c, size * 0.08), 6, fill=(255, 255, 255, 255))
    g = im.filter(ImageFilter.GaussianBlur(size / 25)); gl = Image.new("RGBA", im.size, (120, 210, 255, 0)); gl.putalpha(g.split()[3].point(lambda v: int(v * 0.9)))
    return to_np(Image.alpha_composite(gl, im))

def near_head(F, x, y, r):
    x, y, r = int(x), int(y), int(r)
    return bool(F.hm[max(0, y - r):max(0, y + r), max(0, x - r):max(0, x + r)].any())

def layer_blit(F, im, alpha=1.0, behind=False):
    F.blit(np.asarray(im).astype(np.float32), W / 2, H / 2, alpha=alpha, behind=behind)

@functools.lru_cache(None)
def chapter(text):
    t = text_layer(text, 52, 800, fill=(255, 255, 255), pad=0)
    w, h = t.size[0] + 70, 88
    base = rounded(w, h, 44, (255, 255, 255), grad=((255, 120, 60), (200, 40, 120)), border=((255, 255, 255, 240), 3), shadow=12)
    pad = (base.size[0] - w) // 2
    base.alpha_composite(t, (pad + 35, pad + (h - t.size[1]) // 2 - 3))
    return to_np(base)

rng = random.Random(7)
CRY = [(rng.uniform(40, 1040), rng.uniform(300, 1350), rng.uniform(0, 6.28), rng.randint(40, 130), rng.choice([0, 1])) for _ in range(34)]
SPARK = [(rng.uniform(0, W), rng.uniform(250, 1300), rng.uniform(0, 6.28), rng.uniform(1.5, 4)) for _ in range(120)]

# ------------------------------------------------------------------ per-scene graphics (u = local time in seconds)
def g1(F, u):   # hook
    popin(F, bigt("מה גורם", 104, WHITE), 540, 330, u, 0.15, 6.1, s0=0.3)
    popin(F, bigt("לשקיעות האש?", 112, FIRE), 540, 450, u, 0.45, 6.1, s0=0.3)
    popin(F, icon_shadowed("fire_3d.png", 150)[0], 175, 860, u, 1.6, 6.1, behind=True, bob=8)
    popin(F, icon_shadowed("sunset_3d.png", 190)[0], 900, 860, u, 0.8, 6.1, behind=True, bob=8)
    chip(F, u, 4.1, 6.2, "גם בסוף השבוע הזה", "sparkles_3d.png", "R")
    popin(F, bigt("התשובה הכי לא צפויה...", 66, WHITE), 540, 360, u, 6.3, 7.8, s0=0.5)
    w = R.window(u, 7.85, 9.0, 0.2, 0.2)
    if w:
        pin, a = w
        F.blit(bigt("קרח", 230, ICE), 540, 420, alpha=a, scale=0.2 + 0.8 * eob(pin, 2.8))
        for k, (x, y, ph, s, fr) in enumerate(CRY[:14]):
            q = clamp((u - 7.85) / 1.1)
            ang = k / 14 * 6.28
            cx, cy = 540 + math.cos(ang) * 420 * eoc(q), 420 + math.sin(ang) * 260 * eoc(q)
            F.blit(crystal(s // 2 + 40), cx, cy, alpha=a * (1 - q * 0.6),
                   rot=u * 90, behind=bool(fr) or near_head(F, cx, cy, s // 2 + 45))
        F.blit(icon_shadowed("ice_3d.png", 170)[0], 860, 620, alpha=a, scale=0.4 + 0.6 * eob(pin, 2.5), rot=math.sin(u * 6) * 8)

def g2(F, u):   # midday: short direct path, blue scatters
    popin(F, chapter("הפיזיקה בקצב"), 540, 330, u, 0.0, 1.7, s0=0.5)
    w = R.window(u, 0.9, 7.3, 0.4, 0.3)
    if w:
        _, a = w
        sx, sy = 880, 420; tx, ty = 560, 980
        F.blit(R.glow(380, (255, 230, 140)), sx, sy, alpha=a * 0.7, behind=True)
        F.blit(icon_shadowed("sun_3d.png", 190)[0], sx, sy, alpha=a, behind=True, rot=u * 10)
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        q = clamp((u - 1.0) / 1.0)
        ex, ey = sx + (tx - sx) * q, sy + (ty - sy) * q
        d.line((sx, sy, ex, ey), fill=(255, 245, 200, 230), width=10)
        layer_blit(F, im.filter(ImageFilter.GaussianBlur(2)), a, behind=True)
        if u > 4.4:   # Rayleigh: blue light scattering in all directions from the beam
            for k in range(40):
                f = (k / 40 + u * 0.35) % 1.0
                bx, by = sx + (tx - sx) * (k / 40), sy + (ty - sy) * (k / 40)
                ang = k * 2.39
                r = 30 + 260 * f
                F.blit(soft_dot(9, (90, 170, 255), 1), bx + math.cos(ang) * r, by + math.sin(ang) * r,
                       alpha=a * (1 - f) * clamp((u - 4.4) / 0.5), behind=True)
    chip(F, u, 1.0, 4.3, "דרך קצרה וישירה", "sun_3d.png", "L")
    chip(F, u, 4.44, 5.6, "האור הכחול מתפזר", "sparkles_3d.png", "L")
    popin(F, bigt("בום!", 150, WHITE), 540, 420, u, 5.62, 6.5, s0=0.2, back=3)
    popin(F, bigt("שמיים כחולים", 104, ((150, 220, 255), (40, 120, 255), (6, 30, 90), (10, 50, 140))), 540, 420, u, 6.55, 7.4, s0=0.3)

WAVES = [((150, 90, 255), 0.30), ((60, 130, 255), 0.42), ((60, 210, 220), 0.55), ((120, 230, 90), 0.68),
         ((255, 220, 60), 1.0), ((255, 150, 40), 1.0), ((255, 90, 140), 1.0), ((235, 40, 40), 1.0)]
def g3(F, u):   # sunset: long path, only long waves arrive
    w = R.window(u, 0.2, 9.6, 0.4, 0.3)
    if w:
        _, a = w
        F.blit(R.glow(420, (255, 160, 60)), 930, 760, alpha=a * 0.6, behind=True)
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        # atmosphere band (arc) the light has to cross
        for k in range(6):
            r0 = 1500 + k * 26
            d.arc((540 - r0, 1700 - r0 + 400, 540 + r0, 1700 + r0 + 400), 230, 310, fill=(255, 200, 150, 40 + k * 8), width=14)
        layer_blit(F, im, a * clamp((u - 0.6) / 0.6), behind=True)
        q0 = clamp((u - 5.4) / 0.4)
        if q0 > 0:
            for j, (col, surv) in enumerate(WAVES):
                y0 = 520 + j * 26
                im2 = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d2 = ImageDraw.Draw(im2)
                prog = clamp((u - 5.4 - j * 0.05) / 3.0)
                xs = np.linspace(980, 980 - 900 * prog, 120)
                fade = np.clip((xs - (980 - 900 * surv)) / 80, 0, 1) if surv < 1 else np.ones_like(xs)
                pts = [(x, y0 + 12 * math.sin((x / (18 + j * 4)) + u * 9)) for x in xs]
                for p1, p2, f in zip(pts, pts[1:], fade):
                    if f > 0.02: d2.line((p1, p2), fill=col + (int(255 * f),), width=6)
                layer_blit(F, im2.filter(ImageFilter.GaussianBlur(1)), a * q0, behind=False if False else True)
            if u > 8.0:
                popin(F, icon("chequered_flag_3d.png", 110), 120, 600, u, 8.0, 9.6, s0=0.3)
    chip(F, u, 2.36, 5.5, "השמש צונחת נמוך", "sunset_3d.png", "L")
    chip(F, u, 5.64, 8.3, "מסלול מכשולים ארוך", "dashing_away_3d.png", "R")
    popin(F, bigt("הכחול לא שורד את המרתון", 64, ((200, 220, 255), (110, 140, 255), (10, 20, 70), (20, 40, 120))), 540, 330, u, 6.6, 8.1, s0=0.4)
    popin(F, bigt("רק כתום, ורוד ואדום", 82, FIRE), 540, 330, u, 8.15, 9.6, s0=0.3, back=2.6)

def g4(F, u):   # meet the cirrus
    popin(F, chapter("ענני הצירוס"), 540, 330, u, 0.0, 3.3, s0=0.5)
    popin(F, bigt("הכוכבים האמיתיים", 84, ((255, 236, 140), (255, 180, 60), (90, 40, 0), (140, 60, 0))), 540, 450, u, 1.1, 3.3, s0=0.3)
    popin(F, bigt("צירוס = ענני נוצה", 86, WHITE), 540, 400, u, 3.5, 6.0, s0=0.3)
    popin(F, icon_shadowed("waving_hand_3d_default.png", 170)[0], 860, 840, u, 6.0, 7.2, rot=math.sin(u * 12) * 18)
    chip(F, u, 6.1, 7.2, "אהלן צירוס!", "cloud_3d.png", "L")

def g5(F, u):   # inside the cirrus
    # altitude & temperature gauge (left, front, beside the body)
    a = clamp((u - 0.3) / 0.4) * (1 - clamp((u - 17.8) / 0.4))
    if a > 0:
        h = 10 * eio(clamp((u - 6.6) / 2.4))
        F.blit(R.chip_sprite(f"גובה: {h:.1f} ק״מ" if False else "גובה", "mountain_3d.png"), 190, 330, alpha=a * 0)
        g = R.glass_sprite if hasattr(R, "glass_sprite") else None
        txt = big(f"{h:4.1f}", 84, (255, 255, 255), (200, 230, 255)) if False else None
        lbl = to_np(text_layer("ק״מ", 40, 800, fill=(200, 235, 255), stroke=4, stroke_fill=(10, 25, 60), pad=10))
        num = ltr(f"{h:.1f}", 90, (255, 255, 255))
        F.blit(num, 200, 380, alpha=a); F.blit(lbl, 200, 450, alpha=a)
        tq = clamp((u - 11.9) / 0.6)
        if tq > 0:
            F.blit(ltr("-50°C", 80, (150, 215, 255)), 880, 380, alpha=a * eoc(tq), scale=0.5 + 0.5 * eob(tq, 2.5))
    # ice crystals drifting in front of and behind the presenter
    k_on = clamp((u - 0.5) / 1.0) * (1 - clamp((u - 17.7) / 0.5))
    if k_on > 0:
        dens = 0.4 + 0.6 * clamp((u - 13.9) / 1.0)
        for j, (x, y, ph, s, fr) in enumerate(CRY[: int(len(CRY) * dens)]):
            yy = (y + u * (25 + s * 0.2)) % 1150 + 280
            cx = x + math.sin(u * 0.7 + ph) * 30
            F.blit(crystal(s), cx, yy, alpha=k_on * (0.55 if fr else 0.85), rot=u * (30 + s * 0.3) * (1 if j % 2 else -1),
                   behind=bool(fr) or near_head(F, cx, yy, s * 0.6 + 10), scale=0.8 + 0.2 * math.sin(u + ph))
        for (x, y, ph, sp) in SPARK[:70]:
            v = max(0.0, math.sin(u * sp + ph)) ** 8
            if v > 0.05: F.blit(soft_dot(6, (255, 255, 255), 1), x, y, alpha=k_on * v, behind=True)
    chip(F, u, 0.94, 2.6, "דקים ושקופים", "cloud_3d.png", "L")
    chip(F, u, 3.48, 4.4, "ענני נוצה", "cloud_3d.png", "R")
    chip(F, u, 7.2, 10.6, "מעל 5-7 ק״מ", "mountain_3d.png", "L")
    chip(F, u, 11.94, 13.8, "קפוא שם למעלה", "snowflake_3d.png", "R")
    popin(F, bigt("מיליארדי גבישי קרח", 82, ICE), 540, 330, u, 14.2, 16.4, s0=0.3)
    popin(F, bigt("לא טיפות מים", 92, ICE), 540, 330, u, 16.46, 18.0, s0=0.3, back=2.6)
    popin(F, icon("cross_mark_3d.png", 110), 880, 900, u, 16.46, 18.0)
    popin(F, icon("droplet_3d.png", 120), 880, 900, u, 16.46, 18.0, alpha=0.75)

def g6(F, u):   # optical trick
    popin(F, chapter("הטריק האופטי"), 540, 330, u, 0.0, 1.7, s0=0.5)
    w = R.window(u, 1.5, 22.8, 0.5, 0.4)
    if w:
        _, a = w
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        hy = 900
        # last red rays from the sun below the horizon -> up to the cirrus -> down to us
        q = clamp((u - 5.9) / 1.6)
        if q > 0:
            sx, sy = 1080, hy + 60; cx, cy = 700, 360
            ex, ey = sx + (cx - sx) * q, sy + (cy - sy) * q
            d.line((sx, sy, ex, ey), fill=(255, 80, 60, 230), width=9)
            if u > 13.0:
                q2 = clamp((u - 13.0) / 1.6)
                d.line((cx, cy, cx + (380 - cx) * q2, cy + (900 - cy) * q2), fill=(255, 140, 120, 220), width=7)
        layer_blit(F, im.filter(ImageFilter.GaussianBlur(2)), a, behind=True)
        if u > 7.3:   # cirrus band glowing red from below
            gq = clamp((u - 7.3) / 1.0)
            F.blit(R.glow(520, (255, 70, 60)), 700, 360, alpha=a * gq * 0.8, behind=True)
            for k in range(9):
                F.blit(crystal(46), 520 + k * 45, 350 + math.sin(k + u * 2) * 18, alpha=a * gq * 0.9, rot=u * 40, behind=True)
    chip(F, u, 0.4, 3.9, "אנחנו כבר בצל", "crescent_moon_3d.png", "L")
    chip(F, u, 6.07, 9.6, "הצירוסים עדיין באור", "cloud_3d.png", "R")
    popin(F, bigt("מיליארדי מראות זעירות", 76, ICE), 540, 330, u, 13.4, 16.8, s0=0.3)
    popin(F, icon_shadowed("mirror_3d.png", 150)[0], 170, 860, u, 13.4, 17.6, rot=math.sin(u * 3) * 8)
    chip(F, u, 16.5, 19.6, "מחזירים את האור אלינו", "sparkles_3d.png", "L")
    popin(F, bigt("העננים נדלקים באדום", 82, FIRE), 540, 330, u, 20.1, 22.9, s0=0.3, back=2.6)

def g7(F, u):   # why now
    popin(F, chapter("למה דווקא עכשיו?"), 540, 330, u, 0.0, 2.9, s0=0.5)
    chip(F, u, 3.54, 5.0, "אוויר נקי ויבש", "sparkles_3d.png", "L")
    chip(F, u, 5.12, 9.6, "בלי הלחות העכורה של הקיץ", "fog_3d.png", "R")
    popin(F, icon("cross_mark_3d.png", 90), 880, 1090, u, 5.6, 9.6)
    chip(F, u, 9.78, 13.5, "ובגובה כבר ממש קר", "snowflake_3d.png", "L")
    popin(F, bigt("= מופע מרהיב", 100, FIRE), 540, 360, u, 13.4, 15.0, s0=0.3, back=2.6)
    for k in range(5):
        popin(F, icon("sparkles_3d.png", 80), 200 + k * 170, 560 + (k % 2) * 60, u, 13.5 + k * 0.08, 15.0, behind=True)

def ending(F, u):   # punch + follow card over the held last frame
    popin(F, bigt("אתם לא סתם רואים", 80, WHITE), 540, 330, u, 0.0, 1.05, s0=0.3, fout=0.15)
    popin(F, bigt("סוף של יום...", 80, WHITE), 540, 430, u, 0.35, 1.05, s0=0.3, fout=0.15)
    popin(F, bigt("אתם רואים קרח", 96, ICE), 540, 320, u, 1.1, 9, s0=0.2, back=2.6)
    popin(F, bigt("שבוער בשמיים", 104, FIRE), 540, 445, u, 1.4, 9, s0=0.2, back=2.6)
    if u > 1.9:
        follow(F, u, 1.9, 2.6)

@functools.lru_cache(None)
def ltr(text, size, color=(255, 255, 255)):
    f = R.font(size, 800); d0 = ImageDraw.Draw(Image.new("L", (1, 1)))
    bb = d0.textbbox((0, 0), text, font=f, direction="ltr", stroke_width=4)
    im = Image.new("RGBA", (bb[2] - bb[0] + 40, bb[3] - bb[1] + 40), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.text((20 - bb[0], 20 - bb[1]), text, font=f, fill=color + (255,), direction="ltr", stroke_width=4, stroke_fill=(5, 15, 40, 255))
    g = Image.new("RGBA", im.size, color + (0,)); g.putalpha(im.split()[3].filter(ImageFilter.GaussianBlur(8)).point(lambda v: int(v * 0.6)))
    return to_np(Image.alpha_composite(g, im))

@functools.lru_cache(None)
def follow_card(followed):
    old = R.text_layer
    def patched(text, *a, **k):
        if text == "60.8K": text = "62.5K"
        return old(text, *a, **k)
    R.text_layer = patched
    try: return R.follow_card.__wrapped__(followed)
    finally: R.text_layer = old

def follow(F, u, t0, tap):
    p = clamp((u - t0) / 0.4)
    spr, (bx, by) = follow_card(u >= tap + 0.1)
    press = 1 - 0.06 * max(0, 1 - abs(u - tap - 0.05) / 0.12)
    cy = 1290 + (1 - eob(p, 1.7)) * 300
    F.blit(spr, 540, cy, alpha=eoc(p), scale=press)
    if tap - 0.6 < u < tap + 0.9:
        hp = clamp((u - (tap - 0.6)) / 0.55)
        hy = cy + by + 70 + (1 - eoc(hp)) * 200 + (8 if abs(u - tap) < 0.1 else 0)
        ha = clamp((u - (tap - 0.6)) / 0.15) * (1 - clamp((u - tap - 0.6) / 0.3))
        F.blit(icon_shadowed("backhand_index_pointing_up_3d_default.png", 120, 8, 8, 0.4)[0], 540 + bx + 30, hy, alpha=ha)

ZBASE = {2: (1.26, 450), 3: (1.26, 450), 6: (1.26, 450), 7: (1.26, 450)}   # crop the leftover balcony chair
GFX = {1: g1, 2: g2, 3: g3, 4: g4, 5: g5, 6: g6, 7: g7}

# ------------------------------------------------------------------ subtitles
FIXW = {"כבישי": "גבישי", "הכרך": "הקרח", "-6": "6", "-7": "7", "5": "5"}
def build_words():
    sw = json.load(open(os.path.join(D, "segwords.json")))
    out = []
    for k in range(1, 8):
        prev = None
        for w in sw[str(k)]:
            t = w["w"]; core = t.strip(",.?!")
            if core == "אנחנו" and prev == "ואנחנו": prev = core; continue
            prev = core
            out.append({"t": FIXW.get(core, core) + ("?" if t.endswith("?") else ""), "s": START[k] / FPS + w["s"],
                        "e": START[k] / FPS + min(w["e"], w["s"] + 0.9), "brk": t.endswith((",", ".", "?")), "seg": k})
    return out

def make_chunks():
    ws = build_words(); res = []; cur = []
    for i, w in enumerate(ws):
        cur.append(w)
        L = sum(len(x["t"]) + 1 for x in cur)
        nxt = ws[i + 1] if i + 1 < len(ws) else None
        if nxt is None or nxt["seg"] != w["seg"] or w["brk"] or len(cur) >= 3 or L >= 15:
            res.append(cur); cur = []
    out = []
    for i, c in enumerate(res):
        t0 = c[0]["s"]; t1 = res[i + 1][0]["s"] if i + 1 < len(res) else TOTAL
        out.append((t0, min(t1, c[-1]["e"] + 0.4), c))
    return out
CHUNKS = make_chunks()

def subtitles(F, t):
    for (t0, t1, ws) in CHUNKS:
        if t0 <= t < t1:
            p = clamp((t - t0) / 0.1)
            sprs = [R.sub_word(w["t"], w["s"] <= t < (ws[j + 1]["s"] if j + 1 < len(ws) else w["e"] + 0.2)) for j, w in enumerate(ws)]
            gap = -30; total = sum(s.shape[1] for s in sprs) + gap * (len(sprs) - 1)
            sc = min(1.0, 860 / total) * (0.88 + 0.12 * eob(p, 2.8))
            x = 540 + total * sc / 2
            for s in sprs:
                ww = s.shape[1] * sc; F.blit(s, x - ww / 2, 1480, alpha=eoc(p), scale=sc); x -= ww + gap * sc
            return

# ------------------------------------------------------------------ frame
@functools.lru_cache(None)
def vign():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = 1 - 0.28 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 1.7)) ** 2)
    g = np.clip(1 - 0.45 * np.clip((yy - 1330) / 590, 0, 1), 0, 1)
    return (np.clip(v, 0.65, 1) * g)[..., None]

def render(i, raw):
    t = i / FPS
    end = i >= START[7] + NF[7]
    if end:
        k, u = 7, (NF[7] - 1) / FPS; ue = t - (START[7] + NF[7]) / FPS
    else:
        k, u = seg_of(i)
    j = min(int(round(u * FPS)), NF[k] - 1)
    P, Hd = masks(k)
    jj = min(j, len(P) - 1)
    img = raw.astype(np.float32)
    # slow cinematic push-in per scene (+ final hold drift)
    zb, ay = ZBASE.get(k, (1.0, 700))
    z = zb + 0.06 * (u / max(DUR[k], 1)) + (0.03 * clamp(ue / END_HOLD) if end else 0)
    M = np.array([[z, 0, 540 - z * 540], [0, z, ay - z * ay]], np.float32)
    img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    pm = cv2.warpAffine(cv2.resize(np.asarray(P[jj]), (W, H)).astype(np.float32) / 255, M, (W, H))
    pm = cv2.GaussianBlur(pm, (0, 0), 1.6)
    hm = cv2.warpAffine(cv2.resize(np.asarray(Hd[jj]), (W, H)), M, (W, H))
    hm = cv2.dilate((hm > 90).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    if end:
        img *= (1 - 0.35 * clamp(ue / 0.6))
    img *= vign()
    F = R.Frame(img, pm, hm)
    if not end:
        GFX[k](F, u)
        # flash transition into every new scene
        if k > 1 and u < 0.22:
            F.img += (255 - F.img) * (1 - u / 0.22) * 0.85
    else:
        ending(F, ue)
    subtitles(F, t)
    np.clip(F.img, 0, 255, out=F.img)
    return F.img.astype(np.uint8), F.head_overlap

def frames_iter(o0, o1):
    """yield (i, raw) for output frames o0..o1 from the per-scene upscaled clips"""
    last = None; cur = None; dec = None
    for i in range(o0, o1):
        if i >= START[7] + NF[7]:
            if last is None:
                b = subprocess.run(["ffmpeg", "-v", "error", "-sseof", "-0.05", "-i", os.path.join(D, "out", "c7.mp4"), "-frames:v", "1",
                                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
                last = np.frombuffer(b[:W * H * 3], np.uint8).reshape(H, W, 3)
            yield i, last; continue
        k, u = seg_of(i)
        if cur != k:
            if dec: dec.kill()
            j0 = i - START[k]
            dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{j0 / FPS:.4f}", "-i", os.path.join(D, "out", f"c{k}.mp4"),
                                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
            cur = k
        b = dec.stdout.read(W * H * 3)
        if len(b) == W * H * 3: last = np.frombuffer(b, np.uint8).reshape(H, W, 3)
        yield i, last
    if dec: dec.kill()

def run(o0, o1, path):
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    ov = []
    for i, raw in frames_iter(o0, o1):
        out, o = render(i, raw)
        if o: ov.append((i, o))
        enc.stdin.write(out.tobytes())
    enc.stdin.close(); enc.wait()
    print(f"{o0}-{o1} done, head-overlap frames {len(ov)} {ov[:4]}", flush=True)

if __name__ == "__main__":
    if sys.argv[1] == "still":
        outs = []
        for ts in sys.argv[3:]:
            i = int(round(float(ts) * FPS))
            raw = next(frames_iter(i, i + 1))[1]
            o, ov = render(i, raw); print(ts, "overlap", ov)
            outs.append(Image.fromarray(o).resize((360, 640), Image.LANCZOS))
        c = Image.new("RGB", (360 * len(outs), 640))
        for k, im in enumerate(outs): c.paste(im, (360 * k, 0))
        c.save(sys.argv[2], quality=90)
    elif sys.argv[1] == "info":
        print("DUR", DUR, "START", START, "NFR", NFR, "TOTAL", TOTAL)
    else:
        run(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
