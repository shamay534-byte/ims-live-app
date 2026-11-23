# Magical opening title: stardust swarms in and forms "הג׳מינידים", luminous gold-to-violet letters, rays, glints, sparkles
import math, functools, random, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
A = lambda n: os.path.join(HERE, "assets", n)
W, H = 1080, 1920
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def eoc(x): x = clamp(x); return 1 - (1 - x) ** 3
def eio(x): x = clamp(x); return x * x * (3 - 2 * x)
def eob(x, s=1.7): x = clamp(x); return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2

TX, TY = 540, 392        # hero centre (inside IG safe zone)
KY, DY = 262, 518        # kicker / date

@functools.lru_cache(None)
def font(name, size, wght=None):
    f = ImageFont.truetype(A(name), size)
    if wght: f.set_variation_by_axes([wght])
    return f

def text_mask(text, f, pad):
    d0 = ImageDraw.Draw(Image.new("L", (1, 1)))
    bb = d0.textbbox((0, 0), text, font=f, direction="rtl", language="he")
    w, h = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).text((pad - bb[0], pad - bb[1]), text, font=f, fill=255, direction="rtl", language="he")
    return np.asarray(m, np.float32) / 255

def vgrad(h, w, stops):
    ys = np.linspace(0, 1, h); out = np.zeros((h, 3), np.float32)
    for c in range(3): out[:, c] = np.interp(ys, [s[0] for s in stops], [s[1][c] for s in stops])
    return np.repeat(out[:, None, :], w, 1)

@functools.lru_cache(None)
def hero():
    """returns face rgba (premultiplied-ready), bloom rgb, face mask"""
    m = text_mask("הג׳מינידים", font("SuezOne.ttf", 178), 90)
    h, w = m.shape
    # luminous face: white-gold core fading to rose and violet toward the bottom, with soft inner light
    g = vgrad(h, w, [(0, (255, 255, 245)), (0.38, (255, 236, 170)), (0.55, (255, 198, 140)), (0.72, (240, 150, 200)), (1, (170, 120, 255))])
    inner = cv2.GaussianBlur(m, (0, 0), 6)
    face = g * (0.78 + 0.22 * inner[..., None])
    edge = np.clip(cv2.dilate(m, np.ones((5, 5), np.uint8)) - cv2.erode(m, np.ones((3, 3), np.uint8)), 0, 1)
    face = face * (1 - 0.35 * edge[..., None]) + np.array((255, 250, 235), np.float32) * 0.35 * edge[..., None] * 0.6
    rgba = np.zeros((h, w, 4), np.float32); rgba[..., :3] = face; rgba[..., 3] = m * 255
    # dark soft shadow under letters for legibility
    sh = np.zeros((h, w, 4), np.float32); sh[..., 3] = np.roll(cv2.GaussianBlur(m, (0, 0), 9), 8, 0) * 150
    # bloom: wide warm glow + tight bright glow
    b1 = cv2.GaussianBlur(m, (0, 0), 28)[..., None] * np.array((255, 170, 90), np.float32) * 0.9
    b2 = cv2.GaussianBlur(m, (0, 0), 10)[..., None] * np.array((255, 225, 170), np.float32) * 0.8
    b3 = cv2.GaussianBlur(m, (0, 0), 60)[..., None] * np.array((150, 110, 255), np.float32) * 0.7
    return rgba, sh, b1 + b2 + b3, m

@functools.lru_cache(None)
def target_points(n=2200, seed=5):
    _, _, _, m = hero(); h, w = m.shape
    ys, xs = np.where(m > 0.5)
    rng = np.random.default_rng(seed); idx = rng.choice(len(xs), n, replace=len(xs) < n)
    return xs[idx] - w / 2, ys[idx] - h / 2

@functools.lru_cache(None)
def particles(n=2200, seed=7):
    rng = np.random.default_rng(seed)
    tx, ty = target_points(n)
    # start along a big swirl coming from the upper right (where the meteor enters)
    ang = rng.uniform(0, 2 * np.pi, n); rad = rng.uniform(250, 900, n)
    sx = 420 + np.cos(ang) * rad * 0.8; sy = -380 + np.sin(ang) * rad * 0.5
    delay = rng.uniform(0.0, 0.55, n); dur = rng.uniform(0.55, 0.95, n)
    swirl = rng.uniform(-1.6, 1.6, n)
    col = rng.choice(4, n, p=[0.45, 0.25, 0.15, 0.15])
    size = rng.uniform(0.5, 1.0, n)
    return tx, ty, sx, sy, delay, dur, swirl, col, size
PCOL = np.array([(255, 240, 200), (255, 200, 120), (255, 170, 220), (190, 170, 255)], np.float32)

@functools.lru_cache(None)
def dot(r, col):
    s = r * 4 + 1; yy, xx = np.mgrid[0:s, 0:s].astype(np.float32) - s / 2
    a = np.exp(-(xx ** 2 + yy ** 2) / (2 * (r * 0.6) ** 2))
    return a[..., None] * np.array(col, np.float32)

@functools.lru_cache(None)
def star4(size, col=(255, 250, 235)):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) - size / 2; r = size / 2
    a = np.exp(-np.abs(xx) / (r * 0.05)) * np.exp(-np.abs(yy) / (r * 0.55)) + np.exp(-np.abs(yy) / (r * 0.05)) * np.exp(-np.abs(xx) / (r * 0.55))
    a += np.exp(-(xx ** 2 + yy ** 2) / (2 * (r * 0.12) ** 2))
    return np.clip(a, 0, 1)[..., None] * np.array(col, np.float32)

@functools.lru_cache(None)
def rays(size=1300, n=22, seed=2):
    rng = np.random.default_rng(seed)
    im = Image.new("L", (size, size), 0); d = ImageDraw.Draw(im); c = size / 2
    for k in range(n):
        a0 = rng.uniform(0, 2 * np.pi); wdt = rng.uniform(0.01, 0.035)
        d.polygon([(c, c), (c + c * math.cos(a0 - wdt), c + c * math.sin(a0 - wdt)), (c + c * math.cos(a0 + wdt), c + c * math.sin(a0 + wdt))], fill=int(rng.uniform(120, 255)))
    a = cv2.GaussianBlur(np.asarray(im, np.float32) / 255, (0, 0), 7)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    fade = np.clip(1 - np.hypot(xx - c, (yy - c) * 1.8) / c, 0, 1) ** 1.6
    return (a * fade)[..., None] * np.array((255, 215, 160), np.float32)

def add(F, spr, cx, cy, k=1.0):
    """additive light"""
    if k <= 0.003: return
    h, w = spr.shape[:2]; x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if X1 <= X0 or Y1 <= Y0: return
    F.img[Y0:Y1, X0:X1] += spr[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] * k

def over(F, rgba, cx, cy, a=1.0, scale=1.0):
    if a <= 0.003: return
    if abs(scale - 1) > 1e-3: rgba = cv2.resize(rgba, (int(rgba.shape[1] * scale), int(rgba.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)
    F.blit(rgba, cx, cy, alpha=a)

def scaled(spr, s):
    return spr if abs(s - 1) < 1e-3 else cv2.resize(spr, (int(spr.shape[1] * s), int(spr.shape[0] * s)), interpolation=cv2.INTER_LINEAR)

@functools.lru_cache(None)
def line_text(text, name, size, col, glow):
    m = text_mask(text, font(name, size), 40); h, w = m.shape
    rgba = np.zeros((h, w, 4), np.float32)
    g = cv2.GaussianBlur(m, (0, 0), 9); rgba[..., :3] = glow; rgba[..., 3] = g * 140
    sh = np.roll(cv2.GaussianBlur(m, (0, 0), 3), 3, 0); a = (sh * 0.7)[..., None]
    rgba[..., :3] *= (1 - a); rgba[..., 3:] = rgba[..., 3:] * (1 - a) + 255 * a
    a = m[..., None]; rgba[..., :3] = rgba[..., :3] * (1 - a) + np.array(col, np.float32) * a; rgba[..., 3:] = rgba[..., 3:] * (1 - a) + 255 * a
    return rgba

def reveal_rtl(rgba, p, soft=60):
    """reveal right-to-left (Hebrew reading direction) with a soft edge"""
    h, w = rgba.shape[:2]; x = np.arange(w, dtype=np.float32)
    edge = w - p * (w + soft)
    a = np.clip((x - edge) / soft, 0, 1)[None, :, None]
    out = rgba.copy(); out[..., 3:] *= a; return out, edge

_rng = random.Random(11)
DUST = [(_rng.uniform(60, 1020), _rng.uniform(200, 600), _rng.uniform(0, 6.28), _rng.uniform(4, 11), _rng.uniform(-12, 12), _rng.uniform(-22, -6)) for _ in range(70)]
GLINTS = [(_rng.uniform(-380, 380), _rng.uniform(-50, 40), _rng.uniform(0, 6.28)) for _ in range(9)]

def draw(F, t, t_end, pulse=None, t0=0.15):
    """t: seconds; title assembles from t0, dissolves upward at t_end"""
    u = t - t0
    if u < 0: return
    out = clamp((t - t_end) / 0.45)
    if out >= 1: return
    va = 1 - out
    rgba, sh, bloom, m = hero()
    h, w = m.shape
    # god rays behind the title, slowly turning
    ra = clamp((u - 0.35) / 0.8) * va
    if ra > 0:
        rr = rays()
        M = cv2.getRotationMatrix2D((rr.shape[1] / 2, rr.shape[0] / 2), u * 6, 1.0)
        rot = cv2.warpAffine(rr, M, (rr.shape[1], rr.shape[0]))
        add(F, rot, TX, TY, 0.20 * ra * (0.85 + 0.15 * math.sin(u * 2.3)))
    # stardust swarm -> letters
    tx, ty, sx, sy, delay, dur, swirl, col, size = particles()
    q = np.clip((u - delay) / dur, 0, 1)
    vis = (u > delay) & (q < 1.0) | ((q >= 1.0) & (u < 1.75))
    if vis.any() and out < 1:
        e = 1 - (1 - q) ** 3
        ang = swirl * (1 - e) * 2.2
        dx, dy = sx - tx, sy - ty
        px = TX + tx + dx * (1 - e) * np.cos(ang) - dy * (1 - e) * np.sin(ang)
        py = TY + ty + dx * (1 - e) * np.sin(ang) + dy * (1 - e) * np.cos(ang)
        if out > 0:  # dissolve upward
            py = py - out * (120 + 260 * np.abs(np.sin(tx * 0.05)))
            px = px + out * 40 * np.sin(ty * 0.1)
        land = np.clip((u - delay - dur) / 0.5, 0, 1)
        br = (0.6 + 0.4 * np.sin(u * 9 + tx)) * (1 - land * 0.85)
        if out > 0: br = (0.9 + 0.1 * np.sin(u * 9 + tx)) * (1 - out)
        for i in np.where(vis | (out > 0))[0][::1]:
            if br[i] <= 0.02: continue
            r = 2 if size[i] > 0.75 else 1
            add(F, dot(r, tuple(PCOL[col[i]])), px[i], py[i], 1.4 * br[i])
    # letters condense out of the dust
    la = eio((u - 0.55) / 0.9) * va * (1 - out * 0.0)
    if la > 0:
        breathe = 1 + 0.012 * math.sin(u * 1.7)
        if pulse is not None and 0 <= t - pulse < 0.5: breathe += 0.05 * math.sin(math.pi * (t - pulse) / 0.5)
        add(F, scaled(bloom, breathe) * (la * (0.8 + 0.15 * math.sin(u * 2.1))), TX, TY)
        F.blit(sh, TX, TY, alpha=la * 0.8, scale=breathe)
        F.blit(rgba, TX, TY, alpha=la, scale=breathe)
        # shimmer sweep across letters
        for ts in (1.3, 3.4, 5.4):
            sw = (u - ts) / 0.9
            if 0 < sw < 1:
                xx = np.arange(w, dtype=np.float32)[None, :] + np.arange(h, dtype=np.float32)[:, None] * 0.5
                cx_ = w + h * 0.5 - (w + h * 0.5 + 60) * sw
                band = np.exp(-((xx - cx_) / 46) ** 2) * m
                add(F, scaled(band[..., None] * np.array((255, 245, 225), np.float32), breathe), TX, TY, 0.9 * la)
        # travelling glints sitting on the letters
        for gx, gy, ph in GLINTS:
            tw = max(0.0, math.sin(u * 2.6 + ph)) ** 6
            add(F, star4(90), TX + gx, TY + gy, 1.1 * tw * la)
    # kicker: written in with a glowing pen, right to left
    ka = clamp((u - 1.0) / 0.9)
    if ka > 0:
        k = line_text("מטר המטאורים הצבעוני של השנה", "SuezOne.ttf", 48, (255, 245, 230), (255, 170, 120))
        kr, edge = reveal_rtl(k, eio(ka))
        F.blit(kr, TX, KY, alpha=va)
        if ka < 1: add(F, star4(70, (255, 230, 190)), TX - k.shape[1] / 2 + edge + 20, KY, 1.2)
        lw = int(120 * eoc(ka))
        if lw > 2:
            ln = np.zeros((3, lw, 4), np.float32); ln[..., :3] = (255, 220, 170); ln[..., 3] = np.linspace(0, 230, lw)[None, :]
            kw_ = k.shape[1] - 80
            F.blit(ln, TX + kw_ / 2 + 18 + lw / 2, KY + 2, alpha=va); F.blit(ln[:, ::-1].copy(), TX - kw_ / 2 - 18 - lw / 2, KY + 2, alpha=va)
    # date
    da = clamp((u - 1.5) / 0.6)
    if da > 0:
        d = line_text("13-14 בדצמבר", "SuezOne.ttf", 60, (215, 235, 255), (110, 150, 255))
        F.blit(d, TX, DY + 14 * (1 - eoc(da)), alpha=va * eoc(da))
    # drifting magic dust around the title
    for (x, y, ph, s, vx, vy) in DUST:
        a = clamp((u - 0.8) / 0.8) * va * (0.5 + 0.5 * math.sin(u * 2.2 + ph)) ** 2
        if a > 0.02:
            add(F, dot(2 if s > 8 else 1, (255, 235, 200)), x + vx * u, y + vy * u, 0.9 * a)
    # re-ignite on "תכירו את הג'מינידים": expanding shimmer ring
    if pulse is not None and 0 <= t - pulse < 0.7:
        p = (t - pulse) / 0.7
        r = int(80 + 520 * eoc(p))
        ring = np.zeros((2 * r + 40, 2 * r + 40), np.float32)
        cv2.ellipse(ring, (r + 20, r + 20), (r, int(r * 0.42)), 0, 0, 360, 1.0, 6)
        ring = cv2.GaussianBlur(ring, (0, 0), 5)[..., None] * np.array((255, 225, 180), np.float32)
        add(F, ring, TX, TY, 0.9 * (1 - p))
