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

TX, TY = 540, 335        # hero block centre (two lines, inside IG safe zone)
KY, DY = 0, 512          # (no kicker) / sub-line

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

PAL = {
 "goldpink": dict(face=[(0, (255, 255, 245)), (0.38, (255, 236, 170)), (0.55, (255, 198, 140)), (0.72, (240, 150, 200)), (1, (170, 120, 255))], depth=(70, 40, 140), outline=(255, 214, 120), glow=[(255, 170, 90), (255, 225, 170), (150, 110, 255)]),
 "gold": dict(face=[(0, (255, 253, 235)), (0.35, (255, 232, 160)), (0.5, (255, 246, 210)), (0.56, (238, 190, 90)), (1, (180, 120, 40))], depth=(70, 40, 10), outline=(255, 240, 190), glow=[(255, 170, 60), (255, 220, 150), (255, 140, 40)]),
 "silver": dict(face=[(0, (255, 255, 255)), (0.45, (225, 238, 255)), (0.52, (255, 255, 255)), (0.6, (180, 205, 240)), (1, (130, 160, 220))], depth=(20, 35, 80), outline=(200, 230, 255), glow=[(110, 170, 255), (210, 235, 255), (80, 120, 255)]),
 "cosmic": dict(face=[(0, (240, 255, 255)), (0.35, (150, 235, 255)), (0.65, (150, 160, 255)), (1, (210, 120, 255))], depth=(30, 15, 80), outline=(220, 245, 255), glow=[(80, 200, 255), (190, 230, 255), (190, 90, 255)]),
 "white": dict(face=[(0, (255, 255, 255)), (1, (255, 246, 230))], depth=(40, 30, 60), outline=(255, 230, 170), glow=[(255, 200, 120), (255, 245, 220), (200, 170, 255)]),
}
PALNAME = "cosmic"
@functools.lru_cache(None)
def hero():
    """returns face rgba (premultiplied-ready), bloom rgb, face mask"""
    # two centred lines, auto-sized to fit 880 px
    lines = ("מטר המטאורים", "הצבעוני של השנה")
    size = 150
    while True:
        ms = [text_mask(t, font("SuezOne.ttf", size), 0) for t in lines]
        if max(x.shape[1] for x in ms) <= 930: break
        size -= 4
    gap = int(size * 0.08); pad = 90
    w = max(x.shape[1] for x in ms) + 2 * pad; h = sum(x.shape[0] for x in ms) + gap + 2 * pad
    m = np.zeros((h, w), np.float32); y = pad
    for x in ms:
        x0 = (w - x.shape[1]) // 2; m[y:y + x.shape[0], x0:x0 + x.shape[1]] = np.maximum(m[y:y + x.shape[0], x0:x0 + x.shape[1]], x); y += x.shape[0] + gap
    h, w = m.shape
    # luminous face: white-gold core fading to rose and violet toward the bottom, with soft inner light
    P = PAL[PALNAME]
    g = vgrad(h, w, P['face'])
    inner = cv2.GaussianBlur(m, (0, 0), 6)
    face = g * (0.78 + 0.22 * inner[..., None])
    edge = np.clip(cv2.dilate(m, np.ones((5, 5), np.uint8)) - cv2.erode(m, np.ones((3, 3), np.uint8)), 0, 1)
    face = face * (1 - 0.35 * edge[..., None]) + np.array((255, 250, 235), np.float32) * 0.35 * edge[..., None] * 0.6
    rgba = np.zeros((h, w, 4), np.float32)
    # 3D depth: stacked violet layers down-right, darker further back
    depth = 9
    for k in range(depth, 0, -1):
        mk = np.roll(np.roll(m, k, 0), k // 2, 1)
        c = np.array(P['depth'], np.float32) * (0.55 + 0.45 * (1 - k / depth))
        a = mk[..., None]; rgba[..., :3] = rgba[..., :3] * (1 - a) + c * a; rgba[..., 3:] = np.maximum(rgba[..., 3:], mk[..., None] * 255)
    # thin gold outline
    ol = np.clip(cv2.dilate(m, np.ones((7, 7), np.uint8)) - m, 0, 1)
    ol = cv2.GaussianBlur(ol, (0, 0), 0.7)
    a = ol[..., None]; rgba[..., :3] = rgba[..., :3] * (1 - a) + np.array(P['outline'], np.float32) * a; rgba[..., 3:] = np.maximum(rgba[..., 3:], ol[..., None] * 255)
    a = m[..., None]; rgba[..., :3] = rgba[..., :3] * (1 - a) + face * a; rgba[..., 3:] = np.maximum(rgba[..., 3:], m[..., None] * 255)
    # dark soft shadow under letters for legibility
    sh = np.zeros((h, w, 4), np.float32); sh[..., 3] = np.roll(cv2.GaussianBlur(m, (0, 0), 9), 8, 0) * 150
    # bloom: wide warm glow + tight bright glow
    b1 = cv2.GaussianBlur(m, (0, 0), 28)[..., None] * np.array(P['glow'][0], np.float32) * 0.9
    b2 = cv2.GaussianBlur(m, (0, 0), 10)[..., None] * np.array(P['glow'][1], np.float32) * 0.8
    b3 = cv2.GaussianBlur(m, (0, 0), 60)[..., None] * np.array(P['glow'][2], np.float32) * 0.7
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
    delay = rng.uniform(0.0, 0.2, n); dur = rng.uniform(0.3, 0.55, n)
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

# stylised Gemini stick figure (Castor & Pollux heads at the top) in a 980x520 box
GEM_PTS = [(640, 40), (560, 70), (610, 190), (520, 220), (590, 330), (500, 380), (660, 470), (430, 480),
           (760, 120), (820, 250), (700, 300), (880, 380), (900, 470)]
GEM_PTS = [(x, y * 0.8) for x, y in GEM_PTS]
GEM_LINES = [(0, 2), (2, 4), (4, 6), (4, 5), (5, 7), (1, 3), (3, 5), (0, 8), (8, 9), (9, 11), (11, 12), (9, 10), (0, 1)]
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
    ra = clamp((u - 0.1) / 0.6) * va
    if ra > 0:
        rr = rays()
        M = cv2.getRotationMatrix2D((rr.shape[1] / 2, rr.shape[0] / 2), u * 6, 1.0)
        rot = cv2.warpAffine(rr, M, (rr.shape[1], rr.shape[0]))
        add(F, rot, TX, TY, 0.20 * ra * (0.85 + 0.15 * math.sin(u * 2.3)))
    # Gemini constellation drawing itself faintly behind the title
    ca = clamp(u / 0.9) * va
    if ca > 0:
        lay = np.zeros((430, 980, 3), np.float32)
        for i, (p0, p1) in enumerate(GEM_LINES):
            q = clamp((u - 0.05 * i) / 0.5)
            if q <= 0: continue
            x0, y0 = GEM_PTS[p0]; x1, y1 = GEM_PTS[p1]
            cv2.line(lay, (int(x0), int(y0)), (int(x0 + (x1 - x0) * q), int(y0 + (y1 - y0) * q)), (150, 170, 255), 2, cv2.LINE_AA)
        lay = cv2.GaussianBlur(lay, (0, 0), 1.0)
        add(F, lay, TX, TY - 10, 0.45 * ca)
        for i, (x, y) in enumerate(GEM_PTS):
            tw = 0.6 + 0.4 * math.sin(u * 3 + i)
            add(F, star4(46 if i in (0, 1) else 30, (210, 220, 255)), TX - 490 + x, TY - 10 - 215 + y, 0.9 * ca * tw)
    # stardust swarm -> letters
    tx, ty, sx, sy, delay, dur, swirl, col, size = particles()
    q = np.clip((u - delay) / dur, 0, 1)
    vis = (u > delay) & (q < 1.0) | ((q >= 1.0) & (u < 1.1))
    if vis.any() and out < 1:
        e = 1 - (1 - q) ** 3
        ang = swirl * (1 - e) * 2.2
        dx, dy = sx - tx, sy - ty
        px = TX + tx + dx * (1 - e) * np.cos(ang) - dy * (1 - e) * np.sin(ang)
        py = TY + ty + dx * (1 - e) * np.sin(ang) + dy * (1 - e) * np.cos(ang)
        if out > 0:  # dissolve upward
            py = py - out * (120 + 260 * np.abs(np.sin(tx * 0.05)))
            px = px + out * 40 * np.sin(ty * 0.1)
        land = np.clip((u - delay - dur) / 0.35, 0, 1)
        br = (0.6 + 0.4 * np.sin(u * 9 + tx)) * (1 - land * 0.85)
        if out > 0: br = (0.9 + 0.1 * np.sin(u * 9 + tx)) * (1 - out)
        for i in np.where(vis | (out > 0))[0][::1]:
            if br[i] <= 0.02: continue
            r = 2 if size[i] > 0.75 else 1
            add(F, dot(r, tuple(PCOL[col[i]])), px[i], py[i], 1.4 * br[i])
    # letters condense out of the dust
    la = eio((u - 0.12) / 0.45) * va * (1 - out * 0.0)
    if la > 0:
        breathe = 1 + 0.012 * math.sin(u * 1.7)
        if pulse is not None and 0 <= t - pulse < 0.5: breathe += 0.05 * math.sin(math.pi * (t - pulse) / 0.5)
        add(F, scaled(bloom, breathe) * (la * (0.8 + 0.15 * math.sin(u * 2.1))), TX, TY)
        F.blit(sh, TX, TY, alpha=la * 0.8, scale=breathe)
        F.blit(rgba, TX, TY, alpha=la, scale=breathe)
        # shimmer sweep across letters
        for ts in (0.7, 2.8, 4.8):
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
    # glowing meteor swoosh underlining the title, drawn right to left
    sa = clamp((u - 0.4) / 0.55)
    if sa > 0:
        L = 760; xs = np.linspace(0, 1, 400)
        lay = np.zeros((90, L + 80, 3), np.float32)
        head = eio(sa)
        pts = [(int(40 + L * (1 - x)), int(45 + 18 * math.sin(math.pi * x) * -1)) for x in xs if x <= head]
        for i in range(1, len(pts)):
            f_ = i / max(1, len(pts))
            cv2.line(lay, pts[i - 1], pts[i], (255 * (0.5 + 0.5 * f_), 215 * (0.5 + 0.5 * f_), 160), 2 + int(3 * f_), cv2.LINE_AA)
        lay = lay + cv2.GaussianBlur(lay, (0, 0), 6) * 1.2
        add(F, lay, TX, TY + 140, 0.85 * va)
        if sa < 1 and pts: add(F, star4(80), TX - (L + 80) / 2 + pts[-1][0], TY + 140 - 45 + pts[-1][1], 1.4)
    # date
    da = clamp((u - 0.5) / 0.7)
    if da > 0:
        d = line_text("מטר הג׳ימינידים - 13-14.12", "SuezOne.ttf", 58, (225, 238, 255), (110, 150, 255))
        dr, edge = reveal_rtl(d, eio(da))
        F.blit(dr, TX, DY, alpha=va)
        if da < 1: add(F, star4(70, (200, 220, 255)), TX - d.shape[1] / 2 + edge + 20, DY, 1.2)
    # drifting magic dust around the title
    for (x, y, ph, s, vx, vy) in DUST:
        a = clamp((u - 0.3) / 0.6) * va * (0.5 + 0.5 * math.sin(u * 2.2 + ph)) ** 2
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
