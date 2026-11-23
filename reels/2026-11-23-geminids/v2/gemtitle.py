# Cinematic opening title for the Geminids reel: a meteor strikes, metallic 3D "הג׳מינידים" ignites
import math, functools, random, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
A = lambda n: os.path.join(HERE, "assets", n)
W, H = 1080, 1920

def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def eoc(x): x = clamp(x); return 1 - (1 - x) ** 3
def eob(x, s=1.9): x = clamp(x); return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2

@functools.lru_cache(None)
def rubik(size, wght):
    f = ImageFont.truetype(A("Rubik.ttf"), size); f.set_variation_by_axes([wght]); return f

def _mask(text, f, pad, direction="rtl"):
    d0 = ImageDraw.Draw(Image.new("L", (1, 1)))
    bb = d0.textbbox((0, 0), text, font=f, direction=direction, language="he")
    w, h = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).text((pad - bb[0], pad - bb[1]), text, font=f, fill=255, direction=direction, language="he")
    return m

def _vgrad(h, w, stops):
    ys = np.linspace(0, 1, h)
    out = np.zeros((h, 3), np.float32)
    for c in range(3):
        out[:, c] = np.interp(ys, [s[0] for s in stops], [s[1][c] for s in stops])
    return np.repeat(out[:, None, :], w, 1)

@functools.lru_cache(None)
def hero(text="הג׳מינידים", size=200, depth=22):
    """metallic extruded title; returns (rgba float32, alpha mask of face for sweeps)"""
    f = rubik(size, 900)
    pad = 70
    m = _mask(text, f, pad)
    w, h = m.size; H2 = h + depth
    face = np.asarray(m, np.float32) / 255
    big = np.zeros((H2, w), np.float32); big[:h] = face
    # extrusion (straight down = feels like heavy 3D block letters)
    ext = np.zeros((H2, w), np.float32)
    for k in range(1, depth + 1):
        ext[k:k + h] = np.maximum(ext[k:k + h], face)
    rgba = np.zeros((H2, w, 4), np.float32)
    # outer glow
    gl = cv2.GaussianBlur(np.maximum(ext, big), (0, 0), 26)
    rgba[..., :3] = (255, 120, 30); rgba[..., 3] = gl * 200
    def over(col, a):
        a = a[..., None]; rgba[..., :3] = rgba[..., :3] * (1 - a) + col * a; rgba[..., 3:] = rgba[..., 3:] * (1 - a) + 255 * a
    # drop shadow
    sh = np.zeros_like(big); sh[14:] = cv2.GaussianBlur(ext, (0, 0), 10)[:-14]
    over(np.array((10, 4, 0), np.float32), sh * 0.6)
    # extrusion with darker bottom
    eg = _vgrad(H2, w, [(0, (120, 50, 10)), (0.6, (90, 30, 5)), (1, (40, 10, 0))])
    over(eg, ext)
    # thin dark outline
    st = cv2.dilate(big, np.ones((5, 5), np.uint8)) - big
    over(np.array((40, 12, 0), np.float32), np.clip(st, 0, 1) * 0.9)
    # metallic face: white-hot top, chrome horizon line, gold, fire bottom
    fg = _vgrad(H2, w, [(0, (255, 255, 245)), (pad / H2 + 0.20 * h / H2, (255, 236, 170)), (pad / H2 + 0.46 * h / H2, (255, 196, 80)),
                        (pad / H2 + 0.48 * h / H2, (255, 250, 220)), (pad / H2 + 0.52 * h / H2, (255, 170, 50)),
                        (pad / H2 + 0.85 * h / H2, (240, 80, 20)), (1, (170, 30, 10))])
    # bevel: light from top-left
    blur = cv2.GaussianBlur(big, (0, 0), 3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3); gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    shade = np.clip(-(gx * 0.6 + gy * 0.8) * 1.6, -1, 1)[..., None]
    fg = np.where(shade > 0, fg + (255 - fg) * shade * 0.8, fg * (1 + shade * 0.45))
    over(fg, big)
    return rgba, big

@functools.lru_cache(None)
def line_text(text, size, wght, col0, col1, glowcol, direction="rtl"):
    f = rubik(size, wght); pad = 40
    m = np.asarray(_mask(text, f, pad, direction), np.float32) / 255
    h, w = m.shape
    rgba = np.zeros((h, w, 4), np.float32)
    g = cv2.GaussianBlur(m, (0, 0), 10); rgba[..., :3] = glowcol; rgba[..., 3] = g * 170
    sh = np.zeros_like(m); sh[4:] = cv2.GaussianBlur(m, (0, 0), 3)[:-4]
    a = (sh * 0.8)[..., None]; rgba[..., :3] = rgba[..., :3] * (1 - a); rgba[..., 3:] = rgba[..., 3:] * (1 - a) + 255 * a
    fg = _vgrad(h, w, [(0, col0), (1, col1)])
    a = m[..., None]; rgba[..., :3] = rgba[..., :3] * (1 - a) + fg * a; rgba[..., 3:] = rgba[..., 3:] * (1 - a) + 255 * a
    return rgba

@functools.lru_cache(None)
def soft(size, col, p=2.2):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d = np.hypot(xx - size / 2, yy - size / 2) / (size / 2)
    arr = np.zeros((size, size, 4), np.float32); arr[..., :3] = col; arr[..., 3] = np.clip(1 - d, 0, 1) ** p * 255
    return arr

@functools.lru_cache(None)
def meteor(L=700):
    im = np.zeros((60, L + 60, 4), np.float32)
    xs = np.arange(L)
    for i, x in enumerate(xs):
        q = (i / L)
        im[22:38, 30 + x, :3] = (255, 200 + 55 * q, 120 + 135 * q)
        im[22:38, 30 + x, 3] = 255 * q ** 2.2
    im[..., 3] = cv2.GaussianBlur(im[..., 3], (0, 0), 5)
    core = soft(56, (255, 255, 255), 1.5)
    im[2:58, L + 2:L + 58] = np.maximum(im[2:58, L + 2:L + 58], core)
    return im

@functools.lru_cache(None)
def star4(size, col=(255, 250, 230)):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) - size / 2
    r = size / 2
    a = np.exp(-np.abs(xx) / (r * 0.06)) * np.exp(-np.abs(yy) / (r * 0.5)) + np.exp(-np.abs(yy) / (r * 0.06)) * np.exp(-np.abs(xx) / (r * 0.5))
    a += np.exp(-(xx ** 2 + yy ** 2) / (2 * (r * 0.12) ** 2))
    arr = np.zeros((size, size, 4), np.float32); arr[..., :3] = col; arr[..., 3] = np.clip(a, 0, 1) * 255
    return arr

def _rot(spr, deg):
    h, w = spr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1]); nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(spr, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))

def add_blit(F, spr, cx, cy, alpha=1.0, scale=1.0):
    """additive light blend (for flashes / sparks / sweeps)"""
    if alpha <= 0.003: return
    if abs(scale - 1) > 1e-3: spr = cv2.resize(spr, (max(1, int(spr.shape[1] * scale)), max(1, int(spr.shape[0] * scale))))
    h, w = spr.shape[:2]; x0, y0 = int(cx - w / 2), int(cy - h / 2)
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if X1 <= X0 or Y1 <= Y0: return
    s = spr[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    F.img[Y0:Y1, X0:X1] += s[..., :3] * (s[..., 3:4] / 255 * alpha)

TX, TY = 540, 300          # hero centre
_rng = random.Random(7)
SPARKS = [(_rng.uniform(0, 2 * math.pi), _rng.uniform(250, 900), _rng.uniform(0.5, 1.1), _rng.choice([(255, 230, 160), (255, 160, 70), (180, 220, 255), (255, 255, 255)])) for _ in range(46)]
TWINK = [(_rng.uniform(150, 930), _rng.uniform(150, 560), _rng.uniform(0, 6.28), _rng.uniform(18, 46)) for _ in range(12)]

def burst(F, dt, strength=1.0):
    if not (0 <= dt < 1.2): return
    for ang, sp, life, col in SPARKS:
        if dt > life: continue
        q = dt / life
        x = TX + math.cos(ang) * sp * dt * (1 - 0.35 * q) * 0.9
        y = TY + math.sin(ang) * sp * dt * 0.55 + 260 * dt * dt
        add_blit(F, soft(26, col, 1.6), x, y, alpha=strength * (1 - q) ** 1.3 * 1.4)

def draw(F, u, t_end, pulse=None, impact=0.42):
    """u = scene time (s); title visible until t_end; optional re-ignite pulse time"""
    out = clamp((u - t_end) / 0.35)
    if out >= 1: return
    va = 1 - out
    # 1) meteor streaks in
    if u < impact + 0.05:
        q = eoc(clamp(u / impact) ** 1.4)
        x0, y0 = 1250, -120
        x = x0 + (TX + 40 - x0) * q; y = y0 + (TY - 20 - y0) * q
        ang = math.degrees(math.atan2(TY - y0, TX - x0))
        spr = _rot(meteor(), -ang)
        L = meteor().shape[1] / 2 - 30
        add_blit(F, spr, x - math.cos(math.radians(ang)) * L * 0.98, y - math.sin(math.radians(ang)) * L * 0.98, alpha=1.0)
        add_blit(F, soft(260, (255, 200, 120)), x, y, alpha=0.5)
    dt = u - impact
    if dt < 0: return
    # 2) impact flash + sparks
    if dt < 0.5:
        add_blit(F, soft(900, (255, 220, 160), 1.8), TX, TY, alpha=(1 - dt / 0.5) ** 2 * 1.3)
    burst(F, dt)
    if pulse is not None and u >= pulse:
        burst(F, u - pulse, 0.8)
        if u - pulse < 0.4: add_blit(F, soft(700, (255, 200, 140), 2), TX, TY, alpha=(1 - (u - pulse) / 0.4) ** 2 * 0.9)
    # 3) hero letters slam in
    rgba, face = hero()
    pin = clamp(dt / 0.38)
    sc = 1.0 + 0.32 * (1 - eob(pin, 2.2)) + 0.03 * out
    if pulse is not None and 0 <= u - pulse < 0.35: sc += 0.06 * math.sin(math.pi * (u - pulse) / 0.35)
    sc *= 0.995 + 0.005 * math.sin(u * 2.2)
    F.blit(rgba, TX, TY + 10, alpha=va * eoc(pin * 1.4), scale=sc)
    # 4) light sweep masked to the letter faces
    for ts in (0.55, 2.6, 4.6):
        sw = (dt - ts) / 0.75
        if 0 < sw < 1:
            h, w = face.shape
            xx = np.arange(w, dtype=np.float32)[None, :] + np.arange(h, dtype=np.float32)[:, None] * 0.45
            cxs = -h * 0.45 + (w + h * 0.45) * sw
            band = np.exp(-((xx - cxs) / 38) ** 2) * face
            spr = np.zeros((h, w, 4), np.float32); spr[..., :3] = 255; spr[..., 3] = band * 255
            add_blit(F, spr, TX, TY + 10, alpha=va * 0.85, scale=sc)
            add_blit(F, star4(120), TX - (w * sc) / 2 + (w * sc) * (1 - sw), TY - 40, alpha=va * math.sin(math.pi * sw))
    # 5) kicker above + date below
    ka = clamp((dt - 0.35) / 0.35)
    if ka > 0:
        k = line_text("מטר המטאורים הצבעוני של השנה", 50, 600, (255, 255, 255), (255, 226, 170), (255, 150, 60))
        F.blit(k, TX, 150 - 18 * (1 - eoc(ka)), alpha=va * eoc(ka))
        kw_ = k.shape[1] - 80
        lw = int(110 * eoc(ka))
        if lw > 2:
            ln = np.zeros((4, lw, 4), np.float32); ln[..., :3] = (255, 210, 140); ln[..., 3] = np.linspace(0, 255, lw)[None, :]
            F.blit(ln, TX + kw_ / 2 + 20 + lw / 2, 154, alpha=va)
            F.blit(ln[:, ::-1].copy(), TX - kw_ / 2 - 20 - lw / 2, 154, alpha=va)
            add_blit(F, star4(44), TX + kw_ / 2 + 22 + lw, 154, alpha=va)
            add_blit(F, star4(44), TX - kw_ / 2 - 22 - lw, 154, alpha=va)
    da = clamp((dt - 0.6) / 0.35)
    if da > 0:
        d = line_text("13-14 בדצמבר", 66, 800, (225, 245, 255), (120, 190, 255), (60, 140, 255))
        F.blit(d, TX, 452 + 20 * (1 - eoc(da)), alpha=va * eoc(da), scale=0.8 + 0.2 * eob(da, 2.4))
    # 6) twinkles around
    for (x, y, ph, s) in TWINK:
        tw = 0.5 + 0.5 * math.sin(u * 3.1 + ph)
        add_blit(F, star4(int(s * 2)), x, y, alpha=va * clamp(dt / 0.6) * tw ** 3 * 0.9)
