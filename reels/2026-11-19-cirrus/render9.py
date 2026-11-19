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
END_HOLD = 1.8
NS = 10

# ------------------------------------------------------------------ timeline
DUR = {k: float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                os.path.join(D, "seg", f"s{k}.mp4")], capture_output=True, text=True).stdout) for k in range(1, NS + 1)}
NF = {k: int(round(DUR[k] * FPS)) for k in DUR}
START = {}; acc = 0
for k in range(1, NS + 1): START[k] = acc; acc += NF[k]
NFR = acc + int(END_HOLD * FPS)
TOTAL = NFR / FPS

def seg_of_raw(i):
    for k in range(NS, 0, -1):
        if i >= START[k]: return k
    return 1

# ---------------- jump cuts: drop every pause (silence) from the source timeline
def _keep():
    import wave
    w = wave.open(os.path.join(D, "voice.wav")); sr = w.getframerate()
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2).astype(np.float32).mean(1)
    hop = sr // 100; n = len(a) // hop
    rms = np.sqrt((a[:n * hop].reshape(n, hop) ** 2).mean(1))
    sil = rms < 1300
    endsrc = START[NS] + NF[NS]
    drop = np.zeros(NFR, bool)
    i = 0
    while i < n:
        if sil[i]:
            j = i
            while j < n and sil[j]: j += 1
            a0, a1 = i / 100 + 0.05, j / 100 - 0.05
            if a1 - a0 >= 0.17:
                f0, f1 = int(math.ceil(a0 * FPS)), int(a1 * FPS)
                drop[f0:min(f1, endsrc)] = True
            i = j
        else: i += 1
    drop[endsrc:] = False
    return [i for i in range(NFR) if not drop[i]]
OUT = _keep()
# instant zoom punch alternating on every hard cut inside a scene
PUNCH = {}
_z = 0
for n_, i_ in enumerate(OUT):
    if n_ and i_ != OUT[n_ - 1] + 1:
        _z = 0 if seg_of_raw(i_) != seg_of_raw(OUT[n_ - 1]) else 1 - _z
    PUNCH[i_] = _z

def seg_of(i):
    for k in range(NS, 0, -1):
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

TOPQ = []
def draw_tops(F):
    # only one headline block at a time in the top zone: the most recent one wins
    act = [q for q in TOPQ if R.window(q[4], q[5], q[6], q[7]["fin"], q[7]["fout"])]
    if not act: return
    best = max(act, key=lambda q: q[5])
    grp = [q for q in act if q[6] == best[6] and q[5] >= best[5] - 1.2]
    for (F_, spr, x, y, u, t0, t1, kw) in grp:
        _popin(F_, spr, x, y, u, t0, t1, **kw)

def popin(F, spr, x, y, u, t0, t1, s0=0.4, back=2.2, rot=0.0, alpha=1.0, fin=0.28, fout=0.22, bob=3, behind=False, top=None):
    if top is None: top = (not behind) and y <= 640 and abs(x - 540) < 160
    if top:
        TOPQ.append((F, spr, x, y, u, t0, t1, dict(s0=s0, back=back, rot=rot, alpha=alpha, fin=fin, fout=fout, bob=bob)))
        return 1
    return _popin(F, spr, x, y, u, t0, t1, s0, back, rot, alpha, fin, fout, bob, behind)

def _popin(F, spr, x, y, u, t0, t1, s0=0.4, back=2.2, rot=0.0, alpha=1.0, fin=0.28, fout=0.22, bob=3, behind=False):
    w = R.window(u, t0, t1, fin, fout)
    if not w: return 0
    pin, a = w
    F.blit(spr, x, y + math.sin(u * 3 + x) * bob, alpha=a * alpha, scale=s0 + (1 - s0) * eob(pin, back), rot=rot, behind=behind)
    return a

CUR_PAL = [FIRE]
@functools.lru_cache(None)
def kw_sprite(text, pal_i):
    pal = (FIRE, ICE)[pal_i]
    t = bigt(text, 78, pal)
    sc = min(1.0, 760 / t.shape[1])
    if sc < 1: t = cv2.resize(t, (int(t.shape[1] * sc), int(t.shape[0] * sc)), interpolation=cv2.INTER_AREA)
    return t

def chip(F, u, t0, t1, text, ic, side="L", y=1215):
    w = R.window(u, t0, t1, 0.22, 0.18)
    if not w: return
    pin, a = w
    pal_i = 0 if CUR_PAL[0] is FIRE else 1
    t = kw_sprite(text, pal_i); tw = t.shape[1]
    sgn = -1 if side == "L" else 1
    ic_spr = icon_shadowed(ic, 150)[0]
    # whole group slides in from its side with a snap, slight 3D slant
    slide = (1 - eoc(clamp(pin * 1.6))) * 260 * sgn
    cx = 540 + sgn * 40 + slide
    rot = -4 * sgn * (1 - 0.6 * eoc(clamp(pin * 2)))
    # glow plate behind the words (no box)
    F.blit(R.glow(max(300, tw), (255, 140, 60) if pal_i == 0 else (90, 180, 255)), cx, y, alpha=a * 0.35, scale=1.0)
    F.blit(t, cx, y, alpha=a, scale=0.55 + 0.45 * eob(pin, 2.6), rot=rot)
    # light sweep across the words right after they land
    sw = clamp((u - t0 - 0.25) / 0.45)
    if 0 < sw < 1:
        F.blit(soft_dot(70, (255, 255, 255)), cx - tw / 2 + tw * sw, y, alpha=a * 0.55 * math.sin(math.pi * sw), scale=1.0)
    # 3D icon pops a beat later on the start (right) side of the text, bobbing
    ix = cx + tw / 2 + 40 if cx + tw / 2 + 110 < W else cx - tw / 2 - 40
    ix = min(max(ix, 85), W - 85)
    pi_ = clamp((u - t0 - 0.08) / 0.3)
    if pi_ > 0:
        F.blit(ic_spr, ix, y - 70 + math.sin(u * 4) * 8, alpha=a * eoc(pi_), scale=0.3 + 0.7 * eob(pi_, 3.0), rot=math.sin(u * 3) * 10)

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

@functools.lru_cache(None)
def hexplate(size, sq):
    # a flat hexagonal ice plate seen at an angle (sq = horizontal squash level 1..8)
    S = size * 2 + 8
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    c = S / 2; pts = [(c + size * math.cos(k * math.pi / 3), c + size * math.sin(k * math.pi / 3)) for k in range(6)]
    d.polygon(pts, fill=(255, 200, 175, 45), outline=(255, 235, 220, 150))
    d.line(pts + [pts[0]], fill=(255, 240, 230, 140), width=max(1, size // 16))
    d.polygon([(c, c - size * 0.6), (c + size * 0.5, c), (c, c + size * 0.6)], fill=(255, 255, 255, 55))
    im = im.resize((max(4, int(S * sq / 8)), S), Image.LANCZOS)
    g = im.filter(ImageFilter.GaussianBlur(3)); gl = Image.new("RGBA", im.size, (255, 120, 80, 0)); gl.putalpha(g.split()[3])
    return to_np(Image.alpha_composite(gl, im))

@functools.lru_cache(None)
def starflare(r):
    S = r * 4; im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im); c = S / 2
    d.line((c - r * 1.9, c, c + r * 1.9, c), fill=(255, 235, 200, 255), width=3)
    d.line((c, c - r * 1.9, c, c + r * 1.9), fill=(255, 235, 200, 255), width=3)
    d.line((c - r, c - r, c + r, c + r), fill=(255, 200, 160, 160), width=2)
    d.line((c - r, c + r, c + r, c - r), fill=(255, 200, 160, 160), width=2)
    im = im.filter(ImageFilter.GaussianBlur(1.2))
    return to_np(im)

_mr = random.Random(23)
MIR = [(_mr.uniform(30, 1050), _mr.uniform(240, 1700), _mr.choice([12, 15, 19, 24, 30, 38]), _mr.uniform(0, 6.28),
        _mr.uniform(0.6, 1.6) * _mr.choice([-1, 1]), _mr.random() < 0.5) for _ in range(70)]

def mirrors(F, u, u0, u1):
    a = clamp((u - u0) / 0.35) * (1 - clamp((u - (u1 - 0.3)) / 0.3))
    if a <= 0: return
    tt = u - u0
    ray = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(ray); any_ray = False
    for (x, y, sz, ph, spin, fr) in MIR:
        xx = x + math.sin(tt * 0.5 + ph) * 40 - tt * 14 * (1 if x > 540 else -1)
        yy = y - tt * (18 + sz * 0.35)
        flip = abs(math.cos(tt * spin + ph)); sq = max(1, int(round(flip * 8)))
        fl = max(0.0, math.sin(tt * 2.3 + ph * 3)) ** 10
        behind = bool(fr) or near_head(F, xx, yy, sz * 3.4 + 20)
        F.blit(hexplate(sz, sq), xx, yy, alpha=a * (0.35 + 0.4 * flip), rot=math.degrees(ph) % 60, behind=behind)
        if fl > 0.05:
            F.blit(soft_dot(sz * 1.3, (255, 150, 90)), xx, yy, alpha=a * fl * 0.8, behind=behind)
            F.blit(starflare(int(sz * 1.6)), xx, yy, alpha=a * fl, rot=tt * 20, behind=behind)
            if fl > 0.35:   # the plate bounces a red ray back towards him (drawn behind his body)
                dr.line((xx, yy, 540 + (xx - 540) * 0.25, 760 + (yy - 760) * 0.25), fill=(255, 120, 90, int(150 * fl)), width=5); any_ray = True
    if any_ray:
        layer_blit(F, ray.filter(ImageFilter.GaussianBlur(2.5)), a, behind=True)

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
_er = random.Random(11)
EMBERS = [(_er.uniform(30, 1050), _er.uniform(0, 1500), _er.uniform(0, 6.28), _er.uniform(60, 190), _er.choice([6, 8, 10, 13, 17]), _er.random() < 0.55) for _ in range(60)]
CRY = [(rng.uniform(40, 1040), rng.uniform(300, 1350), rng.uniform(0, 6.28), rng.randint(40, 130), rng.choice([0, 1])) for _ in range(34)]
SPARK = [(rng.uniform(0, W), rng.uniform(250, 1300), rng.uniform(0, 6.28), rng.uniform(1.5, 4)) for _ in range(120)]

# ------------------------------------------------------------------ per-scene graphics (u = local time in seconds)
def g1(F, u):   # hook
    popin(F, bigt("מה גורם", 104, WHITE), 540, 330, u, 0.15, 6.1, s0=0.3)
    popin(F, bigt("לשקיעות האש?", 112, FIRE), 540, 450, u, 0.45, 6.1, s0=0.3)
    # glowing embers drifting up through the burning sky, some behind and some in front of him
    ea = clamp(u / 0.6) * (1 - clamp((u - 7.6) / 0.4))
    if ea > 0:
        for (ex, ey, ph, sp, sz, fr) in EMBERS:
            yy = (ey - u * sp) % 1500 + 250
            xx = ex + math.sin(u * 1.3 + ph) * 26
            fl = 0.6 + 0.4 * math.sin(u * 9 + ph * 3)
            near = (not fr) and near_head(F, xx, yy, sz * 2 + 10)
            F.blit(soft_dot(sz, (255, 200 if sz < 10 else 140, 80), 1), xx, yy, alpha=ea * fl * (0.9 if fr else 0.75), behind=bool(fr) or near)
            F.blit(soft_dot(sz * 3, (255, 110, 30)), xx, yy, alpha=ea * fl * 0.5, behind=bool(fr) or near)
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
    chip(F, u, 1.0, 4.3, "דרך קצרה וישירה", "sun_3d.png", "L")
    chip(F, u, 4.44, 5.6, "האור הכחול מתפזר", "sparkles_3d.png", "L")
    popin(F, bigt("בום!", 150, WHITE), 540, 420, u, 5.62, 6.5, s0=0.2, back=3)
    popin(F, bigt("שמיים כחולים", 104, ((150, 220, 255), (40, 120, 255), (6, 30, 90), (10, 50, 140))), 540, 420, u, 6.55, 7.4, s0=0.3)

WAVES = [((150, 90, 255), 0.30), ((60, 130, 255), 0.42), ((60, 210, 220), 0.55), ((120, 230, 90), 0.68),
         ((255, 220, 60), 1.0), ((255, 150, 40), 1.0), ((255, 90, 140), 1.0), ((235, 40, 40), 1.0)]
def g3(F, u):   # sunset: long path through the atmosphere
    chip(F, u, 2.36, 5.5, "השמש צונחת נמוך", "sunset_3d.png", "L")
    chip(F, u, 5.64, 9.4, "מסלול מכשולים ארוך", "dashing_away_3d.png", "R")

def gM(F, u):   # the light marathon: blue drops out, only the long red waves reach the finish line
    w = R.window(u, 0.15, 9.3, 0.3, 0.3)
    if w:
        _, a = w
        # the survivors flash as they cross the finish line
        fq = clamp((u - 3.9) / 0.5)
        if fq > 0:
            F.blit(R.glow(380, (255, 120, 60)), 110, 700, alpha=a * fq * 0.7 * (0.75 + 0.25 * math.sin(u * 8)), behind=True)
    popin(F, bigt("הכחול לא שורד את המרתון", 66, ((200, 220, 255), (110, 140, 255), (10, 20, 70), (20, 40, 120))), 540, 330, u, 0.7, 3.0, s0=0.4)
    popin(F, bigt("מי מגיע לקו הסיום?", 84, WHITE), 540, 330, u, 3.05, 5.7, s0=0.3, back=2.6)
    popin(F, bigt("רק כתום, ורוד ואדום", 84, FIRE), 540, 330, u, 5.72, 9.3, s0=0.3, back=2.6)
    chip(F, u, 4.5, 7.3, "הגלים הארוכים", "trophy_3d.png", "L")

def g4(F, u):   # meet the cirrus
    popin(F, chapter("ענני הצירוס"), 540, 330, u, 0.0, 3.3, s0=0.5)
    popin(F, bigt("הכוכבים האמיתיים", 84, ((255, 236, 140), (255, 180, 60), (90, 40, 0), (140, 60, 0))), 540, 450, u, 1.1, 3.3, s0=0.3)
    popin(F, bigt("צירוס = ענני נוצה", 86, WHITE), 540, 400, u, 3.5, 6.0, s0=0.3)
    popin(F, icon_shadowed("waving_hand_3d_default.png", 170)[0], 860, 840, u, 6.0, 7.2, rot=math.sin(u * 12) * 18)
    chip(F, u, 6.1, 7.2, "אהלן צירוס!", "cloud_3d.png", "L")

def g5(F, u):   # inside the cirrus
    # altitude & temperature gauge (left, front, beside the body)
    a = clamp((u - 0.3) / 0.4) * (1 - clamp((u - 13.9) / 0.3))
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

def gB(F, u):   # why ice and not water: prism drops vs. the high white screen (cinematic b-roll)
    popin(F, bigt("רגע, למה דווקא קרח?", 88, ICE), 540, 330, u, 0.1, 2.25, s0=0.3, back=2.6)
    popin(F, bigt("כל טיפה = מנסרה קטנטנה", 78, WHITE), 540, 330, u, 2.3, 7.8, s0=0.3, back=2.6)
    chip(F, u, 6.0, 7.8, "ככה נוצרת קשת", "rainbow_3d.png", "L")
    popin(F, bigt("האור כבר מגיע אדום", 84, FIRE), 540, 330, u, 7.85, 12.5, s0=0.3, back=2.6)
    chip(F, u, 10.4, 12.5, "צריך מסך גבוה", "sparkles_3d.png", "R")
    popin(F, bigt("הצל של כדור הארץ", 84, ((200, 215, 255), (110, 130, 200), (10, 15, 50), (20, 30, 90))), 540, 330, u, 12.6, 16.7, s0=0.3, back=2.6)
    chip(F, u, 13.6, 16.7, "ענני מים נמוכים מאפירים", "fog_3d.png", "L")
    popin(F, bigt("צירוסים = מסך קולנוע ענק", 74, ICE), 540, 330, u, 16.75, 21.4, s0=0.3, back=2.6)
    chip(F, u, 17.8, 21.4, "בגובה כמעט 10 ק״מ", "cloud_3d.png", "R")
    popin(F, bigt("נצבעים באדום בוער!", 92, FIRE), 540, 330, u, 21.45, 23.6, s0=0.2, back=2.8)
    for su in (7.8, 12.55, 16.7):   # flash between the shots
        if 0 <= u - su < 0.16:
            F.img += (255 - F.img) * (1 - (u - su) / 0.16) * 0.6

def g6(F, u):   # optical trick
    if len(SPLICE6_) == 2: mirrors(F, u, SPLICE6_[0], SPLICE6_[1])
    popin(F, chapter("הטריק האופטי"), 540, 330, u, 0.0, 1.7, s0=0.5)
    chip(F, u, 0.4, 3.9, "אנחנו כבר בצל", "crescent_moon_3d.png", "L")
    chip(F, u, 6.07, 9.6, "הצירוסים עדיין באור", "cloud_3d.png", "R")
    popin(F, bigt("מיליארדי מראות זעירות", 76, ICE), 540, 330, u, 13.4, 16.8, s0=0.3)
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

def g8(F, u):   # punch line + follow (continues through the end hold)
    popin(F, bigt("כשהשמיים נדלקים באדום", 84, FIRE), 540, 330, u, 1.6, 3.95, s0=0.3, back=2.6)
    w = R.window(u, 4.0, 6.65, 0.2, 0.25)
    if w:
        pin, a = w
        q = clamp((u - 4.0) / 1.0)
        for k, (x, y, ph, s_, fr) in enumerate(CRY[:18]):
            ang = k / 18 * 6.28 + u * 0.4
            F.blit(crystal(s_ // 2 + 40), 540 + math.cos(ang) * 470 * eoc(q), 430 + math.sin(ang) * 230 * eoc(q),
                   alpha=a * (1 - q * 0.5), rot=u * 90, behind=True)
        F.blit(R.glow(700, (255, 120, 40)), 540, 430, alpha=a * 0.5, behind=True)
        _popin(F, icon_shadowed("fire_3d.png", 170)[0], 150, 640, u, 4.6, 6.65, behind=True, bob=10)
        _popin(F, icon_shadowed("ice_3d.png", 160)[0], 930, 640, u, 4.0, 6.65, behind=True, bob=10, rot=math.sin(u * 5) * 8)
    popin(F, bigt("קרח", 190, ICE), 540, 340, u, 4.0, 6.65, s0=0.2, back=2.8)
    popin(F, bigt("שבוער!", 150, FIRE), 540, 520, u, 4.7, 6.65, s0=0.2, back=2.8)
    chip(F, u, 6.72, 9.45, "גם בסוף השבוע הקרוב!", "sunset_3d.png", "R")
    if u > 9.4:
        follow(F, u, 9.4, 10.35)

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

ZBASE = {}
try:
    SPLICE6 = [int(v) / FPS for v in open(os.path.join(D, "splice8.txt")).read().split()]
except Exception:
    SPLICE6 = []
SPLICE6_ = SPLICE6
GFX = {1: g1, 2: g2, 3: g3, 4: gM, 5: g4, 6: g5, 7: gB, 8: g6, 9: g7, 10: g8}

# ------------------------------------------------------------------ subtitles
FIXW = {"כבישי": "גבישי", "וכבישי": "וגבישי", "הכרך": "הקרח", "-6": "6", "-7": "7", "5": "5"}
def build_words():
    sw = json.load(open(os.path.join(D, "segwords.json")))
    out = []
    for k in range(1, NS + 1):
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
    g = np.clip(1 - 0.30 * np.clip((yy - 1330) / 590, 0, 1), 0, 1)
    return (np.clip(v, 0.65, 1) * g)[..., None]

def render(i, raw):
    t = i / FPS
    end = i >= START[NS] + NF[NS]
    if end:
        k, u = NS, (NF[NS] - 1) / FPS; ue = t - (START[NS] + NF[NS]) / FPS
    else:
        k, u = seg_of(i)
    j = min(int(round(u * FPS)), NF[k] - 1)
    P, Hd = masks(k)
    jj = min(j, len(P) - 1)
    img = raw.astype(np.float32)
    # slow cinematic push-in per scene (+ final hold drift)
    zb, ay = ZBASE.get(k, (1.0, 700))
    z = zb * (1.09 if PUNCH.get(i, 0) else 1.0) + 0.05 * (u / max(DUR[k], 1)) + (0.03 * clamp(ue / END_HOLD) if end else 0)
    M = np.array([[z, 0, 540 - z * 540], [0, z, ay - z * ay]], np.float32)
    img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    pm = cv2.warpAffine(cv2.resize(np.asarray(P[jj]), (W, H)).astype(np.float32) / 255, M, (W, H))
    pm = cv2.GaussianBlur(pm, (0, 0), 1.6)
    hm = cv2.warpAffine(cv2.resize(np.asarray(Hd[jj]), (W, H)), M, (W, H))
    hm = cv2.dilate((hm > 90).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    img *= vign()
    F = R.Frame(img, pm, hm)
    TOPQ.clear(); CUR_PAL[0] = ICE if k in (2, 5, 6) else FIRE
    if not end:
        GFX[k](F, u)
        # flash transition into every new scene
        if k > 1 and u < 0.22:
            F.img += (255 - F.img) * (1 - u / 0.22) * 0.85
        if k == 8:   # flash in/out of the inside-the-cirrus mirrors shot
            for su in SPLICE6:
                if 0 <= u - su < 0.2:
                    F.img += (255 - F.img) * (1 - (u - su) / 0.2) * 0.8
    else:
        GFX[NS](F, DUR[NS] + ue)
    draw_tops(F)
    subtitles(F, t)
    np.clip(F.img, 0, 255, out=F.img)
    return F.img.astype(np.uint8), F.head_overlap

def frames_iter(o0, o1):
    """yield (i, raw) for output frames o0..o1 (indices into OUT) from the per-scene clips"""
    last = None; cur = None; dec = None; pos = None
    for o in range(o0, o1):
        i = OUT[o]
        if i >= START[NS] + NF[NS]:
            if last is None or cur != "end":
                b = subprocess.run(["ffmpeg", "-v", "error", "-sseof", "-0.05", "-i", os.path.join(D, "out", f"c{NS}.mp4"), "-frames:v", "1",
                                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
                last = np.frombuffer(b[:W * H * 3], np.uint8).reshape(H, W, 3); cur = "end"
            yield i, last; continue
        k, u = seg_of(i); j = i - START[k]
        if cur != k or j < pos:
            if dec: dec.kill()
            dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{j / FPS:.4f}", "-i", os.path.join(D, "out", f"c{k}.mp4"),
                                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
            cur = k; pos = j
        while pos < j:
            dec.stdout.read(W * H * 3); pos += 1
        b = dec.stdout.read(W * H * 3); pos += 1
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
            o_ = min(int(round(float(ts) * FPS)), len(OUT) - 1); i = OUT[o_]
            raw = next(frames_iter(o_, o_ + 1))[1]
            o, ov = render(i, raw); print(ts, "overlap", ov)
            outs.append(Image.fromarray(o).resize((360, 640), Image.LANCZOS))
        c = Image.new("RGB", (360 * len(outs), 640))
        for k, im in enumerate(outs): c.paste(im, (360 * k, 0))
        c.save(sys.argv[2], quality=90)
    elif sys.argv[1] == "info":
        print("DUR", DUR, "START", START, "NFR", NFR, "TOTAL", TOTAL, "OUT", len(OUT), len(OUT) / FPS)
    else:
        run(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
