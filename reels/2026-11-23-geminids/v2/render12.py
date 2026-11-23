# Geminids reel v2: clean takes, 13 grounded Seedance locations (native 720p, one sharp resample), new cinematic title
import sys, os, json, math, subprocess, functools, importlib.util, bisect
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("R", os.path.join(HERE, "render.py"))
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
import gemtitle as G
clamp, eoc, eob = R.clamp, R.ease_out_cubic, R.ease_out_back
icon_shadowed, to_np = R.icon_shadowed, R.to_np
D = os.path.join(HERE, "r11")
W, H, FPS = 1080, 1920, 30
SP = 1.0           # no speed-up (natural motion); pace comes from pause trimming
END_HOLD = 1.7

# ------------------------------------------------------------------ edit decision list
CLIPS = {n: (a, b) for n, a, b in json.load(open(os.path.join(D, "clips.json")))}
# piece: (kind, clip, src_a, src_b)   kind "c" = his footage, "v" = voice-note cutaway (times in the voice note)
PIECES = [
    ("c", 1, 12.85, 19.35),
    ("c", 2, 21.75, 29.30),
    ("c", 3, 29.30, 34.95),
    ("c", 4, 36.85, 39.94),
    ("c", 5, 58.26, 66.75),
    ("c", 6, 81.75, 87.32),
    ("c", 6, 88.00, 91.12),     # "ומתכת" removed
    ("c", 7, 94.50, 97.41),
    ("v", 0, 3.90, 8.70),       # re-recorded colours line over a fireball cutaway
    ("c", 8, 110.88, 113.62),
    ("c", 9, 116.80, 122.25),
    ("c", 10, 135.85, 144.45),  # the full retake (133.6 was an aborted take)
    ("c", 11, 151.40, 155.56),
    ("c", 12, 155.56, 159.25),
    ("c", 13, 159.50, 161.62),
]

def _wav(path):
    import wave
    w = wave.open(path); a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    return a, w.getframerate()
SRC, SR = _wav(os.path.join(D, "src16.wav"))
VO, VSR = _wav(os.path.join(D, "vo", "fix16.wav"))

def runs_of(kind, a, b, min_gap=0.27, pad=0.05, th=1300):
    """keep [a,b]; silences >= min_gap shrink to pad on each side (removes >= 0.17 s), edges trimmed too"""
    x, sr = (SRC, SR) if kind == "c" else (VO, VSR)
    hop = int(0.01 * sr); i0, i1 = int(a * sr), int(b * sr)
    e = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2)) for i in range(i0, i1, hop)])
    quiet = e < th
    n = len(e); k0 = 0; k1 = n
    while k0 < n and quiet[k0]: k0 += 1
    while k1 > k0 and quiet[k1 - 1]: k1 -= 1
    s0 = a + max(0, k0 * 0.01 - pad); s1 = min(b, a + k1 * 0.01 + pad)
    out = []; cur = s0; k = k0
    while k < k1:
        if quiet[k]:
            j = k
            while j < k1 and quiet[j]: j += 1
            t0, t1 = a + k * 0.01, a + j * 0.01
            if (t1 - t0) >= min_gap:
                out.append((cur, t0 + pad)); cur = t1 - pad
            k = j
        else:
            k += 1
    out.append((cur, s1))
    return [(p, q) for p, q in out if q - p > 0.02]

RUNS = []   # (piece_idx, src_a, src_b, tl_a)
_tl = 0.0
for pi, (kind, n, a, b) in enumerate(PIECES):
    for s, e in runs_of(kind, a, b):
        RUNS.append((pi, s, e, _tl)); _tl += e - s
TL = _tl
NOUT = int(round(TL / SP * FPS)) + int(END_HOLD * FPS)
_RA = [r[3] for r in RUNS]

def at(j):
    """output frame -> (piece, src time, time since piece start in output seconds, end-hold seconds)"""
    tau = j * SP / FPS
    if tau >= TL:
        pi, s, e, ta = RUNS[-1]
        return pi, e - 0.001, (tau - TL) / SP
    r = RUNS[bisect.bisect_right(_RA, tau) - 1]
    return r[0], r[1] + (tau - r[3]), 0.0

def T(src, pi=None):
    """source time (in piece pi, or the first piece containing it) -> output seconds"""
    for (p, s, e, ta) in RUNS:
        if pi is not None and p != pi: continue
        if pi is None and not (PIECES[p][2] - 0.05 <= src <= PIECES[p][3] + 0.05): continue
        if src < s: return ta / SP
        if src <= e: return (ta + src - s) / SP
    for (p, s, e, ta) in RUNS[::-1]:
        if pi is not None and p != pi: continue
        if pi is None and not (PIECES[p][2] - 0.05 <= src <= PIECES[p][3] + 0.05): continue
        return (ta + e - s) / SP
    raise ValueError(src)
def PSTART(pi): return min(r[3] for r in RUNS if r[0] == pi) / SP
def PEND(pi): return max(r[3] + r[2] - r[1] for r in RUNS if r[0] == pi) / SP

# ------------------------------------------------------------------ type helpers (as in v1)
@functools.lru_cache(None)
def big(text, size, g0, g1, sf, ef, extrude=8):
    return to_np(R.text_layer(text, size, 800, gradient=(g0, g1), stroke=5, stroke_fill=sf, extrude=extrude, extrude_fill=ef, shadow=16, pad=80))
STAR = ((255, 248, 210), (255, 196, 90), (60, 30, 0), (110, 60, 10))
ICE = ((235, 250, 255), (120, 200, 255), (10, 40, 90), (20, 70, 140))
FIRE = ((255, 236, 110), (255, 90, 30), (90, 20, 0), (140, 40, 0))
NIGHT = ((230, 225, 255), (150, 130, 255), (20, 10, 60), (40, 25, 110))
YEL = ((255, 250, 190), (255, 205, 40), (80, 50, 0), (130, 90, 0))
GRN = ((210, 255, 200), (60, 220, 90), (0, 60, 10), (10, 100, 30))
RED = ((255, 170, 150), (240, 50, 40), (70, 0, 0), (120, 10, 10))
BLUE = ((180, 225, 255), (40, 120, 255), (0, 20, 70), (10, 40, 120))
def bigt(text, size, pal): return big(text, size, pal[0], pal[1], pal[2], pal[3])

@functools.lru_cache(None)
def ltr(text, size, g0, g1):
    f = R.font(size, 800); d0 = ImageDraw.Draw(Image.new("L", (1, 1)))
    bb = d0.textbbox((0, 0), text, font=f, direction="ltr", stroke_width=6)
    w, h = bb[2] - bb[0] + 80, bb[3] - bb[1] + 80
    m = Image.new("L", (w, h), 0); ImageDraw.Draw(m).text((40 - bb[0], 40 - bb[1]), text, font=f, fill=255, direction="ltr")
    ms = Image.new("L", (w, h), 0); ImageDraw.Draw(ms).text((40 - bb[0], 40 - bb[1]), text, font=f, fill=255, direction="ltr", stroke_width=6, stroke_fill=255)
    yy = np.linspace(0, 1, h)[:, None, None]
    g = (np.array(g0, np.float32) * (1 - yy) + np.array(g1, np.float32) * yy) * np.ones((1, w, 1))
    body = Image.fromarray(g.astype(np.uint8)).convert("RGBA"); body.putalpha(m)
    st = Image.new("RGBA", (w, h), (30, 15, 0, 255)); st.putalpha(ms)
    gl = Image.new("RGBA", (w, h), tuple(int(c) for c in g1) + (0,)); gl.putalpha(ms.filter(ImageFilter.GaussianBlur(14)).point(lambda v: int(v * 0.7)))
    return to_np(Image.alpha_composite(Image.alpha_composite(gl, st), body))

TOPQ = []
CUR = {"n": 1}
@functools.lru_cache(None)
def headtop(n):
    try:
        Hd = np.load(os.path.join(D, "out", f"m{n}_head.npy"), mmap_mode="r")
        tops = [np.where((np.asarray(f) > 90).any(1))[0] for f in Hd[::3]]
        return int(min(t[0] for t in tops if len(t)) * 3)
    except Exception:
        return 600
def TY(base=330):
    return min(base, headtop(CUR["n"]) - 150)
SAFE_W = 740          # visible-content width that survives Instagram's side crop on tall phones
def fitw(spr, maxw=SAFE_W):
    a = spr[..., 3] > 8
    cols = np.where(a.any(0))[0]
    if not len(cols): return spr
    vis = cols[-1] - cols[0]
    if vis <= maxw: return spr
    k = maxw / vis
    return cv2.resize(spr, (int(spr.shape[1] * k), int(spr.shape[0] * k)), interpolation=cv2.INTER_AREA)
def _popin(F, spr, x, y, t, t0, t1, s0=0.4, back=2.2, rot=0.0, alpha=1.0, fin=0.28, fout=0.22, bob=3):
    w = R.window(t, t0, t1, fin, fout)
    if not w: return 0
    pin, a = w
    spr = fitw(spr)
    F.blit(spr, x, y + math.sin(t * 3 + x) * bob, alpha=a * alpha, scale=s0 + (1 - s0) * eob(pin, back), rot=rot)
    return a
def top(F, spr, t, t0, t1, y=330, **kw):
    TOPQ.append((F, spr, 540, TY(y), t, t0, t1, kw))
def draw_tops(F):
    act = [q for q in TOPQ if R.window(q[4], q[5], q[6], 0.28, 0.22)]
    if not act: return
    best = max(act, key=lambda q: q[5])
    for (F_, spr, x, y, t, t0, t1, kw) in [q for q in act if q[6] == best[6] and q[5] >= best[5] - 1.2]:
        _popin(F_, spr, x, y, t, t0, t1, **kw)

@functools.lru_cache(None)
def kw_sprite(text, pal_i):
    pal = (STAR, ICE, FIRE, NIGHT)[pal_i]
    t = bigt(text, 76, pal)
    sc = min(1.0, 600 / t.shape[1])
    if sc < 1: t = cv2.resize(t, (int(t.shape[1] * sc), int(t.shape[0] * sc)), interpolation=cv2.INTER_AREA)
    return t
def kw(F, t, t0, t1, text, ic, side="L", pal=0, y=1215):
    w = R.window(t, t0, t1, 0.22, 0.18)
    if not w: return
    pin, a = w
    s = kw_sprite(text, pal); tw = s.shape[1]
    sgn = -1 if side == "L" else 1
    cx = 540 + sgn * 30 + (1 - eoc(clamp(pin * 1.6))) * 260 * sgn
    rot = -4 * sgn * (1 - 0.6 * eoc(clamp(pin * 2)))
    col = [(255, 190, 90), (90, 180, 255), (255, 120, 50), (160, 130, 255)][pal]
    F.blit(R.glow(min(420, max(300, int(tw * 0.9))), col), cx, y + 20, alpha=a * 0.32)
    F.blit(s, cx, y, alpha=a, scale=0.55 + 0.45 * eob(pin, 2.6), rot=rot)
    sw = clamp((t - t0 - 0.25) / 0.45)
    if 0 < sw < 1: F.blit(R.glow(150, (255, 255, 255)), cx - tw / 2 + tw * sw, y, alpha=a * 0.5 * math.sin(math.pi * sw))
    if ic:
        ix = cx + tw / 2 + 30 if cx + tw / 2 + 110 < W else cx - tw / 2 - 30
        ix = min(max(ix, 200), W - 200)
        pi_ = clamp((t - t0 - 0.08) / 0.3)
        if pi_ > 0:
            F.blit(icon_shadowed(ic, 140)[0], ix, y - 70 + math.sin(t * 4) * 8, alpha=a * eoc(pi_), scale=0.3 + 0.7 * eob(pi_, 3.0), rot=math.sin(t * 3) * 10)

@functools.lru_cache(None)
def streak(L, col):
    im = Image.new("RGBA", (L + 40, 40), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for k in range(L):
        d.line((20 + k, 20, 21 + k, 20), fill=col + (int(255 * (k / L) ** 2),), width=4)
    d.ellipse((L + 8, 12, L + 26, 28), fill=(255, 255, 255, 255))
    g = im.filter(ImageFilter.GaussianBlur(4)); gl = Image.new("RGBA", im.size, col + (0,)); gl.putalpha(g.split()[3])
    return to_np(Image.alpha_composite(gl, im))
def shooting(F, t, t0, n=3, cols=((255, 240, 200),), behind=True, y0=260):
    for j in range(n):
        tt = (t - t0 - j * 0.45) % 2.4
        if tt > 0.7 or t < t0: continue
        q = tt / 0.7
        x = 1100 - q * 900 - j * 60; y = y0 + j * 110 + q * 260
        F.blit(streak(260, cols[j % len(cols)]), x, y, alpha=math.sin(math.pi * q), rot=-16, behind=behind)

@functools.lru_cache(None)
def follow_card(followed):
    old = R.text_layer
    def patched(text, *a, **k):
        if text == "60.8K": text = "62.5K"
        return old(text, *a, **k)
    R.text_layer = patched
    try: return R.follow_card.__wrapped__(followed)
    finally: R.text_layer = old

# ------------------------------------------------------------------ graphics timeline (t = output seconds)
def V(t):  # voice-note time -> output seconds
    return T(t, 8)

def graphics(F, t, end):
    # 1. opening: cinematic title, re-ignites on "תכירו את הג'מינידים"
    G.draw(F, t, PEND(0) - 0.36, pulse=T(18.46), impact=0.30)
    kw(F, t, T(16.6), T(17.95), "בדצמבר!", "calendar_3d.png", "R", 0)
    # 2. Earth crosses the trail
    _popin(F, ltr("13-14.12", 150, (255, 248, 220), (255, 170, 60)), 540, TY(), t, T(22.3), T(24.2), s0=0.3, back=2.6)
    top(F, bigt("כדור הארץ חוצה שובל", 72, ICE), t, T(24.3), T(29.0))
    kw(F, t, T(25.5), T(29.0), "שובל של אבק ושברי סלע", "sparkles_3d.png", "R", 1)
    # 3. sand grains -> shooting stars
    top(F, bigt("בגודל של גרגיר חול", 76, STAR), t, T(29.6), T(30.75))
    kw(F, t, T(30.8), T(32.6), "חודרים לאטמוספירה", "fire_3d.png", "L", 2)
    top(F, bigt("= כוכב נופל", 110, FIRE), t, T(32.7), T(34.95), s0=0.2, back=2.8)
    shooting(F, t, T(32.6), 3, ((255, 220, 160), (255, 255, 255)))
    # 4. what's special
    top(F, bigt("מה מיוחד בהם?", 96, NIGHT), t, T(36.9), T(38.3))
    top(F, bigt("המקור!", 140, FIRE), t, T(38.35), T(39.94), s0=0.2, back=2.8)
    # 5. other showers come from comets
    top(F, bigt("פרסאידים: משביט", 80, ICE), t, T(59.4), T(63.3))
    kw(F, t, T(60.5), T(61.75), "מטר הקיץ", "comet_3d.png", "R", 1)
    top(F, bigt("סלע + הרבה קרח", 90, ICE), t, T(63.4), T(66.7))
    kw(F, t, T(64.6), T(66.7), "כוכב שביט", "ice_3d.png", "L", 1)
    # 6. Phaethon
    top(F, bigt("אבל הג׳מינידים...", 84, FIRE), t, T(81.8), T(82.85))
    top(F, bigt("אסטרואיד פייתון", 96, STAR), t, T(82.9), T(87.3), y=300, back=2.6)
    _popin(F, ltr("3200 Phaethon", 56, (230, 230, 240), (170, 170, 190)), 540, 405, t, T(83.2), T(87.3), s0=0.4)
    kw(F, t, T(86.2), T(87.3), "אסטרואיד סלעי", "rock_3d.png", "R", 0)
    top(F, bigt("נראים אחרת!", 110, FIRE), t, T(88.0), T(91.1), s0=0.2, back=2.8)
    kw(F, t, T(89.4), T(91.1), "בחדירה לאטמוספירה", "fire_3d.png", "L", 2)
    # 7. colourful
    top(F, bigt("צבעוניים במיוחד", 86, STAR), t, T(94.6), T(97.4))
    top(F, bigt("כדור אש!", 130, FIRE), t, V(4.40), V(8.45), s0=0.2, back=2.8, y=300)
    for word, pal, x, y, vt in (("צהוב", YEL, 330, 640, 5.94), ("ירוק", GRN, 560, 560, 6.42), ("אדום", RED, 700, 830, 7.68), ("כחול", BLUE, 760, 1060, 8.02)):
        _popin(F, bigt(word, 124, pal), x, y, t, V(vt), V(8.45), s0=0.2, back=3.0)
    # 8. rate
    t8 = T(111.4)
    if t >= t8 - 0.05:
        q = clamp((t - t8) / 1.4)
        _popin(F, ltr(f"{int(round(100 * eoc(q)))}+", 190, (255, 248, 220), (255, 160, 50)), 540, 330, t, t8, T(113.6), s0=0.4, back=2.4)
    _popin(F, bigt("בשמיים חשוכים לגמרי", 52, ICE), 540, 470, t, T(112.0), T(113.6), s0=0.4)
    kw(F, t, T(111.9), T(113.6), "מטאורים בשעה", "shooting_star_3d.png", "R", 0)
    shooting(F, t, T(110.9), 4)
    # 9. moon
    top(F, bigt("כמעט בלי ירח", 96, NIGHT), t, T(117.3), T(120.2))
    kw(F, t, T(119.1), T(120.2), "שוקע מוקדם", "crescent_moon_3d.png", "R", 3)
    top(F, bigt("תנאי צפייה מצוינים", 80, STAR), t, T(120.26), T(122.2))
    # 10. how to watch
    top(F, bigt("לא צריך שום ציוד", 86, ICE), t, T(136.0), T(137.3))
    kw(F, t, T(137.36), T(138.5), "רק להתרחק מהעיר", "automobile_3d.png", "R", 1)
    top(F, bigt("מדבר = חושך טבעי", 86, STAR), t, T(138.5), T(141.4))
    kw(F, t, T(138.8), T(141.4), "אזורי המדבר", "desert_3d.png", "L", 0)
    top(F, bigt("ולהסתכל למעלה", 96, ICE), t, T(141.45), T(144.4))
    shooting(F, t, T(141.4), 3)
    # 11-12. December weather
    top(F, bigt("אבל... זה דצמבר", 90, ICE), t, T(151.5), T(155.5))
    kw(F, t, T(153.6), T(155.5), "בודקים תחזית", "cloud_with_lightning_3d.png", "R", 1)
    top(F, bigt("סערה = אין מופע", 92, NIGHT), t, T(155.6), T(159.2))
    kw(F, t, T(156.8), T(159.2), "גשם שיחריב את האירוע", "cloud_with_lightning_and_rain_3d.png", "L", 1)
    # 13. outro + follow
    t13 = T(159.5)
    if t >= t13 - 0.05:
        u = t - t13
        top(F, bigt("נתראה מתחת למטר", 84, STAR), t, t13 + 0.05, 99)
        shooting(F, t, t13, 3)
        t0, tap = 0.15, 0.85
        if u > t0:
            p = clamp((u - t0) / 0.4)
            spr, (bx, by) = follow_card(u >= tap + 0.1); bx, by = bx * 0.84, by * 0.84
            press = 1 - 0.06 * max(0, 1 - abs(u - tap - 0.05) / 0.12)
            cy = 1100 + (1 - eob(p, 1.7)) * 300
            F.blit(spr, 540, cy, alpha=eoc(p), scale=press * 0.84)
            if tap - 0.6 < u < tap + 0.9:
                hp = clamp((u - (tap - 0.6)) / 0.55)
                hy = cy + by + 70 + (1 - eoc(hp)) * 200 + (8 if abs(u - tap) < 0.1 else 0)
                ha = clamp((u - (tap - 0.6)) / 0.15) * (1 - clamp((u - tap - 0.6) / 0.3))
                F.blit(icon_shadowed("backhand_index_pointing_up_3d_default.png", 120, 8, 8, 0.4)[0], 540 + bx + 25, hy, alpha=ha)

# ------------------------------------------------------------------ subtitles
FIX = {"אתר": "מטר", "שלווה": "של אבק", "פייטון": "פייתון", "הג'מילידים": "הג'מינידים"}
def load_words():
    full = [w for s in json.load(open(os.path.join(D, "trw.json"))) for w in s["words"]]
    win = json.load(open(os.path.join(D, "w136.json")))
    win = [dict(w, s=w["s"] + 136.0, e=w["e"] + 136.0) for s in win for w in s["words"]]
    full = [w for w in full if not (135.5 <= w["s"] < 138.4) and not (75.5 <= w["s"] < 82.5)] + [w for w in win if w["s"] < 138.4]
    full += [dict(w, s=w["s"] + 110.5, e=w["e"] + 110.5) for s_ in json.load(open(os.path.join(D, "w110.json"))) for w in s_["words"]]
    full += [{"w": " אבל", "s": 81.80, "e": 82.04}, {"w": " הג'מינידים", "s": 82.04, "e": 82.58}]
    full.sort(key=lambda w: w["s"])
    vo = [w for s in json.load(open(os.path.join(D, "vo", "fix.json"))) for w in s["words"]]
    out = []
    for pi, (kind, n, a, b) in enumerate(PIECES):
        src = full if kind == "c" else vo
        for w in src:
            if not (w["e"] > a + 0.08 and w["s"] < b - 0.05): continue
            txt = w["w"].strip()
            if out and out[-1]["pi"] == pi and (txt.startswith("'") or txt.startswith("-")):
                out[-1]["t"] += txt; out[-1]["e"] = T(min(w["e"], b), pi); continue
            out.append({"t": txt, "s": T(max(w["s"], a), pi), "e": T(min(w["e"], b), pi), "pi": pi})
    for w in out:
        core = w["t"].strip(",.?!")
        core = FIX.get(core, core).replace("'", "׳")
        w["brk"] = w["t"].endswith((",", ".", "?")); w["t"] = core
    return out
WORDS = load_words()

def make_chunks():
    res = []; cur = []
    for i, w in enumerate(WORDS):
        cur.append(w); L = sum(len(x["t"]) + 1 for x in cur)
        nxt = WORDS[i + 1] if i + 1 < len(WORDS) else None
        if nxt is None or w["brk"] or len(cur) >= 3 or L >= 15 or nxt["s"] - w["e"] > 0.6:
            res.append(cur); cur = []
    out = []
    for i, c in enumerate(res):
        t0 = c[0]["s"]; t1 = res[i + 1][0]["s"] if i + 1 < len(res) else 999
        out.append((t0, min(t1, c[-1]["e"] + 0.4), c))
    return out
CHUNKS = make_chunks()

def subtitles(F, t):
    for (t0, t1, ws) in CHUNKS:
        if t0 <= t < t1:
            p = clamp((t - t0) / 0.1)
            sprs = [R.sub_word(w["t"], w["s"] <= t < (ws[j + 1]["s"] if j + 1 < len(ws) else w["e"] + 0.2)) for j, w in enumerate(ws)]
            gap = -30; total = sum(s.shape[1] for s in sprs) + gap * (len(sprs) - 1)
            sc = min(1.0, 720 / total) * (0.88 + 0.12 * eob(p, 2.8))
            x = 540 + total * sc / 2
            for s in sprs:
                ww = s.shape[1] * sc; F.blit(s, x - ww / 2, 1390, alpha=eoc(p), scale=sc); x -= ww + gap * sc
            return

# ------------------------------------------------------------------ frames
_masks = {}
def masks(n):
    if n not in _masks:
        _masks[n] = (np.load(os.path.join(D, "out", f"m{n}_person.npy"), mmap_mode="r"), np.load(os.path.join(D, "out", f"m{n}_head.npy"), mmap_mode="r"))
    return _masks[n]

@functools.lru_cache(None)
def vign():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = 1 - 0.28 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 1.7)) ** 2)
    g = np.clip(1 - 0.30 * np.clip((yy - 1330) / 590, 0, 1), 0, 1)
    return (np.clip(v, 0.62, 1) * g)[..., None]

@functools.lru_cache(None)
def cutaway_img():
    im = cv2.cvtColor(cv2.imread(os.path.join(D, "img", "cutaway.jpg")), cv2.COLOR_BGR2RGB)
    return cv2.resize(im, (W, H), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)

def sharpen(img, amt=0.45):
    bl = cv2.GaussianBlur(img, (0, 0), 1.4)
    return img + (img - bl) * amt

class Src:
    """sequential reader of a clip's 720p stretched frames"""
    def __init__(self): self.n = None; self.p = None; self.pos = 0; self.last = None
    def get(self, n, k):
        if n != self.n or k < self.pos - 1:
            if self.p: self.p.kill()
            self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", os.path.join(D, "out", f"c{n}.mp4"), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
            self.n, self.pos, self.last = n, 0, None
        if k == self.pos - 1 and self.last is not None: return self.last
        while self.pos <= k:
            b = self.p.stdout.read(720 * 1280 * 3)
            if len(b) < 720 * 1280 * 3: break
            self.last = np.frombuffer(b, np.uint8).reshape(1280, 720, 3); self.pos += 1
        return self.last

def render(j, src):
    t = j / FPS
    pi, s, hold = at(j)
    kind, n, a, b = PIECES[pi]
    CUR["n"] = n if kind == "c" else 7
    ps = PSTART(pi); pe = PEND(pi)
    u = t - ps
    TOPQ.clear()
    if kind == "c":
        ca = CLIPS[n][0]
        k = int(round((s - ca) * FPS))
        fr = src.get(n, k).astype(np.float32)
        P, Hd = masks(n); kk = min(k, len(P) - 1)
        z = 1.0 + 0.035 * clamp(u / max(pe - ps, 1)) + 0.02 * clamp(hold / END_HOLD)
        sx = W / 720 * z
        M = np.array([[sx, 0, 540 - sx * 360], [0, sx, 900 - sx * 600]], np.float32)
        img = cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        img = sharpen(img)
        Ms = M.copy(); Ms[:2, :2] *= 2
        pm = cv2.warpAffine(np.asarray(P[kk]).astype(np.float32) / 255, Ms, (W, H))
        pm = cv2.GaussianBlur(pm, (0, 0), 1.6)
        hm = cv2.warpAffine(np.asarray(Hd[kk]), Ms, (W, H))
        hm = cv2.dilate((hm > 90).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
        hm[headtop(n) + 380:] = False   # hands are sometimes classed as face-skin; the head is never that low
    else:
        base = cutaway_img()
        z = 1.08 + 0.10 * clamp(u / max(pe - ps, 1))
        M = np.array([[z, 0, 540 - z * 540], [0, z, 760 - z * 760 - 40 * clamp(u / 4)]], np.float32)
        img = cv2.warpAffine(base, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # light flicker from the fireballs
        img *= 1.0 + 0.10 * math.sin(u * 17) * math.sin(u * 5.3)
        pm = np.zeros((H, W), np.float32); hm = np.zeros((H, W), bool)
    img *= vign()
    F = R.Frame(img, pm, hm)
    if kind == "v":
        shooting(F, t, ps, 4, ((255, 220, 60), (80, 255, 120), (255, 80, 70), (90, 160, 255)), behind=False, y0=200)
    graphics(F, t, pi == len(PIECES) - 1)
    # soft flash on every location change
    uf = t - ps
    if pi > 0 and 0 <= uf < 0.18 and PIECES[pi - 1][1] != n:
        F.img += (255 - F.img) * (1 - uf / 0.18) * 0.55
    draw_tops(F)
    subtitles(F, t)
    np.clip(F.img, 0, 255, out=F.img)
    return F.img.astype(np.uint8), F.head_overlap

def run(o0, o1, path):
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "15", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    src = Src(); ov = []
    for j in range(o0, o1):
        out, o = render(j, src)
        if o: ov.append((j, o))
        enc.stdin.write(out.tobytes())
    enc.stdin.close(); enc.wait()
    print(f"{o0}-{o1} done, head-overlap frames {len(ov)} {ov[:5]}", flush=True)

if __name__ == "__main__":
    if sys.argv[1] == "info":
        print("timeline", round(TL, 2), "out frames", NOUT, "=", round(NOUT / FPS, 2), "s")
        for pi in range(len(PIECES)): print(pi, PIECES[pi], round(PSTART(pi), 2), round(PEND(pi), 2))
        print(" ".join(w["t"] for w in WORDS))
    elif sys.argv[1] == "still":
        outs = []; src = Src()
        for ts in sys.argv[3:]:
            j = int(round(float(ts) * FPS)); o, ov = render(j, src); print(ts, "overlap", ov)
            outs.append(Image.fromarray(o).resize((360, 640), Image.LANCZOS))
        c = Image.new("RGB", (360 * len(outs), 640))
        for k, im in enumerate(outs): c.paste(im, (360 * k, 0))
        c.save(sys.argv[2], quality=90)
    else:
        run(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
