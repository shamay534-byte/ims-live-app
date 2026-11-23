# Geminids reel: 21 short Seedance scenes, cinematic titles, 3D keywords, subtitles, follow card
import sys, os, json, math, random, subprocess, functools, importlib.util
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("R", os.path.join(HERE, "render.py"))
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
clamp, eoc, eio, eob = R.clamp, R.ease_out_cubic, R.ease_in_out, R.ease_out_back
text_layer, icon, icon_shadowed, to_np = R.text_layer, R.icon, R.icon_shadowed, R.to_np
D = os.path.join(HERE, "r10")
W, H, FPS = 1080, 1920, 30
NS = 21
END_HOLD = 1.6

# ------------------------------------------------------------------ timeline
DUR = {k: float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                os.path.join(D, "seg", f"s{k}.mp4")], capture_output=True, text=True).stdout) for k in range(1, NS + 1)}
NF = {k: int(round(DUR[k] * FPS)) for k in DUR}
START = {}; _acc = 0
for k in range(1, NS + 1): START[k] = _acc; _acc += NF[k]
NFR = _acc + int(END_HOLD * FPS)

def seg_of_raw(i):
    for k in range(NS, 0, -1):
        if i >= START[k]: return k
    return 1

def seg_of(i):
    k = seg_of_raw(i); return k, (i - START[k]) / FPS

TRIM = [(6, 2.92, None), (11, 0.0, 0.40)]   # stumble at the end of 6, repeated word at the start of 11

def _keep():
    import wave
    w = wave.open(os.path.join(D, "voice.wav")); sr = w.getframerate()
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2).astype(np.float32).mean(1)
    hop = sr // 100; n = len(a) // hop
    rms = np.sqrt((a[:n * hop].reshape(n, hop) ** 2).mean(1)); sil = rms < 1300
    endsrc = START[NS] + NF[NS]
    drop = np.zeros(NFR, bool); i = 0
    while i < n:
        if sil[i]:
            j = i
            while j < n and sil[j]: j += 1
            a0, a1 = i / 100 + 0.05, j / 100 - 0.05
            if a1 - a0 >= 0.17: drop[int(math.ceil(a0 * FPS)):min(int(a1 * FPS), endsrc)] = True
            i = j
        else: i += 1
    for k_, u0_, u1_ in TRIM:
        drop[START[k_] + int(round(u0_ * FPS)): START[k_] + (NF[k_] if u1_ is None else int(round(u1_ * FPS)))] = True
    drop[endsrc:] = False
    return [i for i in range(NFR) if not drop[i]]
OUT = _keep()
FIRST = {}
for _i in OUT: FIRST.setdefault(seg_of_raw(_i), _i)
PUNCH = {}; _z = 0
for n_, i_ in enumerate(OUT):
    if n_ and i_ != OUT[n_ - 1] + 1:
        _z = 0 if seg_of_raw(i_) != seg_of_raw(OUT[n_ - 1]) else 1 - _z
    PUNCH[i_] = _z

_masks = {}
def masks(k):
    if k not in _masks:
        _masks[k] = (np.load(os.path.join(D, f"m{k}_person.npy"), mmap_mode="r"), np.load(os.path.join(D, f"m{k}_head.npy"), mmap_mode="r"))
    return _masks[k]

# ------------------------------------------------------------------ type
@functools.lru_cache(None)
def big(text, size, g0, g1, sf, ef, extrude=8):
    return to_np(text_layer(text, size, 800, gradient=(g0, g1), stroke=5, stroke_fill=sf, extrude=extrude, extrude_fill=ef, shadow=16, pad=80))
STAR = ((255, 248, 210), (255, 196, 90), (60, 30, 0), (110, 60, 10))
ICE = ((235, 250, 255), (120, 200, 255), (10, 40, 90), (20, 70, 140))
FIRE = ((255, 236, 110), (255, 90, 30), (90, 20, 0), (140, 40, 0))
NIGHT = ((230, 225, 255), (150, 130, 255), (20, 10, 60), (40, 25, 110))
RED = ((255, 150, 140), (240, 40, 40), (70, 0, 0), (120, 10, 10))
BLUE = ((170, 220, 255), (40, 120, 255), (0, 20, 70), (10, 40, 120))
PURP = ((235, 190, 255), (160, 70, 255), (40, 0, 70), (70, 20, 120))
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

# ------------------------------------------------------------------ animation helpers
TOPQ = []
def _popin(F, spr, x, y, u, t0, t1, s0=0.4, back=2.2, rot=0.0, alpha=1.0, fin=0.28, fout=0.22, bob=3, behind=False):
    w = R.window(u, t0, t1, fin, fout)
    if not w: return 0
    pin, a = w
    F.blit(spr, x, y + math.sin(u * 3 + x) * bob, alpha=a * alpha, scale=s0 + (1 - s0) * eob(pin, back), rot=rot, behind=behind)
    return a

def popin(F, spr, x, y, u, t0, t1, s0=0.4, back=2.2, rot=0.0, alpha=1.0, fin=0.28, fout=0.22, bob=3, behind=False, top=None):
    if top is None: top = (not behind) and y <= 640 and abs(x - 540) < 160
    if top:
        TOPQ.append((F, spr, x, y, u, t0, t1, dict(s0=s0, back=back, rot=rot, alpha=alpha, fin=fin, fout=fout, bob=bob))); return 1
    return _popin(F, spr, x, y, u, t0, t1, s0, back, rot, alpha, fin, fout, bob, behind)

def draw_tops(F):
    act = [q for q in TOPQ if R.window(q[4], q[5], q[6], q[7]["fin"], q[7]["fout"])]
    if not act: return
    best = max(act, key=lambda q: q[5])
    for (F_, spr, x, y, u, t0, t1, kw) in [q for q in act if q[6] == best[6] and q[5] >= best[5] - 1.2]:
        _popin(F_, spr, x, y, u, t0, t1, **kw)

@functools.lru_cache(None)
def kw_sprite(text, pal_i):
    pal = (STAR, ICE, FIRE, NIGHT)[pal_i]
    t = bigt(text, 76, pal)
    sc = min(1.0, 740 / t.shape[1])
    if sc < 1: t = cv2.resize(t, (int(t.shape[1] * sc), int(t.shape[0] * sc)), interpolation=cv2.INTER_AREA)
    return t

def kw(F, u, t0, t1, text, ic, side="L", pal=0, y=1215):
    w = R.window(u, t0, t1, 0.22, 0.18)
    if not w: return
    pin, a = w
    t = kw_sprite(text, pal); tw = t.shape[1]
    sgn = -1 if side == "L" else 1
    cx = 540 + sgn * 30 + (1 - eoc(clamp(pin * 1.6))) * 260 * sgn
    rot = -4 * sgn * (1 - 0.6 * eoc(clamp(pin * 2)))
    col = [(255, 190, 90), (90, 180, 255), (255, 120, 50), (160, 130, 255)][pal]
    F.blit(R.glow(min(420, max(300, int(tw * 0.9))), col), cx, y + 20, alpha=a * 0.32)
    F.blit(t, cx, y, alpha=a, scale=0.55 + 0.45 * eob(pin, 2.6), rot=rot)
    sw = clamp((u - t0 - 0.25) / 0.45)
    if 0 < sw < 1: F.blit(R.glow(150, (255, 255, 255)), cx - tw / 2 + tw * sw, y, alpha=a * 0.5 * math.sin(math.pi * sw))
    if ic:
        ix = cx + tw / 2 + 30 if cx + tw / 2 + 110 < W else cx - tw / 2 - 30
        ix = min(max(ix, 85), W - 85)
        pi_ = clamp((u - t0 - 0.08) / 0.3)
        if pi_ > 0:
            F.blit(icon_shadowed(ic, 140)[0], ix, y - 70 + math.sin(u * 4) * 8, alpha=a * eoc(pi_), scale=0.3 + 0.7 * eob(pi_, 3.0), rot=math.sin(u * 3) * 10)

@functools.lru_cache(None)
def streak(L, col):
    im = Image.new("RGBA", (L + 40, 40), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for k in range(L):
        a = int(255 * (k / L) ** 2)
        d.line((20 + k, 20, 21 + k, 20), fill=col + (a,), width=4)
    d.ellipse((L + 8, 12, L + 26, 28), fill=(255, 255, 255, 255))
    g = im.filter(ImageFilter.GaussianBlur(4)); gl = Image.new("RGBA", im.size, col + (0,)); gl.putalpha(g.split()[3])
    return to_np(Image.alpha_composite(gl, im))

def shooting(F, u, t0, n=3, cols=((255, 240, 200),)):
    """a few graphic shooting stars crossing the upper sky behind him"""
    for j in range(n):
        tt = (u - t0 - j * 0.45) % 2.4
        if tt > 0.7: continue
        q = tt / 0.7
        x = 1100 - q * 900 - j * 60; y = 260 + j * 110 + q * 260
        F.blit(streak(260, cols[j % len(cols)]), x, y, alpha=math.sin(math.pi * q), rot=-16, behind=True)

# ------------------------------------------------------------------ scenes (u = scene time in seconds)
def g1(F, u):
    popin(F, bigt("מטר המטאורים", 86, STAR), 540, 300, u, 0.0, 4.6, s0=0.3)
    popin(F, bigt("הכי צבעוני של השנה", 70, ICE), 540, 400, u, 0.3, 4.6, s0=0.3)
    popin(F, bigt("הג׳מינידים", 150, FIRE), 540, 340, u, 4.9, 6.5, s0=0.2, back=2.8)
    kw(F, u, 3.75, 4.7, "בדצמבר!", "calendar_3d.png", "R", 0)
    shooting(F, u, 0.2, 3)

def g2(F, u):
    popin(F, ltr("13-14.12", 150, (255, 248, 220), (255, 170, 60)), 540, 330, u, 0.3, 2.9, s0=0.3, back=2.6)
    kw(F, u, 0.8, 2.9, "לילה אחד בדצמבר", "calendar_3d.png", "L", 0)

def g3(F, u):
    popin(F, bigt("כדור הארץ חוצה שובל", 72, ICE), 540, 330, u, 0.2, 4.5, s0=0.3)
    kw(F, u, 1.6, 4.5, "שובל של אבק ושברי סלע", "sparkles_3d.png", "R", 1)

def g4(F, u):
    popin(F, bigt("בגודל של גרגיר חול", 76, STAR), 540, 330, u, 0.2, 2.1, s0=0.3)
    kw(F, u, 0.6, 2.1, "חלקיקים זעירים", "rock_3d.png", "L", 0)

def g5(F, u):
    popin(F, bigt("= כוכב נופל", 110, FIRE), 540, 340, u, 2.0, 4.0, s0=0.2, back=2.8)
    kw(F, u, 0.4, 2.0, "חודרים לאטמוספירה", "fire_3d.png", "R", 2)
    shooting(F, u, 1.8, 3, ((255, 220, 160), (255, 255, 255)))

def g6(F, u):
    popin(F, bigt("מה מיוחד בהם?", 96, NIGHT), 540, 330, u, 0.1, 3.0, s0=0.3)

def g7(F, u):
    popin(F, bigt("המקור!", 140, FIRE), 540, 340, u, 0.5, 2.5, s0=0.2, back=2.8)
    kw(F, u, 0.6, 2.5, "מאיפה הם מגיעים", "telescope_3d.png", "L", 3)

def g8(F, u):
    popin(F, bigt("פרסאידים: משביט", 80, ICE), 540, 330, u, 1.6, 5.4, s0=0.3)
    kw(F, u, 2.4, 5.4, "מטר הקיץ", "comet_3d.png", "R", 1)

def g9(F, u):
    popin(F, bigt("סלע + הרבה קרח", 90, ICE), 540, 330, u, 0.3, 3.0, s0=0.3)
    kw(F, u, 0.9, 3.0, "כוכב שביט", "ice_3d.png", "L", 1)

def g10(F, u):
    popin(F, bigt("אבל הג׳מינידים...", 84, FIRE), 540, 330, u, 0.1, 2.3, s0=0.3)

def g11(F, u):
    popin(F, bigt("אסטרואיד פייתון", 96, STAR), 540, 300, u, 0.5, 5.4, s0=0.3, back=2.6)
    popin(F, ltr("3200 Phaethon", 56, (230, 230, 240), (170, 170, 190)), 540, 405, u, 0.8, 5.4, s0=0.4)
    kw(F, u, 2.6, 5.4, "סלע ומתכת", "rock_3d.png", "R", 0)

def g12(F, u):
    popin(F, bigt("נראים אחרת!", 110, FIRE), 540, 340, u, 0.2, 2.8, s0=0.2, back=2.8)
    kw(F, u, 0.8, 2.8, "בחדירה לאטמוספירה", "fire_3d.png", "L", 2)

def g13(F, u):
    popin(F, bigt("צבעוניים במיוחד", 86, STAR), 540, 330, u, 0.2, 2.6, s0=0.3)
    popin(F, bigt("כדור אש!", 130, FIRE), 540, 340, u, 2.6, 4.3, s0=0.2, back=2.8)

def g14(F, u):
    ws = {w["w"].strip(",.?!"): w["s"] for w in SW["14"]}
    for word, pal, x, y in (("אדום", RED, 800, 330), ("כחול", BLUE, 540, 430), ("סגול", PURP, 280, 330)):
        t0 = ws.get(word, 0.5)
        _popin(F, bigt(word, 120, pal), x, y, u, t0, 3.5, s0=0.2, back=3.0)
    shooting(F, u, 0.2, 3, ((255, 80, 80), (80, 150, 255), (190, 90, 255)))

def g15(F, u):
    q = clamp((u - 0.9) / 1.6)
    if u > 0.8:
        n = int(round(100 * eoc(q)))
        _popin(F, ltr(f"{n}+", 190, (255, 248, 220), (255, 160, 50)), 540, 340, u, 0.8, 3.0, s0=0.4, back=2.4)
    kw(F, u, 1.4, 3.0, "מטאורים בשעה", "shooting_star_3d.png", "R", 0)
    shooting(F, u, 0.0, 4)

def g16(F, u):
    popin(F, bigt("כמעט בלי ירח", 96, NIGHT), 540, 330, u, 0.2, 2.9, s0=0.3)
    popin(F, bigt("תנאי צפייה מצוינים", 80, STAR), 540, 330, u, 3.0, 5.2, s0=0.3)
    kw(F, u, 1.6, 2.9, "שוקע מוקדם", "crescent_moon_3d.png", "R", 3)
    kw(F, u, 3.1, 5.2, "שמיים חשוכים", "glowing_star_3d.png", "L", 3)

def g17(F, u):
    popin(F, bigt("לא צריך שום ציוד", 86, ICE), 540, 330, u, 0.2, 2.2, s0=0.3)
    kw(F, u, 2.6, 5.1, "רק להתרחק מהעיר", "automobile_3d.png", "R", 1)

def g18(F, u):
    popin(F, bigt("מדבר = חושך טבעי", 86, STAR), 540, 330, u, 0.1, 3.4, s0=0.3)
    popin(F, bigt("ולהסתכל למעלה", 96, ICE), 540, 330, u, 3.5, 6.0, s0=0.3)
    kw(F, u, 0.5, 3.4, "אזורי המדבר", "desert_3d.png", "L", 0)
    shooting(F, u, 3.4, 3)

def g19(F, u):
    popin(F, bigt("אבל... זה דצמבר", 90, ICE), 540, 330, u, 0.1, 4.3, s0=0.3)
    kw(F, u, 1.8, 4.3, "בודקים תחזית", "cloud_with_lightning_3d.png", "R", 1)

def g20(F, u):
    popin(F, bigt("סערה = אין מופע", 92, NIGHT), 540, 330, u, 0.6, 3.9, s0=0.3)
    kw(F, u, 1.6, 3.9, "גשם שיחריב את האירוע", "cloud_with_lightning_and_rain_3d.png", "L", 1)

@functools.lru_cache(None)
def follow_card(followed):
    old = R.text_layer
    def patched(text, *a, **k):
        if text == "60.8K": text = "62.5K"
        return old(text, *a, **k)
    R.text_layer = patched
    try: return R.follow_card.__wrapped__(followed)
    finally: R.text_layer = old

def g21(F, u):
    popin(F, bigt("נתראה מתחת למטר", 84, STAR), 540, 330, u, 0.1, 9, s0=0.3)
    shooting(F, u, 0.0, 3)
    t0, tap = 0.15, 0.85
    if u > t0:
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

GFX = {k: globals()[f"g{k}"] for k in range(1, NS + 1)}

# ------------------------------------------------------------------ subtitles
SW = json.load(open(os.path.join(D, "segwords.json")))
def make_chunks():
    ws = []
    for k in range(1, NS + 1):
        for w in SW[str(k)]:
            t = w["w"]; core = t.strip(",.?!")
            ws.append({"t": core + ("?" if t.endswith("?") else ""), "s": START[k] / FPS + w["s"], "e": START[k] / FPS + min(w["e"], w["s"] + 0.9),
                       "brk": t.endswith((",", ".", "?")), "seg": k})
    res = []; cur = []
    for i, w in enumerate(ws):
        cur.append(w); L = sum(len(x["t"]) + 1 for x in cur)
        nxt = ws[i + 1] if i + 1 < len(ws) else None
        if nxt is None or nxt["seg"] != w["seg"] or w["brk"] or len(cur) >= 3 or L >= 15:
            res.append(cur); cur = []
    out = []
    for i, c in enumerate(res):
        t0 = c[0]["s"]; t1 = res[i + 1][0]["s"] if i + 1 < len(res) else NFR / FPS
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
    v = 1 - 0.30 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 1.7)) ** 2)
    g = np.clip(1 - 0.30 * np.clip((yy - 1330) / 590, 0, 1), 0, 1)
    return (np.clip(v, 0.62, 1) * g)[..., None]

def render(i, raw):
    t = i / FPS
    end = i >= START[NS] + NF[NS]
    if end:
        k, u = NS, (NF[NS] - 1) / FPS; ue = t - (START[NS] + NF[NS]) / FPS
    else:
        k, u = seg_of(i); ue = 0
    P, Hd = masks(k)
    jj = min(int(round(u * FPS)), len(P) - 1)
    img = raw.astype(np.float32)
    z = (1.08 if PUNCH.get(i, 0) else 1.0) + 0.06 * (u / max(DUR[k], 1)) + (0.03 * clamp(ue / END_HOLD) if end else 0)
    M = np.array([[z, 0, 540 - z * 540], [0, z, 760 - z * 760]], np.float32)
    img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    pm = cv2.warpAffine(cv2.resize(np.asarray(P[jj]), (W, H)).astype(np.float32) / 255, M, (W, H))
    pm = cv2.GaussianBlur(pm, (0, 0), 1.6)
    hm = cv2.warpAffine(cv2.resize(np.asarray(Hd[jj]), (W, H)), M, (W, H))
    hm = cv2.dilate((hm > 90).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    img *= vign()
    F = R.Frame(img, pm, hm)
    TOPQ.clear()
    GFX[k](F, u + (ue if end else 0))
    uf = (i - FIRST.get(k, i)) / FPS
    if k > 1 and 0 <= uf < 0.2 and not end:
        F.img += (255 - F.img) * (1 - uf / 0.2) * 0.8
    draw_tops(F)
    subtitles(F, t)
    np.clip(F.img, 0, 255, out=F.img)
    return F.img.astype(np.uint8), F.head_overlap

def frames_iter(o0, o1):
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
        print("NFR", NFR, "OUT", len(OUT), len(OUT) / FPS)
    else:
        run(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
