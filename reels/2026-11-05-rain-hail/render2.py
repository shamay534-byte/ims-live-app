# Reel #2: how rain forms & why hail falls on a hot day
import sys, os, json, math, random, subprocess, functools, importlib.util
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("R", os.path.join(HERE, "render.py"))
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
clamp, eoc, eio, eob = R.clamp, R.ease_out_cubic, R.ease_in_out, R.ease_out_back
text_layer, rounded, icon, icon_shadowed, to_np = R.text_layer, R.rounded, R.icon, R.icon_shadowed, R.to_np
A = R.A
D = os.path.join(HERE, "r2")
W, H, FPS = 1080, 1920, 30
INTRO = 8.3
VO_DUR = 65.43
DUR = INTRO + VO_DUR
NFR = int(round(DUR * FPS))

SP = np.load(os.path.join(D, "seed_person.npy"), mmap_mode="r")
SH = np.load(os.path.join(D, "seed_head.npy"), mmap_mode="r")

# ------------------------------------------------------------------ scenes (VO time v)
SCENES = [(0.0, 13.6, "bg1.jpg"), (13.6, 31.42, "bg2.jpg"), (31.42, 43.8, "bg3.jpg"), (43.8, 59.0, "bg4.jpg"),
          (59.0, 99, "OUTRO")]
XF = 0.4  # crossfade

@functools.lru_cache(None)
def bg_img(name):
    if name == "OUTRO":
        raw = subprocess.run(["ffmpeg", "-v", "error", "-sseof", "-0.1", "-i", os.path.join(D, "seed1080.mp4"),
                              "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        im = np.frombuffer(raw[:W * H * 3], np.uint8).reshape(H, W, 3)
        return cv2.resize(im, (1536, 2731), interpolation=cv2.INTER_CUBIC)
    return np.asarray(Image.open(os.path.join(D, "bg", name)).convert("RGB"))

DIM = {"bg1.jpg": 0.0, "bg2.jpg": 0.08, "bg3.jpg": 0.1, "bg4.jpg": 0.32, "OUTRO": 0.15}

@functools.lru_cache(None)
def outro_head_src():
    m = cv2.resize(np.asarray(SH[len(SH) - 1]), (1536, 2731))
    return m

def kb_matrix(ih, iw, u, seed):
    base = max(W / iw, H / ih); z = base * (1.0 + 0.10 * u)
    rnd = random.Random(seed); dx, dy = rnd.uniform(-30, 30), rnd.uniform(-40, 40)
    cx, cy = iw / 2 + dx * u / base, ih / 2 + dy * u / base
    return np.array([[z, 0, W / 2 - z * cx], [0, z, H / 2 - z * cy]], np.float32)

def outro_hm(v):
    s0 = SCENES[4][0]; u = (v - s0) / max(1, min(SCENES[4][1], VO_DUR) - s0)
    m = cv2.warpAffine(outro_head_src(), kb_matrix(2731, 1536, u, 4), (W, H))
    return cv2.dilate((m > 90).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0

def kenburns(name, u, seed):
    img = bg_img(name)
    ih, iw = img.shape[:2]
    base = max(W / iw, H / ih)
    z = base * (1.0 + 0.10 * u)
    rnd = random.Random(seed)
    dx, dy = rnd.uniform(-30, 30), rnd.uniform(-40, 40)
    cx, cy = iw / 2 + dx * u / base, ih / 2 + dy * u / base
    M = np.array([[z, 0, W / 2 - z * cx], [0, z, H / 2 - z * cy]], np.float32)
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    d = DIM[name]
    if d: out *= (1 - d)
    return out

@functools.lru_cache(None)
def vignette():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = 1 - 0.35 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 1.6)) ** 2)
    g = np.ones((H, W), np.float32)
    g *= np.clip(1 - 0.45 * np.clip((yy - 1250) / 670, 0, 1), 0, 1)   # darker bottom for subtitles
    g *= np.clip(1 - 0.25 * np.clip((420 - yy) / 420, 0, 1), 0, 1)
    return (np.clip(v, 0.6, 1) * g)[..., None]

def scene_bg(v):
    def one(k):
        s0, s1, nm = SCENES[k]
        return kenburns(nm, (v - s0) / max(1, min(s1, VO_DUR) - s0), k)
    k = max(i for i, s in enumerate(SCENES) if v >= s[0])
    img = one(k)
    if k + 1 < len(SCENES) and v > SCENES[k + 1][0] - XF:
        a = eio((v - (SCENES[k + 1][0] - XF)) / XF)
        img = img * (1 - a) + one(k + 1) * a
    return img * vignette()

# ------------------------------------------------------------------ helpers
class F2(R.Frame):
    pass

def lay(t, t0, t1, fin=0.3, fout=0.25):
    return R.window(t, t0, t1, fin, fout)

def pop(F, spr, x, y, t, t0, t1, s0=0.5, back=2.0, fin=0.3, fout=0.25, rot=0.0, bob=0.0, alpha=1.0):
    w = lay(t, t0, t1, fin, fout)
    if not w: return
    pin, a = w
    sc = s0 + (1 - s0) * eob(pin, back)
    F.blit(spr, x, y + math.sin(t * 2.2) * bob, alpha=a * alpha, scale=sc, rot=rot)

@functools.lru_cache(None)
def big_text(text, size, g0, g1, stroke=(12, 28, 70), ext=(16, 40, 110), extrude=8):
    return to_np(text_layer(text, size, 800, gradient=(g0, g1), stroke=5, stroke_fill=stroke, extrude=extrude,
                            extrude_fill=ext, shadow=14, pad=80))

@functools.lru_cache(None)
def plain(text, size, color=(255, 255, 255), weight=800, stroke=5, sf=(0, 0, 0)):
    return to_np(text_layer(text, size, weight, fill=color, stroke=stroke, stroke_fill=sf, shadow=10, pad=40))

@functools.lru_cache(None)
def soft_dot(r, color=(255, 255, 255), hard=0.0):
    s = int(r * 2 + 4)
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    d = np.hypot(xx - s / 2, yy - s / 2) / r
    a = np.clip(1 - d, 0, 1) ** (0.6 if hard else 1.6) * 255
    out = np.zeros((s, s, 4), np.float32); out[..., :3] = color; out[..., 3] = a
    return out

def layer():
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); return im, ImageDraw.Draw(im)

def blit_layer(F, im, alpha=1.0):
    arr = np.asarray(im).astype(np.float32)
    F.blit(arr, W / 2, H / 2, alpha=alpha)

def chip(F, t, t0, t1, text, ic, y=1250, x=540):
    spr = R.chip_sprite(text, ic)
    pop(F, spr, x, y, t, t0, t1, s0=0.6, back=2.2, bob=3)

@functools.lru_cache(None)
def droplet_sprite(r):
    s = int(r * 2 + 8); im = Image.new("RGBA", (s, s), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    c = s / 2
    d.ellipse((c - r, c - r, c + r, c + r), fill=(120, 190, 255, 150), outline=(220, 240, 255, 255), width=max(2, r // 10))
    d.ellipse((c - r * 0.55, c - r * 0.6, c - r * 0.15, c - r * 0.2), fill=(255, 255, 255, 200))
    return to_np(im.filter(ImageFilter.GaussianBlur(0.8)))

@functools.lru_cache(None)
def hail_sprite(r, layers):
    s = int(r * 2 + 10); im = Image.new("RGBA", (s, s), (0, 0, 0, 0)); d = ImageDraw.Draw(im); c = s / 2
    for k in range(layers, 0, -1):
        rr = r * k / layers
        col = (235, 245, 255, 245) if k % 2 else (175, 205, 235, 245)
        d.ellipse((c - rr, c - rr, c + rr, c + rr), fill=col, outline=(130, 170, 215, 255), width=2)
    d.ellipse((c - r * 0.5, c - r * 0.62, c - r * 0.1, c - r * 0.25), fill=(255, 255, 255, 170))
    return to_np(im)

# ------------------------------------------------------------------ data
rng = random.Random(3)
VAPOR = [(rng.uniform(140, 940), rng.uniform(0, 1), rng.uniform(0.5, 1.0), rng.uniform(8, 20)) for _ in range(70)]
MOLS = [(rng.uniform(0, 2 * math.pi), rng.uniform(300, 420), rng.uniform(0, 1.2)) for _ in range(46)]
FLOAT = [(rng.uniform(120, 960), rng.uniform(420, 1300), rng.uniform(0, 6.28), rng.randint(26, 54)) for _ in range(16)]

def cloud_targets(n):
    a = icon("cloud_3d.png", 420)[..., 3]
    ys, xs = np.where(a > 160)
    r = random.Random(5); idx = [r.randrange(len(xs)) for _ in range(n)]
    return [(540 - 210 + xs[i], 760 - 210 + ys[i]) for i in idx]
CT = cloud_targets(260)
CS = [(rng.uniform(80, 1000), rng.uniform(300, 1300)) for _ in range(260)]

# ------------------------------------------------------------------ subtitles
FIXW = {"כשלוהד": "כשלוהט", "מתאבים": "מתעבים", "לכבישי": "לגבישי", "זהירים": "זעירים", "לברת": "לברד",
        "50": "50", "10": "10"}
def words():
    ws = []
    for s in json.load(open(os.path.join(D, "tr_video.json"))):
        for w in s["words"]:
            ws.append((w["w"], w["s"], w["e"]))
    prev = None
    for s in json.load(open(os.path.join(D, "tr_vocut.json"))):
        for w in s["words"]:
            tx = w["w"].strip()
            if tx == "שעות" and prev == "שעות": continue
            prev = tx
            ws.append((w["w"], w["s"] + INTRO, w["e"] + INTRO))
    out = []
    for txt, s, e in ws:
        txt = txt.strip(); brk = txt.endswith((",", ".", "?", "!"))
        core = txt.strip(",.?!")
        core = FIXW.get(core, core)
        if core == "50": core = "50"
        out.append({"t": core + ("?" if txt.endswith("?") else ""), "s": s, "e": min(e, s + 1.0), "brk": brk})
    return out

def make_chunks():
    ws = words(); res = []; cur = []
    for i, w in enumerate(ws):
        cur.append(w)
        gap = ws[i + 1]["s"] - w["e"] if i + 1 < len(ws) else 9
        L = sum(len(x["t"]) + 1 for x in cur)
        cross = i + 1 < len(ws) and w["s"] < INTRO <= ws[i + 1]["s"]
        if cross or w["brk"] or len(cur) >= 4 or L >= 17 or gap > 0.35:
            res.append(cur); cur = []
    if cur: res.append(cur)
    out = []
    for i, c in enumerate(res):
        t0 = c[0]["s"]; t1 = res[i + 1][0]["s"] if i + 1 < len(res) else DUR
        out.append((t0, min(t1, c[-1]["e"] + 0.5), c))
    return out
CHUNKS = make_chunks()

def subtitles(F, t):
    for (t0, t1, ws) in CHUNKS:
        if t0 <= t < t1:
            p = clamp((t - t0) / 0.12)
            sprs = [R.sub_word(w["t"], w["s"] <= t < (ws[j + 1]["s"] if j + 1 < len(ws) else w["e"] + 0.2))
                    for j, w in enumerate(ws)]
            gap = -30
            total = sum(s.shape[1] for s in sprs) + gap * (len(sprs) - 1)
            sc = min(1.0, 860 / total) * (0.9 + 0.1 * eob(p, 2.5))
            x = 540 + total * sc / 2
            for s in sprs:
                ww = s.shape[1] * sc
                F.blit(s, x - ww / 2, 1470, alpha=eoc(p), scale=sc)
                x -= ww + gap * sc
            return

# ------------------------------------------------------------------ intro overlay
def intro(F, t):
    # title
    t1 = 6.65
    if t < t1 + 0.35:
        out = clamp((t - t1) / 0.3); oa = 1 - eio(out)
        l1 = big_text("איך נוצר גשם", 116, (255, 255, 255), (205, 228, 255))
        l2 = big_text("ולמה ברד יכול לרדת ביום שרב?", 70, (255, 236, 120), (255, 160, 40),
                      stroke=(70, 30, 0), ext=(120, 55, 0), extrude=6)
        for spr, t0, cy in [(l1, 0.15, 368), (l2, 0.55, 492)]:
            p = clamp((t - t0) / 0.4)
            if p <= 0: continue
            F.blit(spr, 540, round(cy - (1 - eoc(p)) * 40 - eio(out) * 50), alpha=eoc(p) * oa,
                   scale=0.4 + 0.6 * eob(p, 1.3))
    # "פינת העשרה" badge
    if t >= 6.8:
        badge = badge_sprite()
        p = clamp((t - 6.8) / 0.4)
        F.blit(badge, 540, 420, alpha=eoc(p), scale=0.3 + 0.7 * eob(p, 2.2))
        for k in range(6):
            q = clamp((t - 6.85 - k * 0.04) / 0.8)
            if 0 < q < 1:
                ang = k * math.pi / 3 + 0.4
                F.blit(icon("sparkles_3d.png", 70), 540 + math.cos(ang) * (230 + 120 * q),
                       420 + math.sin(ang) * (70 + 60 * q), alpha=1 - q, scale=0.6 + 0.4 * q)

@functools.lru_cache(None)
def badge_sprite():
    tl = text_layer("פינת העשרה", 78, 800, fill=(255, 255, 255), pad=0)
    w, h = tl.size[0] + 210, 130
    base = rounded(w, h, 65, (255, 255, 255), grad=((40, 200, 255), (110, 70, 255)),
                   border=((255, 255, 255, 240), 4), shadow=16)
    pad = (base.size[0] - w) // 2
    base.alpha_composite(tl, (pad + 50, pad + (h - tl.size[1]) // 2 - 4))
    ic = Image.open(A("test_tube_3d.png")).convert("RGBA").resize((150, 150), Image.LANCZOS)
    can = Image.new("RGBA", (base.size[0] + 40, base.size[1] + 40), (0, 0, 0, 0))
    can.alpha_composite(base, (0, 20)); can.alpha_composite(ic, (pad + w - 150 + 30, 20 + pad + (h - 150) // 2))
    return to_np(can)

# ------------------------------------------------------------------ scene graphics
def scene1(F, v, t):
    # sun + vapor (evaporation)
    w = lay(v, 0.0, 5.3, 0.5, 0.4)
    if w:
        pin, a = w
        F.blit(R.glow(460, (255, 220, 120)), 860, 430, alpha=a * 0.8)
        F.blit(R.rays(420), 860, 430, alpha=a, rot=t * 12)
        F.blit(icon_shadowed("sun_3d.png", 210)[0], 860, 430, alpha=a, scale=0.6 + 0.4 * eob(pin))
        for (x0, ph, sp, r) in VAPOR:
            q = ((v * 0.22 * sp + ph) % 1.0)
            y = 1380 - q * 760
            x = x0 + math.sin(q * 9 + ph * 6) * 26
            F.blit(soft_dot(r * (1 + q)), x, y, alpha=a * math.sin(q * math.pi) * 0.75)
    chip(F, v, 0.6, 2.9, "אידוי מהים ומהיבשה", "droplet_3d.png")
    # altitude / cooling gauge
    w = lay(v, 2.95, 5.3, 0.35, 0.3)
    if w:
        pin, a = w
        im, d = layer()
        x0, yb, yt = 120, 1180, 430
        for k in range(60):
            y = yb - (yb - yt) * k / 60; c = (int(255 - 190 * k / 60), int(120 + 60 * k / 60), int(70 + 185 * k / 60), 255)
            d.rectangle((x0 - 16, y - (yb - yt) / 60, x0 + 16, y), fill=c)
        d.rounded_rectangle((x0 - 18, yt - 18, x0 + 18, yb + 18), 18, outline=(255, 255, 255, 230), width=4)
        for km in range(4):
            y = yb - (yb - yt) * km / 3
            d.line((x0 + 22, y, x0 + 46, y), fill=(255, 255, 255, 230), width=4)
        blit_layer(F, im, a)
        for km in range(4):
            y = yb - (yb - yt) * km / 3
            F.blit(plain(f"{km} ק״מ", 42, weight=800, stroke=5), x0 + 120, y, alpha=a)
        q = eio((v - 3.1) / 1.9)
        y = yb - (yb - yt) * q
        temp = int(round(25 - 19.5 * q * 3 / 3))
        F.blit(soft_dot(30, (255, 255, 255), 1), x0, y, alpha=a)
        F.blit(plain(f"{temp}°", 60, color=(255, 255, 255)), x0 + 230, y, alpha=a)
    chip(F, v, 3.0, 5.3, "עולה למעלה ומתקרר", "snowflake_3d.png")
    # condensation magnifier
    w = lay(v, 5.4, 10.9, 0.4, 0.35)
    if w:
        pin, a = w
        sc = 0.5 + 0.5 * eob(pin, 1.6)
        im, d = layer()
        cx, cy, r = 540, 780, 280
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(10, 30, 70, 175), outline=(255, 255, 255, 255), width=8)
        nuc = clamp((v - 6.2) / 0.5)
        if nuc > 0:   # condensation nucleus: dust grain + salt crystal
            pts = [(cx + 56 * math.cos(k * 0.9) * (1 + 0.25 * math.sin(k * 2.3)), cy + 50 * math.sin(k * 0.9) * (1 + 0.2 * math.cos(k * 1.7))) for k in range(7)]
            d.polygon([(cx + (px - cx) * nuc, cy + (py - cy) * nuc) for px, py in pts], fill=(160, 110, 70, 255))
        blit_layer(F, im, a * 1.0)
        F.blit(icon("magnifying_glass_tilted_left_3d.png", 150), cx + 220, cy - 220, alpha=a, scale=sc)
        # water molecules converge to the nucleus
        if v > 6.4:
            for (ang, rad, dl) in MOLS:
                q = clamp((v - 6.5 - dl) / 2.6)
                if q <= 0: continue
                rr = rad * (1 - eio(q)) * 0.62 + 40
                x, y = cx + math.cos(ang + v * 0.3) * rr, cy + math.sin(ang + v * 0.3) * rr
                if q < 1: F.blit(soft_dot(15, (120, 200, 255), 1), x, y, alpha=a * min(1, q * 4) * (1 - max(0, (v - 9.3) / 0.6)))
        g = clamp((v - 9.3) / 1.4)
        if g > 0:
            F.blit(droplet_sprite(140), cx, cy, alpha=a * eoc(g), scale=0.15 + 0.85 * eob(g, 1.4))
        F.blit(big_text("גרעין התעבות", 76, (255, 255, 255), (200, 230, 255)), 540, 420, alpha=a * eoc(clamp((v - 5.9) / 0.4)))
    chip(F, v, 7.85, 10.9, "אבק", "rock_3d.png", y=1170, x=300)
    chip(F, v, 8.42, 10.9, "מלח מהים", "water_wave_3d.png", y=1170, x=760)
    # billions -> cloud
    if 10.9 < v < 13.8:
        q = eio((v - 11.0) / 1.4)
        fade = 1 - clamp((v - 13.35) / 0.3)
        for (sx, sy), (tx, ty) in zip(CS, CT):
            F.blit(soft_dot(7, (255, 255, 255), 1), sx + (tx - sx) * q, sy + (ty - sy) * q, alpha=0.85 * fade * (1 - clamp((v - 12.4) / 0.5)))
        c = clamp((v - 12.2) / 0.5)
        if c > 0: F.blit(icon_shadowed("cloud_3d.png", 420)[0], 540, 760, alpha=c * fade, scale=0.9 + 0.1 * eob(c))
        F.blit(big_text("מיליארדי טיפונת", 92, (255, 255, 255), (210, 232, 255)), 540, 400,
               alpha=eoc(clamp((v - 11.0) / 0.35)) * fade, scale=0.6 + 0.4 * eob(clamp((v - 11.0) / 0.35)))
        F.blit(big_text("= ענן", 96, (255, 236, 120), (255, 170, 40), (70, 30, 0), (120, 55, 0)), 540, 1110,
               alpha=eoc(clamp((v - 12.45) / 0.3)) * fade, scale=0.5 + 0.5 * eob(clamp((v - 12.45) / 0.3), 2.4))

def fmt_int(n): return f"{n:,}"

def scene2(F, v, t):
    # floating tiny droplets
    w = lay(v, 13.9, 17.45, 0.5, 0.3)
    if w:
        pin, a = w
        for (x, y, ph, r) in FLOAT:
            F.blit(droplet_sprite(r // 2 + 6), x + math.sin(v * 0.9 + ph) * 22, y + math.cos(v * 0.7 + ph) * 18, alpha=a * 0.95)
    chip(F, v, 14.4, 17.4, "קוטר טיפונת: ~0.02 מ״מ", "microscope_3d.png")
    # size comparison x1,000,000
    w = lay(v, 17.5, 19.6, 0.35, 0.3)
    if w:
        pin, a = w
        F.blit(droplet_sprite(10), 260, 820, alpha=a)
        F.blit(plain("טיפת ענן", 44), 260, 930, alpha=a)
        g = eob(clamp((v - 17.7) / 0.9), 1.4)
        F.blit(icon_shadowed("droplet_3d.png", 300)[0], 760, 790, alpha=a, scale=0.1 + 0.9 * g)
        F.blit(plain("טיפת גשם", 44), 760, 1000, alpha=a)
        q = clamp((v - 17.7) / 1.5)
        n = int(round(10 ** (6 * eio(q))))
        n = 1000000 if q >= 1 else n
        F.blit(big_text(f"×{fmt_int(n)}", 104, (255, 236, 120), (255, 160, 40), (70, 30, 0), (120, 55, 0)), 540, 470,
               alpha=a, scale=0.6 + 0.4 * eob(clamp((v - 17.6) / 0.3)))
        F.blit(plain("בנפח!", 46, color=(255, 236, 160)), 540, 590, alpha=a * eoc(clamp((v - 19.0) / 0.25)))
    # collision & coalescence
    w = lay(v, 19.62, 21.0, 0.2, 0.3)
    if w:
        pin, a = w
        q = clamp((v - 19.65) / 0.55)
        if q < 1:
            off = 330 * (1 - eio(q))
            F.blit(icon("droplet_3d.png", 130), 540 - off, 760, alpha=a)
            F.blit(icon("droplet_3d.png", 130), 540 + off, 760, alpha=a)
        else:
            F.blit(icon_shadowed("droplet_3d.png", 190)[0], 540, 760, alpha=a, scale=0.7 + 0.3 * eob(clamp((v - 20.2) / 0.3), 2.5))
            b = clamp((v - 20.2) / 0.5)
            if b < 1: F.blit(icon("collision_3d.png", 300), 540, 740, alpha=(1 - b), scale=0.6 + 0.6 * b)
        F.blit(big_text("מתנגשות ומתאחדות", 78, (255, 255, 255), (200, 230, 255)), 540, 440, alpha=a)
    # ice crystals in the upper cloud
    w = lay(v, 21.0, 26.1, 0.4, 0.3)
    if w:
        pin, a = w
        im, d = layer()
        d.rounded_rectangle((70, 300, 1010, 760), 40, fill=(160, 210, 255, 55), outline=(220, 240, 255, 200), width=4)
        blit_layer(F, im, a)
        F.blit(plain("החלק העליון של הענן · מתחת ל־0°", 40, color=(225, 242, 255)), 540, 340, alpha=a)
        for k, (x, y, s) in enumerate([(220, 520, 120), (420, 610, 90), (620, 500, 140), (840, 600, 100), (330, 700, 70), (760, 700, 80)]):
            g = clamp((v - 23.0 - k * 0.18) / 0.5)
            if g <= 0: continue
            grow = 1 + 0.35 * clamp((v - 24.9) / 1.0)
            F.blit(icon("snowflake_3d.png", s), x, y + math.sin(v * 1.5 + k) * 8, alpha=a * eoc(g),
                   scale=(0.3 + 0.7 * eob(g, 2)) * grow, rot=v * 25 * (1 if k % 2 else -1))
    chip(F, v, 24.35, 26.1, "גבישי קרח שגדלים מהר", "ice_3d.png", y=1060)
    # falling through the 0°C line -> rain
    w = lay(v, 26.1, 31.5, 0.3, 0.3)
    if w:
        pin, a = w
        im, d = layer()
        for x in range(90, 990, 44): d.line((x, 900, x + 26, 900), fill=(255, 255, 255, 235), width=5)
        blit_layer(F, im, a)
        F.blit(plain("0°C", 54, color=(255, 255, 255)), 950, 850, alpha=a)
        F.blit(plain("קרח ❄", 44, color=(220, 240, 255)) if False else plain("קרח", 44, color=(220, 240, 255)), 130, 850, alpha=a)
        F.blit(plain("מים", 44, color=(180, 220, 255)), 130, 950, alpha=a)
        q = clamp((v - 26.6) / 2.6)
        y = 420 + 840 * eio(q)
        melt = clamp((y - 860) / 90)
        if melt < 1: F.blit(icon("snowflake_3d.png", 140), 540, y, alpha=a * (1 - melt), rot=v * 60)
        if melt > 0: F.blit(icon("droplet_3d.png", 140), 540, y, alpha=a * melt)
    chip(F, v, 28.2, 29.6, "הקרח נמס בדרך", "thermometer_3d.png", y=1180)
    if v > 29.5 and v < 31.6:
        q = clamp((v - 29.5) / 0.3); fade = 1 - clamp((v - 31.2) / 0.3)
        rain(F, v, 0.9 * q * fade)
        F.blit(big_text("גשם!", 130, (255, 255, 255), (170, 215, 255)), 540, 560, alpha=eoc(q) * fade,
               scale=0.4 + 0.6 * eob(q, 2.4))
        F.blit(icon_shadowed("cloud_with_rain_3d.png", 300)[0], 540, 380 + 0 * q, alpha=0)

DROPS = [(rng.uniform(-100, W), rng.uniform(0, H), rng.uniform(1600, 2300), rng.uniform(50, 100)) for _ in range(160)]
def rain(F, v, amt):
    if amt <= 0.01: return
    lay_ = np.zeros((H, W), np.uint8)
    for (x0, y0, sp, ln) in DROPS[: int(len(DROPS) * amt)]:
        y = (y0 + sp * v) % (H + 200) - 100; x = x0 + 0.15 * y
        cv2.line(lay_, (int(x), int(y)), (int(x + 0.15 * ln), int(y + ln)), 200, 2, cv2.LINE_AA)
    a = (lay_.astype(np.float32) / 255 * 0.6 * amt)[..., None]
    F.img += (np.array([230, 240, 255], np.float32) - F.img) * a

@functools.lru_cache(None)
def chevron(color):
    im = Image.new("RGBA", (120, 90), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.line((10, 80, 60, 20, 110, 80), fill=color, width=20, joint="curve")
    return to_np(im.filter(ImageFilter.GaussianBlur(1)))

def scene3(F, v, t):
    # "ברד" title
    w = lay(v, 31.5, 33.55, 0.3, 0.3)
    if w:
        pin, a = w
        F.blit(icon_shadowed("ice_3d.png", 300)[0], 540, 820, alpha=a, scale=0.4 + 0.6 * eob(pin, 2.2), rot=math.sin(v * 3) * 6)
        F.blit(big_text("ועכשיו... ברד!", 112, (230, 245, 255), (140, 200, 255)), 540, 470, alpha=a, scale=0.5 + 0.5 * eob(pin, 2.0))
    # altitude ruler: hot ground vs -50 aloft
    w = lay(v, 32.7, 38.1, 0.4, 0.3)
    if w:
        pin, a = w
        x0, yb, yt = 130, 1250, 420
        g = eio(clamp((v - 32.8) / 1.6))
        ytop = yb - (yb - yt) * g
        im, d = layer()
        n = 70
        for k in range(n):
            if yb - (yb - yt) * k / n < ytop: break
            y = yb - (yb - yt) * k / n
            c = (int(255 - 200 * k / n), int(110 + 90 * k / n), int(40 + 215 * k / n), 255)
            d.rectangle((x0 - 18, y - (yb - yt) / n - 1, x0 + 18, y), fill=c)
        d.rounded_rectangle((x0 - 20, ytop - 20, x0 + 20, yb + 20), 20, outline=(255, 255, 255, 235), width=4)
        blit_layer(F, im, a)
        pop(F, plain("+35°C", 64, color=(255, 200, 120)), x0 + 160, yb - 10, v, 32.9, 38.1)
        pop(F, icon("fire_3d.png", 90), x0 + 300, yb - 14, v, 32.95, 38.1)
        pop(F, plain("10 ק״מ", 56), x0 + 150, yt + 40, v, 33.8, 38.1)
        pop(F, plain("‎-50°C", 72, color=(170, 220, 255)), x0 + 170, yt + 120, v, 36.8, 38.1, back=2.6)
        pop(F, icon("snowflake_3d.png", 100), x0 + 350, yt + 110, v, 36.85, 38.1, rot=v * 30)
        pop(F, plain("0 ק״מ", 44, weight=700), x0 + 130, yb + 70, v, 32.9, 38.1)
    # updrafts
    w = lay(v, 38.06, 43.7, 0.4, 0.3)
    if w:
        pin, a = w
        for col, x in enumerate([380, 560, 740]):
            for k in range(6):
                q = ((v * 0.55 + k / 6 + col * 0.13) % 1.0)
                y = 1300 - q * 900
                F.blit(chevron((255, 255, 255, 255)), x, y, alpha=a * math.sin(q * math.pi) * 0.95, scale=1.0 + 0.3 * (col == 1))
        chip_spr = R.chip_sprite("זרמי אוויר עולים חזקים", "dashing_away_3d.png")
        pop(F, chip_spr, 540, 380, v, 38.3, 43.7, s0=0.6, back=2.2)
    w = lay(v, 41.3, 43.75, 0.35, 0.3)
    if w:
        pin, a = w
        im, d = layer()
        for k in range(5):
            pts = [(x, 1330 - k * 26 + math.sin(x / 50 + v * 6 + k) * 10) for x in range(80, 1010, 12)]
            d.line(pts, fill=(255, 150, 60, 200 - k * 30), width=6)
        blit_layer(F, im.filter(ImageFilter.GaussianBlur(2)), a)
        chip(F, v, 41.4, 43.7, "החום בקרקע מחזק אותם", "fire_3d.png", y=1190)

def scene4(F, v, t):
    # hailstone loops up & down, growing a new ice layer every lap
    w = lay(v, 43.85, 49.95, 0.4, 0.3)
    if w:
        pin, a = w
        cx, cy, rx, ry = 390, 800, 200, 330
        im, d = layer()
        for k in range(0, 360, 8):
            th0, th1 = math.radians(k), math.radians(k + 4)
            d.line((cx + rx * math.cos(th0), cy + ry * math.sin(th0), cx + rx * math.cos(th1), cy + ry * math.sin(th1)),
                   fill=(255, 255, 255, 220), width=6)
        blit_layer(F, im, a)
        F.blit(icon("up_arrow_3d.png", 90), cx + rx + 60, cy, alpha=a)
        F.blit(icon("down_arrow_3d.png", 90), cx - rx - 60, cy, alpha=a)
        laps = clamp((v - 44.4) / 5.2) * 3
        th = math.pi / 2 - laps * 2 * math.pi   # start bottom, go up the right side
        layers = 1 + int(laps)
        r = 26 + 14 * laps
        F.blit(hail_sprite(int(r), layers), cx + rx * math.cos(th), cy + ry * math.sin(th), alpha=a)
        # cross-section
        F.blit(hail_sprite(130, max(1, layers + (1 if v > 48.6 else 0))), 830, 640, alpha=a,
               scale=0.4 + 0.6 * eob(clamp((v - 46.4) / 0.4), 1.6) if v > 46.4 else 0.01)
        if v > 46.4:
            F.blit(plain("חתך של אבן ברד", 40), 830, 830, alpha=a * eoc(clamp((v - 46.6) / 0.3)))
        F.blit(big_text("שוב ושוב", 90, (255, 255, 255), (200, 230, 255)), 540, 380, alpha=a)
    pop(F, icon_shadowed("onion_3d.png", 150)[0], 830, 960, v, 49.2, 49.95, back=2.6)
    chip(F, v, 47.4, 49.9, "עוד שכבת קרח בכל סיבוב", "ice_3d.png", y=1190)
    # falling fast: speedometer
    w = lay(v, 49.95, 53.95, 0.35, 0.3)
    if w:
        pin, a = w
        cx, cy, r = 540, 860, 250
        im, d = layer()
        d.arc((cx - r, cy - r, cx + r, cy + r), 150, 390, fill=(255, 255, 255, 120), width=34)
        q = eio(clamp((v - 51.4) / 1.4))
        d.arc((cx - r, cy - r, cx + r, cy + r), 150, 150 + 240 * (0.05 + 0.75 * q), fill=(255, 120, 60, 255), width=34)
        ang = math.radians(150 + 240 * (0.05 + 0.75 * q))
        d.line((cx, cy, cx + (r - 50) * math.cos(ang), cy + (r - 50) * math.sin(ang)), fill=(255, 255, 255, 255), width=12)
        d.ellipse((cx - 22, cy - 22, cx + 22, cy + 22), fill=(255, 255, 255, 255))
        blit_layer(F, im, a)
        F.blit(plain("קמ״ש", 44), cx, cy + 110, alpha=a)
        F.blit(big_text("עשרות קמ״ש!", 96, (255, 236, 120), (255, 150, 40), (70, 30, 0), (120, 55, 0)), 540, 420,
               alpha=a * eoc(clamp((v - 52.5) / 0.3)), scale=0.5 + 0.5 * eob(clamp((v - 52.5) / 0.3), 2.4))
        # falling hailstone with streak
        fy = ((v - 49.95) * 900) % 1300 + 200
        for k in range(8):
            F.blit(soft_dot(18 - k * 2, (230, 245, 255)), 900, fy - k * 26, alpha=a * (0.5 - k * 0.06))
        F.blit(hail_sprite(34, 4), 900, fy, alpha=a)
    # no time to melt -> impact
    w = lay(v, 53.95, 58.5, 0.35, 0.2)
    if w:
        pin, a = w
        F.blit(icon_shadowed("stopwatch_3d.png", 220)[0], 540, 640, alpha=a, scale=0.5 + 0.5 * eob(pin, 2), rot=math.sin(v * 14) * 4)
        F.blit(big_text("אין זמן להפשיר", 96, (255, 255, 255), (200, 230, 255)), 540, 400, alpha=a)
        q = clamp((v - 56.6) / 1.8)
        y = 820 + 450 * (q ** 2)
        if q < 1:
            F.blit(hail_sprite(44, 5), 540, y, alpha=a * clamp((v - 56.6) / 0.2))
        b = clamp((v - 58.4) / 0.5)
        if 0 < b < 1:
            F.blit(icon("collision_3d.png", 260), 540, 1270, alpha=1 - b, scale=0.5 + 0.8 * b)

def scene5(F, v, t):
    rain(F, v, 0.45)
    w = lay(v, 60.9, 62.15, 0.3, 0.25)
    if w:
        pin, a = w
        F.blit(big_text("עכשיו אתם יודעים!", 92, (255, 236, 120), (255, 160, 40), (70, 30, 0), (120, 55, 0)), 540, 450,
               alpha=a, scale=0.5 + 0.5 * eob(pin, 2.2))
        F.blit(icon("light_bulb_3d.png", 120), 900, 330, alpha=a, scale=0.5 + 0.5 * eob(pin, 2.5), rot=12)
    if v >= 62.2:
        follow(F, t, INTRO + 62.2, INTRO + 63.15)

@functools.lru_cache(None)
def follow_card(followed):
    R_fc = R.follow_card.__wrapped__(followed)   # reuse layout
    return R_fc

def follow(F, t, t0, tap):
    p = clamp((t - t0) / 0.45)
    spr, (bx, by) = follow_card_609(t >= tap + 0.12)
    press = 1 - 0.06 * max(0, 1 - abs(t - tap - 0.05) / 0.12)
    cy = 1262 + (1 - eob(p, 1.7)) * 260
    F.blit(spr, 540, cy, alpha=eoc(p), scale=press)
    if tap - 0.9 < t < tap + 0.9:
        hp = clamp((t - (tap - 0.9)) / 0.8)
        hy = cy + by + 70 + (1 - eoc(hp)) * 220 + (8 if abs(t - tap) < 0.1 else 0)
        ha = clamp((t - (tap - 0.9)) / 0.2) * (1 - clamp((t - tap - 0.6) / 0.3))
        F.blit(icon_shadowed("backhand_index_pointing_up_3d_default.png", 120, 8, 8, 0.4)[0], 540 + bx + 30, hy, alpha=ha)
    if tap < t < tap + 1.2:
        q = (t - tap) / 1.2
        for k in range(5):
            F.blit(icon("red_heart_3d.png", 56), 540 + bx - 70 + (k - 2) * 26 - q * 110, cy - 40 - q * (40 + k * 14),
                   alpha=(1 - q), scale=0.6 + 0.6 * q, rot=(k - 2) * 12)

@functools.lru_cache(None)
def follow_card_609(followed):
    # same card as reel #1, with 60.9K followers
    src = R.follow_card.__wrapped__
    old = R.text_layer
    def patched(text, *a, **k):
        if text == "60.8K": text = "60.9K"
        return old(text, *a, **k)
    R.text_layer = patched
    try:
        return src(followed)
    finally:
        R.text_layer = old

# ------------------------------------------------------------------ frame
def render(i, raw):
    t = i / FPS
    if t < INTRO:
        img = raw.astype(np.float32)
        k = min(i, len(SP) - 1)
        pm = cv2.resize(np.asarray(SP[k]), (W, H)).astype(np.float32) / 255
        hm = cv2.resize(np.asarray(SH[k]), (W, H))
        hm = cv2.dilate((hm > 90).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
        # crossfade into scene 1
    else:
        img = scene_bg(t - INTRO)
        pm = np.zeros((H, W), np.float32)
        hm = outro_hm(t - INTRO) if t - INTRO > 59.0 else np.zeros((H, W), bool)
    F = R.Frame(img, pm, hm)
    if t < INTRO:
        intro(F, t)
        if t > INTRO - XF:   # blend start of scene 1 over the intro
            a = eio((t - (INTRO - XF)) / XF)
            F.img[:] = F.img * (1 - a) + scene_bg(0.0) * a
    else:
        v = t - INTRO
        if v < 13.9: scene1(F, v, t)
        if 13.4 < v < 31.8: scene2(F, v, t)
        if 31.2 < v < 44.2: scene3(F, v, t)
        if 43.6 < v < 59.4: scene4(F, v, t)
        if v > 58.8: scene5(F, v, t)
    subtitles(F, t)
    np.clip(F.img, 0, 255, out=F.img)
    return F.img.astype(np.uint8), F.head_overlap

def run(o0, o1, path):
    dec = None
    if o0 < int(INTRO * FPS):
        dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{o0 / FPS:.4f}", "-i", os.path.join(D, "seed1080.mp4"),
                                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", path],
                           stdin=subprocess.PIPE)
    ov = []; last = None
    for i in range(o0, o1):
        raw = None
        if i < int(INTRO * FPS):
            b = dec.stdout.read(W * H * 3)
            raw = np.frombuffer(b, np.uint8).reshape(H, W, 3) if len(b) == W * H * 3 else last
            last = raw
        out, o = render(i, raw)
        if o: ov.append((i, o))
        enc.stdin.write(out.tobytes())
    enc.stdin.close(); enc.wait()
    print(f"{o0}-{o1} done, head-overlap frames {len(ov)} {ov[:4]}", flush=True)

if __name__ == "__main__":
    if sys.argv[1] == "still":
        outs = []
        for ts in sys.argv[3:]:
            tt = float(ts); i = int(tt * FPS); raw = None
            if tt < INTRO:
                b = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{tt:.3f}", "-i", os.path.join(D, "seed1080.mp4"), "-frames:v", "1",
                                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
                raw = np.frombuffer(b, np.uint8).reshape(H, W, 3)
            o, ov = render(i, raw); print(ts, "overlap", ov)
            outs.append(Image.fromarray(o).resize((360, 640), Image.LANCZOS))
        c = Image.new("RGB", (360 * len(outs), 640))
        for k, im in enumerate(outs): c.paste(im, (360 * k, 0))
        c.save(sys.argv[2], quality=90)
    else:
        run(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
