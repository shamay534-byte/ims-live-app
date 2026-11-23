# Section 1 only, full quality: 1080p Seedance plate, camera moves, real-looking meteors behind him, magical title
import sys, os, math, subprocess, functools
import numpy as np, cv2
sys.argv_saved = sys.argv; sys.argv = ["x", "info"]
import importlib.util
spec = importlib.util.spec_from_file_location("m", os.path.join(os.path.dirname(os.path.abspath(__file__)), "render12.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sys.argv = sys.argv_saved
import gemtitle4 as G2
R = m.R; D = m.D; W, H, FPS = 1080, 1920, 30
clamp, eoc = m.clamp, m.eoc
def eio(x): x = clamp(x); return x * x * (3 - 2 * x)

PI = 0
CA = m.CLIPS[1][0]
TEND = m.PEND(PI)
TAIL = 0.6                              # tiny hold so the last word breathes
NF = int(round((TEND + TAIL) * FPS))
T = lambda s: m.T(s, PI)

P = np.load(os.path.join(D, "out", "m1hd_person.npy"), mmap_mode="r")
HD = np.load(os.path.join(D, "out", "m1hd_head.npy"), mmap_mode="r")

# ------------------------------------------------------------------ camera
IMPACT = 0.30
def cam(t):
    """one slow, smooth cinematic push-in; a little extra drift toward his face on the name reveal"""
    tk = T(17.98)
    z = 1.0 + 0.055 * eio(t / tk)
    if t >= tk:
        z += 0.05 * eio((t - tk) / 1.6)
    return z, 540, 760 + 200 / z, 0.0      # zoom anchored on his face (face stays put, frame closes in)

def warp_mat(z, cx, cy, rot):
    M = cv2.getRotationMatrix2D((cx, cy), rot, z)
    M[0, 2] += W / 2 - cx; M[1, 2] += H / 2 - cy
    return M

# ------------------------------------------------------------------ meteors (world space, behind him)
@functools.lru_cache(None)
def meteor_sprite(L, col, w=3):
    h = 40; spr = np.zeros((h, L + 40, 3), np.float32)
    x = np.arange(L, dtype=np.float32); q = x / L
    tail = (q ** 2.4)[None, :]
    core = np.zeros((h, L), np.float32); core[h // 2 - w // 2:h // 2 + w // 2 + 1, :] = 1
    core = cv2.GaussianBlur(core, (0, 0), 0.9) * tail
    glow = cv2.GaussianBlur(core, (0, 0), 5) * 1.6
    c = np.array(col, np.float32)
    spr[:, 20:20 + L] = core[..., None] * np.array((255, 255, 255), np.float32) * 0.9 + glow[..., None] * c
    hx = 20 + L - 2
    yy, xx = np.mgrid[0:h, 0:L + 40].astype(np.float32)
    head = np.exp(-((xx - hx) ** 2 + (yy - h / 2) ** 2) / (2 * 3.2 ** 2)) + 0.35 * np.exp(-((xx - hx) ** 2 + (yy - h / 2) ** 2) / (2 * 11 ** 2))
    spr += head[..., None] * (np.array((255, 255, 255), np.float32) * 0.7 + c * 0.6)
    return spr

def rot_sprite(spr, deg):
    h, w = spr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    cs, sn = abs(M[0, 0]), abs(M[0, 1]); nw, nh = int(h * sn + w * cs) + 2, int(h * cs + w * sn) + 2
    M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(spr, M, (nw, nh))

_r = np.random.default_rng(14)
RADIANT = (1250.0, -250.0)              # Gemini radiant up and to the right, off frame
METEORS = []
tt = 0.05
while tt < 9:
    tt += _r.uniform(0.18, 0.55)
    ang = _r.uniform(math.radians(118), math.radians(160))      # direction of travel (down-left)
    d0 = _r.uniform(250, 900)
    x0 = RADIANT[0] + math.cos(ang) * d0; y0 = RADIANT[1] + math.sin(ang) * d0
    L = int(_r.uniform(260, 760)); dur = _r.uniform(0.35, 0.75); spd = _r.uniform(900, 1700)
    col = [(140, 255, 170), (150, 190, 255), (255, 230, 170), (255, 255, 255), (255, 160, 120)][_r.choice(5, p=[0.22, 0.2, 0.28, 0.2, 0.1])]
    br = _r.uniform(0.55, 1.1) * (1.6 if _r.uniform() < 0.12 else 1.0)
    METEORS.append((tt, x0, y0, ang, L, dur, spd, col, br))

def draw_meteors(plate, pm, t):
    layer = np.zeros_like(plate)
    for (t0, x0, y0, ang, L, dur, spd, col, br) in METEORS:
        u = t - t0
        if not (0 <= u < dur): continue
        q = u / dur
        hx = x0 + math.cos(ang) * spd * u; hy = y0 + math.sin(ang) * spd * u
        a = br * math.sin(math.pi * min(1, q * 1.3)) ** 0.7 * (1 - q ** 3)
        spr = rot_sprite(meteor_sprite(L, col), -math.degrees(ang))
        sh, sw = spr.shape[:2]
        # sprite head is at its right end before rotation -> place so head sits at (hx,hy)
        cx = hx - math.cos(ang) * (L / 2); cy = hy - math.sin(ang) * (L / 2)
        x0_, y0_ = int(cx - sw / 2), int(cy - sh / 2)
        X0, Y0, X1, Y1 = max(0, x0_), max(0, y0_), min(W, x0_ + sw), min(1150, y0_ + sh)
        if X1 > X0 and Y1 > Y0:
            layer[Y0:Y1, X0:X1] += spr[Y0 - y0_:Y1 - y0_, X0 - x0_:X1 - x0_] * a
    # only in the sky, and behind his body
    sky = np.clip((1150 - np.arange(H, dtype=np.float32)) / 120, 0, 1)[:, None, None]
    plate += layer * sky * (1 - pm[..., None])
    return plate

# ------------------------------------------------------------------ styled subtitles (same font, one baseline, no boxes)
from PIL import Image, ImageDraw, ImageFont, ImageFilter
SUBY = 1390
@functools.lru_cache(None)
def sub_word2(text, state):
    """state 0 upcoming, 1 active, 2 spoken. Returns sprite and baseline offset from the sprite top."""
    f = R.font(76, 800); pad = 34; st = 6
    asc, desc = f.getmetrics()
    d0 = ImageDraw.Draw(Image.new("L", (1, 1)))
    bb = d0.textbbox((0, 0), text, font=f, direction="rtl", language="he", stroke_width=st)
    w = bb[2] - bb[0] + 2 * pad; h = asc + desc + 2 * st + 2 * pad
    org = (pad - bb[0], pad + st)
    def mask(sw):
        mm = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mm).text(org, text, font=f, fill=255, direction="rtl", language="he", stroke_width=sw, stroke_fill=255)
        return np.asarray(mm, np.float32) / 255
    body = mask(0); stroke = mask(st)
    out = np.zeros((h, w, 4), np.float32)
    # soft drop shadow
    shd = np.roll(cv2.GaussianBlur(stroke, (0, 0), 7), 6, 0)
    out[..., 3] = shd * 200
    if state == 1:   # warm glow behind the active word
        gl = cv2.GaussianBlur(stroke, (0, 0), 14)
        a = (gl * 0.85)[..., None]
        out[..., :3] = out[..., :3] * (1 - a) + np.array((255, 170, 40), np.float32) * a; out[..., 3:] = np.maximum(out[..., 3:], gl[..., None] * 230)
    sc = np.array((40, 22, 8) if state == 1 else (12, 14, 30), np.float32)
    a = stroke[..., None]; out[..., :3] = out[..., :3] * (1 - a) + sc * a; out[..., 3:] = np.maximum(out[..., 3:], stroke[..., None] * 255)
    ys = np.linspace(0, 1, h)[:, None, None]
    if state == 1: fill = np.array((255, 246, 170), np.float32) * (1 - ys) + np.array((255, 176, 30), np.float32) * ys
    else: fill = np.array((255, 255, 255), np.float32) * (1 - ys) + np.array((226, 228, 255), np.float32) * ys
    fill = fill * np.ones((1, w, 1), np.float32)
    a = body[..., None]; out[..., :3] = out[..., :3] * (1 - a) + fill * a
    if state == 0: out[..., 3] *= 0.62
    return out, org[1] + asc

def subtitles2(F, t):
    for ci, (t0, t1, ws) in enumerate(m.CHUNKS):
        if t0 <= t < t1:
            p = clamp((t - t0) / 0.12)
            hook = 1.45 if ci == 0 else 1.0      # the opening hook line is bigger
            items = []
            for j, w in enumerate(ws):
                end = ws[j + 1]["s"] if j + 1 < len(ws) else w["e"] + 0.2
                state = 1 if w["s"] <= t < end else (2 if t >= end else 0)
                spr, bl = sub_word2(w["t"], state)
                pop = 1.0
                if state == 1: pop = 1.0 + 0.10 * m.eob(clamp((t - w["s"]) / 0.14), 2.4)
                items.append((spr, bl, pop))
            gap = -38; total = sum(s.shape[1] for s, _, _ in items) + gap * (len(items) - 1)
            sc = min(hook, 720 / total) * (0.9 + 0.1 * m.eob(p, 2.8))
            x = 540 + total * sc / 2
            for spr, bl, pop in items:
                ww = spr.shape[1] * sc; s2 = sc * pop
                # keep every word on the same baseline even while the active word pops
                cy = SUBY + (spr.shape[0] / 2 - bl) * s2 + (bl - spr.shape[0] / 2) * sc
                F.blit(spr, x - ww / 2, cy, alpha=eoc(p), scale=s2)
                x -= ww + gap * sc
            return

# ------------------------------------------------------------------ frame
class Src:
    def __init__(self):
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", os.path.join(D, "out", "c1hd.mp4"), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        self.pos = 0; self.last = None
    def get(self, k):
        while self.pos <= k:
            b = self.p.stdout.read(W * H * 3)
            if len(b) < W * H * 3: break
            self.last = np.frombuffer(b, np.uint8).reshape(H, W, 3); self.pos += 1
        return self.last

def sharpen(img, amt=0.3):
    return img + (img - cv2.GaussianBlur(img, (0, 0), 1.2)) * amt

def render(j, src):
    t = j / FPS
    if t < TEND: _, s, _ = m.at(j)
    else: s = m.RUNS[[r[0] for r in m.RUNS].index(PI) + sum(1 for r in m.RUNS if r[0] == PI) - 1][2]
    k = int(round((s - CA) * FPS))
    fr = src.get(k).astype(np.float32)
    kk = min(k, len(P) - 1)
    pm0 = cv2.resize(np.asarray(P[kk]).astype(np.float32) / 255, (W, H), interpolation=cv2.INTER_LINEAR)
    pm0 = cv2.GaussianBlur(pm0, (0, 0), 1.5)
    plate = draw_meteors(fr, pm0, t)
    z, cx, cy, rot = cam(t)
    # motion blur during the opening whip
    if False:
        acc = np.zeros_like(plate)
        for dt in (-0.012, -0.006, 0.0):
            acc += cv2.warpAffine(plate, warp_mat(*cam(max(0, t + dt))), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img = acc / 3
    else:
        img = cv2.warpAffine(plate, warp_mat(z, cx, cy, rot), (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    img = sharpen(img)
    Mw = warp_mat(z, cx, cy, rot)
    pm = cv2.warpAffine(pm0, Mw, (W, H))
    hm = cv2.warpAffine(cv2.resize(np.asarray(HD[kk]), (W, H)), Mw, (W, H))
    hm = cv2.dilate((hm > 90).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    ht = np.where(hm.any(1))[0]
    if len(ht): hm[ht[0] + 420:] = False
    img *= m.vign()
    F = R.Frame(img, pm, hm)
    m.TOPQ.clear()
    G2.draw(F, t, TEND - 0.3, pulse=T(18.46), t0=0.0)
    m.kw(F, t, T(16.6), T(17.95), "בדצמבר!", "calendar_3d.png", "R", 0)
    if 0.1 < t < TEND - 0.05: subtitles2(F, t)
    np.clip(F.img, 0, 255, out=F.img)
    return F.img.astype(np.uint8), F.head_overlap

if __name__ == "__main__":
    out = sys.argv[1]
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    src = Src(); ov = []
    for j in range(NF):
        o, h = render(j, src)
        if h: ov.append((j, h))
        enc.stdin.write(o.tobytes())
    enc.stdin.close(); enc.wait()
    print("frames", NF, "dur", NF / FPS, "head-overlap", len(ov), ov[:5])
