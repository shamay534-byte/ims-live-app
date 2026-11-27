# Custom 3D atmosphere visualisation (own design) driven by the ECMWF 500hPa fields extracted from the user's maps.
import numpy as np, cv2, math, os, functools
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "r15", "maps")
U0, U1, V0, V1, S = 430, 944, 210, 688, 3
CW, CH = (U1 - U0) * S, (V1 - V0) * S
SW, SH = 1080, 1920
def c2(u, v): return ((u - U0) * S, (v - V0) * S)
HF = [cv2.resize(np.load(f"{D}/h{i}.npy")[V0:V1, U0:U1], (CW, CH), interpolation=cv2.INTER_CUBIC) for i in range(1, 5)]
HF = [cv2.GaussianBlur(h, (0, 0), 5) for h in HF]
ISRAEL = c2(913, 489)
# low centres per map (canvas px)
LOWS = [c2(641, 434), c2(805, 468), c2(815, 455), c2(812, 410)]
TRANS = [(23.0, 25.2), (37.4, 39.8), (50.4, 52.8)]
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ss(x): x = clamp(x); return x * x * (3 - 2 * x)
def map_w(t):
    """index + blend fraction between maps"""
    k = 0
    for i, (a, b) in enumerate(TRANS):
        if t >= b: k = i + 1
        elif t > a: return i, ss((t - a) / (b - a))
    return k, 0.0
def field(t):
    k, f = map_w(t)
    if f <= 0 or k >= 3: return HF[min(k, 3)]
    return HF[k] * (1 - f) + HF[k + 1] * f
def low_pos(t):
    k, f = map_w(t)
    if f <= 0 or k >= 3: return LOWS[min(k, 3)]
    a, b = LOWS[k], LOWS[k + 1]
    return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)

# ---------------- static base (own design: dark glass ocean, slate land, glowing coastlines, grid)
@functools.lru_cache(None)
def base():
    land = cv2.resize(np.load(f"{D}/land.npy")[V0:V1, U0:U1].astype(np.float32), (CW, CH), interpolation=cv2.INTER_LINEAR)
    land = cv2.GaussianBlur(land, (0, 0), 1.2)
    coast = cv2.resize(np.load(f"{D}/coastmask.npy")[V0:V1, U0:U1].astype(np.float32), (CW, CH), interpolation=cv2.INTER_LINEAR)
    coast = cv2.GaussianBlur(coast, (0, 0), 1.0); coast = np.clip((coast - 0.18) * 2.2, 0, 1)
    yy, xx = np.mgrid[0:CH, 0:CW].astype(np.float32)
    ocean = np.dstack([8 + 10 * yy / CH, 22 + 22 * yy / CH, 52 + 40 * yy / CH])
    rng = np.random.default_rng(1); nz = cv2.GaussianBlur(rng.standard_normal((CH, CW)).astype(np.float32), (0, 0), 18) * 40
    landc = np.dstack([34 + nz * 0.15, 48 + nz * 0.2, 72 + nz * 0.25])
    img = ocean * (1 - land[..., None]) + landc * land[..., None]
    # is a line pixel a coastline (sea nearby) or a border (land on both sides)?
    seaN = cv2.dilate((land < 0.5).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
    coastline = coast * seaN; border = coast * (1 - seaN)
    glow = cv2.GaussianBlur(coastline, (0, 0), 6)
    img += np.array([60, 170, 255]) * glow[..., None] * 0.9
    img = img * (1 - coastline[..., None]) + np.array([150, 230, 255]) * coastline[..., None]
    img = img * (1 - 0.35 * border[..., None]) + np.array([120, 150, 200]) * 0.35 * border[..., None]
    # faint lat/lon style grid
    g = ((xx % 90 < 1.6) | (yy % 90 < 1.6)).astype(np.float32)
    img += np.array([40, 90, 140]) * g[..., None] * 0.35
    # soft edge vignette (alpha)
    e = np.minimum.reduce([xx / 160, (CW - xx) / 160, yy / 260, (CH - yy) / 120]); a = np.clip(e, 0, 1) ** 1.5
    return img.astype(np.float32), a.astype(np.float32), land

# ---------------- flow particles (geostrophic wind along height contours), precomputed sequentially
NP = 1500
@functools.lru_cache(None)
def particles(nfr=2300, fps=30):
    rng = np.random.default_rng(5)
    V = []
    for h in HF:
        gy, gx = np.gradient(cv2.GaussianBlur(h, (0, 0), 10))
        V.append((gy.astype(np.float32), (-gx).astype(np.float32)))   # NH geostrophic: low to the left (image y down)
    P = np.zeros((nfr, NP, 2), np.float32); A = np.zeros((nfr, NP), np.int16); age = rng.uniform(0, 80, NP)
    p = np.stack([rng.uniform(0, CW, NP), rng.uniform(0, CH, NP)], 1).astype(np.float32)
    for i in range(nfr):
        t = i / fps; k, f = map_w(t)
        xi = np.clip(p[:, 0], 0, CW - 1).astype(int); yi = np.clip(p[:, 1], 0, CH - 1).astype(int)
        def vel(k): return np.stack([V[k][0][yi, xi], V[k][1][yi, xi]], 1)
        v = vel(min(k, 3)) if (f <= 0 or k >= 3) else vel(k) * (1 - f) + vel(k + 1) * f
        p = p + v * 55.0
        age += 1
        dead = (age > 90) | (p[:, 0] < 0) | (p[:, 0] >= CW) | (p[:, 1] < 0) | (p[:, 1] >= CH)
        n = int(dead.sum())
        if n:
            p[dead] = np.stack([rng.uniform(0, CW, n), rng.uniform(0, CH, n)], 1); age[dead] = 0
        P[i] = p; A[i] = age
    return P, A

def ground(t, i):
    img, alpha, land = base()
    img = img.copy()
    h = field(t)
    cold = np.clip((576 - h) / 24, 0, 1); warm = np.clip((h - 580) / 9, 0, 1)
    pulse = 0.92 + 0.08 * math.sin(t * 2.0)
    cc = cold[..., None]
    icy = np.array([90, 200, 255]) * (1 - cc) + np.array([150, 90, 255]) * cc      # cyan rim -> violet core
    img = img * (1 - 0.55 * cc ** 0.8) + icy * 0.55 * cc ** 0.8 * pulse * 1.25
    img += np.array([255, 140, 40]) * warm[..., None] * 0.30
    # contour lines every 4 dam, highlight the cold core
    q = np.floor(h / 4)
    ed = ((q != np.roll(q, 1, 0)) | (q != np.roll(q, 1, 1))).astype(np.float32)
    ed = cv2.GaussianBlur(ed, (0, 0), 0.9)
    img = img * (1 - 0.22 * ed[..., None]) + 255 * 0.22 * ed[..., None]
    # cloud deck around the cold low (drifts with it)
    lx, ly = low_pos(t)
    yy, xx = np.ogrid[0:CH, 0:CW]
    r = np.sqrt((xx - lx) ** 2 + ((yy - ly) * 1.15) ** 2)
    cl = np.clip(1 - r / 330, 0, 1) ** 1.4 * cloud_noise(t)
    img = img * (1 - 0.55 * cl[..., None]) + np.array([225, 232, 245]) * 0.55 * cl[..., None]
    return img, alpha

@functools.lru_cache(None)
def _noise():
    rng = np.random.default_rng(9)
    return cv2.GaussianBlur(rng.standard_normal((CH * 2, CW * 2)).astype(np.float32), (0, 0), 28)
def cloud_noise(t):
    n = _noise(); ox, oy = int(t * 9) % CW, int(t * 5) % CH
    z = n[oy:oy + CH, ox:ox + CW]
    return np.clip(z * 9 + 0.35, 0, 1)

def air(t, i):
    """particle streaks layer (RGBA) floating above the ground"""
    P, A = particles()
    lay = np.zeros((CH, CW), np.float32)
    a, b = P[i], P[max(0, i - 6)]
    ok = (A[i] >= 7) & (np.linalg.norm(a - b, axis=1) < 140)
    for (x0, y0), (x1, y1) in zip(b[ok], a[ok]):
        cv2.line(lay, (int(x0), int(y0)), (int(x1), int(y1)), 1.0, 2, cv2.LINE_AA)
    lay = np.maximum(lay * 0.8, cv2.GaussianBlur(lay, (0, 0), 3) * 1.2)
    rgba = np.zeros((CH, CW, 4), np.float32)
    rgba[..., :3] = np.array([210, 240, 255]); rgba[..., 3] = np.clip(lay, 0, 1) * 150
    return rgba

# ---------------- camera
CAMS = [  # t, target(u,v map px), screen anchor (x,y), distance, pitch(deg), yaw(deg)
    (0.0, (780, 472), (560, 980), 2600, 40, 0),
    (2.6, (780, 472), (560, 950), 1750, 50, 0),
    (13.4, (775, 470), (560, 950), 1700, 50, 0),
    (15.2, (655, 446), (230, 930), 1250, 52, -4),
    (23.2, (665, 448), (230, 930), 1200, 52, -4),
    (26.2, (860, 476), (540, 940), 1000, 52, 0),
    (37.4, (862, 476), (540, 940), 980, 52, 0),
    (40.0, (868, 478), (520, 950), 880, 55, 2),
    (50.3, (868, 478), (520, 950), 860, 55, 2),
    (53.0, (862, 470), (540, 940), 1000, 52, 0),
    (99.0, (862, 470), (540, 940), 980, 52, 0),
]
def cam_at(t):
    for (t0, *a), (t1, *b) in zip(CAMS, CAMS[1:]):
        if t0 <= t <= t1:
            f = ss((t - t0) / (t1 - t0)) if t1 - t0 < 4 else ss((t - t0) / (t1 - t0)) * 0.0 + (t - t0) / (t1 - t0)
            def L(x, y): return x + (y - x) * f
            (tu, tv), (sx, sy), d, p, y = a; (tu2, tv2), (sx2, sy2), d2, p2, y2 = b
            return (L(tu, tu2), L(tv, tv2)), (L(sx, sx2), L(sy, sy2)), L(d, d2), L(p, p2), L(y, y2)
    return CAMS[-1][1:]
def homography(t, height=0.0, F=1500):
    (tu, tv), (ax, ay), d, pitch, yaw = cam_at(t)
    tx, ty = c2(tu, tv)
    p, yw = math.radians(pitch), math.radians(yaw)
    fwd = np.array([math.sin(yw) * math.cos(p), -math.sin(p), math.cos(yw) * math.cos(p)])
    cam = -fwd * d
    right = np.cross(np.array([0, 1.0, 0]), fwd); right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    def proj(x, y):
        P = np.array([x - tx, height, -(y - ty)]) - cam
        zc = P @ fwd
        return ax + F * (P @ right) / zc, ay - F * (P @ up) / zc
    src = np.float32([[0, 0], [CW, 0], [CW, CH], [0, CH]])
    dst = np.float32([proj(*q) for q in src])
    return cv2.getPerspectiveTransform(src, dst), proj

def render_layer(t, i):
    """screen RGBA (premultiplied colour, alpha 0..1) of the whole visualisation"""
    Hg, proj = homography(t, 0.0); Ha, proja = homography(t, 60.0)
    g, ga = ground(t, i)
    intro = ss(t / 1.2)
    out_c = cv2.warpPerspective(g * ga[..., None], Hg, (SW, SH), flags=cv2.INTER_LINEAR)
    out_a = cv2.warpPerspective(ga, Hg, (SW, SH), flags=cv2.INTER_LINEAR)
    ar = air(t, i)
    ac = cv2.warpPerspective(ar[..., :3] * (ar[..., 3:] / 255) * ga[..., None], Ha, (SW, SH))
    aa = cv2.warpPerspective(ar[..., 3] / 255 * ga, Ha, (SW, SH))
    out_c = out_c * (1 - aa[..., None] * 0.7) + ac * 0.9
    out_a = np.clip(out_a + aa * 0.5, 0, 1)
    # horizon fade: the far part of the plane dissolves into the sky
    yy = np.arange(SH, dtype=np.float32)[:, None]
    fade = np.clip((yy - 600) / 150, 0, 1) * np.clip((1520 - yy) / 260, 0, 1)
    out_a *= fade[:, :] * 0.94 * intro; out_c *= (fade * 0.94 * intro)[..., None]
    return out_c, out_a, proj, proja

if __name__ == "__main__":
    import sys, time
    t0 = time.time(); particles(); print("particles", time.time() - t0)
    for t in map(float, sys.argv[1:]):
        t1 = time.time()
        c, a, proj, _ = render_layer(t, int(t * 30))
        bg = np.zeros((SH, SW, 3), np.float32) + np.array([150, 170, 200])
        img = bg * (1 - a[..., None]) + c
        x, y = proj(*ISRAEL); cv2.circle(img, (int(x), int(y)), 9, (255, 60, 60), -1)
        Image.fromarray(img.clip(0, 255).astype(np.uint8)).resize((540, 960)).save(f"stills/viz_{t}.jpg")
        print(t, "render", round(time.time() - t1, 2))
