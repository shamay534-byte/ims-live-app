# Clean stylised 3D map + storm system that travels Italy -> Aegean -> Israel, camera follows the storm to the presenter's hand.
import numpy as np, cv2, math, os, functools
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "r15", "maps")
U0, U1, V0, V1, S = 430, 944, 250, 688, 3
CW, CH = (U1 - U0) * S, (V1 - V0) * S
SW, SH = 1080, 1920
def c2(u, v): return ((u - U0) * S, (v - V0) * S)
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ss(x): x = clamp(x); return x * x * (3 - 2 * x)
PLACES = {"איטליה": (662, 452), "יוון": (768, 474), "טורקיה": (852, 432), "ישראל": (913, 489)}

@functools.lru_cache(None)
def base():
    land = np.load(f"{D}/land2.npy")[V0:V1, U0:U1].astype(np.float32)
    land = cv2.resize(land, (CW, CH), interpolation=cv2.INTER_LINEAR)
    land = np.clip((cv2.GaussianBlur(land, (0, 0), 2.0) - 0.5) * 3 + 0.5, 0, 1)
    coast = cv2.resize(np.load(f"{D}/coastmask.npy")[V0:V1, U0:U1].astype(np.float32), (CW, CH), interpolation=cv2.INTER_LINEAR)
    coast = np.clip((cv2.GaussianBlur(coast, (0, 0), 1.0) - 0.18) * 2.2, 0, 1)
    yy, xx = np.mgrid[0:CH, 0:CW].astype(np.float32)
    rng = np.random.default_rng(3)
    n1 = cv2.GaussianBlur(rng.standard_normal((CH, CW)).astype(np.float32), (0, 0), 30)
    n2 = cv2.GaussianBlur(rng.standard_normal((CH, CW)).astype(np.float32), (0, 0), 6)
    relief = n1 * 60 + n2 * 6
    gy, gx = np.gradient(relief); shade = np.clip(1 + (-gx * 0.9 - gy * 0.7), 0.75, 1.2)
    # sea: deep blue with lighter shallows near the coast
    dist = cv2.distanceTransform((land < 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    shallow = np.exp(-dist / 28)[..., None]
    sea = np.array([18, 70, 150]) * (1 - shallow) + np.array([40, 140, 205]) * shallow
    sea = sea + n2[..., None] * 25
    # land: warm sand / olive, desert to the south
    south = np.clip((yy - CH * 0.62) / (CH * 0.25), 0, 1)[..., None]
    landc = (np.array([150, 160, 110]) * (1 - south) + np.array([214, 186, 132]) * south) * shade[..., None]
    img = sea * (1 - land[..., None]) + landc * land[..., None]
    seaN = cv2.dilate((land < 0.5).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
    cl = coast * seaN; bd = coast * (1 - seaN)
    img = img * (1 - 0.85 * cl[..., None]) + 255 * 0.85 * cl[..., None]
    img = img * (1 - 0.25 * bd[..., None]) + np.array([90, 80, 60]) * 0.25 * bd[..., None]
    e = np.minimum.reduce([xx / 180, (CW - xx) / 120, yy / 220, (CH - yy) / 160]); a = np.clip(e, 0, 1) ** 1.5
    return img.astype(np.float32), a.astype(np.float32)

if __name__ == "__main__":
    img, a = base()
    im = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    from PIL import ImageDraw, ImageFont
    d = ImageDraw.Draw(im); f = ImageFont.truetype(os.path.join(HERE, "assets", "Rubik.ttf"), 40)
    for k, (u, v) in PLACES.items():
        x, y = c2(u, v); d.ellipse((x - 10, y - 10, x + 10, y + 10), fill=(255, 0, 0)); d.text((x + 14, y - 20), k[::-1], fill=(0, 0, 0), font=f)
    im.resize((CW // 2, CH // 2)).save("stills/viz17_base.jpg")

PLACES = {"איטליה": (688, 466), "יוון": (757, 468), "טורקיה": (850, 432), "ישראל": (913, 489)}
# ---- storm track (map px) following the ECMWF fields: Italy -> Aegean -> E. Med next to Israel, then lingers
TRACK = [(0.0, (660, 448)), (23.6, (660, 448)), (31.0, (770, 470)), (37.7, (850, 474)), (41.0, (893, 481)), (50.6, (893, 481)), (55.0, (872, 472)), (99, (872, 472))]
def storm_pos(t):
    for (t0, a), (t1, b) in zip(TRACK, TRACK[1:]):
        if t0 <= t <= t1:
            f = ss((t - t0) / (t1 - t0)) if t1 > t0 else 0
            x, y = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
            w = 4 * math.sin(t * 0.7) if t > 41 else 0
            return (x + w, y + 0.6 * w)
    return TRACK[-1][1]
def storm_strength(t):      # 0..1 : forming over Italy -> mature
    return 0.45 + 0.55 * ss((t - 13.0) / 4.0)
# ---- camera keyframes: (t, target map px or "storm", screen anchor, distance, pitch)
CAMS = [
    (0.0, (785, 468), (560, 980), 2600, 46),
    (2.4, (785, 468), (560, 940), 2050, 52),
    (13.0, (782, 468), (560, 940), 2000, 52),
    (16.0, "storm", (905, 880), 1050, 55),
    (23.6, "storm", (905, 880), 1050, 55),
    (27.0, "storm", (830, 900), 1350, 53),
    (37.0, "storm", (830, 900), 1350, 53),
    (40.0, "storm", (900, 870), 1050, 55),
    (50.6, "storm", (900, 870), 1050, 55),
    (53.5, "storm", (850, 890), 1250, 53),
    (99.0, "storm", (850, 890), 1250, 53),
]
def cam_at(t):
    for (t0, ta, aa, da, pa), (t1, tb, ab, db, pb) in zip(CAMS, CAMS[1:]):
        if t0 <= t <= t1:
            f = ss((t - t0) / (t1 - t0))
            A = storm_pos(t) if ta == "storm" else ta; B = storm_pos(t) if tb == "storm" else tb
            L = lambda x, y: x + (y - x) * f
            return (L(A[0], B[0]), L(A[1], B[1])), (L(aa[0], ab[0]), L(aa[1], ab[1])), L(da, db), L(pa, pb)
    raise ValueError
def homography(t, height=0.0, F=1500):
    (tu, tv), (ax, ay), d, pitch = cam_at(t)
    tx, ty = c2(tu, tv); p = math.radians(pitch)
    fwd = np.array([0.0, -math.sin(p), math.cos(p)]); cam = -fwd * d
    right = np.array([1.0, 0, 0]); up = np.cross(fwd, right)
    def proj(x, y, h=height):
        P = np.array([x - tx, h, -(y - ty)]) - cam; zc = P @ fwd
        return ax + F * (P @ right) / zc, ay - F * (P @ up) / zc, F / zc
    src = np.float32([[0, 0], [CW, 0], [CW, CH], [0, CH]])
    dst = np.float32([proj(*q)[:2] for q in src])
    return cv2.getPerspectiveTransform(src, dst), proj

@functools.lru_cache(None)
def spiral(R=360):
    yy, xx = np.mgrid[-R:R, -R:R].astype(np.float32)
    r = np.sqrt(xx ** 2 + yy ** 2) + 1; th = np.arctan2(yy, xx)
    rng = np.random.default_rng(4); n = cv2.GaussianBlur(rng.standard_normal((2 * R, 2 * R)).astype(np.float32), (0, 0), 7)
    return r, th, n
def ground(t):
    img, alpha = base(); img = img.copy()
    sx, sy = c2(*storm_pos(t)); st = storm_strength(t)
    R = 360; r, th, n = spiral(R)
    # cyclonic (counter-clockwise) comma-shaped cloud spiral around the low
    band = 0.5 + 0.5 * np.sin(2.0 * (th + t * 0.55) + np.log(r) * 2.6 + n * 2.2)
    fall = np.clip(1 - r / R, 0, 1) ** 0.9
    cl = np.clip(band ** 2.2 * fall * 1.5, 0, 1) * (0.55 + 0.45 * st)
    core = np.exp(-(r / (R * 0.32)) ** 2) * 0.7 * st
    cl = np.clip(cl + core, 0, 1)
    x0, y0 = int(sx - R), int(sy - R)
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(CW, x0 + 2 * R), min(CH, y0 + 2 * R)
    if X1 > X0 and Y1 > Y0:
        c = cl[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0, None]
        reg = img[Y0:Y1, X0:X1]
        sh = cv2.GaussianBlur(c[..., 0], (0, 0), 6)[..., None]
        reg *= (1 - 0.35 * sh)                                   # cloud shadow / wet ground
        reg[:] = reg * (1 - 0.82 * c) + np.array([238, 242, 250]) * 0.82 * c
    # track arrow Italy -> Israel (shown while he describes the move east)
    a = ss((t - 24.0) / 1.0) * (1 - ss((t - 41.0) / 1.5))
    if a > 0.01:
        lay = np.zeros((CH, CW), np.float32)
        pts = [c2(*storm_pos(tt)) for tt in np.linspace(23.6, 41.0, 60)]
        k = int(len(pts) * clamp((t - 24.0) / 3.0))
        for j in range(0, max(0, k - 1), 2):
            cv2.line(lay, tuple(map(int, pts[j])), tuple(map(int, pts[j + 1])), 1.0, 9, cv2.LINE_AA)
        if k > 3:
            p, q = np.array(pts[k - 1]), np.array(pts[k - 4]); d_ = (p - q) / (np.linalg.norm(p - q) + 1e-6); nrm = np.array([-d_[1], d_[0]])
            tri = np.array([p + d_ * 30, p - d_ * 10 + nrm * 26, p - d_ * 10 - nrm * 26], np.int32); cv2.fillPoly(lay, [tri], 1.0)
        g = cv2.GaussianBlur(lay, (0, 0), 8)
        img += np.array([255, 210, 80]) * (g[..., None] * 0.9 * a)
        img = img * (1 - 0.9 * a * lay[..., None]) + np.array([255, 230, 120]) * 0.9 * a * lay[..., None]
    return img, alpha

def render_layer(t):
    Hg, proj = homography(t, 0.0)
    g, ga = ground(t)
    intro = ss(t / 1.0)
    out_c = cv2.warpPerspective(g * ga[..., None], Hg, (SW, SH), flags=cv2.INTER_LINEAR)
    out_a = cv2.warpPerspective(ga, Hg, (SW, SH), flags=cv2.INTER_LINEAR)
    yy = np.arange(SH, dtype=np.float32)[:, None]
    fade = np.clip((yy - 600) / 140, 0, 1) * np.clip((1560 - yy) / 260, 0, 1) * 0.96 * intro
    out_a = out_a * fade; out_c = out_c * fade[..., None]
    return out_c, out_a, proj
