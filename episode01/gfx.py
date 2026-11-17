#!/usr/bin/env python3
"""Motion graphics for episode 1: animated main title, concept card, end card. Renders PNG sequences."""
import math, os, random, sys
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops, ImageEnhance

os.chdir(os.path.dirname(os.path.abspath(__file__)))
W, H, S, FPS = 854, 480, 2, 30          # render at 2x then downscale
WW, HH = W * S, H * S
F_LOGO = "fonts/secular-full.ttf"
F_BOLD = "fonts/heebo-800.ttf"
F_MED = "fonts/heebo-700.ttf"
F_REG = "fonts/heebo-400.ttf"
NAVY = (16, 30, 52)
GOLD = (242, 179, 61)
CREAM = (255, 246, 214)
SKY = (120, 190, 235)


def font(path, size): return ImageFont.truetype(path, size * S // 2 * 2 // 2 if False else int(size * S))


def ease_out_back(x, s=1.9):
    x = min(max(x, 0), 1) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def ease_out(x): x = min(max(x, 0), 1); return 1 - (1 - x) ** 3
def clamp01(x): return min(max(x, 0.0), 1.0)


def text_mask(txt, fnt, pad=40):
    d = ImageDraw.Draw(Image.new("L", (10, 10)))
    l, t, r, b = d.textbbox((0, 0), txt, font=fnt, direction="rtl", language="he")
    m = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
    ImageDraw.Draw(m).text((pad - l, pad - t), txt, font=fnt, fill=255, direction="rtl", language="he")
    return m


def vgrad(size, top, bot):
    w, h = size
    g = Image.new("RGB", (1, h))
    for y in range(h):
        k = y / max(1, h - 1)
        g.putpixel((0, y), tuple(int(top[i] + (bot[i] - top[i]) * k) for i in range(3)))
    return g.resize((w, h))


def styled(txt, fnt, top=CREAM, bot=GOLD, outline=7, glow=(110, 180, 255), shadow=True):
    """Gradient-filled text with dark outline, outer glow and drop shadow -> RGBA."""
    o = outline * S
    m = text_mask(txt, fnt, pad=40 + 3 * o)
    big = m.filter(ImageFilter.MaxFilter(o * 2 + 1))
    out = Image.new("RGBA", m.size, (0, 0, 0, 0))
    if glow:
        g = big.filter(ImageFilter.GaussianBlur(14 * S)).point(lambda v: int(v * 0.85))
        out.paste(Image.new("RGBA", m.size, glow + (255,)), (0, 0), g)
    if shadow:
        sh = Image.new("L", m.size, 0); sh.paste(big, (0, 5 * S))
        sh = sh.filter(ImageFilter.GaussianBlur(4 * S)).point(lambda v: int(v * 0.7))
        out.paste(Image.new("RGBA", m.size, (0, 0, 0, 255)), (0, 0), sh)
    out.paste(Image.new("RGBA", m.size, NAVY + (255,)), (0, 0), big)
    fill = vgrad(m.size, top, bot).convert("RGBA")
    out.paste(fill, (0, 0), m)
    # inner highlight line on top half
    hi = ImageChops.multiply(m, vgrad(m.size, (255, 255, 255), (0, 0, 0)).convert("L").point(lambda v: 255 if v > 150 else 0))
    hi = ImageChops.subtract(hi, m.filter(ImageFilter.MinFilter(3 * S + 1 if (3 * S) % 2 == 0 else 3 * S)))
    out.paste(Image.new("RGBA", m.size, (255, 255, 255, 140)), (0, 0), hi)
    return out, m


def shine(img, mask, pos):
    """Diagonal light sweep clipped to the text mask. pos 0..1 across."""
    w, h = img.size
    band = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(band)
    x = -0.3 * w + pos * 1.6 * w
    d.polygon([(x, 0), (x + 60 * S, 0), (x + 60 * S - h * 0.5, h), (x - h * 0.5, h)], fill=200)
    band = band.filter(ImageFilter.GaussianBlur(8 * S))
    a = ImageChops.multiply(band, mask)
    img = img.copy(); img.paste(Image.new("RGBA", img.size, (255, 255, 255, 255)), (0, 0), a)
    return img


def place(canvas, img, cx, cy, scale=1.0, alpha=1.0, rot=0):
    if scale <= 0.01 or alpha <= 0.01: return
    im = img
    if scale != 1.0: im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
    if rot: im = im.rotate(rot, resample=Image.BICUBIC, expand=True)
    if alpha < 1:
        if im is img: im = im.copy()
        im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
    canvas.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))


def bolt(seed, x0, y0, x1, y1, segs=9, jag=38):
    rnd = random.Random(seed); pts = [(x0, y0)]
    for i in range(1, segs):
        k = i / segs
        pts.append((x0 + (x1 - x0) * k + rnd.uniform(-jag, jag) * S, y0 + (y1 - y0) * k))
    pts.append((x1, y1)); return pts


def draw_bolt(canvas, pts, a):
    lay = Image.new("RGBA", canvas.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    d.line(pts, fill=(170, 210, 255, int(255 * a)), width=16 * S, joint="curve")
    lay = lay.filter(ImageFilter.GaussianBlur(10 * S))
    d = ImageDraw.Draw(lay)
    d.line(pts, fill=(255, 255, 240, int(255 * a)), width=4 * S, joint="curve")
    canvas.alpha_composite(lay)


class Sparkles:
    def __init__(self, n, seed, area):
        r = random.Random(seed); x0, y0, x1, y1 = area
        self.p = [(r.uniform(x0, x1), r.uniform(y0, y1), r.uniform(0, 6.28), r.uniform(0.6, 1.6), r.uniform(2, 5))
                  for _ in range(n)]

    def draw(self, canvas, t, a=1.0):
        d = ImageDraw.Draw(canvas)
        for x, y, ph, sp, sz in self.p:
            k = (math.sin(t * sp * 3 + ph) + 1) / 2
            if k < 0.55: continue
            al = int(255 * a * (k - 0.55) / 0.45)
            s = sz * S * k; yy = y - t * 6 * S * sp
            d.line([(x - s * 2, yy), (x + s * 2, yy)], fill=(255, 240, 190, al), width=max(1, S))
            d.line([(x, yy - s * 2), (x, yy + s * 2)], fill=(255, 240, 190, al), width=max(1, S))
            d.ellipse([x - s / 2, yy - s / 2, x + s / 2, yy + s / 2], fill=(255, 255, 230, al))


def pill(txt, fnt, fg=GOLD, bg=NAVY + (225,), border=GOLD, padx=26, pady=10):
    m = text_mask(txt, fnt, pad=0)
    w, h = m.width + 2 * padx * S, m.height + 2 * pady * S
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=h // 2, fill=bg, outline=border, width=2 * S)
    im.paste(Image.new("RGBA", m.size, fg + (255,)), (padx * S, pady * S), m)
    return im


def save(canvas, path, rgb=False):
    im = canvas.resize((W, H), Image.LANCZOS)
    (im.convert("RGB") if rgb else im).save(path, compress_level=1)


# ---------------------------------------------------------------- MAIN TITLE (overlay, 12s)
def main_title():
    os.makedirs("work/title", exist_ok=True)
    w1, m1 = styled("שומרי", font(F_LOGO, 104))
    w2, m2 = styled("הרקיע", font(F_LOGO, 104))
    gap = 18 * S
    # RTL: first word on the right
    cx, cy = WW / 2, HH / 2 - 40 * S
    padpx = (40 + 3 * 7 * S) * 2
    r1, r2 = w1.width - padpx, w2.width - padpx
    gap = 30 * S
    total = r1 + gap + r2
    x1 = cx + total / 2 - r1 / 2
    x2 = cx - total / 2 + r2 / 2
    badge = pill("פרק ראשון", font(F_MED, 19))
    sub, _ = styled("התחנה הנטושה", font(F_BOLD, 40), top=(255, 255, 255), bot=(200, 225, 255), outline=4, glow=None)
    spark = Sparkles(70, 7, (WW * 0.15, HH * 0.15, WW * 0.85, HH * 0.75))
    bpts = bolt(3, cx + 30 * S, cy - 200 * S, cx - 60 * S, cy + 140 * S)
    n = int(12 * FPS)
    vb = Image.new("L", (WW, HH), 0)
    ImageDraw.Draw(vb).ellipse([WW * 0.12, HH * 0.18, WW * 0.88, HH * 0.82], fill=150)
    vb = vb.filter(ImageFilter.GaussianBlur(70 * S)); dark = Image.new("RGBA", (WW, HH), (5, 12, 25, 255))
    only = [int(x) for x in os.environ.get("FRAMES", "").split(",") if x]
    for f in range(n):
        if only and f not in only: continue
        t = f / FPS
        c = Image.new("RGBA", (WW, HH), (0, 0, 0, 0))
        out = clamp01((11.7 - t) / 0.9)                          # global fade-out
        va = clamp01((t - 0.8) / 1.2) * out
        if va > 0: c.paste(dark, (0, 0), vb.point(lambda v: int(v * va)))
        # lightning (behind logo) at 2.55s
        if 2.5 < t < 3.05:
            fl = 1.0 if int((t - 2.5) * 30) % 3 != 1 else 0.35
            draw_bolt(c, bpts, fl * clamp01((3.05 - t) / 0.3))
        spark.draw(c, t, a=clamp01((t - 2.6) / 1.0) * out)
        # words drop in with overshoot
        for img, msk, x, t0 in ((w1, m1, x1, 1.2), (w2, m2, x2, 1.75)):
            k = ease_out_back((t - t0) / 0.75)
            if t < t0: continue
            y = cy - (1 - k) * 160 * S
            sc = 0.6 + 0.4 * k if k < 1 else 1.0 + 0.012 * math.sin((t - t0) * 2.2)
            im = img
            if 3.6 < t < 4.9: im = shine(img, msk.resize(img.size), (t - 3.6) / 1.3)
            place(c, im, x, y, scale=max(0.05, sc) * (1 + 0.04 * (1 - out)), alpha=clamp01((t - t0) / 0.25) * out)
        # badge + subtitle
        kb = ease_out((t - 4.6) / 0.6)
        if t > 4.6:
            place(c, badge, cx, cy + 118 * S + (1 - kb) * 30 * S, alpha=kb * out)
        ks = ease_out((t - 5.1) / 0.7)
        if t > 5.1:
            place(c, sub, cx, cy + 178 * S + (1 - ks) * 30 * S, alpha=ks * out)
        # full-frame white flash on the strike
        if 2.55 < t < 2.95:
            c.alpha_composite(Image.new("RGBA", (WW, HH), (235, 245, 255, int(170 * (2.95 - t) / 0.4))))
        save(c, f"work/title/{f:04d}.png")
    print("title", n)


# ---------------------------------------------------------------- CONCEPT CARD (6s, full frame)
def cloud_icon(d, cx, cy, s, rain=True):
    for dx, dy, r in ((-0.55, 0.1, 0.42), (0, -0.2, 0.55), (0.55, 0.1, 0.42), (0, 0.25, 0.5)):
        d.ellipse([cx + (dx - r) * s, cy + (dy - r) * s, cx + (dx + r) * s, cy + (dy + r) * s], fill=(235, 240, 248))
    if rain:
        for i in range(-2, 3):
            x = cx + i * 0.32 * s
            d.line([(x, cy + 0.85 * s), (x - 0.12 * s, cy + 1.2 * s)], fill=SKY, width=5 * S)


def wave_icon(d, cx, cy, s):
    pts = [(cx - 1.1 * s + i * 2.2 * s / 60, cy + math.sin(i / 60 * 2 * math.pi * 1.5) * 0.45 * s) for i in range(61)]
    r = random.Random(5)
    for x, y in pts[::3]:
        yy = y + r.uniform(-0.45, 0.45) * s
        d.ellipse([x - 3 * S, yy - 3 * S, x + 3 * S, yy + 3 * S], fill=(255, 220, 140))
    d.line(pts, fill=GOLD, width=6 * S, joint="curve")


def concept_card(bg_key="k31", dur=7.0):
    os.makedirs("work/concept", exist_ok=True)
    bg = Image.open(f"keys/{bg_key}.png").convert("RGB").resize((WW, HH))
    bg = ImageEnhance.Brightness(bg.filter(ImageFilter.GaussianBlur(12 * S))).enhance(0.45).convert("RGBA")
    head, hm = styled("המושג של הפרק", font(F_BOLD, 34), outline=4, glow=(255, 190, 90))
    cw, ch = 330 * S, 250 * S

    def card(title, line1, line2, icon, accent):
        im = Image.new("RGBA", (cw, ch), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        d.rounded_rectangle([0, 0, cw - 1, ch - 1], radius=28 * S, fill=(14, 26, 46, 235), outline=accent, width=3 * S)
        icon(d, cw / 2, 50 * S, 36 * S)
        for txt, f, y, col in ((title, font(F_BOLD, 34), 122, accent), (line1, font(F_MED, 20), 172, (240, 240, 240)),
                               (line2, font(F_REG, 18), 204, (190, 205, 225))):
            m = text_mask(txt, f, pad=0)
            im.paste(Image.new("RGBA", m.size, col + (255,)), (int(cw / 2 - m.width / 2), int(y * S - m.height / 2)), m)
        return im

    c_weather = card("מזג אוויר", "מה קורה עכשיו, היום, כאן", "כמו מצב רוח, משתנה מהר",
                     lambda d, x, y, s: cloud_icon(d, x, y, s), SKY)
    c_climate = card("אקלים", "מה קורה בדרך כלל", "ממוצע של שלושים שנה, כמו אופי",
                     lambda d, x, y, s: wave_icon(d, x, y, s), GOLD)
    vs = pill("לעומת", font(F_MED, 18), fg=(255, 255, 255), bg=(180, 60, 60, 240), border=(255, 255, 255))
    spark = Sparkles(40, 11, (0, 0, WW, HH))
    n = int(dur * FPS)
    only = [int(x) for x in os.environ.get("FRAMES", "").split(",") if x]
    for f in range(n):
        if only and f not in only: continue
        t = f / FPS; out = clamp01((dur - t) / 0.5)
        c = bg.copy()
        spark.draw(c, t, a=0.6 * out)
        place(c, head, WW / 2, 62 * S - (1 - ease_out(t / 0.5)) * 20 * S, alpha=clamp01(t / 0.4) * out)
        k1 = ease_out_back((t - 0.4) / 0.7); k2 = ease_out_back((t - 0.9) / 0.7)
        if t > 0.4: place(c, c_weather, WW / 2 + 190 * S + (1 - k1) * 300 * S, HH / 2 + 30 * S, alpha=out, rot=-2 * (1 - k1))
        if t > 0.9: place(c, c_climate, WW / 2 - 190 * S - (1 - k2) * 300 * S, HH / 2 + 30 * S, alpha=out, rot=2 * (1 - k2))
        kv = ease_out_back((t - 1.6) / 0.5)
        if t > 1.6: place(c, vs, WW / 2, HH / 2 + 30 * S, scale=max(0.05, kv), alpha=out)
        if t < 0.25: c.alpha_composite(Image.new("RGBA", (WW, HH), (0, 0, 0, int(255 * (1 - t / 0.25)))))
        save(c, f"work/concept/{f:04d}.png", rgb=True)
    print("concept", n)


# ---------------------------------------------------------------- END CARD (7s, full frame)
def end_card(dur=7.0):
    os.makedirs("work/endcard", exist_ok=True)
    bg = Image.open("keys/k05.png").convert("RGB").resize((int(WW * 1.15), int(HH * 1.15)))
    bg = ImageEnhance.Brightness(bg).enhance(0.55)
    logo, lm = styled("שומרי הרקיע", font(F_LOGO, 64), outline=5)
    nxt = pill("בפרק הבא", font(F_MED, 18))
    ttl, _ = styled("האוקיינוס שמעלינו", font(F_BOLD, 40), top=(255, 255, 255), bot=(190, 220, 255), outline=4, glow=(90, 160, 255))
    tease = text_mask("למה אנחנו לא מרגישים טון של אוויר על הראש?", font(F_MED, 20), pad=0)
    spark = Sparkles(60, 21, (0, 0, WW, HH))
    n = int(dur * FPS)
    only = [int(x) for x in os.environ.get("FRAMES", "").split(",") if x]
    for f in range(n):
        if only and f not in only: continue
        t = f / FPS; k = t / dur
        c = bg.crop((int(k * 0.1 * WW), int(k * 0.06 * HH), int(k * 0.1 * WW) + WW, int(k * 0.06 * HH) + HH)).convert("RGBA")
        place(c, shine(logo, lm.resize(logo.size), (t - 1.0) / 1.4) if 1 < t < 2.4 else logo,
              WW / 2, 120 * S, scale=0.9 + 0.1 * ease_out(t / 1.0), alpha=clamp01(t / 0.6))
        if t > 1.4: place(c, nxt, WW / 2, 235 * S + (1 - ease_out((t - 1.4) / 0.5)) * 20 * S, alpha=clamp01((t - 1.4) / 0.4))
        if t > 1.9: place(c, ttl, WW / 2, 300 * S, scale=max(0.05, ease_out_back((t - 1.9) / 0.6)), alpha=clamp01((t - 1.9) / 0.3))
        if t > 2.8:
            a = clamp01((t - 2.8) / 0.6)
            c.paste(Image.new("RGBA", tease.size, (255, 230, 170, 255)), (int(WW / 2 - tease.width / 2), int(375 * S)),
                    tease.point(lambda v: int(v * a)))
        spark.draw(c, t, a=0.8)
        fo = clamp01((t - (dur - 0.8)) / 0.8)
        if fo: c.alpha_composite(Image.new("RGBA", (WW, HH), (0, 0, 0, int(255 * fo))))
        if t < 0.4: c.alpha_composite(Image.new("RGBA", (WW, HH), (0, 0, 0, int(255 * (1 - t / 0.4)))))
        save(c, f"work/endcard/{f:04d}.png", rgb=True)
    print("endcard", n)


if __name__ == "__main__":
    which = sys.argv[1:] or ["title", "concept", "end"]
    if "preview" in which:
        # render single frames for checking
        globals()["FPS_PREVIEW"] = True
    if "title" in which: main_title()
    if "concept" in which: concept_card()
    if "end" in which: end_card()
