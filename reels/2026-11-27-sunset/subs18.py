# subtitles-only overlay for r16 (sky above the heads), composited on the untouched 60fps original
import sys, json, subprocess, math
sys.argv = ["x", "info"]
import importlib.util
spec = importlib.util.spec_from_file_location("m", "render17.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import numpy as np
FIX = {"מכבישי": "מגבישי", "גשר": "גשם"}
DUR = 29.603333
def load_words():
    ws = []
    for s in json.load(open("r16/trw.json")):
        for w in s["words"]:
            txt = w["w"].strip(); brk = txt.endswith((",", ".", "?", "!")); core = txt.strip(",.?!")
            ws.append({"t": FIX.get(core, core), "s": w["s"], "e": w["e"], "brk": brk})
    return ws
m.load_words = load_words; m.CUTS = []; m.DUR = DUR
CH = m.chunks()
SUB_Y = 600
W, H = 1080, 1920
def frame(t):
    img = np.zeros((H, W, 3), np.float32); F = m.Frame(img, np.zeros((H, W), np.float32), np.zeros((H, W), bool))
    alpha = np.zeros((H, W), np.float32)
    # render on black and white backgrounds to recover straight alpha
    out = []
    for bgv in (0.0, 255.0):
        F.img = np.full((H, W, 3), bgv, np.float32)
        for (t0, t1, ws) in CH:
            if t0 <= t < t1:
                p = m.clamp((t - t0) / 0.12); items = []
                for j, w_ in enumerate(ws):
                    end = ws[j + 1]["s"] if j + 1 < len(ws) else w_["e"] + 0.2
                    st = 1 if w_["s"] <= t < end else (2 if t >= end else 0)
                    spr, bl = m.sub_word2(w_["t"], st)
                    pop = 1.0 + (0.10 * m.ease_out_back(m.clamp((t - w_["s"]) / 0.14), 2.4) if st == 1 else 0)
                    items.append((spr, bl, pop))
                gap = -38; total = sum(q[0].shape[1] for q in items) + gap * (len(items) - 1)
                sc = min(1.12, 760 / total) * (0.9 + 0.1 * m.ease_out_back(p, 2.8))
                x = 540 + total * sc / 2
                for spr, bl, pop in items:
                    ww = spr.shape[1] * sc; s2 = sc * pop
                    cy = SUB_Y + (spr.shape[0] / 2 - bl) * s2 + (bl - spr.shape[0] / 2) * sc
                    F.blit(spr, x - ww / 2, cy, alpha=m.ease_out_cubic(p), scale=s2); x -= ww + gap * sc
                break
        out.append(F.img.copy())
    a = 1 - (out[1] - out[0]) / 255.0
    a = np.clip(a.mean(-1), 0, 1)
    col = np.where(a[..., None] > 1e-3, out[0] / np.maximum(a[..., None], 1e-3), 0)
    return np.dstack([np.clip(col, 0, 255), a * 255]).astype(np.uint8)
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "still":
        pass
    n = int(DUR * 30) + 1
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", "1080x1920", "-r", "30", "-i", "-",
                            "-c:v", "qtrle", "r16/subs.mov"], stdin=subprocess.PIPE)
    for i in range(n):
        enc.stdin.write(frame(i / 30).tobytes())
    enc.stdin.close(); enc.wait(); print("frames", n, "chunks", [(round(a, 2), " ".join(w["t"] for w in c)) for a, b, c in CH])
