# voice track (clean takes + re-recorded colours line) and SFX for render12, all in OUTPUT time
import numpy as np, wave, subprocess, sys, os
from scipy.signal import lfilter
sys.argv = ["x", "info"]
import importlib.util
spec = importlib.util.spec_from_file_location("m", "render12.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
SR = 48000
def load(path, extra=()):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, *extra, "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"], capture_output=True)
    return np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32768
src = load("r11/../r10/dl/src.mov")
# voice note: match the camera mic (cut lows, gentle presence, light room) and loudness
vo = load("r11/vo/fix.m4a", ("-af", "highpass=f=110,lowpass=f=9500,equalizer=f=3000:t=q:w=1.2:g=-2,aecho=0.6:0.35:28:0.12"))
def rms(x): return np.sqrt(np.mean(x[np.abs(x) > 0.01] ** 2))
ref = src[int(94.5 * SR):int(97.4 * SR)]
vo *= rms(ref) / rms(vo[int(3.9 * SR):int(8.7 * SR)])
parts = []; f = int(0.008 * SR)
for (pi, s, e, ta) in m.RUNS:
    x = src if m.PIECES[pi][0] == "c" else vo
    seg = x[int(s * SR):int(e * SR)].copy()
    seg[:f] *= np.linspace(0, 1, f); seg[-f:] *= np.linspace(1, 0, f)
    parts.append(seg)
tl = np.concatenate(parts)
tl = np.concatenate([tl, np.zeros(int(m.END_HOLD * m.SP * SR) + SR)])
w = wave.open("r11/voice_tl.wav", "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
w.writeframes((np.clip(tl / np.abs(tl).max() * 0.9, -1, 1) * 32767).astype(np.int16).tobytes()); w.close()
# speed-up to output time (pitch preserved)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "r11/voice_tl.wav", "-af", f"atempo={m.SP}", "r11/voice_out.wav"], check=True)

NN = int((m.NOUT / 30 + 0.5) * SR); out = np.zeros(NN); rng = np.random.default_rng(3)
def lp(x, a): return lfilter([a], [1, -(1 - a)], x)
def whoosh(d=0.45):
    n = int(d * SR); t = np.linspace(0, 1, n); x = rng.standard_normal(n); s = lp(x, 0.08) * (1 - t) + (x - lp(x, 0.02)) * 0.35 * t; s *= np.sin(np.pi * t) ** 2; return s / np.abs(s).max()
def pop(fq=900):
    n = int(0.09 * SR); t = np.arange(n) / SR; return np.sin(2 * np.pi * np.cumsum(fq * np.exp(-t * 18)) / SR) * np.exp(-t * 45)
def click():
    n = int(0.05 * SR); t = np.arange(n) / SR; return (np.sin(2 * np.pi * 2200 * t) * 0.6 + rng.standard_normal(n) * 0.4) * np.exp(-t * 120)
def shimmer(d=1.4):
    n = int(d * SR); t = np.arange(n) / SR; s = np.zeros(n)
    for fq in [2093, 2637, 3136, 3951, 4186]:
        ph = rng.uniform(0, 6.28); s += np.sin(2 * np.pi * fq * t + ph) * np.exp(-t * 2.5) * (0.5 + 0.5 * np.sin(t * 30 + ph))
    s *= (1 - np.exp(-t * 60)); return s / np.abs(s).max()
def boom(d=1.6):
    n = int(d * SR); t = np.arange(n) / SR
    s = np.sin(2 * np.pi * np.cumsum(70 * np.exp(-t * 1.5) + 30) / SR) * np.exp(-t * 2.8) + lp(rng.standard_normal(n), 0.02) * np.exp(-t * 4) * 3
    s *= (1 - np.exp(-t * 200)); return s / np.abs(s).max()
def add(ts, s, g):
    i = int(max(0, ts) * SR); n = min(len(s), NN - i); out[i:i + n] += s[:n] * g
T = m.T
# title: meteor whoosh into impact boom + shimmer, re-ignite
add(0.0, whoosh(0.4), 0.18); add(0.28, boom(), 0.16); add(0.3, shimmer(1.8), 0.11); add(T(18.46), boom(1.2), 0.09)
# location changes
for pi in range(1, len(m.PIECES)):
    if m.PIECES[pi][1] != m.PIECES[pi - 1][1]: add(m.PSTART(pi) - 0.05, whoosh(0.45), 0.15)
for ts in (T(32.7), T(38.35), T(88.0), m.V(4.4)): add(ts, boom(), 0.11)
for ts in (T(22.3), T(82.9), T(111.4), T(120.26), T(141.45)): add(ts, shimmer(1.2), 0.09)
for ts in (T(16.6), T(25.5), T(30.8), T(60.5), T(64.6), T(86.2), T(89.4), T(111.9), T(119.1), T(137.36), T(138.8), T(153.6), T(156.8)): add(ts, pop(850), 0.11)
for vt, fq in ((5.94, 650), (6.42, 800), (7.68, 950), (8.02, 1150)): add(m.V(vt), pop(fq), 0.15)
t13 = T(159.5); add(t13 + 0.1, whoosh(0.35), 0.13); add(t13 + 0.85, click(), 0.3); add(t13 + 0.87, pop(1300), 0.15)
out = np.clip(out, -1, 1)
w = wave.open("r11/sfx.wav", "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((out * 32767).astype(np.int16).tobytes()); w.close()
print("ok", m.NOUT / 30, len(tl) / SR / m.SP)
