import numpy as np, wave, json, bisect
from scipy.signal import lfilter
SR = 48000; D = "r15"
KEEP = [tuple(k) for k in json.load(open(f"{D}/keep.json")) if k[1] - k[0] > 5]
OUT = [f for a, b in KEEP for f in range(a, b)]
w = wave.open(f"{D}/a48.wav"); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
f = int(0.012 * SR); parts = []
for a, b in KEEP:
    seg = x[int(a / 30 * SR):int(b / 30 * SR)].copy(); seg[:f] *= np.linspace(0, 1, f); seg[-f:] *= np.linspace(1, 0, f); parts.append(seg)
v = np.concatenate(parts); v = v / np.abs(v).max() * 0.89
o = wave.open(f"{D}/voice_cut.wav", "wb"); o.setnchannels(1); o.setsampwidth(2); o.setframerate(SR); o.writeframes((v * 32767).astype(np.int16).tobytes()); o.close()
def s2o(ts): return bisect.bisect_left(OUT, int(round(ts * 30))) / 30
NN = len(v) + SR; out = np.zeros(NN); rng = np.random.default_rng(3)
def lp(x, a): return lfilter([a], [1, -(1 - a)], x)
def whoosh(d=0.45):
    n = int(d * SR); t = np.linspace(0, 1, n); z = rng.standard_normal(n); s = lp(z, 0.08) * (1 - t) + (z - lp(z, 0.02)) * 0.35 * t; s *= np.sin(np.pi * t) ** 2; return s / np.abs(s).max()
def pop(fq=900):
    n = int(0.09 * SR); t = np.arange(n) / SR; return np.sin(2 * np.pi * np.cumsum(fq * np.exp(-t * 18)) / SR) * np.exp(-t * 45)
def click():
    n = int(0.05 * SR); t = np.arange(n) / SR; return (np.sin(2 * np.pi * 2200 * t) * 0.6 + rng.standard_normal(n) * 0.4) * np.exp(-t * 120)
def thunder(d=1.8):
    n = int(d * SR); t = np.arange(n) / SR; s = lp(rng.standard_normal(n), 0.01) * np.exp(-t * 2.2) * (1 - np.exp(-t * 30)); return s / np.abs(s).max()
def boom(d=1.2):
    n = int(d * SR); t = np.arange(n) / SR
    s = np.sin(2 * np.pi * np.cumsum(70 * np.exp(-t * 1.5) + 30) / SR) * np.exp(-t * 3.2) + lp(rng.standard_normal(n), 0.02) * np.exp(-t * 5) * 3
    s *= (1 - np.exp(-t * 200)); return s / np.abs(s).max()
def add(ts, s, g):
    i = int(max(0, s2o(ts)) * SR); n = min(len(s), NN - i); out[i:i + n] += s[:n] * g
T0 = KEEP[0][0] / 30
add(T0, boom(), 0.2); add(T0 + 0.02, whoosh(0.5), 0.12)
for ts in (23.6, 37.7, 50.6): add(ts - 0.05, whoosh(0.5), 0.14)
for ts in (22.76, 23.12, 44.04, 44.36, 57.8): add(ts, thunder(), 0.14)
for ts in (3.4, 9.38, 16.44, 21.4, 25.1, 30.6, 35.4, 38.92, 42.66, 47.58, 53.24, 57.22, 64.44, 68.96):
    add(ts + 0.02, pop(850), 0.1)

add(73.3, click(), 0.3); add(73.32, pop(1300), 0.15)
out = np.clip(out, -1, 1)
o = wave.open(f"{D}/sfx.wav", "wb"); o.setnchannels(1); o.setsampwidth(2); o.setframerate(SR); o.writeframes((out * 32767).astype(np.int16).tobytes()); o.close()
print("voice", len(v) / SR, "frames", len(OUT) / 30)
