# match the phone voice-note to the camera mic: long-term spectrum EQ + room tone + level
import numpy as np, subprocess
from scipy.signal import welch
SR = 48000
def load(path, extra=()):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, *extra, "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"], capture_output=True)
    return np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32768
src = load("r10/dl/src.mov"); vo = load("r11/vo/fix.m4a")
def seg(x, a, b): return x[int(a * SR):int(b * SR)]
ref = np.concatenate([seg(src, 94.5, 97.4), seg(src, 110.9, 113.6), seg(src, 88.0, 91.1)])
cur = seg(vo, 1.1, 8.7)
def active(x):  # keep only voiced 20ms frames
    fr = x[:len(x) // 960 * 960].reshape(-1, 960); e = np.sqrt((fr ** 2).mean(1))
    return fr[e > np.percentile(e, 40)].ravel()
f, Pr = welch(active(ref), SR, nperseg=4096); _, Pc = welch(active(cur), SR, nperseg=4096)
g = 10 * np.log10((Pr + 1e-12) / (Pc + 1e-12))
# smooth over ~1/3 octave
lf = np.log2(np.maximum(f, 20)); gs = np.array([g[np.abs(lf - l) < 1 / 4].mean() for l in lf])
gs -= np.interp(1000, f, gs)          # keep level at 1 kHz, shape only
gs = np.clip(gs, -14, 12); gs[f < 70] = np.minimum(gs[f < 70], -12)
# apply as zero-phase FFT filter
n = len(vo); N = 1 << int(np.ceil(np.log2(n)))
F = np.fft.rfft(vo, N); ff = np.fft.rfftfreq(N, 1 / SR)
F *= 10 ** (np.interp(ff, f, gs) / 20)
y = np.fft.irfft(F, N)[:n]
# gentle spectral noise reduction using the voice note's own pause as the noise profile
from scipy.signal import stft, istft
fq, tt, Z = stft(y, SR, nperseg=1024)
nz = np.abs(stft(seg(y, 0.15, 0.95), SR, nperseg=1024)[2]).mean(1, keepdims=True)
mag = np.abs(Z); gain = np.maximum(1 - 1.6 * nz / (mag + 1e-9), 0.12)
gain = np.minimum(1, gain)
y = istft(Z * gain, SR, nperseg=1024)[1][:n]
# level match on voiced parts
def rms(x): return np.sqrt(np.mean(active(x) ** 2))
y *= rms(ref) / rms(seg(y, 1.1, 8.7))
# room tone from a pause in the camera recording, looped with crossfades
rt = seg(src, 19.45, 21.45)
reps = int(np.ceil(n / len(rt))) + 1; tone = np.concatenate([rt] * reps)[:n]
# compare noise floors: source pause vs voice-note pause
vo_floor = np.sqrt(np.mean(seg(y, 0.2, 0.9) ** 2)); src_floor = np.sqrt(np.mean(rt ** 2))
y = y + tone * np.sqrt(max(0.0, 1.0 - (vo_floor / max(src_floor, 1e-6)) ** 2))
# second pass: correct the remaining spectral difference (after denoise + room tone)
for _ in range(2):
    _, Py = welch(active(seg(y, 1.1, 8.7)), SR, nperseg=4096)
    g2 = 10 * np.log10((Pr + 1e-12) / (Py + 1e-12))
    g2 = np.array([g2[np.abs(lf - l) < 1 / 4].mean() for l in lf]); g2 -= np.interp(1000, f, g2)
    g2 = np.clip(g2, -8, 6); g2[f < 70] = 0
    Fy = np.fft.rfft(y, N); Fy *= 10 ** (np.interp(ff, f, g2) / 20); y = np.fft.irfft(Fy, N)[:n]
    y *= rms(ref) / rms(seg(y, 1.1, 8.7))
np.save("r11/vo/vo_matched.npy", y.astype(np.float32))
print("EQ dB @ 150/500/2k/5k/9k:", [round(float(np.interp(x, f, gs)), 1) for x in (150, 500, 2000, 5000, 9000)], "floors", round(float(vo_floor), 5), round(float(src_floor), 5))
