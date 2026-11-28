import subprocess, numpy as np, json, os
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
D = "r17"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{D}/dl/src.mov", "-map", "0:v:0", "-map", "0:a:0", "-vf", "fps=30", "-c:v", "libx264", "-crf", "12",
                "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", f"{D}/video.mp4"], check=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{D}/video.mp4", "-vn", "-ac", "1", "-ar", "48000", f"{D}/a48.wav"], check=True)
# pause trimming (house rule: 10ms RMS < 1300, keep 0.05 s pads, only remove >= 0.17 s)
import wave
w = wave.open(f"{D}/a48.wav"); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32); sr = w.getframerate()
hop = sr // 100; e = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2)) for i in range(0, len(x) - hop, hop)])
q = e < 800; n = len(e); keep = np.ones(n, bool); k = 0
while k < n:
    if q[k]:
        j = k
        while j < n and q[j]: j += 1
        a0, a1 = k + 9, j - 9
        if k == 0: a0 = 0
        if j == n: a1 = n
        if a1 - a0 >= 22: keep[a0:a1] = False
        k = j
    else: k += 1
runs = []; k = 0
while k < n:
    if keep[k]:
        j = k
        while j < n and keep[j]: j += 1
        runs.append((k / 100, j / 100)); k = j
    else: k += 1
fr = [(int(round(a * 30)), int(round(b * 30))) for a, b in runs]
json.dump(fr, open(f"{D}/keep.json", "w"))
print("runs", len(fr), "kept s", round(sum(b - a for a, b in fr) / 30, 2), "of", round(n / 100, 2))
# masks
W, H = 360, 640
seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path="assets/seg.tflite"),
                                               running_mode=vision.RunningMode.VIDEO, output_confidence_masks=True))
p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", f"{D}/video.mp4", "-vf", f"scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
P = []; Hd = []; Fs = []; i = 0
while True:
    b = p.stdout.read(W * H * 3)
    if len(b) < W * H * 3: break
    r = seg.segment_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.frombuffer(b, np.uint8).reshape(H, W, 3).copy()), int(i * 1000 / 30))
    cm = [np.squeeze(m.numpy_view()) for m in r.confidence_masks]
    P.append(((1 - cm[0]) * 255).clip(0, 255).astype(np.uint8)); Hd.append(((cm[1] + cm[3]) * 255).clip(0, 255).astype(np.uint8))
    Fs.append((cm[3] * 255).clip(0, 255).astype(np.uint8)); i += 1
np.save(f"{D}/person.npy", np.array(P)); np.save(f"{D}/head.npy", np.array(Hd)); np.save(f"{D}/face.npy", np.array(Fs))
print("frames", i, flush=True)
os._exit(0)
