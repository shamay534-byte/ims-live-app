# download a Seedance clip, keep it at native 720p, stretch to the source clip duration (lossless-ish), compute masks
import sys, subprocess, numpy as np, os, json
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
D = os.path.dirname(os.path.abspath(__file__))
n = sys.argv[1]; url = sys.argv[2]
os.makedirs(f"{D}/out", exist_ok=True)
raw = f"{D}/out/raw{n}.mp4"
if not os.path.exists(raw):
    subprocess.run(["curl", "-sSL", "-o", raw, url], check=True)
dur = lambda p: float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p], capture_output=True, text=True).stdout)
din = dur(f"{D}/seg/c{n}.mp4"); dout = dur(raw)
f = din / dout
# stretch only (no scaling) -> near-lossless intermediate at 30 fps
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-vf", f"setpts=PTS*{f:.6f},fps=30", "-t", f"{din:.3f}", "-an",
                "-c:v", "libx264", "-crf", "8", "-preset", "slow", "-pix_fmt", "yuv444p", f"{D}/out/c{n}.mp4"], check=True)
W, H = 360, 640
seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=f"{D}/../assets/seg.tflite"), running_mode=vision.RunningMode.VIDEO, output_confidence_masks=True))
p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", f"{D}/out/c{n}.mp4", "-vf", f"scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
P = []; Hd = []; i = 0
while True:
    b = p.stdout.read(W * H * 3)
    if len(b) < W * H * 3: break
    r = seg.segment_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.frombuffer(b, np.uint8).reshape(H, W, 3).copy()), int(i * 1000 / 30))
    cm = [np.squeeze(m.numpy_view()) for m in r.confidence_masks]
    P.append(((1 - cm[0]) * 255).clip(0, 255).astype(np.uint8)); Hd.append(((cm[1] + cm[3]) * 255).clip(0, 255).astype(np.uint8)); i += 1
np.save(f"{D}/out/m{n}_person.npy", np.array(P)); np.save(f"{D}/out/m{n}_head.npy", np.array(Hd))
print(f"clip {n}: in {din:.2f}s out {dout:.2f}s stretch {f:.4f}, frames {i}", flush=True)
os._exit(0)
