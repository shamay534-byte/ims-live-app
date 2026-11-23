import sys, subprocess, numpy as np, os
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
D=os.path.dirname(os.path.abspath(__file__))
k=sys.argv[1]; url=sys.argv[2]
raw=f"{D}/out/raw{k}.mp4"
subprocess.run(["curl","-sSL","-o",raw,url],check=True)
dur=lambda p: float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",p],capture_output=True,text=True).stdout)
din=dur(f"{D}/seg/s{k}.mp4")
xin=dur(f"{D}/seg/x{k}.mp4") if os.path.exists(f"{D}/seg/x{k}.mp4") else din
dout=float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",raw],capture_output=True,text=True).stdout)
f=xin/dout
subprocess.run(["ffmpeg","-v","error","-y","-i",raw,"-vf",f"setpts=PTS*{f:.5f},scale=1080:1920:flags=lanczos,fps=30","-t",f"{din:.3f}","-an",
                "-c:v","libx264","-crf","12","-preset","medium","-pix_fmt","yuv420p",f"{D}/out/c{k}.mp4"],check=True)
W,H=360,640
seg=vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=f"{D}/../assets/seg.tflite"),running_mode=vision.RunningMode.VIDEO,output_confidence_masks=True))
p=subprocess.Popen(["ffmpeg","-v","error","-i",f"{D}/out/c{k}.mp4","-vf",f"scale={W}:{H}","-f","rawvideo","-pix_fmt","rgb24","-"],stdout=subprocess.PIPE)
P=[];Hd=[];i=0
while True:
    b=p.stdout.read(W*H*3)
    if len(b)<W*H*3: break
    r=seg.segment_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.frombuffer(b,np.uint8).reshape(H,W,3).copy()),int(i*1000/30))
    cm=[np.squeeze(m.numpy_view()) for m in r.confidence_masks]
    P.append(((1-cm[0])*255).clip(0,255).astype(np.uint8)); Hd.append(((cm[1]+cm[3])*255).clip(0,255).astype(np.uint8)); i+=1
np.save(f"{D}/m{k}_person.npy",np.array(P)); np.save(f"{D}/m{k}_head.npy",np.array(Hd))
print(f"seg {k}: in {din:.2f}s out {dout:.2f}s stretch {f:.4f}, frames {i}", flush=True)
