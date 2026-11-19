import numpy as np, subprocess
F0=314
n=int(subprocess.run(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames","-of","csv=p=0","out/c6b.mp4"],capture_output=True,text=True).stdout)
F1=F0+n
print("6b frames",n,"F1",F1)
subprocess.run(["ffmpeg","-v","error","-y","-i","out/c6_full.mp4","-i","out/c6b.mp4","-filter_complex",
 f"[0:v]trim=end_frame={F0},setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[0:v]trim=start_frame={F1},setpts=PTS-STARTPTS[c];[a][b][c]concat=n=3:v=1[v]",
 "-map","[v]","-c:v","libx264","-crf","12","-preset","medium","-pix_fmt","yuv420p","out/c6.mp4"],check=True)
for nm in ["person","head"]:
    A=np.load(f"m6_{nm}_full.npy"); B=np.load(f"m6b_{nm}.npy")
    np.save(f"m6_{nm}.npy",np.concatenate([A[:F0],B,A[F1:]]))
print("total",F0+n+len(A[F1:]))
open("splice6.txt","w").write(f"{F0} {F1}\n")
