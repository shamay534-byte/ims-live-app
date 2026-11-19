# Inside-the-cirrus mirrors shot: the lip-synced presenter from the rooftop take (c7_full)
# composited over the moving crystal-cloud plate, graded to the red light.
import numpy as np, subprocess, cv2
W,H=1080,1920; F0,F1=[int(v) for v in open("splice8.txt").read().split()]
P=np.load("m8_person_full.npy",mmap_mode="r")
def reader(path, ss=0.0):
    return subprocess.Popen(["ffmpeg","-v","error","-ss",f"{ss:.4f}","-i",path,"-vf","scale=1080:1920:flags=lanczos,fps=30","-f","rawvideo","-pix_fmt","rgb24","-"],stdout=subprocess.PIPE)
src=reader("out/c8_full.mp4"); plate=reader("out/plate.mp4")
enc=subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","1080x1920","-r","30","-i","-","-c:v","libx264","-crf","12","-preset","medium","-pix_fmt","yuv420p","out/c8.mp4"],stdin=subprocess.PIPE)
N=W*H*3; last_plate=None
for j in range(len(P)):
    b=src.stdout.read(N)
    if len(b)<N: break
    img=np.frombuffer(b,np.uint8).reshape(H,W,3).astype(np.float32)
    if F0<=j<F1:
        pb=plate.stdout.read(N)
        if len(pb)==N: last_plate=np.frombuffer(pb,np.uint8).reshape(H,W,3).astype(np.float32)
        bg=last_plate
        m=cv2.resize(np.asarray(P[j]).astype(np.float32)/255,(W,H),interpolation=cv2.INTER_LINEAR)
        m=np.clip((m-0.5)/0.3,0,1); m=cv2.erode(m,np.ones((5,5),np.uint8)); m=cv2.GaussianBlur(m,(0,0),1.6)[...,None]
        person=np.clip(img*np.array([1.16,0.97,0.92],np.float32)+8,0,255)        # red sunset light from below
        glow=cv2.GaussianBlur(bg,(0,0),18)
        edge=np.clip(m*(1-m)*4,0,1)                             # light wrap at the silhouette
        comp=bg*(1-m)+person*m
        comp=comp*(1-edge*0.25)+glow*edge*0.25
        img=np.clip(comp,0,255)
    enc.stdin.write(img.astype(np.uint8).tobytes())
enc.stdin.close(); enc.wait(); src.kill(); plate.kill()
print("composed", F0, F1)
