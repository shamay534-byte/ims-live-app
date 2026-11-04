import subprocess, numpy as np, mediapipe as mp, sys
from mediapipe.tasks.python import vision, BaseOptions
W,H=540,960
opts=vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path="assets/seg.tflite"),running_mode=vision.RunningMode.VIDEO,output_confidence_masks=True,output_category_mask=False)
seg=vision.ImageSegmenter.create_from_options(opts)
p=subprocess.Popen(["ffmpeg","-v","error","-i","dl/video.mp4","-vf",f"fps=30,scale={W}:{H}","-f","rawvideo","-pix_fmt","rgb24","-"],stdout=subprocess.PIPE)
person=[];head=[];i=0
while True:
    b=p.stdout.read(W*H*3)
    if len(b)<W*H*3: break
    fr=np.frombuffer(b,np.uint8).reshape(H,W,3)
    r=seg.segment_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(fr)),int(i*1000/30))
    cm=[m.numpy_view() for m in r.confidence_masks]  # 0 bg,1 hair,2 body-skin,3 face-skin,4 clothes,5 others
    bg=cm[0]
    person.append(((1-bg)*255).clip(0,255).astype(np.uint8))
    head.append(((cm[1]+cm[3])*255).clip(0,255).astype(np.uint8))
    i+=1
    if i%300==0: print(i,flush=True)
np.savez_compressed("masks.npz",person=np.array(person),head=np.array(head))
print("done",i)
