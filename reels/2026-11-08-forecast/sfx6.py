import numpy as np, wave, importlib.util, subprocess
from scipy.signal import lfilter
spec=importlib.util.spec_from_file_location("m","render6.py"); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
SR=48000; KEEP=m.KEEP
def o2s(): pass
def s2o(ts):  # source time -> output time (None if inside a cut)
    off=0
    for a,b in KEEP:
        if a/30 <= ts < b/30: return off+ts-a/30
        off+=(b-a)/30
    return None
OUTDUR=len(m.OUT)/30; N=int((OUTDUR+1)*SR); out=np.zeros(N); rng=np.random.default_rng(6)
def lp(x,a): return lfilter([a],[1,-(1-a)],x)
def whoosh(d=0.45):
    n=int(d*SR); t=np.linspace(0,1,n); x=rng.standard_normal(n); s=lp(x,0.08)*(1-t)+(x-lp(x,0.02))*0.35*t; s*=np.sin(np.pi*t)**2; return s/np.abs(s).max()
def pop(f=900):
    n=int(0.09*SR); t=np.arange(n)/SR; return np.sin(2*np.pi*np.cumsum(f*np.exp(-t*18))/SR)*np.exp(-t*45)
def tick():
    n=int(0.06*SR); t=np.arange(n)/SR; return np.sin(2*np.pi*1800*t)*np.exp(-t*70)
def thunder(d=2.2):
    n=int(d*SR); t=np.arange(n)/SR; x=lp(lp(rng.standard_normal(n),0.01),0.03); s=x*(1-np.exp(-t*30))*np.exp(-t*1.6)*(1+0.5*np.sin(t*9)); return s/np.abs(s).max()
def click():
    n=int(0.05*SR); t=np.arange(n)/SR; return (np.sin(2*np.pi*2200*t)*0.6+rng.standard_normal(n)*0.4)*np.exp(-t*120)
def add(ts,s,g):
    t=s2o(ts)
    if t is None: return
    i=int(t*SR); out[i:i+len(s)]+=s[:N-i]*g
add(0.0,whoosh(),0.15); add(0.08,pop(700),0.15); add(0.4,pop(800),0.15)
for a0,a1 in m.SCREEN_WINDOWS: add(a0-0.1,whoosh(0.5),0.17); add(a1,whoosh(0.4),0.12)
for k in range(5): add(m.SCREEN_WINDOWS[0][0]+0.9+k*0.28,tick(),0.1)
add(m.HILITE[m.LEADER],pop(1300),0.15)
add(m.STRIP_T0,whoosh(0.35),0.1)
for k in range(6): add(m.STRIP_T0+k*0.06,pop(1100+k*50),0.05)
for c in m.DAY_CHANGES: add(c,tick(),0.12)
for c in m.CHIPS: add(c[0],pop(850),0.12)
for lt in m.LIGHTNING: add(lt-0.02,thunder(),0.2)
add(m.FOLLOW_T0,whoosh(0.35),0.13); add(m.TAP_T,click(),0.3); add(m.TAP_T+0.02,pop(1300),0.15)
# zoom-punch whooshes on the jump cuts (very soft)
for a,b in KEEP[1:]: add(a/30+0.001,whoosh(0.18),0.05)
out=np.clip(out,-1,1)
w=wave.open("r5/sfx6.wav","wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((out*32767).astype(np.int16).tobytes()); w.close()
# voice cut to the same KEEP ranges
raw=subprocess.run(["ffmpeg","-v","error","-i",m.VIDEO,"-vn","-ac","2","-ar",str(SR),"-f","s16le","-"],capture_output=True).stdout
a=np.frombuffer(raw,np.int16).reshape(-1,2).astype(np.float32); f=int(0.006*SR); parts=[]
for k,(s0,s1) in enumerate(KEEP):
    seg=a[int(s0/30*SR):int(s1/30*SR)].copy()
    if k>0: seg[:f]*=np.linspace(0,1,f)[:,None]
    if k<len(KEEP)-1: seg[-f:]*=np.linspace(1,0,f)[:,None]
    parts.append(seg)
v=np.concatenate(parts); print("voice", len(v)/SR, "video", OUTDUR)
o=wave.open("r5/voice6.wav","wb"); o.setnchannels(2); o.setsampwidth(2); o.setframerate(SR); o.writeframes(v.astype(np.int16).tobytes()); o.close()
