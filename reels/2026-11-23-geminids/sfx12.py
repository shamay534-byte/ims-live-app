import numpy as np, wave, subprocess, sys, importlib.util, bisect
from scipy.signal import lfilter
SR=48000; NS=21
START={}; acc=0
for k in range(1,NS+1):
    d=float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",f"r10/seg/s{k}.mp4"],capture_output=True,text=True).stdout)
    START[k]=acc; acc+=int(round(d*30))
S={k:v/30 for k,v in START.items()}; TOTAL=(acc+int(1.6*30))/30; N=int((TOTAL+0.5)*SR)
def rd(p):
    w=wave.open(p); return np.frombuffer(w.readframes(w.getnframes()),np.int16).reshape(-1,w.getnchannels()).astype(np.float32)
v=np.zeros((N,2),np.float32); f=int(0.006*SR)
for k in range(1,NS+1):
    a=rd(f"r10/seg/a{k}.wav"); a[:f]*=np.linspace(0,1,f)[:,None]; a[-f:]*=np.linspace(1,0,f)[:,None]
    i=int(S[k]*SR); n=min(len(a),N-i); v[i:i+n]+=a[:n]
v*=0.89*32767/np.abs(v).max()
o=wave.open("r10/voice.wav","wb"); o.setnchannels(2); o.setsampwidth(2); o.setframerate(SR); o.writeframes(v.astype(np.int16).tobytes()); o.close()
sys.argv=["x","info"]
spec=importlib.util.spec_from_file_location("m","render11.py"); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
OUT=m.OUT; NO=len(OUT)
def s2o(ts): return bisect.bisect_left(OUT,int(round(ts*30)))/30
runs=[]; a0=OUT[0]
for n in range(1,NO+1):
    if n==NO or OUT[n]!=OUT[n-1]+1: runs.append((a0,OUT[n-1]+1)); a0=OUT[n] if n<NO else None
parts=[]
for r0,r1 in runs:
    seg=v[int(r0/30*SR):int(r1/30*SR)].copy()
    seg[:f]*=np.linspace(0,1,f)[:,None]; seg[-f:]*=np.linspace(1,0,f)[:,None]; parts.append(seg)
vc=np.concatenate(parts)
o=wave.open("r10/voice_cut.wav","wb"); o.setnchannels(2); o.setsampwidth(2); o.setframerate(SR); o.writeframes(vc.astype(np.int16).tobytes()); o.close()
NN=int((NO/30+0.5)*SR); out=np.zeros(NN); rng=np.random.default_rng(3)
def lp(x,a): return lfilter([a],[1,-(1-a)],x)
def whoosh(d=0.45):
    n=int(d*SR); t=np.linspace(0,1,n); x=rng.standard_normal(n); s=lp(x,0.08)*(1-t)+(x-lp(x,0.02))*0.35*t; s*=np.sin(np.pi*t)**2; return s/np.abs(s).max()
def pop(fq=900):
    n=int(0.09*SR); t=np.arange(n)/SR; return np.sin(2*np.pi*np.cumsum(fq*np.exp(-t*18))/SR)*np.exp(-t*45)
def click():
    n=int(0.05*SR); t=np.arange(n)/SR; return (np.sin(2*np.pi*2200*t)*0.6+rng.standard_normal(n)*0.4)*np.exp(-t*120)
def shimmer(d=1.4):
    n=int(d*SR); t=np.arange(n)/SR; s=np.zeros(n)
    for fq in [2093,2637,3136,3951,4186]:
        ph=rng.uniform(0,6.28); s+=np.sin(2*np.pi*fq*t+ph)*np.exp(-t*2.5)*(0.5+0.5*np.sin(t*30+ph))
    s*=(1-np.exp(-t*60)); return s/np.abs(s).max()
def boom(d=1.6):
    n=int(d*SR); t=np.arange(n)/SR; s=np.sin(2*np.pi*np.cumsum(70*np.exp(-t*1.5)+30)/SR)*np.exp(-t*2.8)+lp(rng.standard_normal(n),0.02)*np.exp(-t*4)*3
    s*=(1-np.exp(-t*200)); return s/np.abs(s).max()
def add(ts,s,g):
    i=int(max(0,s2o(ts))*SR); n=min(len(s),NN-i); out[i:i+n]+=s[:n]*g
add(0,whoosh(0.6),0.14); add(0.05,shimmer(1.6),0.1)
for k in range(2,NS+1): add(S[k]+0.01,whoosh(0.45),0.15)
for k,t in [(1,4.9),(5,2.0),(7,0.5),(12,0.2),(13,2.6)]: add(S[k]+t,boom(),0.11)
for k,t in [(2,0.3),(11,0.5),(15,0.8),(16,3.0),(18,3.5)]: add(S[k]+t,shimmer(1.2),0.09)
for k,t in [(1,3.75),(2,0.8),(3,1.6),(4,0.6),(5,0.4),(7,0.6),(8,2.4),(9,0.9),(11,2.6),(12,0.8),(16,1.6),(16,3.1),(17,2.6),(18,0.5),(19,1.8),(20,1.6)]: add(S[k]+t,pop(850),0.11)
ws={w["w"].strip(",.?!"):w["s"] for w in m.SW["14"]}
for word,fq in (("אדום",700),("כחול",950),("סגול",1200)): add(S[14]+ws.get(word,0.5),pop(fq),0.14)
add(S[21]+0.1,whoosh(0.35),0.13); add(S[21]+0.85,click(),0.3); add(S[21]+0.87,pop(1300),0.15)
out=np.clip(out,-1,1)
w=wave.open("r10/sfx.wav","wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((out*32767).astype(np.int16).tobytes()); w.close()
print("ok", NO/30, len(vc)/SR)
