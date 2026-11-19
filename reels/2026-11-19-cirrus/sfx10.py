import numpy as np, wave
from scipy.signal import lfilter
SR=48000; rng=np.random.default_rng(9)
import subprocess as _sp
NS=10; START={}; _acc=0
for k in range(1,NS+1):
    _d=float(_sp.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",f"r8/seg/s{k}.mp4"],capture_output=True,text=True).stdout)
    START[k]=_acc; _acc+=int(round(_d*30))
S={k:v/30 for k,v in START.items()}
TOTAL=(_acc+int(1.8*30))/30; N=int((TOTAL+0.5)*SR)
def rd(p):
    w=wave.open(p); a=np.frombuffer(w.readframes(w.getnframes()),np.int16).reshape(-1,w.getnchannels()).astype(np.float32); return a
# voice: segments placed at their scene start
v=np.zeros((N,2),np.float32); f=int(0.006*SR)
for k in range(1,NS+1):
    a=rd(f"r8/seg/a{k}.wav"); a[:f]*=np.linspace(0,1,f)[:,None]; a[-f:]*=np.linspace(1,0,f)[:,None]
    i=int(S[k]*SR); n=min(len(a),N-i); v[i:i+n]+=a[:n]
pk=np.abs(v).max(); v*=0.89*32767/pk
_o=wave.open("r8/voice.wav","wb"); _o.setnchannels(2); _o.setsampwidth(2); _o.setframerate(SR); _o.writeframes(v.astype(np.int16).tobytes()); _o.close()
import sys, importlib.util, bisect
sys.argv=["x","info"]
spec=importlib.util.spec_from_file_location("m","render9.py"); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
OUT=m.OUT; NO=len(OUT)
def s2o(ts):
    return bisect.bisect_left(OUT, int(round(ts*30)))/30
# cut voice by kept runs
runs=[]; a0=OUT[0]
for n in range(1,NO+1):
    if n==NO or OUT[n]!=OUT[n-1]+1: runs.append((a0,OUT[n-1]+1)); a0=OUT[n] if n<NO else None
f2=int(0.006*SR); parts=[]
for r0,r1 in runs:
    seg=v[int(r0/30*SR):int(r1/30*SR)].copy()
    seg[:f2]*=np.linspace(0,1,f2)[:,None]; seg[-f2:]*=np.linspace(1,0,f2)[:,None]; parts.append(seg)
vc=np.concatenate(parts)
o=wave.open("r8/voice_cut.wav","wb"); o.setnchannels(2); o.setsampwidth(2); o.setframerate(SR); o.writeframes(vc.astype(np.int16).tobytes()); o.close()
N=int((NO/30+0.5)*SR); TOTAL=NO/30
out=np.zeros(N)
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
def pad(d):
    n=int(d*SR); t=np.arange(n)/SR; s=np.zeros(n)
    for fq in [110,164.8,220,277.2,329.6]:
        s+=np.sin(2*np.pi*fq*t+np.sin(t*0.3)*2)*(0.6+0.4*np.sin(2*np.pi*t/9+fq))
    s=lp(s,0.05); env=np.minimum(1,t/3)*np.minimum(1,(d-t)/2); return s*env/np.abs(s).max()
def add(t,s,g):
    t=s2o(t); i=int(max(0,t)*SR); n=min(len(s),N-i); out[i:i+n]+=s[:n]*g
add(0,pad(TOTAL-0.05),0.05)
add(0,whoosh(0.5),0.15)
for k in range(2,NS+1): add(S[k]-0.12,whoosh(0.5),0.16)
add(S[5]-0.05,boom(),0.12)
add(7.85,shimmer(),0.13); add(7.85,boom(),0.1)
CH={1:[4.1],2:[1.0,4.44],3:[2.36,5.64],4:[4.5],5:[6.1],6:[0.94,3.48,7.2,11.94],7:[6.0,10.4,13.6,17.8],8:[0.4,6.07,16.5],9:[3.54,5.12,9.78],10:[6.72]}
for k,ts in CH.items():
    for t in ts: add(S[k]+t,pop(850),0.12)
for k,t in [(2,5.62),(4,0.7),(4,3.05),(4,5.72),(6,14.2),(6,16.46),(7,0.1),(7,2.3),(7,7.85),(7,12.6),(7,16.75),(8,13.4),(8,20.1),(9,13.4)]: add(S[k]+t,pop(1200),0.1)
add(S[4]+0.2,whoosh(1.2),0.12); add(S[4]+3.9,boom(),0.1); add(S[4]+3.0,pop(900),0.1)
add(S[6]+14.2,shimmer(),0.08); add(S[8]+13.4,shimmer(),0.07)
for t in (7.8,12.55,16.7): add(S[7]+t-0.12,whoosh(0.4),0.13)
add(S[7]+21.45,boom(),0.1)
add(S[8]+314/30-0.1,whoosh(0.5),0.16); add(S[8]+314/30,shimmer(1.8),0.12); add(S[8]+579/30-0.1,whoosh(0.4),0.13)
add(S[10]+4.0,boom(),0.14); add(S[10]+4.0,shimmer(),0.12); add(S[10]+4.7,whoosh(0.5),0.1)
add(S[10]+9.3,whoosh(0.35),0.13); add(S[10]+10.35,click(),0.3); add(S[10]+10.37,pop(1300),0.15)
for r0,r1 in runs[1:]:
    t_=s2o(r0/30)
    i=int(t_*SR); w_=whoosh(0.16); out[i:i+len(w_)]+=w_*0.05
out=np.clip(out,-1,1)
w=wave.open("r8/sfx.wav","wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((out*32767).astype(np.int16).tobytes()); w.close()
print("ok",TOTAL)
