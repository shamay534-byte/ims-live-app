import numpy as np, wave
from scipy.signal import lfilter
SR=48000; rng=np.random.default_rng(9)
START={1:0,2:262,3:486,4:772,5:985,6:1526,7:2215}; S={k:v/30 for k,v in START.items()}
TOTAL=2760/30; N=int((TOTAL+0.5)*SR)
def rd(p):
    w=wave.open(p); a=np.frombuffer(w.readframes(w.getnframes()),np.int16).reshape(-1,w.getnchannels()).astype(np.float32); return a
# voice: segments placed at their scene start
v=np.zeros((N,2),np.float32); f=int(0.006*SR)
for k in range(1,8):
    a=rd(f"r8/seg/a{k}.wav"); a[:f]*=np.linspace(0,1,f)[:,None]; a[-f:]*=np.linspace(1,0,f)[:,None]
    i=int(S[k]*SR); n=min(len(a),N-i); v[i:i+n]+=a[:n]
pk=np.abs(v).max(); v*=0.89*32767/pk
o=wave.open("r8/voice.wav","wb"); o.setnchannels(2); o.setsampwidth(2); o.setframerate(SR); o.writeframes(v.astype(np.int16).tobytes()); o.close()
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
    i=int(max(0,t)*SR); n=min(len(s),N-i); out[i:i+n]+=s[:n]*g
add(0,pad(TOTAL),0.05)
add(0,whoosh(0.5),0.15)
for k in range(2,8): add(S[k]-0.12,whoosh(0.5),0.16)
add(S[5]-0.05,boom(),0.12)
add(7.85,shimmer(),0.13); add(7.85,boom(),0.1)
CH={1:[4.1],2:[1.0,4.44],3:[2.36,5.64],4:[6.1],5:[0.94,3.48,7.2,11.94],6:[0.4,6.07,16.5],7:[3.54,5.12,9.78]}
for k,ts in CH.items():
    for t in ts: add(S[k]+t,pop(850),0.12)
for k,t in [(2,5.62),(3,8.15),(5,14.2),(5,16.46),(6,13.4),(6,20.1),(7,13.4)]: add(S[k]+t,pop(1200),0.1)
add(S[5]+14.2,shimmer(),0.08); add(S[6]+13.4,shimmer(),0.07)
E=2215/30+14.77
add(E+1.0,boom(),0.12); add(E+1.0,shimmer(),0.1)
add(E+1.9-0.1,whoosh(0.35),0.13); add(E+2.6,click(),0.3); add(E+2.62,pop(1300),0.15)
out=np.clip(out,-1,1)
w=wave.open("r8/sfx.wav","wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((out*32767).astype(np.int16).tobytes()); w.close()
print("ok",TOTAL,E)
