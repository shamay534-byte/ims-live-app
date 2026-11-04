import numpy as np, wave, sys
sys.path.insert(0,'.')
import importlib.util
spec=importlib.util.spec_from_file_location("r","render.py"); r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
SR=48000; N=int(r.DUR*SR)+SR; out=np.zeros(N)
rng=np.random.default_rng(1)
def lp(x,a):
    y=np.zeros_like(x); acc=0
    for i,v in enumerate(x): acc+=a*(v-acc); y[i]=acc
    return y
def whoosh(d=0.45):
    n=int(d*SR); t=np.linspace(0,1,n); x=rng.standard_normal(n)
    a=np.interp(t,[0,1],[0.03,0.5]); lo=lp(x,0.08); hi=x-lp(x,0.02)
    s=lo*(1-t)+hi*0.35*t; env=np.sin(np.pi*t)**2
    return s*env/np.abs(s*env).max()
def pop(f=900):
    n=int(0.09*SR); t=np.arange(n)/SR; fr=f*np.exp(-t*18)
    return np.sin(2*np.pi*np.cumsum(fr)/SR)*np.exp(-t*45)
def thunder():
    n=int(2.0*SR); t=np.arange(n)/SR; x=lp(lp(rng.standard_normal(n),0.01),0.03)
    env=(1-np.exp(-t*30))*np.exp(-t*1.8)*(1+0.5*np.sin(t*9))
    s=x*env; return s/np.abs(s).max()
def click():
    n=int(0.05*SR); t=np.arange(n)/SR
    return (np.sin(2*np.pi*2200*t)*0.6+rng.standard_normal(n)*0.4)*np.exp(-t*120)
def add(t,s,g):
    i=int(t*SR); out[i:i+len(s)]+=s[:N-i]*g
W_=whoosh(); 
for t in [0.0,4.15]+[c-0.02 for c in r.DAY_CHANGES]: add(max(0,t),W_,0.16)
for t in [0.05,0.3,0.62]: add(t,pop(700),0.18)
for k in range(7): add(r.STRIP_T0+k*0.06,pop(1100+k*60),0.06)
for c in r.CHIPS: add(c[0],pop(850),0.13)
for lt in r.LIGHTNING: add(lt-0.02,thunder(),0.22)
add(r.FOLLOW_T0,whoosh(0.35),0.14); add(r.TAP_T,click(),0.3); add(r.TAP_T+0.02,pop(1300),0.15)
out=np.clip(out,-1,1)
w=wave.open("sfx.wav","wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((out*32767).astype(np.int16).tobytes()); w.close()
print("ok")
