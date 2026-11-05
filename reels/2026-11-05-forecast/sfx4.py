import numpy as np, wave, importlib.util
from scipy.signal import lfilter
spec=importlib.util.spec_from_file_location("m","render4.py"); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
SR=48000; DUR=m.NFR/30; N=int((DUR+1)*SR); out=np.zeros(N); rng=np.random.default_rng(5)
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
def add(t,s,g):
    i=int(t*SR); out[i:i+len(s)]+=s[:N-i]*g
add(0.0,whoosh(),0.15); add(0.08,pop(700),0.15); add(0.4,pop(800),0.15)
for a0,a1 in m.SCREEN_WINDOWS: add(a0-0.1,whoosh(0.5),0.17); add(a1,whoosh(0.4),0.12)
add(m.STRIP_T0,whoosh(0.35),0.1)
for k in range(4): add(m.STRIP_T0+k*0.06,pop(1100+k*60),0.06)
for c in m.DAY_CHANGES: add(c-0.02,whoosh(0.35),0.12)
for s in m.SCREEN_SEQ: add(s[0]+0.02,tick(),0.12)
for c in m.CHIPS: add(c[0],pop(850),0.12)
for lt in m.LIGHTNING: add(lt-0.02,thunder(),0.2)
add(m.FOLLOW_T0,whoosh(0.35),0.13); add(m.TAP_T,click(),0.3); add(m.TAP_T+0.02,pop(1300),0.15)
out=np.clip(out,-1,1)
w=wave.open("r3/sfx4.wav","wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((out*32767).astype(np.int16).tobytes()); w.close(); print("ok",DUR)
