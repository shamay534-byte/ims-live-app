import numpy as np, wave
SR=48000; INTRO=8.3; DUR=INTRO+65.43; N=int((DUR+1)*SR); out=np.zeros(N); rng=np.random.default_rng(2)
def lp(x,a):
    from scipy.signal import lfilter
    return lfilter([a],[1,-(1-a)],x)
def whoosh(d=0.45):
    n=int(d*SR); t=np.linspace(0,1,n); x=rng.standard_normal(n)
    s=lp(x,0.08)*(1-t)+(x-lp(x,0.02))*0.35*t; env=np.sin(np.pi*t)**2; s=s*env; return s/np.abs(s).max()
def pop(f=900):
    n=int(0.09*SR); t=np.arange(n)/SR; fr=f*np.exp(-t*18); return np.sin(2*np.pi*np.cumsum(fr)/SR)*np.exp(-t*45)
def thunder(d=2.2):
    n=int(d*SR); t=np.arange(n)/SR; x=lp(lp(rng.standard_normal(n),0.01),0.03)
    s=x*(1-np.exp(-t*30))*np.exp(-t*1.6)*(1+0.5*np.sin(t*9)); return s/np.abs(s).max()
def rain(d):
    n=int(d*SR); x=rng.standard_normal(n); s=x-lp(x,0.05); s=lp(s,0.5)
    env=np.minimum(1,np.minimum(np.arange(n)/(0.6*SR),(n-np.arange(n))/(0.6*SR))); s=s*env; return s/np.abs(s).max()
def thud():
    n=int(0.35*SR); t=np.arange(n)/SR; return (np.sin(2*np.pi*60*t*np.exp(-t*4))*np.exp(-t*12)+0.4*lp(rng.standard_normal(n),0.1)*np.exp(-t*20))
def click():
    n=int(0.05*SR); t=np.arange(n)/SR; return (np.sin(2*np.pi*2200*t)*0.6+rng.standard_normal(n)*0.4)*np.exp(-t*120)
def chime():
    n=int(0.8*SR); t=np.arange(n)/SR; return sum(np.sin(2*np.pi*f*t)*np.exp(-t*4)/(k+1) for k,f in enumerate([1318,1760,2637]))
def add(t,s,g):
    i=int(t*SR); out[i:i+len(s)]+=s[:N-i]*g
add(0,rain(INTRO+0.4),0.05); add(0,thunder(2.5),0.12)
for t in [0.15,0.55]: add(t,pop(700),0.16)
add(6.8,chime(),0.07); add(6.8,pop(1000),0.14)
for v in [-0.25,13.4,31.2,43.6,58.8]: add(INTRO+v,whoosh(),0.15)
for v in [0.6,3.0,5.5,7.85,8.42,11.0,12.45,14.4,17.6,19.0,23.0,24.35,28.2,32.9,33.8,36.8,38.3,41.4,46.4,47.4,49.2,52.5,54.0,60.9]:
    add(INTRO+v,pop(850),0.12)
add(INTRO+20.2,thud(),0.25); add(INTRO+29.5,rain(2.0),0.06); add(INTRO+31.5,thunder(),0.25)
add(INTRO+58.4,thud(),0.35); add(INTRO+62.2,whoosh(0.35),0.13); add(INTRO+63.15,click(),0.3); add(INTRO+63.17,pop(1300),0.15)
add(INTRO+59.0,rain(6.5),0.04)
out=np.clip(out,-1,1)
w=wave.open("r2/sfx2.wav","wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((out*32767).astype(np.int16).tobytes()); w.close(); print("ok")
