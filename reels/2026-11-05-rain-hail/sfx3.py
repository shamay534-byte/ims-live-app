import numpy as np, wave, importlib.util, sys
from scipy.signal import lfilter
spec=importlib.util.spec_from_file_location("m","render3.py"); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
SR=48000; DUR=m.DUR; N=int((DUR+0.5)*SR); out=np.zeros(N); rng=np.random.default_rng(4); VO0=m.VO0
def lp(x,a): return lfilter([a],[1,-(1-a)],x)
def whoosh(d=0.45):
    n=int(d*SR); t=np.linspace(0,1,n); x=rng.standard_normal(n); s=lp(x,0.08)*(1-t)+(x-lp(x,0.02))*0.35*t
    s*=np.sin(np.pi*t)**2; return s/np.abs(s).max()
def pop(f=900):
    n=int(0.09*SR); t=np.arange(n)/SR; return np.sin(2*np.pi*np.cumsum(f*np.exp(-t*18))/SR)*np.exp(-t*45)
def crack():
    n=int(2.6*SR); t=np.arange(n)/SR; x=rng.standard_normal(n)
    hi=(x-lp(x,0.2))*np.exp(-t*25); lo=lp(lp(x,0.01),0.02); lo=lo/np.abs(lo).max()*(1-np.exp(-t*40))*np.exp(-t*1.5)
    s=hi*0.9+lo*1.2; return s/np.abs(s).max()
def rainbed(d):
    n=int(d*SR); x=rng.standard_normal(n); s=lp(x-lp(x,0.05),0.5); return s/np.abs(s).max()
def thud():
    n=int(0.5*SR); t=np.arange(n)/SR
    return np.sin(2*np.pi*55*t*np.exp(-t*3))*np.exp(-t*9)+0.5*lp(rng.standard_normal(n),0.15)*np.exp(-t*18)
def click():
    n=int(0.05*SR); t=np.arange(n)/SR; return (np.sin(2*np.pi*2200*t)*0.6+rng.standard_normal(n)*0.4)*np.exp(-t*120)
def nt(vold):  # old voiceover time -> new time
    if vold < 46.6: return vold
    if vold < 51.62: return 46.6 + (vold - 46.6) * 0.7231
    return vold - 1.39
def add(t,s,g):
    i=int(t*SR)
    if i<0: s=s[-i:]; i=0
    out[i:i+len(s)]+=s[:N-i]*g
# intro
add(0, rainbed(m.FLASH1)*np.linspace(0.6,1,int(m.FLASH1*SR)), 0.035)
for t in [0.1,0.32,0.54,0.9,1.03,1.16,1.29,1.42,1.55]: add(t,pop(650+200*(t>0.8)),0.11)
add(m.FLASH1, crack(), 0.55); add(m.FLASH2, crack()[:int(1.2*SR)], 0.3)
add(m.FLASH1, rainbed(VO0+1.0-m.FLASH1), 0.09)
add(m.FLASH1+0.01, pop(1100), 0.15)
add(7.85, whoosh(0.6), 0.22)
# wind bed following camera speed
ts=np.arange(0, DUR, 1/30); spd=[]
for t in ts:
    v=t-VO0
    if v<-0.3: spd.append(0); continue
    a=m.cam(m.vmap(v)); b=m.cam(m.vmap(v+1/30)); spd.append(abs(b[1]-a[1])*1080/a[2])
spd=np.clip(np.array(spd)/40,0,1.2)
env=np.interp(np.arange(N)/SR, ts, spd); env=lp(env,0.0005)
x=rng.standard_normal(N); wind=lp(x,0.02)*0.7+lp(x-lp(x,0.01),0.08)*0.3; wind/=np.abs(wind).max()
start=int((VO0-0.2)*SR); wind[:start]=0
out+= wind*(0.05+0.5*env)*0.5
# pops on every graphic
POPS=[1.25,3.0,4.4,5.6,7.4,7.95,10.35,11.9,12.75,15.4,17.6,21.6,22.55,24.4,25.45,27.25,27.85,28.9,30.1,32.4,33.65,34.75,36.45,38.6,40.3,41.75,43.4,44.5,46.55,48.5,52.5]
for v in POPS: add(VO0+nt(v)+0.01, pop(900), 0.12)
for v in [17.95]: add(VO0+v, thud(), 0.25)
for v in [27.25,32.4]: add(VO0+v, whoosh(0.3), 0.12)
add(VO0+26.0, rainbed(2.6)*np.sin(np.linspace(0,np.pi,int(2.6*SR))), 0.08)
add(VO0+nt(51.62), thud(), 0.55); add(VO0+nt(51.62), crack()[:int(0.6*SR)], 0.15)
add(VO0+nt(55.25), whoosh(0.35), 0.13); add(VO0+nt(55.85), click(), 0.3); add(VO0+nt(55.87), pop(1300), 0.15)
# soft pulse beat (112 bpm) under the voiceover
bpm=112; beat=60/bpm
kick_n=int(0.25*SR); kt=np.arange(kick_n)/SR; kick=np.sin(2*np.pi*np.cumsum(50+90*np.exp(-kt*30))/SR)*np.exp(-kt*10)
hat_n=int(0.05*SR); hat=(lambda x:(x-lp(x,0.6)))(rng.standard_normal(hat_n))*np.exp(-np.arange(hat_n)/SR*90)
t=VO0
while t<DUR-0.3:
    add(t,kick,0.13); add(t+beat/2,hat,0.035); t+=beat
out=np.clip(out,-1,1)
w=wave.open("r2/sfx3.wav","wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((out*32767).astype(np.int16).tobytes()); w.close(); print("ok", DUR)
