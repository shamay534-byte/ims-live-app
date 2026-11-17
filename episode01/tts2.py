import json,subprocess,re,sys,os,concurrent.futures as cf
V={"M":"s0T0DT6EAPeX57MQpn3I","O":"PBHwEnXdzx0yfhIO2UCH","N":"AZzja2JCoYeMe9vDQBll","G":"ZiofZHcEBTn5sRySe5h8","A":"EXAVITQu4vr4xnSDxMaL"}
P={"M":"rubberband=pitch=1.06:formant=preserved","O":"rubberband=pitch=1.04:formant=preserved",
   "N":"rubberband=pitch=1.16:tempo=1.0517:formant=shifted,rubberband=pitch=1.0:formant=preserved","G":"rubberband=pitch=0.93:formant=preserved","A":"anull"}
P["N"]="rubberband=pitch=1.17:tempo=1.0517:formant=preserved"
TRIM="silenceremove=start_periods=1:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse"
L={l[0]:l for l in json.load(open("lines.json"))}
def dur(f): return float(subprocess.run(["ffprobe","-v","0","-show_entries","format=duration","-of","csv=p=0",f],capture_output=True,text=True).stdout)
def clean(t):
    t=re.sub(r"\[[^\]]*\]","",t); t="".join(c for c in t if not("֑"<=c<="ֽ" or "ֿ"<=c<="ׇ"))
    return " ".join(re.sub(r"[^א-ת ]"," ",t.replace("־"," ")).split())
def stt(f):
    import time
    for i in range(5):
        r=subprocess.run(["curl","-sS","-X","POST","https://api.elevenlabs.io/v1/speech-to-text","-F","model_id=scribe_v1","-F","language_code=he","-F",f"file=@{f}"],capture_output=True,text=True)
        if '"text"' in r.stdout: break
        time.sleep(4+3*i)
    r=subprocess.run(["curl","-sS","-X","POST","https://api.elevenlabs.io/v1/speech-to-text","-F","model_id=scribe_v1","-F","language_code=he","-F",f"file=@{f}"],capture_output=True,text=True)
    try: return clean(json.loads(r.stdout)["text"])
    except: return ""
def wer(a,b):
    a,b=a.split(),b.split(); d=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        p,d[0]=d[0],i
        for j,y in enumerate(b,1): p,d[j]=d[j],min(d[j]+1,d[j-1]+1,p+(x!=y))
    return d[-1]/max(1,len(a))
def process(raw,spk,out,target=None,lid=""):
    f=P[spk]+","+TRIM
    if lid in("g3","g4"): f+=",highpass=f=350,lowpass=f=3000,acompressor,volume=1.3"
    tmp=out+".tmp.wav"
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",raw,"-af",f,"-ar","44100","-ac","1",tmp],check=True)
    post="loudnorm=I=-16:TP=-1.5"
    if target:
        r=dur(tmp)/target; post=f"rubberband=tempo={r:.5f}:formant=preserved,apad=whole_dur={target},atrim=0:{target},"+post
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",tmp,"-af",post,"-ar","44100","-ac","1",out],check=True); os.remove(tmp)
def take(lid,text,k):
    spk=L[lid][1]; raw=f"work/takes/{lid}_{k}.mp3"
    body=json.dumps({"text":text,"model_id":"eleven_v3","language_code":"he","seed":1000+k})
    import time
    for i in range(6):
        c=subprocess.run(["curl","-sS","-o",raw,"-w","%{http_code}","-X","POST",f"https://api.elevenlabs.io/v1/text-to-speech/{V[spk]}?output_format=mp3_44100_128","-H","Content-Type: application/json","-d",body],capture_output=True,text=True).stdout
        if c=="200": break
        time.sleep(4+3*i)
    out=f"work/takes/{lid}_{k}.wav"; process(raw,spk,out,lid=lid)
    return k,raw,out,wer(clean(text),stt(out)),dur(out)
if __name__=="__main__":
    os.makedirs("work/takes",exist_ok=True)
    mode=sys.argv[1]
    if mode=="reprocess":   # re-pitch untouched lines from their raw recordings, keep exact old duration
        fixes=json.load(open("fix_lines.json"))
        def rp(lid):
            process(f"audio/raw_{lid}.mp3",L[lid][1],f"audio/{lid}.wav",target=dur(f"audio_v1/{lid}.wav"),lid=lid); return lid
        with cf.ThreadPoolExecutor(6) as ex: print(len(list(ex.map(rp,[l for l in L if l not in fixes]))),"reprocessed")
    else:
        fixes=json.load(open("fix_lines.json"))
        def best(lid):
            ts=[take(lid,fixes[lid],k) for k in range(3)]
            old=dur(f"audio_v1/{lid}.wav")
            ts.sort(key=lambda t:(t[3],abs(t[4]-old)))
            k,raw,out,w,d=ts[0]
            process(raw,L[lid][1],f"audio/{lid}.wav",target=old,lid=lid)
            return lid,w,round(d,2),round(old,2),[round(t[3],2) for t in ts]
        with cf.ThreadPoolExecutor(2) as ex:
            for r in ex.map(best,list(fixes)): print(r,flush=True)
