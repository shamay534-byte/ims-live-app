import json,subprocess,sys,os
V={"M":"s0T0DT6EAPeX57MQpn3I","O":"PBHwEnXdzx0yfhIO2UCH","N":"AZzja2JCoYeMe9vDQBll","G":"ZiofZHcEBTn5sRySe5h8","A":"EXAVITQu4vr4xnSDxMaL"}
# post filters: pitch shift via asetrate+atempo
F={"M":"asetrate=44100*1.06,aresample=44100,atempo=1/1.06","O":"asetrate=44100*1.04,aresample=44100,atempo=1/1.04",
   "N":"asetrate=44100*1.22,aresample=44100,atempo=1/1.16","G":"asetrate=44100*0.93,aresample=44100,atempo=1/0.93","A":"anull"}
only=set(sys.argv[1:])
for i,s,t in json.load(open("lines.json")):
    if only and i not in only: continue
    raw=f"audio/raw_{i}.mp3"
    if not os.path.exists(raw) or only:
        body=json.dumps({"text":t,"model_id":"eleven_v3","language_code":"he"})
        r=subprocess.run(["curl","-sS","-o",raw,"-w","%{http_code}","-X","POST",f"https://api.elevenlabs.io/v1/text-to-speech/{V[s]}?output_format=mp3_44100_128","-H","Content-Type: application/json","-d",body],capture_output=True,text=True)
        if r.stdout!="200": print(i,"FAIL",r.stdout,open(raw).read()[:200]); continue
    f=F[s]
    if i in("g3","g4"): f+=",highpass=f=350,lowpass=f=3000,acompressor,volume=1.3"
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",raw,"-af",f+",silenceremove=start_periods=1:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse,loudnorm=I=-16:TP=-1.5","-ar","44100","-ac","1",f"audio/{i}.wav"],check=True)
    d=subprocess.run(["ffprobe","-v","0","-show_entries","format=duration","-of","csv=p=0",f"audio/{i}.wav"],capture_output=True,text=True).stdout.strip()
    print(i,s,d)
