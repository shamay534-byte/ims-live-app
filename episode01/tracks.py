import json,subprocess,sys
T={"t11":["a1","a2"],"t12":["m1","m2","m3"],"t13":["o1","o2","o3"],"t22":["o4"],"t21":["m4"],
"t26":["n1","n2","n4","n5","n7","n10","n16"],"t27":["m5","m6","m8","m9","m10","m15","m17"],
"t28":["o5","o6","o7","o8","o12","o13","o14"],"t29":["n3","n9"],"t30":["m7"],"t32":["n6","n8","n11"],
"t45":["n12","n13","n14","n15"],"t44":["m11","m12","m13","m18"],"t42":["o9"],"t51":["o10","o11"],"t52":["m16"],
"t62":["m19","m20"],"t63":["n17"],"t72":["n18","n19"],"t73":["m21","m22"]}
LEAD,GAP=0.8,1.6
def dur(f): return float(subprocess.run(["ffprobe","-v","0","-show_entries","format=duration","-of","csv=p=0",f],capture_output=True,text=True).stdout)
off={}
for t,ls in T.items():
    ins=[];fc=[];pos=LEAD
    for n,l in enumerate(ls):
        d=dur(f"audio/{l}.wav"); off[l]={"track":t,"start":round(pos,3),"dur":round(d,3)}
        ins+=["-i",f"audio/{l}.wav"]; fc.append(f"[{n}]adelay={int(pos*1000)}:all=1[a{n}]"); pos+=d+GAP
    total=max(pos-GAP+1.0,5.4)
    fc.append("".join(f"[a{n}]" for n in range(len(ls)))+f"amix=inputs={len(ls)}:normalize=0,apad=whole_dur={total},atrim=0:{total}[o]")
    subprocess.run(["ffmpeg","-y","-loglevel","error",*ins,"-filter_complex",";".join(fc),"-map","[o]","-ar","44100","-ac","1","-b:a","128k",f"tracks/{t}.mp3"],check=True)
    print(t,round(total,1),subprocess.run(["stat","-c","%s",f"tracks/{t}.mp3"],capture_output=True,text=True).stdout.strip())
json.dump(off,open("offsets.json","w"),indent=0)
