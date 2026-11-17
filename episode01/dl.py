import sys,json,subprocess
u=json.load(open("urls.json"))
for a in sys.argv[1:]:
    k,url=a.split("=",1)
    if not url.startswith("http"): url="https://v3b.fal.media/files/b/"+url
    ext=url.rsplit(".",1)[-1]
    d="keys" if ext in("png","jpg") else "clips"
    subprocess.run(["curl","-sS","-o",f"{d}/{k}.{ext}",url],check=True); u[k]=url
json.dump(u,open("urls.json","w"),indent=0)
