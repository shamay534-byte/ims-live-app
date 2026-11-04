import json, sys
from faster_whisper import WhisperModel
name = sys.argv[1]
m = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=4)
import numpy as np, wave; wf=wave.open(sys.argv[2]); a=np.frombuffer(wf.readframes(wf.getnframes()),dtype=np.int16).astype(np.float32)/32768
segs, info = m.transcribe(a, language="he", word_timestamps=True, beam_size=5, vad_filter=False)
out=[]
for s in segs:
    out.append({"start":s.start,"end":s.end,"text":s.text,"words":[{"w":w.word,"s":w.start,"e":w.end} for w in s.words]})
    print(f"{s.start:6.2f}-{s.end:6.2f} {s.text}", flush=True)
json.dump(out, open(sys.argv[3],"w"), ensure_ascii=False, indent=1)
