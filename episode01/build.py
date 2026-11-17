#!/usr/bin/env python3
"""Edit & mix 'שומרי הרקיע' episode 1 from generated clips, stills, dialogue, music and sfx."""
import json, os, subprocess, sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
OFF = json.load(open("offsets.json"))
LINES = {l[0]: l for l in json.load(open("lines.json"))}
W, H, FPS = 854, 480, 30
os.makedirs("work/seg", exist_ok=True)


def dur(f):
    return float(subprocess.run(["ffprobe", "-v", "0", "-show_entries", "format=duration", "-of", "csv=p=0", f],
                                capture_output=True, text=True).stdout)


def clip(name, a=0, b=None, **kw):
    return dict(kind="clip", src=f"clips/{name}.mp4", a=a, b=b, **kw)


def still(key, d, move="in", **kw):
    src = f"keys/{key}.png" if os.path.exists(f"keys/{key}.png") else f"refs/{key}.png"
    return dict(kind="still", src=src, dur=d, move=move, **kw)


def lip(track, *lines, **kw):
    """Cut of a lip-sync clip covering the given consecutive lines of its track."""
    s = OFF[lines[0]]["start"]; e = OFF[lines[-1]]["start"] + OFF[lines[-1]]["dur"]
    a = max(0, s - 0.3); b = e + 0.4
    if b - a < 1.4: b = a + 1.4
    evs = [(l, OFF[l]["start"] - a) for l in lines]
    return dict(kind="clip", src=f"clips/{track}.mp4", a=a, b=b, lines=evs, native=0, **kw)


def wan_or_still(name, key, d, **kw):
    return clip(name, 0, d, **kw) if os.path.exists(f"clips/{name}.mp4") else still(key, d, **kw)


TITLE = "שומרי הרקיע"
EDL = [
    # ---------- COLD OPEN ----------
    clip("w01", 0, 7.5, native=1.0, music=("", 0)),
    clip("w02", 0, 11.8, native=0.5, lines=[("g1", 0.9)]),
    clip("w03", 0, 8, native=0.9, lines=[("g2", 0.5)]),
    clip("w04", 0, 6, native=0.8, fadeout=0.8),
    # ---------- TITLE ----------
    clip("w05", 0, 12, native=0, title=True, fadein=0.6, fadeout=0.8, music=("theme", 0.95)),
    # ---------- SCENE 1: arrival ----------
    clip("w10", 0, 6.5, native=0.6, amb=("rain_out", 0.35), music=("", 0)),
    lip("L11", "a1"), lip("L12", "m1"), lip("L11", "a2"),
    lip("L13", "o1"), lip("L12", "m2"), lip("L13", "o2"),
    clip("w14", 0, 5.5, native=0.6),
    lip("L13", "o3"), lip("L12", "m3"),
    # ---------- SCENE 2: the station ----------
    clip("w15", 0, 9, native=0.7, music=("mystery", 0.22)),
    clip("w20", 0, 7, native=0.6),
    lip("L22", "o4"), lip("L21", "m4"),
    clip("w23", 0, 6, native=0.5, amb=("rain_room", 0.30)),
    lip("L28", "o5"),
    clip("w24", 0, 5.6, native=0.7, sfx=[("pop", 5.0, 0.9)]),
    clip("w25", 0, 7, native=0.9),
    lip("L26", "n1"), lip("L27", "m5"), lip("L26", "n2"), lip("L27", "m6"), lip("L29", "n3"),
    clip("w31", 0, 6, native=0.4),
    # ---------- SCENE 3: weather vs climate ----------
    lip("L30", "m7"), lip("L26", "n4"), lip("L27", "m8"), lip("L26", "n5"), lip("L28", "o6"),
    lip("L32", "n6"), lip("L27", "m9"), lip("L26", "n7"), lip("L28", "o7"), lip("L32", "n8"),
    lip("L27", "m10"), lip("L29", "n9"), lip("L28", "o8"), lip("L26", "n10"),
    # ---------- SCENE 4: forty years of dots ----------
    lip("L32", "n11", music=("wonder", 0.40)),
    clip("w40", 0, 8, native=0.6, sfx=[("chime", 1.0, 0.5)]),
    lip("L42", "o9"),
    clip("w41", 0, 9.5, native=0.5, lines=[("n12", 1.2)]),
    wan_or_still("w43", "k43", 6, native=0.6, sfx=[("chime", 0.3, 0.7)]),
    lip("L44", "m11"), lip("L45", "n13"), lip("L44", "m12"), lip("L45", "n14"),
    lip("L44", "m13"), lip("L45", "n15"),
    # ---------- SCENE 5: the lock ----------
    wan_or_still("w50", "k50", 6.2, native=0.5, lines=[("m14", 0.8)], music=("mystery", 0.20, 35)),
    lip("L51", "o10", "o11", sfx=[("locked", 3.75, 0.8)]),
    lip("L27", "m15"),
    lip("L52", "m16", sfx=[("pages", 0.0, 0.5)]),
    lip("L28", "o12"), lip("L27", "m17"), lip("L26", "n16"), lip("L44", "m18"),
    wan_or_still("w54", "k54", 6, move="in_fast", native=0.8, sfx=[("lock", 0.2, 0.9)]),
    lip("L28", "o13"),
    # ---------- SCENE 6: the map room ----------
    wan_or_still("w60", "k60", 7, native=0.6, music=("tender", 0.35)),
    still("loc_map", 4.5, move="pan"),
    wan_or_still("w61", "k61", 5, native=0.5),
    lip("L62", "m19"), lip("L63", "n17"),
    wan_or_still("w64", "k64", 12, native=0.6, lines=[("g3", 1.2)]),
    still("loc_map", 5, move="in", lines=[("g4", 1.6)], sfx=[("tape", 3.5, 0.8)]),
    lip("L62", "m20"),
    # ---------- SCENE 7: cliffhanger ----------
    wan_or_still("w70", "k70", 5, move="in_fast", native=0.8, sfx=[("tick", 0.0, 0.9)], music=("ominous", 0.45)),
    lip("L28", "o14"), lip("L72", "n18"), lip("L73", "m21"), lip("L72", "n19"),
    wan_or_still("w73", "k73", 5.5, native=0.4, lines=[("a3", 0.4)]),
    lip("L73", "m22"),
    wan_or_still("w74", "k74", 10, native=0.7, sfx=[("wind", 0.0, 0.8)], amb=("", 0), fadeout=1.5),
    dict(kind="end", dur=6),
]


def render_segment(i, s):
    out = f"work/seg/{i:03d}.mp4"
    vf_tail = f"scale={W}:{H}:flags=lanczos,setsar=1,fps={FPS},format=yuv420p"
    if s["kind"] == "clip":
        b = s["b"] if s["b"] is not None else dur(s["src"])
        b = min(b, dur(s["src"]))
        d = b - s["a"]
        cmd = ["-ss", f"{s['a']:.3f}", "-t", f"{d:.3f}", "-i", s["src"]]
        vf = vf_tail
    elif s["kind"] == "still":
        d = s["dur"]; n = int(d * FPS)
        z = {"in": f"1.0+0.12*on/{n}", "in_fast": f"1.0+0.25*on/{n}", "pan": "1.18"}[s["move"]]
        x = f"(iw-iw/zoom)*on/{n}" if s["move"] == "pan" else "iw/2-(iw/zoom/2)"
        cmd = ["-i", s["src"]]
        vf = (f"scale=2752:1536,zoompan=z='{z}':x='{x}':y='ih/2-(ih/zoom/2)':d={n}:s={W}x{H}:fps={FPS},"
              + vf_tail)
    else:  # end card
        d = s["dur"]
        cmd = ["-f", "lavfi", "-t", f"{d}", "-i", f"color=c=black:s={W}x{H}:r={FPS}"]
        vf = (f"drawtext=fontfile=fonts/secular-one.ttf:text='{TITLE}':fontcolor=white:fontsize=54:"
              f"x=(w-tw)/2:y=h/2-70:alpha='if(lt(t,1),t,1)',"
              f"drawtext=fontfile=fonts/heebo.ttf:text='בפרק הבא   האוקיינוס שמעלינו':fontcolor=0xF2C14E:"
              f"fontsize=26:x=(w-tw)/2:y=h/2+20:alpha='if(lt(t,1.5),0,min(1,t-1.5))',"
              + vf_tail)
    if s.get("title"):
        vf += (f",drawtext=fontfile=fonts/secular-one.ttf:text='{TITLE}':fontcolor=white:fontsize=78:"
               f"borderw=3:bordercolor=0x1d2b3a:x=(w-tw)/2:y=h/2-95:"
               f"alpha='if(lt(t,1.5),0,if(lt(t,3),(t-1.5)/1.5,if(gt(t,10.6),max(0,(11.6-t)),1)))',"
               f"drawtext=fontfile=fonts/heebo.ttf:text='פרק ראשון   התחנה הנטושה':fontcolor=0xF2C14E:fontsize=30:"
               f"borderw=2:bordercolor=0x1d2b3a:x=(w-tw)/2:y=h/2+5:"
               f"alpha='if(lt(t,3.5),0,if(lt(t,5),(t-3.5)/1.5,if(gt(t,10.6),max(0,(11.6-t)),1)))'")
    if s.get("fadein"): vf += f",fade=t=in:st=0:d={s['fadein']}"
    if s.get("fadeout"): vf += f",fade=t=out:st={d - s['fadeout']:.3f}:d={s['fadeout']}"
    vf += ",eq=saturation=1.05:contrast=1.03"
    if not os.path.exists(out) or "-f" in sys.argv:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *cmd, "-vf", vf, "-an", "-c:v", "libx264",
                        "-preset", "medium", "-crf", "17", "-r", str(FPS), out], check=True)
    return out, dur(out)


# ---------- video ----------
segs, t = [], 0.0
timeline = []  # (shot, start, dur)
for i, s in enumerate(EDL):
    out, d = render_segment(i, s)
    segs.append(out); timeline.append((s, t, d)); t += d
TOTAL = t
open("work/concat.txt", "w").write("".join(f"file 'seg/{os.path.basename(x)}'\n" for x in segs))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "work/concat.txt",
                "-c", "copy", "work/video.mp4"], check=True)
print(f"video {TOTAL:.1f}s, {len(EDL)} shots")

# ---------- audio events ----------
inputs, filt, labels = [], [], []
NIN = [0]


def add(src, start, vol, extra="", a=None, d=None, loop=False):
    k = NIN[0]; NIN[0] += 1
    if loop: inputs.extend(["-stream_loop", "-1"])
    if a is not None: inputs.extend(["-ss", f"{a:.3f}"])
    if d is not None: inputs.extend(["-t", f"{d:.3f}"])
    inputs.extend(["-i", src])
    chain = f"[{k}:a]aresample=48000,aformat=channel_layouts=stereo{extra},volume={vol},adelay={int(start*1000)}:all=1[a{k}]"
    filt.append(chain); labels.append(f"[a{k}]")


subs = []
amb = None; mus = None


def close_bed(bed, end):
    if not bed: return
    name, vol, start, off, folder = bed
    d = end - start
    if d <= 0.2: return
    fo = min(2.0, d / 3)
    add(f"{folder}/{name}.mp3", start, vol, f",afade=t=in:d=1.0,afade=t=out:st={max(0, d - fo):.2f}:d={fo:.2f}",
        a=off, d=d, loop=(folder == "sfx"))


for s, st, d in timeline:
    if "amb" in s:
        close_bed(amb, st)
        amb = (s["amb"][0], s["amb"][1], st, 0, "sfx") if s["amb"][0] else None
    if "music" in s:
        m = s["music"]
        if mus: close_bed(mus, st + 1.5)
        mus = (m[0], m[1], st, m[2] if len(m) > 2 else 0, "music") if m[0] else None
    if s["kind"] == "clip" and s.get("native", 0) > 0:
        add(s["src"], st, s["native"], a=s["a"], d=d)
    for l, off in s.get("lines", []):
        extra = ",highpass=f=300,lowpass=f=2600,aecho=0.8:0.7:60|120:0.35|0.2" if l == "a3" else ""
        add(f"audio/{l}.wav", st + off, 1.0, extra)
        ld = dur(f"audio/{l}.wav")
        txt = LINES[l][2]
        import re
        txt = re.sub(r"\[[^\]]*\]", "", txt).replace("ְ", "").strip()
        txt = "".join(c for c in txt if not ("֑" <= c <= "ׇ"))  # strip niqqud
        subs.append((st + off, st + off + ld, " ".join(txt.split())))
    for name, off, vol in s.get("sfx", []):
        add(f"sfx/{name}.mp3", st + off, vol)
close_bed(amb, TOTAL); close_bed(mus, TOTAL)

n = len(labels)
filt.append("".join(labels) + f"amix=inputs={n}:normalize=0:dropout_transition=0,apad=whole_dur={TOTAL:.3f},"
            f"atrim=0:{TOTAL:.3f},alimiter=limit=0.89,loudnorm=I=-16:TP=-1.5:LRA=11[mix]")
open("work/audio_filter.txt", "w").write(";\n".join(filt))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex_script", "work/audio_filter.txt",
                "-map", "[mix]", "-ar", "48000", "-c:a", "aac", "-b:a", "192k", "work/audio.m4a"], check=True)
print(f"audio: {n} events")


# ---------- subtitles ----------
def ts(x):
    h, r = divmod(x, 3600); m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"


subs.sort()
with open("episode01_he.srt", "w") as f:
    for i, (a, b, txt) in enumerate(subs, 1):
        f.write(f"{i}\n{ts(a)} --> {ts(b + 0.2)}\n‫{txt}‬\n\n")

# ---------- mux ----------
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "work/video.mp4", "-i", "work/audio.m4a",
                "-i", "episode01_he.srt", "-map", "0:v", "-map", "1:a", "-map", "2:s", "-c:v", "libx264", "-preset", "slow", "-crf", "21", "-c:a", "copy",
                "-c:s", "mov_text", "-metadata:s:s:0", "language=heb", "-metadata", f"title={TITLE} - פרק 1",
                "-movflags", "+faststart", "episode01.mp4"], check=True)
print("done", dur("episode01.mp4"))
