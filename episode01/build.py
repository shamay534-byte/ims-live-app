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


def rlip(lid, **kw):
    """Single-line re-recorded lip-sync clip (line starts 0.5s into the clip)."""
    d = dur(f"audio/{lid}.wav")
    return dict(kind="clip", src=f"clips/R_{lid}.mp4", a=0.2, b=0.5 + d + 0.45, lines=[(lid, 0.3)], native=0, **kw)


def frames(folder, d, **kw):
    return dict(kind="frames", src=folder, dur=d, **kw)


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
    rlip("o5"),
    clip("w24", 0, 5.6, native=0.7, sfx=[("pop", 5.0, 0.9)]),
    clip("w25", 0, 7, native=0.9),
    lip("L26", "n1"), lip("L27", "m5"), rlip("n2"), lip("L27", "m6"), lip("L29", "n3"),
    clip("w31", 0, 6, native=0.4),
    # ---------- SCENE 3: weather vs climate ----------
    lip("L30", "m7"), lip("L26", "n4"), lip("L27", "m8"), lip("L26", "n5"), lip("L28", "o6"),
    lip("L32", "n6"), lip("L27", "m9"), lip("L26", "n7"), lip("L28", "o7"), lip("L32", "n8"),
    lip("L27", "m10"), lip("L29", "n9"), rlip("o8"), lip("L26", "n10"),
    frames("work/concept", 7.0, sfx=[("chime", 0.35, 0.6)]),
    # ---------- SCENE 4: forty years of dots ----------
    lip("L32", "n11", music=("wonder", 0.40)),
    clip("w40", 0, 8, native=0.6, sfx=[("chime", 1.0, 0.5)]),
    lip("L42", "o9"),
    clip("w41", 0, 9.5, native=0.5, lines=[("n12", 1.2)]),
    wan_or_still("w43", "k43", 6, native=0.6, sfx=[("chime", 0.3, 0.7)]),
    rlip("m11"), lip("L45", "n13"), lip("L44", "m12"), rlip("n14"),
    lip("L44", "m13"), lip("L45", "n15"),
    # ---------- SCENE 5: the lock ----------
    wan_or_still("w50", "k50", 6.2, native=0.5, lines=[("m14", 0.8)], music=("mystery", 0.20, 35)),
    lip("L51", "o10", sfx=[("locked", 2.9, 0.8)]), rlip("o11"),
    rlip("m15"),
    lip("L52", "m16", sfx=[("pages", 0.0, 0.5)]),
    rlip("o12"), rlip("m17"), lip("L26", "n16"), lip("L44", "m18"),
    wan_or_still("w54", "k54", 6, move="in_fast", native=0.8, sfx=[("lock", 0.2, 0.9)]),
    lip("L28", "o13"),
    # ---------- SCENE 6: the map room ----------
    wan_or_still("w60", "k60", 7, native=0.6, music=("tender", 0.35)),
    still("loc_map", 4.5, move="pan"),
    wan_or_still("w61", "k61", 5, native=0.5),
    lip("L62", "m19"), lip("L63", "n17"),
    wan_or_still("w64", "k64", 12, native=0.6, lines=[("g3", 1.2)]),
    still("loc_map", 5, move="in", lines=[("g4", 1.6)], sfx=[("tape", 3.5, 0.8)]),
    rlip("m20"),
    # ---------- SCENE 7: cliffhanger ----------
    wan_or_still("w70", "k70", 5, move="in_fast", native=0.8, sfx=[("tick", 0.0, 0.9)], music=("ominous", 0.45)),
    lip("L28", "o14"), lip("L72", "n18"), lip("L73", "m21"), lip("L72", "n19"),
    wan_or_still("w73", "k73", 5.5, native=0.4, lines=[("a3", 0.4)]),
    lip("L73", "m22"),
    wan_or_still("w74", "k74", 10, native=0.7, sfx=[("wind", 0.0, 0.8)], amb=("", 0), fadeout=1.5),
    frames("work/endcard", 7.0),
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
    elif s["kind"] == "frames":
        d = s["dur"]
        cmd = ["-framerate", str(FPS), "-i", f"{s['src']}/%04d.png"]
        vf = vf_tail
    else:  # end card
        d = s["dur"]
        cmd = ["-f", "lavfi", "-t", f"{d}", "-i", f"color=c=black:s={W}x{H}:r={FPS}"]
        vf = (f"drawtext=fontfile=fonts/secular-one.ttf:text='{TITLE}':fontcolor=white:fontsize=54:"
              f"x=(w-tw)/2:y=h/2-70:alpha='if(lt(t,1),t,1)',"
              f"drawtext=fontfile=fonts/heebo.ttf:text='בפרק הבא   האוקיינוס שמעלינו':fontcolor=0xF2C14E:"
              f"fontsize=26:x=(w-tw)/2:y=h/2+20:alpha='if(lt(t,1.5),0,min(1,t-1.5))',"
              + vf_tail)
    if s.get("title"):
        cmd += ["-framerate", str(FPS), "-i", "work/title/%04d.png"]
        vf = f"[0:v]{vf}[b];[b][1:v]overlay=0:0:shortest=1"
    if s.get("fadein"): vf += f",fade=t=in:st=0:d={s['fadein']}"
    if s.get("fadeout"): vf += f",fade=t=out:st={d - s['fadeout']:.3f}:d={s['fadeout']}"
    vf += ",eq=saturation=1.05:contrast=1.03"
    if not os.path.exists(out) or "-f" in sys.argv:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *cmd, "-filter_complex" if s.get("title") else "-vf", vf, "-an", "-c:v", "libx264",
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
        subs.append((st + off, st + off + ld, l))
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
import re

SUB_OVERRIDE = {"g1": "יומן תחנה, רישום 14,611. לחץ האוויר צונח. זו לא עוד סערה. חיכיתי לה ארבעים שנה.",
                "m4": "14 מעלות. הוא מדד פה דברים.",
                "n6": "מה שאת רואה בחלון, זה מזג אוויר. מה שקורה עכשיו, פה, היום. גשם, 14 מעלות, רוח. מחר זה יכול להתהפך.",
                "o10": "קל. 14 מעלות. אפס, אחת, ארבע.",
                "n13": "האופי של המקום. חורף גשום ומתון, קיץ חם ויבש. ככה זה פה בדרך כלל, גם אם היום נראה אחרת. וכדי לקרוא לזה אקלים, צריך לפחות 30 שנה של מדידות.",
                "n8": "היום הוא צרח כמו קומקום. זה מצב הרוח שלו, היום. אבל אם תכירי אותו 30 שנה, תדעי איזה בן אדם הוא באמת.",
                "n10": "את זה נדע בעוד 30 שנה.", "n19": "שאת צריכה ללמוד הרבה. ואין לנו 30 שנה."}
FIXES = json.load(open("fix_lines.json")) if os.path.exists("fix_lines.json") else {}


def sub_text(l):
    t = SUB_OVERRIDE.get(l) or FIXES.get(l) or LINES[l][2]
    t = re.sub(r"\[[^\]]*\]", "", t)
    t = t.replace("־", " ")                                   # maqaf -> space (never glue words)
    t = "".join(c for c in t if not ("֑" <= c <= "ֽ" or "ֿ" <= c <= "ׇ"))  # niqqud only
    t = t.replace("...", "…")
    return " ".join(t.split())


def chunks(text, maxc=40):
    """Split into subtitle cards of <=2 lines x maxc chars, breaking at sentence then word boundaries."""
    sents = re.findall(r"[^.?!…]+[.?!…]*", text)
    cards, cur = [], ""
    for snt in (x.strip() for x in sents if x.strip()):
        if len(cur) + len(snt) + 1 <= 2 * maxc: cur = (cur + " " + snt).strip()
        else:
            if cur: cards.append(cur)
            while len(snt) > 2 * maxc:
                cut = snt.rfind(" ", 0, 2 * maxc); cards.append(snt[:cut]); snt = snt[cut + 1:]
            cur = snt
    if cur: cards.append(cur)
    out = []
    for c in cards:
        if len(c) > maxc:
            mid = len(c) // 2
            cut = min((k for k in range(len(c)) if c[k] == " "), key=lambda k: abs(k - mid))
            c = c[:cut] + "\\N" + c[cut + 1:]
        out.append(c)
    return out


def ass_t(x):
    h, r = divmod(max(0, x), 3600); m, s_ = divmod(r, 60)
    return f"{int(h)}:{int(m):02d}:{s_:05.2f}"


def srt_t(x):
    h, r = divmod(x, 3600); m, s_ = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s_):02d},{int((s_ % 1) * 1000):03d}"


events = []
for a, b, l in sorted(subs):
    cards = chunks(sub_text(l)); total = sum(len(c) for c in cards); t0 = a
    for c in cards:
        t1 = t0 + (b - a) * len(c) / total
        events.append((t0, t1 + (0.25 if c is cards[-1] else 0.0), c)); t0 = t1
# avoid overlaps
for k in range(len(events) - 1):
    if events[k][1] > events[k + 1][0] - 0.04:
        events[k] = (events[k][0], events[k + 1][0] - 0.04, events[k][2])

ass = [
    "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 2", "",
    "[V4+ Styles]",
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
    "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
    "Style: Default,Heebo,25,&H00FFFFFF,&H00FFFFFF,&H00341E10,&H96000000,-1,0,0,0,100,100,0,0,1,2.4,1.2,2,40,40,20,1",
    "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
for a, b, c in events:
    ass.append(f"Dialogue: 0,{ass_t(a)},{ass_t(b)},Default,,0,0,0,,{c}")
open("episode01_he.ass", "w").write("\n".join(ass) + "\n")
with open("episode01_he.srt", "w") as f:
    for k, (a, b, c) in enumerate(events, 1):
        f.write(f"{k}\n{srt_t(a)} --> {srt_t(b)}\n{c.replace(chr(92) + 'N', chr(10))}\n\n")
print("subtitle cards:", len(events))

# ---------- mux (subtitles burned in, plus soft track) ----------
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "work/video.mp4", "-i", "work/audio.m4a",
                "-i", "episode01_he.srt", "-filter_complex", "[0:v]subtitles=episode01_he.ass:fontsdir=fonts[v]",
                "-map", "[v]", "-map", "1:a", "-map", "2:s", "-c:v", "libx264", "-preset", "slow", "-crf", "21",
                "-c:a", "copy", "-c:s", "mov_text", "-metadata:s:s:0", "language=heb",
                "-metadata", f"title={TITLE} - פרק 1", "-movflags", "+faststart", "episode01.mp4"], check=True)
print("done", dur("episode01.mp4"))
