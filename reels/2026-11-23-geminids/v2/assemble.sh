# HD opening (frames 0-181) + v3 middle (182-2048) + HD ending (2049-2149), one encode, then audio mix
set -e
cd "$(dirname "$0")"
ffmpeg -v error -y -i s1_video.mp4 -i video.mp4 -filter_complex "\
[0:v]trim=start_frame=0:end_frame=182,setpts=PTS-STARTPTS[a];\
[1:v]trim=start_frame=182,setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1:a=0[v]" -map "[v]" -r 30 -c:v libx264 -preset slow -crf 17 -pix_fmt yuv420p final_v.mp4
D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 final_v.mp4)
FO=$(python3 -c "print($D-1.8)")
ffmpeg -v error -y -i final_v.mp4 -i voice_out.wav -i sfx.wav -i ../r10/music.mp3 -filter_complex "\
[1:a]aformat=sample_rates=48000:channel_layouts=stereo,asplit=2[v][vsc];\
[2:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=0.8[s];\
[3:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=0.30,afade=t=out:st=$FO:d=1.8[m];\
[m][vsc]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=350[md];\
[v][s][md]amix=inputs=3:normalize=0:duration=first,alimiter=limit=0.95[a]" \
 -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 192k -t $D -movflags +faststart geminids_reel_v4.mp4
ffprobe -v error -show_entries format=duration -of csv=p=0 geminids_reel_v4.mp4
