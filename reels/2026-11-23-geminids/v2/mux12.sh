set -e
cd "$(dirname "$0")"
printf "file 'q0.mp4'\nfile 'q540.mp4'\nfile 'q1080.mp4'\nfile 'q1620.mp4'\n" > parts/list.txt
ffmpeg -v error -y -f concat -safe 0 -i parts/list.txt -c copy video.mp4
D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 video.mp4)
FO=$(python3 -c "print($D-1.8)")
ffmpeg -v error -y -i video.mp4 -i voice_out.wav -i sfx.wav -i ../r10/music.mp3 -filter_complex "\
[1:a]aformat=sample_rates=48000:channel_layouts=stereo,asplit=2[v][vsc];\
[2:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=0.8[s];\
[3:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=0.30,afade=t=out:st=$FO:d=1.8[m];\
[m][vsc]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=350[md];\
[v][s][md]amix=inputs=3:normalize=0:duration=first,alimiter=limit=0.95[a]" \
 -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 192k -t $D -movflags +faststart geminids_reel_v2.mp4
ffprobe -v error -show_entries format=duration -of csv=p=0 geminids_reel_v2.mp4
