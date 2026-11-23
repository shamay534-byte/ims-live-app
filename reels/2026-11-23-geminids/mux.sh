set -e
cd "$(dirname "$0")"
printf "file 'p0.mp4'\nfile 'p572.mp4'\nfile 'p1144.mp4'\nfile 'p1716.mp4'\n" > parts/list.txt
ffmpeg -v error -y -f concat -safe 0 -i parts/list.txt -c copy video.mp4
SP=1.07
ffmpeg -v error -y -i video.mp4 -i voice_cut.wav -i sfx.wav -i music.mp3 -filter_complex "\
[1:a]aformat=sample_rates=48000:channel_layouts=stereo,asplit=2[v][vsc];\
[2:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=0.8[s];\
[3:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=0.30,afade=t=out:st=74.5:d=1.7[m];\
[m][vsc]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=350[md];\
[v][s][md]amix=inputs=3:normalize=0:duration=first,alimiter=limit=0.95,atempo=$SP[a];\
[0:v]setpts=PTS/$SP[vv]" -map "[vv]" -map "[a]" -c:v libx264 -crf 19 -preset medium -pix_fmt yuv420p -r 30 -c:a aac -b:a 192k -movflags +faststart geminids_reel.mp4
ffprobe -v error -show_entries format=duration -of csv=p=0 geminids_reel.mp4
