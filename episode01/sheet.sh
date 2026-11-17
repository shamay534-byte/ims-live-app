#!/bin/bash
# sheet.sh out.jpg k1 k2 ... : contact sheet 4 per row
out=$1; shift; args=(); for k in "$@"; do args+=(-i keys/$k.png); done; n=$#
ffmpeg -y -loglevel error "${args[@]}" -filter_complex "$(for i in $(seq 0 $((n-1))); do echo -n "[$i]scale=480:268,drawtext=text='$(eval echo \${$((i+1))})':fontcolor=yellow:fontsize=20:x=6:y=6:box=1:boxcolor=black@0.6[v$i];"; done)$(for i in $(seq 0 $((n-1))); do echo -n "[v$i]"; done)xstack=inputs=$n:layout=$(python3 -c "print('|'.join(f'{(i%4)*480}_{(i//4)*268}' for i in range($n)))"):fill=black" $out
