'use client';
import { useMemo, useRef } from 'react';
import { Icon } from './Icons';
import { dayKey, longValid, shortDay } from '@/lib/time';

export interface Tick {
  ms: number; // valid time
  lead: number; // hours since run start
}

interface Props {
  ticks: Tick[];
  index: number;
  onChange: (i: number) => void;
  playing: boolean;
  onTogglePlay: () => void;
}

export function Timeline({ ticks, index, onChange, playing, onTogglePlay }: Props) {
  const track = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const days = useMemo(
    () => ticks.map((t, i) => (i === 0 || dayKey(t.ms) !== dayKey(ticks[i - 1].ms) ? shortDay(t.ms) : '')),
    [ticks],
  );

  const pickAt = (clientX: number) => {
    const el = track.current;
    if (!el || !ticks.length) return;
    const r = el.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
    const i = Math.min(ticks.length - 1, Math.floor(f * ticks.length));
    if (i !== index) onChange(i);
  };

  const cur = ticks[index];

  return (
    <div className="timeline glass">
      <button className="play" onClick={onTogglePlay} aria-label={playing ? 'עצירה' : 'הפעלת אנימציה'}>
        <Icon name={playing ? 'pause' : 'play'} size={22} />
      </button>
      <div className="tl-now">
        <span className="tl-valid">{cur ? longValid(cur.ms) : '—'}</span>
        <span className="tl-lead">{cur ? `+${cur.lead} ש׳ מתחילת הריצה · שעון ישראל` : ''}</span>
      </div>
      {/* RTL row: this button sits right of the left-to-right track, i.e. next to later times */}
      <button className="ibtn tl-step" onClick={() => onChange(Math.min(ticks.length - 1, index + 1))} aria-label="שעה הבאה">
        <Icon name="next" />
      </button>
      <div
        className="tl-track"
        dir="ltr"
        ref={track}
        role="slider"
        tabIndex={0}
        aria-label="זמן תחזית"
        aria-valuemin={0}
        aria-valuemax={Math.max(0, ticks.length - 1)}
        aria-valuenow={index}
        aria-valuetext={cur ? longValid(cur.ms) : ''}
        onPointerDown={(e) => {
          dragging.current = true;
          (e.target as Element).setPointerCapture?.(e.pointerId);
          pickAt(e.clientX);
        }}
        onPointerMove={(e) => dragging.current && pickAt(e.clientX)}
        onPointerUp={() => (dragging.current = false)}
        onPointerCancel={() => (dragging.current = false)}
      >
        {ticks.map((t, i) => (
          <span key={t.lead} className={`tick${days[i] ? ' day' : ''}${i === index ? ' on' : ''}`}>
            {days[i] && <b>{days[i]}</b>}
            <i />
          </span>
        ))}
      </div>
      <button className="ibtn tl-step" onClick={() => onChange(Math.max(0, index - 1))} aria-label="שעה קודמת">
        <Icon name="prev" />
      </button>
    </div>
  );
}
