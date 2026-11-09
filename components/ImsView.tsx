'use client';
import { useEffect, useState } from 'react';
import type { Frame } from '@/lib/ims';

/**
 * Shows the pre-rendered IMS map for the selected frame. All frames of the current selection
 * are preloaded so stepping / animating is instant.
 */
export default function ImsView({ frames, index, title }: { frames: Frame[]; index: number; title: string }) {
  const [loaded, setLoaded] = useState<Record<string, boolean>>({});
  const [failed, setFailed] = useState<Record<string, boolean>>({});

  useEffect(() => {
    let dead = false;
    const imgs = frames.map((f) => {
      const im = new Image();
      im.onload = () => !dead && setLoaded((s) => ({ ...s, [f.url]: true }));
      im.onerror = () => !dead && setFailed((s) => ({ ...s, [f.url]: true }));
      im.src = f.url;
      return im;
    });
    return () => {
      dead = true;
      imgs.forEach((im) => (im.onload = im.onerror = null));
    };
  }, [frames]);

  const cur = frames[index];
  const done = frames.filter((f) => loaded[f.url]).length;

  return (
    <div className="ims-view">
      {cur && !failed[cur.url] && (
        // eslint-disable-next-line @next/next/no-img-element
        <img key={cur.url} src={cur.url} alt={`${title} · +${cur.lead} ש׳`} className="ims-img" />
      )}
      {cur && failed[cur.url] && <div className="ims-msg">המפה לשעה זו לא נמצאה באתר השירות המטאורולוגי</div>}
      {!cur && <div className="ims-msg">אין מפות זמינות לבחירה זו</div>}
      {frames.length > 0 && done < frames.length && (
        <div className="ims-progress" role="status">
          טוען מפות {done}/{frames.length}
        </div>
      )}
    </div>
  );
}
