const TZ = 'Asia/Jerusalem';
const DAY_SHORT = ['א׳', 'ב׳', 'ג׳', 'ד׳', 'ה׳', 'ו׳', 'ש׳'];
const DAY_LONG = ['ראשון', 'שני', 'שלישי', 'רביעי', 'חמישי', 'שישי', 'שבת'];

export function parseRun(iso: string): number {
  // "2026-11-09T00:00Z"
  return Date.parse(iso.replace('Z', ':00Z'));
}

interface Parts {
  y: number;
  mo: number;
  d: number;
  h: number;
  dow: number;
}

const fmt = new Intl.DateTimeFormat('en-GB', {
  timeZone: TZ, year: 'numeric', month: 'numeric', day: 'numeric', hour: 'numeric', hourCycle: 'h23', weekday: 'short',
});
const WD: Record<string, number> = { Sun: 0, Mon: 1, Tue: 2, Wed: 3, Thu: 4, Fri: 5, Sat: 6 };

export function localParts(ms: number): Parts {
  const p: Record<string, string> = {};
  for (const x of fmt.formatToParts(new Date(ms))) p[x.type] = x.value;
  return { y: +p.year, mo: +p.month, d: +p.day, h: +p.hour, dow: WD[p.weekday] ?? 0 };
}

export function dayKey(ms: number) {
  const p = localParts(ms);
  return `${p.y}-${p.mo}-${p.d}`;
}

export function shortDay(ms: number) {
  const p = localParts(ms);
  return `${DAY_SHORT[p.dow]} ${p.d}.${p.mo}`;
}

export function longValid(ms: number) {
  const p = localParts(ms);
  return `יום ${DAY_LONG[p.dow]} ${p.d}.${p.mo} · ${String(p.h).padStart(2, '0')}:00`;
}

export function runLabel(ms: number) {
  const d = new Date(ms);
  const p2 = (n: number) => String(n).padStart(2, '0');
  return `${p2(d.getUTCHours())}Z · ${p2(d.getUTCDate())}.${p2(d.getUTCMonth() + 1)}.${d.getUTCFullYear()}`;
}
