// Colour scales: value stops -> CSS colours (rgba allowed, so low values can be transparent).
export type Stop = [number, string];
export interface Scale {
  stops: Stop[];
  /** labels shown under the legend bar */
  ticks: number[];
}

function parseColor(c: string): [number, number, number, number] {
  if (c.startsWith('#')) {
    const h = c.slice(1);
    const n = h.length === 3 ? h.split('').map((x) => x + x).join('') : h;
    return [parseInt(n.slice(0, 2), 16), parseInt(n.slice(2, 4), 16), parseInt(n.slice(4, 6), 16), 255];
  }
  const m = c.match(/rgba?\(([^)]+)\)/);
  if (!m) return [0, 0, 0, 0];
  const p = m[1].split(',').map((x) => parseFloat(x));
  return [p[0], p[1], p[2], Math.round((p[3] ?? 1) * 255)];
}

export interface Lut {
  min: number;
  max: number;
  rgba: Uint8ClampedArray; // 1024 * 4
}

const LUT_N = 1024;
const lutCache = new WeakMap<Scale, Lut>();

export function lutFor(scale: Scale): Lut {
  const hit = lutCache.get(scale);
  if (hit) return hit;
  const stops = scale.stops.map(([v, c]) => [v, parseColor(c)] as const);
  const min = stops[0][0];
  const max = stops[stops.length - 1][0];
  const rgba = new Uint8ClampedArray(LUT_N * 4);
  for (let i = 0; i < LUT_N; i++) {
    const v = min + ((max - min) * i) / (LUT_N - 1);
    let k = 0;
    while (k < stops.length - 2 && v > stops[k + 1][0]) k++;
    const [v0, c0] = stops[k];
    const [v1, c1] = stops[k + 1];
    const t = v1 === v0 ? 0 : Math.min(1, Math.max(0, (v - v0) / (v1 - v0)));
    for (let ch = 0; ch < 4; ch++) rgba[i * 4 + ch] = c0[ch] + (c1[ch] - c0[ch]) * t;
  }
  const lut = { min, max, rgba };
  lutCache.set(scale, lut);
  return lut;
}

export function cssGradient(scale: Scale): string {
  const { stops } = scale;
  const min = stops[0][0];
  const max = stops[stops.length - 1][0];
  const parts = stops.map(([v, c]) => `${c} ${(((v - min) / (max - min)) * 100).toFixed(1)}%`);
  return `linear-gradient(90deg, ${parts.join(', ')})`;
}

/** Re-map a scale defined on 0..1 to a data range (used for geopotential height). */
export function stretch(scale: Scale, lo: number, hi: number, tickStep: number): Scale {
  const stops: Stop[] = scale.stops.map(([f, c]) => [lo + f * (hi - lo), c]);
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / tickStep) * tickStep; v <= hi; v += tickStep) ticks.push(v);
  return { stops, ticks };
}

export const SCALES = {
  temp: {
    stops: [
      [-60, '#f5f3ff'], [-45, '#c4b5fd'], [-30, '#8b5cf6'], [-20, '#5b21b6'], [-10, '#1e40af'],
      [-3, '#2563eb'], [0, '#0ea5e9'], [5, '#06b6d4'], [10, '#14b8a6'], [15, '#65a30d'],
      [20, '#ca8a04'], [25, '#ea580c'], [30, '#dc2626'], [35, '#9f1239'], [40, '#701a75'], [46, '#3b0764'],
    ],
    ticks: [-40, -20, -10, 0, 10, 20, 30, 40],
  },
  wind: {
    stops: [
      [0, '#3b4a8a'], [10, '#2f6fb7'], [20, '#1b9aa0'], [30, '#3fae5a'], [45, '#b9b81c'],
      [60, '#e2861a'], [80, '#d33f2b'], [100, '#a3197a'], [130, '#6d28d9'], [180, '#e9d5ff'],
    ],
    ticks: [0, 20, 40, 60, 80, 100, 130],
  },
  precip: {
    stops: [
      [0, 'rgba(96,165,250,0)'], [0.1, 'rgba(96,165,250,0)'], [0.3, 'rgba(125,211,252,.75)'],
      [1, '#3b82f6'], [2, '#1d4ed8'], [4, '#7c3aed'], [8, '#c026d3'], [15, '#e11d48'],
      [30, '#f59e0b'], [50, '#fef08a'],
    ],
    ticks: [0.3, 1, 2, 4, 8, 15, 30, 50],
  },
  clouds: {
    stops: [
      [0, 'rgba(15,23,42,0)'], [15, 'rgba(148,163,184,0)'], [40, 'rgba(176,190,206,.4)'],
      [70, 'rgba(214,222,232,.68)'], [100, 'rgba(244,247,251,.88)'],
    ],
    ticks: [0, 25, 50, 75, 100],
  },
  rh: {
    stops: [
      [0, '#7c2d12'], [20, '#c2410c'], [40, '#ca8a04'], [60, '#65a30d'], [75, '#0d9488'],
      [90, '#2563eb'], [100, '#1e3a8a'],
    ],
    ticks: [0, 20, 40, 60, 80, 100],
  },
  msl: {
    stops: [
      [975, '#4c1d95'], [990, '#1d4ed8'], [1000, '#0891b2'], [1008, '#0d9488'], [1013, '#4d7c0f'],
      [1018, '#a16207'], [1025, '#c2410c'], [1035, '#b91c1c'], [1045, '#7f1d1d'],
    ],
    ticks: [980, 990, 1000, 1010, 1020, 1030, 1040],
  },
  cape: {
    stops: [
      [0, 'rgba(0,0,0,0)'], [50, 'rgba(101,163,13,0)'], [150, 'rgba(101,163,13,.7)'],
      [400, '#facc15'], [800, '#f97316'], [1500, '#dc2626'], [2500, '#a21caf'], [4000, '#f0abfc'],
    ],
    ticks: [150, 400, 800, 1500, 2500, 4000],
  },
  // normalised 0..1, stretched to each level's height range at runtime
  gh: {
    stops: [
      [0, '#4c1d95'], [0.15, '#1d4ed8'], [0.3, '#0891b2'], [0.45, '#0d9488'], [0.6, '#65a30d'],
      [0.75, '#ca8a04'], [0.88, '#ea580c'], [1, '#b91c1c'],
    ],
    ticks: [],
  },
} satisfies Record<string, Scale>;
