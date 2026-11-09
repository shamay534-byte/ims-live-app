import { lutFor, Scale } from './colors';
import type { Source } from './catalog';

export const BASE = process.env.NEXT_PUBLIC_BASE_PATH || '';

export interface Grid {
  lat0: number; // northernmost row
  lon0: number; // westernmost column
  dlat: number; // negative (rows go south)
  dlon: number;
  nlat: number;
  nlon: number;
}

export interface FieldMeta {
  file: string;
  scale: number;
  offset: number;
  units: string;
  min: number | null;
  max: number | null;
}

export interface Manifest {
  model: string;
  resolution: string;
  source: string;
  run: string;
  runId: string;
  generated: string;
  grid: Grid;
  steps: number[];
  levels: string[];
  fields: Record<string, FieldMeta>;
}

export interface Field {
  meta: FieldMeta;
  q: Int16Array;
  n: number; // points per step
}

export async function loadManifest(): Promise<Manifest> {
  const r = await fetch(`${BASE}/data/ecmwf/manifest.json`, { cache: 'no-cache' });
  if (!r.ok) throw new Error(`manifest ${r.status}`);
  return r.json();
}

const cache = new Map<string, Promise<Field>>();

export function loadField(man: Manifest, id: string): Promise<Field> {
  const meta = man.fields[id];
  if (!meta) return Promise.reject(new Error(`missing field ${id}`));
  const key = `${man.runId}/${id}`;
  let p = cache.get(key);
  if (!p) {
    p = fetch(`${BASE}/data/ecmwf/${meta.file}`)
      .then(async (r) => {
        if (!r.ok) throw new Error(`${id}: ${r.status}`);
        // Hosts that only serve text types get the same int16 grid as base64 in a .txt file
        if (meta.file.endsWith('.txt')) {
          const bin = atob((await r.text()).trim());
          const bytes = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
          return bytes.buffer;
        }
        return r.arrayBuffer();
      })
      .then((buf) => ({ meta, q: new Int16Array(buf), n: man.grid.nlat * man.grid.nlon }));
    p.catch(() => cache.delete(key));
    cache.set(key, p);
  }
  return p;
}

const MISSING = -32768;

/** Decode one step of a field into a Float32Array (NaN = missing), optionally scaled. */
export function stepValues(f: Field, step: number, factor = 1): Float32Array {
  const out = new Float32Array(f.n);
  const o = step * f.n;
  const { scale, offset } = f.meta;
  for (let i = 0; i < f.n; i++) {
    const q = f.q[o + i];
    out[i] = q === MISSING ? NaN : (q * scale + offset) * factor;
  }
  return out;
}

export function speedValues(u: Field, v: Field, step: number, factor: number): Float32Array {
  const a = stepValues(u, step);
  const b = stepValues(v, step);
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = Math.hypot(a[i], b[i]) * factor;
  return out;
}

export function valuesFor(src: Source, fields: Record<string, Field>, step: number): Float32Array | null {
  if (src.kind === 'scalar') {
    const f = fields[src.field];
    return f ? stepValues(f, step, src.factor ?? 1) : null;
  }
  const u = fields[src.u];
  const v = fields[src.v];
  return u && v ? speedValues(u, v, step, src.factor) : null;
}

/** Bilinear sample at lon/lat. Returns NaN outside the grid. */
export function sample(g: Grid, vals: Float32Array, lon: number, lat: number): number {
  const x = (lon - g.lon0) / g.dlon;
  const y = (lat - g.lat0) / g.dlat;
  if (x < 0 || y < 0 || x > g.nlon - 1 || y > g.nlat - 1) return NaN;
  const x0 = Math.min(Math.floor(x), g.nlon - 2);
  const y0 = Math.min(Math.floor(y), g.nlat - 2);
  const fx = x - x0;
  const fy = y - y0;
  const i = y0 * g.nlon + x0;
  const a = vals[i];
  const b = vals[i + 1];
  const c = vals[i + g.nlon];
  const d = vals[i + g.nlon + 1];
  return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}

/** Outer edges of the grid cells, for placing the image on the map. */
export function bounds(g: Grid) {
  const west = g.lon0 - g.dlon / 2;
  const east = g.lon0 + g.dlon * (g.nlon - 1) + g.dlon / 2;
  const north = g.lat0 - g.dlat / 2;
  const south = g.lat0 + g.dlat * (g.nlat - 1) + g.dlat / 2;
  return { west, east, north, south };
}

const mercY = (lat: number) => Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360));
const invMercY = (y: number) => (360 / Math.PI) * Math.atan(Math.exp(y)) - 90;

/**
 * Paint the field into a canvas laid out in Web-Mercator rows, so that MapLibre's linear
 * image placement lines up exactly with the lat/lon grid. Values are bilinearly smoothed.
 */
export function renderField(canvas: HTMLCanvasElement, g: Grid, vals: Float32Array, scale: Scale, upsample = 4) {
  const lut = lutFor(scale);
  const { west, east, north, south } = bounds(g);
  const W = g.nlon * upsample;
  const yN = mercY(north);
  const yS = mercY(south);
  const H = Math.round(W * ((yN - yS) / (((east - west) * Math.PI) / 180)));
  if (canvas.width !== W || canvas.height !== H) {
    canvas.width = W;
    canvas.height = H;
  }
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(W, H);
  const px = img.data;
  const span = lut.max - lut.min;
  const xs = new Float32Array(W);
  for (let x = 0; x < W; x++) {
    const lon = west + ((x + 0.5) / W) * (east - west);
    xs[x] = (lon - g.lon0) / g.dlon;
  }
  for (let row = 0; row < H; row++) {
    const lat = invMercY(yN - ((row + 0.5) / H) * (yN - yS));
    let y = (lat - g.lat0) / g.dlat;
    y = Math.max(0, Math.min(g.nlat - 1.0001, y));
    const y0 = Math.floor(y);
    const fy = y - y0;
    const r0 = y0 * g.nlon;
    const r1 = r0 + g.nlon;
    for (let x = 0; x < W; x++) {
      const xx = Math.max(0, Math.min(g.nlon - 1.0001, xs[x]));
      const x0 = Math.floor(xx);
      const fx = xx - x0;
      const v =
        (vals[r0 + x0] * (1 - fx) + vals[r0 + x0 + 1] * fx) * (1 - fy) +
        (vals[r1 + x0] * (1 - fx) + vals[r1 + x0 + 1] * fx) * fy;
      const o = (row * W + x) * 4;
      if (Number.isNaN(v)) {
        px[o + 3] = 0;
        continue;
      }
      let k = Math.round(((v - lut.min) / span) * 1023);
      k = k < 0 ? 0 : k > 1023 ? 1023 : k;
      px[o] = lut.rgba[k * 4];
      px[o + 1] = lut.rgba[k * 4 + 1];
      px[o + 2] = lut.rgba[k * 4 + 2];
      px[o + 3] = lut.rgba[k * 4 + 3];
    }
  }
  ctx.putImageData(img, 0, 0);
}

/** Value at one point for every step (used for the picker sparkline). */
export function pointSeries(src: Source, fields: Record<string, Field>, g: Grid, lon: number, lat: number): number[] {
  const x = (lon - g.lon0) / g.dlon;
  const y = (lat - g.lat0) / g.dlat;
  if (x < 0 || y < 0 || x > g.nlon - 1 || y > g.nlat - 1) return [];
  const x0 = Math.min(Math.floor(x), g.nlon - 2);
  const y0 = Math.min(Math.floor(y), g.nlat - 2);
  const fx = x - x0;
  const fy = y - y0;
  const idx = [y0 * g.nlon + x0, y0 * g.nlon + x0 + 1, (y0 + 1) * g.nlon + x0, (y0 + 1) * g.nlon + x0 + 1];
  const w = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy];
  const at = (f: Field, s: number) => {
    let acc = 0;
    for (let k = 0; k < 4; k++) {
      const q = f.q[s * f.n + idx[k]];
      if (q === MISSING) return NaN;
      acc += (q * f.meta.scale + f.meta.offset) * w[k];
    }
    return acc;
  };
  const anyField = src.kind === 'scalar' ? fields[src.field] : fields[src.u];
  if (!anyField) return [];
  const steps = anyField.q.length / anyField.n;
  const out: number[] = [];
  for (let s = 0; s < steps; s++) {
    if (src.kind === 'scalar') out.push(at(fields[src.field], s) * (src.factor ?? 1));
    else {
      const u = fields[src.u];
      const v = fields[src.v];
      out.push(u && v ? Math.hypot(at(u, s), at(v, s)) * src.factor : NaN);
    }
  }
  return out;
}
