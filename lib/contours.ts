import type { Grid } from './ecmwf';

export interface ContourLine {
  level: number;
  /** flat [lon, lat, lon, lat, ...] pairs of independent segments */
  segs: Float32Array;
}

/**
 * Marching squares over a regular lat/lon grid. Produces short segments per contour level;
 * they are drawn as-is (the grid is fine enough that joining them is not needed).
 */
export function contours(g: Grid, vals: Float32Array, step: number): ContourLine[] {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < vals.length; i++) {
    const v = vals[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  const out: ContourLine[] = [];
  const { nlat, nlon, lat0, lon0, dlat, dlon } = g;
  for (let level = Math.ceil(lo / step) * step; level <= hi; level += step) {
    const segs: number[] = [];
    for (let r = 0; r < nlat - 1; r++) {
      for (let c = 0; c < nlon - 1; c++) {
        const i = r * nlon + c;
        const a = vals[i];
        const b = vals[i + 1];
        const d = vals[i + nlon];
        const e = vals[i + nlon + 1];
        if (Number.isNaN(a + b + d + e)) continue;
        // corners: a=top-left, b=top-right, e=bottom-right, d=bottom-left
        const idx = (a > level ? 8 : 0) | (b > level ? 4 : 0) | (e > level ? 2 : 0) | (d > level ? 1 : 0);
        if (idx === 0 || idx === 15) continue;
        const pts: [number, number][] = [];
        const lerp = (v0: number, v1: number) => (level - v0) / (v1 - v0);
        const top = (): [number, number] => [c + lerp(a, b), r];
        const right = (): [number, number] => [c + 1, r + lerp(b, e)];
        const bottom = (): [number, number] => [c + lerp(d, e), r + 1];
        const left = (): [number, number] => [c, r + lerp(a, d)];
        switch (idx) {
          case 1: case 14: pts.push(left(), bottom()); break;
          case 2: case 13: pts.push(bottom(), right()); break;
          case 3: case 12: pts.push(left(), right()); break;
          case 4: case 11: pts.push(top(), right()); break;
          case 6: case 9: pts.push(top(), bottom()); break;
          case 7: case 8: pts.push(left(), top()); break;
          case 5: pts.push(left(), top(), bottom(), right()); break;
          case 10: pts.push(left(), bottom(), top(), right()); break;
        }
        for (const [x, y] of pts) segs.push(lon0 + x * dlon, lat0 + y * dlat);
      }
    }
    if (segs.length) out.push({ level, segs: new Float32Array(segs) });
  }
  return out;
}
