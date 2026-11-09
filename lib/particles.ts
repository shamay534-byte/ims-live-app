import type { Map as MlMap } from 'maplibre-gl';
import { sample, type Grid } from './ecmwf';

const TRAIL = 10; // screen positions kept per particle

interface P {
  lon: number;
  lat: number;
  age: number;
  max: number;
  /** recent screen positions (device px), oldest first: x0,y0,x1,y1,... */
  xy: number[];
}

/**
 * Animated wind streaks drawn on a 2D canvas laid over the map (Windy-style).
 * Positions live in lon/lat; each frame they move with the bilinearly sampled wind.
 */
export class WindParticles {
  private ctx: CanvasRenderingContext2D;
  private ps: P[] = [];
  private raf = 0;
  private running = false;
  private moving = false;
  private grid: Grid | null = null;
  private u: Float32Array | null = null;
  private v: Float32Array | null = null;

  constructor(private canvas: HTMLCanvasElement, private map: MlMap) {
    this.ctx = canvas.getContext('2d')!;
    map.on('movestart', this.onMoveStart);
    map.on('moveend', this.onMoveEnd);
    map.on('resize', this.resize);
    this.resize();
  }

  setWind(grid: Grid, u: Float32Array, v: Float32Array) {
    this.grid = grid;
    this.u = u;
    this.v = v;
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.seed();
    const tick = () => {
      if (!this.running) return;
      this.frame();
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  stop() {
    this.running = false;
    cancelAnimationFrame(this.raf);
    this.clear();
  }

  destroy() {
    this.stop();
    this.map.off('movestart', this.onMoveStart);
    this.map.off('moveend', this.onMoveEnd);
    this.map.off('resize', this.resize);
  }

  private resize = () => {
    const c = this.map.getCanvas();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(c.clientWidth * dpr);
    this.canvas.height = Math.round(c.clientHeight * dpr);
    this.canvas.style.width = `${c.clientWidth}px`;
    this.canvas.style.height = `${c.clientHeight}px`;
    this.seed();
  };

  private onMoveStart = () => {
    this.moving = true;
    this.clear();
  };

  private onMoveEnd = () => {
    this.moving = false;
    this.seed();
  };

  private clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  private count() {
    const area = this.canvas.width * this.canvas.height;
    return Math.min(4500, Math.max(600, Math.round(area / 900)));
  }

  private spawn(p: P) {
    const b = this.map.getBounds();
    const g = this.grid;
    let w = b.getWest(), e = b.getEast(), s = b.getSouth(), n = b.getNorth();
    if (g) {
      w = Math.max(w, g.lon0);
      e = Math.min(e, g.lon0 + g.dlon * (g.nlon - 1));
      n = Math.min(n, g.lat0);
      s = Math.max(s, g.lat0 + g.dlat * (g.nlat - 1));
    }
    if (e <= w || n <= s) {
      p.age = p.max;
      return;
    }
    p.lon = w + Math.random() * (e - w);
    p.lat = s + Math.random() * (n - s);
    p.age = 0;
    p.xy.length = 0;
    p.max = 40 + Math.random() * 70;
  }

  private seed() {
    const n = this.count();
    this.ps = Array.from({ length: n }, () => {
      const p: P = { lon: 0, lat: 0, age: 0, max: 1, xy: [] };
      this.spawn(p);
      p.age = Math.random() * p.max;
      return p;
    });
  }

  private frame() {
    const { ctx, canvas, map, grid, u, v } = this;
    // Redraw from scratch every frame: trails come from each particle's short history, so no
    // ghosting builds up (fading a canvas in place leaves 8-bit alpha residue behind).
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (this.moving || !grid || !u || !v) return;
    const dpr = window.devicePixelRatio || 1;
    // degrees per screen pixel at this zoom (MapLibre uses 512px tiles)
    const degPerPx = 360 / (512 * Math.pow(2, map.getZoom()));
    const k = 0.25 * degPerPx; // px per frame per (m/s)
    for (const p of this.ps) {
      if (p.age++ > p.max) {
        this.spawn(p);
        continue;
      }
      const uu = sample(grid, u, p.lon, p.lat);
      const vv = sample(grid, v, p.lon, p.lat);
      if (Number.isNaN(uu) || Number.isNaN(vv)) {
        this.spawn(p);
        continue;
      }
      p.lon += uu * k;
      p.lat += vv * k * Math.cos((p.lat * Math.PI) / 180);
      const b = map.project([p.lon, p.lat]);
      p.xy.push(b.x * dpr, b.y * dpr);
      if (p.xy.length > TRAIL * 2) p.xy.splice(0, 2);
    }
    // one stroke per trail segment index, newest segments brightest
    ctx.lineWidth = 1.15 * dpr;
    ctx.lineCap = 'round';
    for (let j = 0; j < TRAIL - 1; j++) {
      ctx.strokeStyle = `rgba(255,255,255,${(0.85 * (j + 1)) / (TRAIL - 1)})`;
      ctx.beginPath();
      for (const p of this.ps) {
        const n = p.xy.length / 2;
        // align so the newest segment of every particle gets j = TRAIL - 2
        const seg = j - (TRAIL - n);
        if (seg < 0) continue;
        const o = seg * 2;
        ctx.moveTo(p.xy[o], p.xy[o + 1]);
        ctx.lineTo(p.xy[o + 2], p.xy[o + 3]);
      }
      ctx.stroke();
    }
  }
}
