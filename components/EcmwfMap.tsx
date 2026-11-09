'use client';
import { useEffect, useMemo, useRef, useState } from 'react';
import maplibregl, { type Map as MlMap, type ImageSource, type LayerSpecification, type StyleSpecification } from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { fieldsFor, unitFor, type LevelId, type ParamDef } from '@/lib/catalog';
import {
  bounds, loadField, pointSeries, renderField, stepValues, valuesFor,
  type Field, type Manifest,
} from '@/lib/ecmwf';
import { contours, type ContourLine } from '@/lib/contours';
import { WindParticles } from '@/lib/particles';
import type { Scale } from '@/lib/colors';

export interface Pick {
  lon: number;
  lat: number;
  name?: string;
}

interface Props {
  manifest: Manifest;
  param: ParamDef;
  level: LevelId;
  step: number;
  scale: Scale;
  particles: boolean;
  isolines: boolean;
  pick: Pick;
  onPick: (p: Pick) => void;
  onLoading: (b: boolean) => void;
  onError: (msg: string | null) => void;
}

const BLANK =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';

const STYLE_URL = 'https://tiles.openfreemap.org/styles/dark';
const RTL_PLUGIN = 'https://unpkg.com/@mapbox/mapbox-gl-rtl-text@0.2.3/mapbox-gl-rtl-text.min.js';

let rtlRequested = false;
function ensureRtl() {
  if (rtlRequested) return;
  rtlRequested = true;
  try {
    maplibregl.setRTLTextPlugin(RTL_PLUGIN, true);
  } catch {
    /* already set */
  }
}

/**
 * Take the OpenFreeMap dark style, prefer Hebrew place names, drop road clutter and slot the
 * model field (an image source) in below borders and labels.
 */
async function buildStyle(coords: [number, number][]): Promise<StyleSpecification> {
  const style = (await (await fetch(STYLE_URL)).json()) as StyleSpecification;
  // Hebrew where OSM has it, otherwise a Latin name (the font stack has no Arabic glyphs)
  const hebrew = ['coalesce', ['get', 'name:he'], ['get', 'name:en'], ['get', 'name:latin'], ['get', 'name']];
  const layers: LayerSpecification[] = [];
  for (const l of style.layers) {
    if (/^(highway|road|railway|aeroway|building)/.test(l.id) && !/name/.test(l.id)) continue;
    if (l.type === 'symbol' && l.layout && 'text-field' in l.layout) {
      l.layout = { ...l.layout, 'text-field': hebrew as unknown as string };
    }
    if (l.id === 'boundary_state') {
      layers.push({ id: 'field', type: 'raster', source: 'field', paint: { 'raster-resampling': 'linear', 'raster-fade-duration': 0, 'raster-opacity': 0.8 } });
    }
    layers.push(l);
  }
  if (!layers.some((l) => l.id === 'field')) {
    const firstSymbol = layers.findIndex((l) => l.type === 'symbol');
    layers.splice(firstSymbol < 0 ? layers.length : firstSymbol, 0, { id: 'field', type: 'raster', source: 'field', paint: { 'raster-resampling': 'linear', 'raster-fade-duration': 0, 'raster-opacity': 0.8 } });
  }
  style.sources = { ...style.sources, field: { type: 'image', url: BLANK, coordinates: coords as [[number, number], [number, number], [number, number], [number, number]] } };
  style.layers = layers;
  return style;
}

export default function EcmwfMap(props: Props) {
  const { manifest, param, level, step, scale, particles, isolines, pick, onPick, onLoading, onError } = props;
  const box = useRef<HTMLDivElement>(null);
  const isoCanvas = useRef<HTMLCanvasElement>(null);
  const windCanvas = useRef<HTMLCanvasElement>(null);
  const mapRef = useRef<MlMap | null>(null);
  const windRef = useRef<WindParticles | null>(null);
  const fieldCanvas = useRef<HTMLCanvasElement | null>(null);
  const linesRef = useRef<{ lines: ContourLine[]; unit: string } | null>(null);
  const [ready, setReady] = useState(false);
  const [fields, setFields] = useState<Record<string, Field> | null>(null);
  const [bubble, setBubble] = useState<{ x: number; y: number } | null>(null);
  const grid = manifest.grid;

  // ---- map setup -----------------------------------------------------------------------
  useEffect(() => {
    if (!box.current) return;
    const b = bounds(grid);
    const small = window.innerWidth < 760;
    const coords: [number, number][] = [[b.west, b.north], [b.east, b.north], [b.east, b.south], [b.west, b.south]];
    ensureRtl();
    const map = new maplibregl.Map({
      container: box.current,
      style: { version: 8, sources: {}, layers: [{ id: 'bg', type: 'background', paint: { 'background-color': '#0b1626' } }] },
      maxBounds: [[b.west - 6, b.south - 4], [b.east + 6, b.north + 4]],
      center: [35.1, 31.6],
      zoom: small ? 5.7 : 6.4,
      minZoom: 4,
      maxZoom: 11,
      attributionControl: false,
      dragRotate: false,
      pitchWithRotate: false,
    });
    map.touchZoomRotate.disableRotation();
    map.addControl(
      new maplibregl.AttributionControl({
        compact: true,
        customAttribution: 'ECMWF open data (CC-BY-4.0)',
      }),
      'bottom-left',
    );
    buildStyle(coords)
      .then((st) => {
        map.setStyle(st);
        map.once('styledata', () => {
          setReady(true);
          // start with the attribution collapsed to its (i) button
          box.current?.querySelector('.maplibregl-ctrl-attrib')?.classList.remove('maplibregl-compact-show');
        });
      })
      .catch(() => {
        // basemap unavailable: show the field on the plain background
        const add = () => {
          map.addSource('field', { type: 'image', url: BLANK, coordinates: coords as [[number, number], [number, number], [number, number], [number, number]] });
          map.addLayer({ id: 'field', type: 'raster', source: 'field', paint: { 'raster-resampling': 'linear', 'raster-fade-duration': 0 } });
          setReady(true);
        };
        if (map.isStyleLoaded()) add();
        else map.once('load', add);
      });
    map.on('click', (e) => onPickRef.current({ lon: e.lngLat.lng, lat: e.lngLat.lat }));
    mapRef.current = map;
    windRef.current = windCanvas.current ? new WindParticles(windCanvas.current, map) : null;
    return () => {
      windRef.current?.destroy();
      map.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [manifest.runId]);

  const onPickRef = useRef(onPick);
  onPickRef.current = onPick;

  // ---- load the fields needed for this param/level -------------------------------------
  const needed = useMemo(() => fieldsFor(param, level, particles), [param, level, particles]);
  useEffect(() => {
    let dead = false;
    onLoading(true);
    Promise.all(needed.filter((id) => manifest.fields[id]).map((id) => loadField(manifest, id).then((f) => [id, f] as const)))
      .then((list) => {
        if (dead) return;
        setFields(Object.fromEntries(list));
        onError(null);
      })
      .catch((e) => !dead && onError(String(e.message || e)))
      .finally(() => !dead && onLoading(false));
    return () => {
      dead = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [manifest, needed.join(',')]);

  const src = useMemo(() => param.source(level), [param, level]);

  // ---- draw field / contours / wind for the current step ---------------------------------
  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map || !fields) return;
    const vals = valuesFor(src, fields, step);
    if (vals) {
      if (!fieldCanvas.current) fieldCanvas.current = document.createElement('canvas');
      renderField(fieldCanvas.current, grid, vals, scale);
      const b = bounds(grid);
      (map.getSource('field') as ImageSource).updateImage({
        url: fieldCanvas.current.toDataURL(),
        coordinates: [[b.west, b.north], [b.east, b.north], [b.east, b.south], [b.west, b.south]],
      });
      map.setPaintProperty('field', 'raster-opacity', param.opacity);
    }
    const c = param.contours?.(level);
    if (isolines && c && fields[c.field]) {
      linesRef.current = { lines: contours(grid, stepValues(fields[c.field], step), c.step), unit: c.unit };
    } else linesRef.current = null;
    drawIso();

    const w = windRef.current;
    const uId = level === 'sfc' ? 'u10' : `u_${level}`;
    const vId = level === 'sfc' ? 'v10' : `v_${level}`;
    if (w && particles && fields[uId] && fields[vId]) {
      w.setWind(grid, stepValues(fields[uId], step), stepValues(fields[vId], step));
      w.start();
    } else w?.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, fields, step, src, scale, isolines, particles]);

  // ---- isolines canvas (redrawn while the map moves) -----------------------------------
  const drawIso = () => {
    const map = mapRef.current;
    const cv = isoCanvas.current;
    if (!map || !cv) return;
    const dpr = window.devicePixelRatio || 1;
    const mc = map.getCanvas();
    const W = Math.round(mc.clientWidth * dpr);
    const H = Math.round(mc.clientHeight * dpr);
    if (cv.width !== W || cv.height !== H) {
      cv.width = W;
      cv.height = H;
      cv.style.width = `${mc.clientWidth}px`;
      cv.style.height = `${mc.clientHeight}px`;
    }
    const ctx = cv.getContext('2d')!;
    ctx.clearRect(0, 0, W, H);
    const data = linesRef.current;
    if (!data) return;
    ctx.lineWidth = 1.5 * dpr;
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.beginPath();
    const labels: { x: number; y: number; t: string }[] = [];
    for (const ln of data.lines) {
      const s = ln.segs;
      let n = 0;
      for (let i = 0; i < s.length; i += 4) {
        const a = map.project([s[i], s[i + 1]]);
        const b = map.project([s[i + 2], s[i + 3]]);
        ctx.moveTo(a.x * dpr, a.y * dpr);
        ctx.lineTo(b.x * dpr, b.y * dpr);
        if (n++ % 70 === 35) {
          const x = (a.x + b.x) / 2;
          const y = (a.y + b.y) / 2;
          if (x > 20 && y > 20 && x < mc.clientWidth - 20 && y < mc.clientHeight - 20 &&
            labels.every((l) => Math.hypot(l.x - x, l.y - y) > 90)) {
            labels.push({ x, y, t: String(Math.round(ln.level)) });
          }
        }
      }
    }
    ctx.stroke();
    ctx.font = `600 ${11 * dpr}px Rubik, system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const l of labels) {
      ctx.lineWidth = 3.5 * dpr;
      ctx.strokeStyle = 'rgba(5,10,20,0.85)';
      ctx.strokeText(l.t, l.x * dpr, l.y * dpr);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(l.t, l.x * dpr, l.y * dpr);
    }
  };

  // ---- keep overlays + picker bubble in place while panning ---------------------------
  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map) return;
    const place = () => {
      const p = map.project([pick.lon, pick.lat]);
      setBubble({ x: p.x, y: p.y });
    };
    const onMove = () => {
      place();
      drawIso();
    };
    place();
    map.on('move', onMove);
    map.on('resize', onMove);
    return () => {
      map.off('move', onMove);
      map.off('resize', onMove);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, pick.lon, pick.lat]);

  // ---- picker value + sparkline --------------------------------------------------------
  const series = useMemo(
    () => (fields ? pointSeries(src, fields, grid, pick.lon, pick.lat) : []),
    [fields, src, grid, pick.lon, pick.lat],
  );
  const value = series[step];
  const unit = unitFor(param, level);
  const fmt = (v: number) =>
    Number.isFinite(v) ? (param.decimals ? v.toFixed(param.decimals) : String(Math.round(v))) : '—';
  const knots = param.unit === 'קמ״ש' && Number.isFinite(value) ? ` · ${Math.round(value / 1.852)} kt` : '';

  const spark = useMemo(() => {
    const vals = series.filter(Number.isFinite);
    if (vals.length < 2) return null;
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const W = 150;
    const H = 34;
    const n = series.length;
    const pts = series
      .map((v, i) => (Number.isFinite(v) ? `${((i / (n - 1)) * W).toFixed(1)},${(H - 2 - ((v - lo) / (hi - lo || 1)) * (H - 4)).toFixed(1)}` : null))
      .filter(Boolean)
      .join(' ');
    return { pts, W, H, x: (step / (n - 1)) * W, lo, hi };
  }, [series, step]);

  const inside = bubble && box.current && bubble.x > 0 && bubble.y > 0 &&
    bubble.x < box.current.clientWidth && bubble.y < box.current.clientHeight;

  return (
    <div className="map-wrap">
      <div ref={box} className="map" />
      <canvas ref={isoCanvas} className="overlay" />
      <canvas ref={windCanvas} className="overlay" />
      {inside && (
        <div className="picker" style={{ left: bubble!.x, top: bubble!.y }}>
          <div className="picker-card glass">
            <span className="picker-name">{pick.name ?? `${pick.lat.toFixed(2)}°N ${pick.lon.toFixed(2)}°E`}</span>
            <span className="picker-val">
              {fmt(value)}
              <small> {unit}</small>
            </span>
            {knots && <span className="picker-sub">{knots.slice(3)}</span>}
            {spark && (
              <svg width={spark.W} height={spark.H} className="spark" aria-hidden="true">
                <polyline points={spark.pts} fill="none" stroke="#7dd3fc" strokeWidth="1.5" />
                <line x1={spark.x} x2={spark.x} y1="0" y2={spark.H} stroke="#fbbf24" strokeWidth="1.5" />
              </svg>
            )}
            {spark && (
              <span className="picker-sub">
                טווח בריצה: <bdi dir="ltr">{fmt(spark.lo)} – {fmt(spark.hi)}</bdi> {unit}
              </span>
            )}
          </div>
          <div className="picker-stem" />
          <div className="picker-dot" />
        </div>
      )}
    </div>
  );
}

