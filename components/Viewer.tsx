'use client';
import dynamic from 'next/dynamic';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  LEVELS, MODELS, PARAMS, ecmwfFor, imsTypeFor, paramById, unitFor,
  type LevelId, type ModelId, type ParamId,
} from '@/lib/catalog';
import { cssGradient } from '@/lib/colors';
import { loadManifest, type Manifest } from '@/lib/ecmwf';
import { framesFor, loadImsManifest, typesFor, type ImsManifest } from '@/lib/ims';
import { parseRun, runLabel } from '@/lib/time';
import { CITIES } from '@/lib/cities';
import { Icon } from './Icons';
import { Timeline, type Tick } from './Timeline';
import ImsView from './ImsView';
import type { Pick } from './EcmwfMap';

const EcmwfMap = dynamic(() => import('./EcmwfMap'), { ssr: false });

const DEFAULT_PICK: Pick = { lon: 34.78, lat: 32.08, name: 'תל אביב' };

function nearest(ticks: Tick[], ms: number) {
  let best = 0;
  ticks.forEach((t, i) => {
    if (Math.abs(t.ms - ms) < Math.abs(ticks[best].ms - ms)) best = i;
  });
  return best;
}

function readHash() {
  if (typeof window === 'undefined') return {} as Record<string, string>;
  return Object.fromEntries(new URLSearchParams(window.location.hash.slice(1)));
}

export default function Viewer() {
  const [model, setModel] = useState<ModelId>('ecmwf');
  const [param, setParam] = useState<ParamId>('temp');
  const [level, setLevel] = useState<LevelId>('sfc');
  const [ecIdx, setEcIdx] = useState(0);
  const [particles, setParticles] = useState(true);
  const [isolines, setIsolines] = useState(true);
  const [imsArea, setImsArea] = useState('ilplus');
  const [imsType, setImsType] = useState('temp2wind10');
  const [imsRun, setImsRun] = useState<Record<string, string>>({});
  const [imsIdx, setImsIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [pick, setPick] = useState<Pick>(DEFAULT_PICK);
  const [loading, setLoading] = useState(false);
  const [mapError, setMapError] = useState<string | null>(null);
  const [ec, setEc] = useState<Manifest | null>(null);
  const [ecErr, setEcErr] = useState<string | null>(null);
  const [ims, setIms] = useState<ImsManifest | null>(null);
  const [imsErr, setImsErr] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const pendingLead = useRef<number | null>(null);

  // Restore the view from the URL hash after mount (keeps server and client render identical)
  useEffect(() => {
    const h = readHash();
    const m = MODELS.find((x) => x.id === h.m)?.id;
    if (m) setModel(m);
    const pp = PARAMS.find((x) => x.id === h.p)?.id;
    if (pp) setParam(pp);
    const ll = LEVELS.find((x) => x.id === h.l)?.id;
    if (ll) setLevel(ll);
    if (h.a) setImsArea(h.a);
    if (h.k) setImsType(h.k);
    if (h.t) pendingLead.current = Number(h.t);
  }, []);

  useEffect(() => {
    loadManifest().then(setEc).catch((e) => setEcErr(String(e.message || e)));
    loadImsManifest().then(setIms).catch((e) => setImsErr(String(e.message || e)));
  }, []);

  const isIms = model !== 'ecmwf';
  const p = paramById(param);
  const lvl: LevelId = p.levels.includes(level) ? level : 'sfc';

  // ---- ECMWF ticks ---------------------------------------------------------------------
  const ecTicks: Tick[] = useMemo(() => {
    if (!ec) return [];
    const r = parseRun(ec.run);
    return ec.steps.map((s) => ({ ms: r + s * 3600e3, lead: s }));
  }, [ec]);

  // ---- IMS selection -------------------------------------------------------------------
  const run = isIms && ims ? imsRun[model] ?? ims.models[model]?.runs[0] : undefined;
  const areas = isIms && ims ? (ims.models[model]?.areas ?? []).filter((a) => ims.maps[model]?.[run!]?.[a]) : [];
  const area = areas.includes(imsArea) ? imsArea : areas[0];
  const groups = useMemo(
    () => (isIms && ims && run && area ? typesFor(ims, model, run, area) : []),
    [isIms, ims, model, run, area],
  );
  const allTypes = groups.flatMap((g) => g.types.map((t) => t.code));
  const type = allTypes.includes(imsType) ? imsType : allTypes[0];
  const frames = useMemo(
    () => (isIms && ims && run && area && type ? framesFor(ims, model, run, area, type) : []),
    [isIms, ims, model, run, area, type],
  );
  const imsTicks: Tick[] = useMemo(() => {
    if (!run) return [];
    const r = parseRun(run);
    return frames.map((f) => ({ ms: r + f.lead * 3600e3, lead: f.lead }));
  }, [frames, run]);

  const ticks = isIms ? imsTicks : ecTicks;
  const idx = Math.min(isIms ? imsIdx : ecIdx, Math.max(0, ticks.length - 1));
  const setIdx = useCallback((i: number) => (isIms ? setImsIdx(i) : setEcIdx(i)), [isIms]);

  // Restore lead from the URL once the ticks are known
  useEffect(() => {
    if (pendingLead.current == null || !ticks.length) return;
    const i = ticks.findIndex((t) => t.lead === pendingLead.current);
    if (i >= 0) setIdx(i);
    pendingLead.current = null;
  }, [ticks, setIdx]);

  // Keep the URL hash in sync (shareable links)
  useEffect(() => {
    const q = new URLSearchParams({ m: model });
    if (isIms) {
      if (area) q.set('a', area);
      if (type) q.set('k', type);
    } else {
      q.set('p', param);
      q.set('l', lvl);
    }
    if (ticks[idx]) q.set('t', String(ticks[idx].lead));
    try {
      history.replaceState(null, '', `#${q.toString()}`);
    } catch {
      /* sandboxed frames may refuse history changes */
    }
  }, [model, param, lvl, area, type, idx, ticks, isIms]);

  const pendingMs = useRef<number | null>(null);
  // ---- model switching keeps the valid time and the closest matching parameter --------
  const switchModel = (m: ModelId) => {
    if (m === model) return;
    const ms = ticks[idx]?.ms;
    if (m === 'ecmwf') {
      if (type) {
        const back = ecmwfFor(type);
        setParam(back.p);
        setLevel(back.l);
      }
      if (ms != null && ecTicks.length) setEcIdx(nearest(ecTicks, ms));
    } else {
      if (!isIms) setImsType(imsTypeFor(param, lvl));
      if (ms != null) pendingMs.current = ms;
    }
    setModel(m);
  };
  useEffect(() => {
    if (pendingMs.current == null || !isIms || !imsTicks.length) return;
    setImsIdx(nearest(imsTicks, pendingMs.current));
    pendingMs.current = null;
  }, [imsTicks, isIms]);

  // ---- animation ------------------------------------------------------------------------
  useEffect(() => {
    if (!playing || ticks.length < 2) return;
    const t = setInterval(() => setIdx(((isIms ? imsIdx : ecIdx) + 1) % ticks.length), isIms ? 600 : 650);
    return () => clearInterval(t);
  }, [playing, ticks.length, isIms, imsIdx, ecIdx, setIdx]);

  // keyboard: arrows step, space plays
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      if (e.key === 'ArrowRight') setIdx(Math.min(ticks.length - 1, idx + 1));
      else if (e.key === 'ArrowLeft') setIdx(Math.max(0, idx - 1));
      else if (e.key === ' ') setPlaying((x) => !x);
      else return;
      e.preventDefault();
      if (e.key !== ' ') setPlaying(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [idx, ticks.length, setIdx]);

  // ---- legend ---------------------------------------------------------------------------
  const srcField = (() => {
    const s = p.source(lvl);
    return s.kind === 'scalar' ? s.field : s.u;
  })();
  const range: [number, number] | undefined =
    ec?.fields[srcField]?.min != null ? [ec.fields[srcField].min!, ec.fields[srcField].max!] : undefined;
  const scale = useMemo(() => p.scale(lvl, range), [p, lvl, range?.[0], range?.[1]]); // eslint-disable-line react-hooks/exhaustive-deps

  const onSearch = (name: string) => {
    setQuery(name);
    const c = CITIES.find((x) => x.name === name.trim());
    if (c) {
      setPick({ lon: c.lon, lat: c.lat, name: c.name });
      setQuery('');
    }
  };

  const runText = isIms ? (run ? runLabel(parseRun(run)) : '—') : ec ? runLabel(parseRun(ec.run)) : '—';
  const modelInfo = isIms
    ? { src: `שמ״ט · ${ims?.models[model]?.name ?? ''}`, res: 'מפות מוכנות מאתר השירות המטאורולוגי · כל 3 ש׳' }
    : { src: 'ECMWF Open Data · GRIB2', res: ec ? `IFS ${ec.resolution} (~9 ק״מ) · עד ${ec.steps[ec.steps.length - 1]} ש׳` : '' };

  return (
    <div className="app" dir="rtl">
      {/* -------- map / image area -------- */}
      {!isIms && ec && (
        <EcmwfMap
          manifest={ec}
          param={p}
          level={lvl}
          step={idx}
          scale={scale}
          particles={particles}
          isolines={isolines}
          pick={pick}
          onPick={setPick}
          onLoading={setLoading}
          onError={setMapError}
        />
      )}
      {!isIms && !ec && (
        <div className="empty">
          {ecErr ? (
            <>
              <b>אין עדיין נתוני ECMWF</b>
              <span>הנתונים נוצרים ע״י <code>npm run data:ecmwf</code> (או אוטומטית ב-GitHub Actions).</span>
            </>
          ) : (
            <span>טוען נתוני ECMWF…</span>
          )}
        </div>
      )}
      {isIms && ims && <ImsView frames={frames} index={idx} title={ims.mapTypes[type ?? '']?.title ?? ''} />}
      {isIms && !ims && <div className="empty">{imsErr ? <b>לא ניתן לטעון את רשימת המפות של שמ״ט</b> : <span>טוען…</span>}</div>}

      {/* -------- top bar -------- */}
      <header className="topbar">
        <div className="tb-group">
          <div className="brand glass">
            <span className="brand-icon"><Icon name="logo" size={24} /></span>
            <span className="brand-name">מפות מודלים</span>
          </div>
          <div className="models glass" role="group" aria-label="בחירת מודל">
            {MODELS.map((m) => (
              <button key={m.id} className={`mdl${model === m.id ? ' on' : ''}`} onClick={() => switchModel(m.id)} aria-pressed={model === m.id}>
                <b>{m.label}</b>
                <span>{m.sub}</span>
              </button>
            ))}
          </div>
        </div>
        <div className="tb-group">
          {!isIms && (
            <>
              <label className="sr-only" htmlFor="place">חיפוש יישוב</label>
              <input
                id="place"
                className="search"
                list="cities"
                placeholder="חיפוש יישוב…"
                value={query}
                onChange={(e) => onSearch(e.target.value)}
              />
              <datalist id="cities">
                {CITIES.map((c) => <option key={c.name} value={c.name} />)}
              </datalist>
            </>
          )}
          {isIms && ims && run ? (
            <label className="run glass">
              <span>ריצה</span>
              <select value={run} onChange={(e) => setImsRun((s) => ({ ...s, [model]: e.target.value }))}>
                {ims.models[model].runs.filter((r) => ims.maps[model]?.[r]).map((r) => (
                  <option key={r} value={r}>{runLabel(parseRun(r))}</option>
                ))}
              </select>
            </label>
          ) : (
            <div className="run glass">
              <span>ריצה</span>
              <b>{runText}</b>
            </div>
          )}
        </div>
      </header>

      {/* -------- dock: params, levels, legend, timeline (rail on desktop, sheet on mobile) -------- */}
      <div className="dock">
        {!isIms ? (
          <nav className="params glass" aria-label="פרמטר">
            <span className="caption">פרמטר</span>
            <div className="param-list">
              {PARAMS.map((x) => (
                <button key={x.id} className={`prm${param === x.id ? ' on' : ''}`} onClick={() => setParam(x.id)} aria-pressed={param === x.id}>
                  <Icon name={x.icon} />
                  <span className="lbl-long">{x.label}</span>
                  <span className="lbl-short">{x.short}</span>
                </button>
              ))}
            </div>
            <div className="rail-extra">
              <label className="toggle"><input type="checkbox" checked={particles} onChange={(e) => setParticles(e.target.checked)} />חלקיקי רוח</label>
              <label className="toggle"><input type="checkbox" checked={isolines} onChange={(e) => setIsolines(e.target.checked)} />קווי שווה לחץ / גובה</label>
              <div className="source">
                <span className="caption">מקור הנתונים</span>
                <b>{modelInfo.src}</b>
                <span>{modelInfo.res}</span>
              </div>
            </div>
          </nav>
        ) : (
          <nav className="params glass ims" aria-label="סוג מפה">
            {areas.length > 1 && (
              <div className="areas" role="group" aria-label="אזור">
                {areas.map((a) => (
                  <button key={a} className={`chip${a === area ? ' on' : ''}`} onClick={() => setImsArea(a)}>
                    {ims?.areas[a] ?? a}
                  </button>
                ))}
              </div>
            )}
            <div className="param-list">
              {groups.map((g) => (
                <div key={g.cat} className="grp">
                  <span className="caption">{g.label}</span>
                  {g.types.map((t) => (
                    <button key={t.code} className={`prm${type === t.code ? ' on' : ''}`} onClick={() => setImsType(t.code)}>
                      <span className="lbl-long">{t.title}</span>
                      <span className="lbl-short">{t.title}</span>
                    </button>
                  ))}
                </div>
              ))}
            </div>
            <div className="rail-extra">
              {type && ims?.mapTypes[type]?.text && <p className="ims-text">{ims.mapTypes[type].text}</p>}
              <div className="source">
                <span className="caption">מקור</span>
                <b>{modelInfo.src}</b>
                <span>{modelInfo.res}</span>
              </div>
            </div>
          </nav>
        )}

        {!isIms && (
          <div className="levels glass" role="group" aria-label="גובה">
            <span className="caption">גובה (hPa)</span>
            {LEVELS.map((l) => {
              const ok = p.levels.includes(l.id);
              return (
                <button
                  key={l.id}
                  className={`lvl${lvl === l.id ? ' on' : ''}${ok ? '' : ' off'}`}
                  disabled={!ok}
                  onClick={() => setLevel(l.id)}
                  aria-pressed={lvl === l.id}
                >
                  {l.label}
                </button>
              );
            })}
          </div>
        )}

        {!isIms && (
          <div className="legend glass">
            <div className="legend-head">
              <span>{p.label} · {p.levelTitle(lvl)}</span>
              <span className="muted">{unitFor(p, lvl)}</span>
            </div>
            <div className="legend-bar" style={{ background: cssGradient(scale) }} />
            <div className="legend-ticks" dir="ltr">
              {scale.ticks.map((t) => <span key={t}>{t}</span>)}
            </div>
            {p.levels.length === 1 && <span className="note">פרמטר זה מוצג בפני השטח בלבד</span>}
          </div>
        )}

        <Timeline ticks={ticks} index={idx} onChange={(i) => { setPlaying(false); setIdx(i); }} playing={playing} onTogglePlay={() => setPlaying((x) => !x)} />
      </div>

      {(loading || mapError) && !isIms && (
        <div className={`status${mapError ? ' err' : ''}`} role="status">{mapError ? `שגיאה בטעינת נתונים: ${mapError}` : 'טוען שכבה…'}</div>
      )}
    </div>
  );
}
