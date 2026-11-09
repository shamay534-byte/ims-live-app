import { SCALES, Scale, stretch } from './colors';

export type LevelId = 'sfc' | '925' | '850' | '700' | '500' | '300' | '250';
export const LEVELS: { id: LevelId; label: string }[] = [
  { id: 'sfc', label: 'קרקע' },
  { id: '925', label: '925' },
  { id: '850', label: '850' },
  { id: '700', label: '700' },
  { id: '500', label: '500' },
  { id: '300', label: '300' },
  { id: '250', label: '250' },
];

export type ParamId = 'temp' | 'wind' | 'gust' | 'precip' | 'clouds' | 'rh' | 'pressure' | 'cape';
export type IconName = 'temp' | 'wind' | 'gust' | 'drop' | 'cloud' | 'humidity' | 'gauge' | 'bolt';

/** How a parameter is read from the ECMWF fields. */
export type Source =
  | { kind: 'scalar'; field: string; factor?: number }
  | { kind: 'speed'; u: string; v: string; factor: number };

export interface ParamDef {
  id: ParamId;
  label: string;
  short: string;
  icon: IconName;
  levels: LevelId[];
  unit: string;
  decimals: number;
  source: (lvl: LevelId) => Source;
  scale: (lvl: LevelId, range?: [number, number]) => Scale;
  /** optional contour overlay: field id and interval */
  contours?: (lvl: LevelId) => { field: string; step: number; unit: string } | null;
  opacity: number;
  levelTitle: (lvl: LevelId) => string;
}

const ALL: LevelId[] = ['sfc', '925', '850', '700', '500', '300', '250'];
const PL = (l: LevelId) => `${l} hPa`;

export const PARAMS: ParamDef[] = [
  {
    id: 'temp', label: 'טמפרטורה', short: 'טמפ׳', icon: 'temp', levels: ALL, unit: '°C', decimals: 0,
    source: (l) => ({ kind: 'scalar', field: l === 'sfc' ? 't2m' : `t_${l}` }),
    scale: () => SCALES.temp,
    contours: (l) => (l === 'sfc' ? null : { field: `gh_${l}`, step: l === '925' || l === '850' ? 3 : 6, unit: 'dam' }),
    opacity: 0.78,
    levelTitle: (l) => (l === 'sfc' ? '2 מ׳' : PL(l)),
  },
  {
    id: 'wind', label: 'רוח', short: 'רוח', icon: 'wind', levels: ALL, unit: 'קמ״ש', decimals: 0,
    source: (l) =>
      l === 'sfc' ? { kind: 'speed', u: 'u10', v: 'v10', factor: 3.6 } : { kind: 'speed', u: `u_${l}`, v: `v_${l}`, factor: 3.6 },
    scale: () => SCALES.wind,
    opacity: 0.78,
    levelTitle: (l) => (l === 'sfc' ? '10 מ׳' : PL(l)),
  },
  {
    id: 'gust', label: 'משבי רוח', short: 'משבים', icon: 'gust', levels: ['sfc'], unit: 'קמ״ש', decimals: 0,
    source: () => ({ kind: 'scalar', field: 'gust', factor: 3.6 }),
    scale: () => SCALES.wind,
    opacity: 0.78,
    levelTitle: () => '10 מ׳',
  },
  {
    id: 'precip', label: 'משקעים', short: 'משקעים', icon: 'drop', levels: ['sfc'], unit: 'מ״מ', decimals: 1,
    source: () => ({ kind: 'scalar', field: 'tp' }),
    scale: () => SCALES.precip,
    contours: () => ({ field: 'msl', step: 4, unit: 'hPa' }),
    opacity: 0.9,
    levelTitle: () => 'מצטבר מהצעד הקודם',
  },
  {
    id: 'clouds', label: 'עננות', short: 'עננות', icon: 'cloud', levels: ['sfc'], unit: '%', decimals: 0,
    source: () => ({ kind: 'scalar', field: 'tcc' }),
    scale: () => SCALES.clouds,
    opacity: 1,
    levelTitle: () => 'כיסוי כולל',
  },
  {
    id: 'rh', label: 'לחות יחסית', short: 'לחות', icon: 'humidity', levels: ALL, unit: '%', decimals: 0,
    source: (l) => ({ kind: 'scalar', field: l === 'sfc' ? 'rh2m' : `r_${l}` }),
    scale: () => SCALES.rh,
    opacity: 0.72,
    levelTitle: (l) => (l === 'sfc' ? '2 מ׳' : PL(l)),
  },
  {
    id: 'pressure', label: 'לחץ / גובה', short: 'לחץ', icon: 'gauge', levels: ALL, unit: 'hPa', decimals: 0,
    source: (l) => ({ kind: 'scalar', field: l === 'sfc' ? 'msl' : `gh_${l}` }),
    scale: (l, range) => {
      if (l === 'sfc') return SCALES.msl;
      const [lo, hi] = range ?? [0, 1];
      const step = hi - lo > 60 ? 12 : 6;
      return stretch(SCALES.gh, lo, hi, step);
    },
    contours: (l) => (l === 'sfc' ? { field: 'msl', step: 2, unit: 'hPa' } : { field: `gh_${l}`, step: l === '925' || l === '850' ? 3 : 6, unit: 'dam' }),
    opacity: 0.7,
    levelTitle: (l) => (l === 'sfc' ? 'מפלס הים' : `גובה ${PL(l)}`),
  },
  {
    id: 'cape', label: 'CAPE', short: 'CAPE', icon: 'bolt', levels: ['sfc'], unit: 'J/kg', decimals: 0,
    source: () => ({ kind: 'scalar', field: 'cape' }),
    scale: () => SCALES.cape,
    opacity: 0.85,
    levelTitle: () => 'Most-unstable',
  },
];

export const paramById = (id: ParamId) => PARAMS.find((p) => p.id === id)!;

export function unitFor(p: ParamDef, l: LevelId) {
  return p.id === 'pressure' && l !== 'sfc' ? 'dam' : p.unit;
}

/** Fields that must be loaded to show a parameter at a level (incl. particles + contours). */
export function fieldsFor(p: ParamDef, l: LevelId, particles: boolean): string[] {
  const s = p.source(l);
  const out = new Set<string>(s.kind === 'scalar' ? [s.field] : [s.u, s.v]);
  const c = p.contours?.(l);
  if (c) out.add(c.field);
  if (particles) {
    out.add(l === 'sfc' ? 'u10' : `u_${l}`);
    out.add(l === 'sfc' ? 'v10' : `v_${l}`);
  }
  return [...out];
}

// ---- IMS (ICON-IL / COSMO-IL) map types --------------------------------------------------

export type ModelId = 'ecmwf' | 'icon' | 'cosmo';
export const MODELS: { id: ModelId; label: string; sub: string }[] = [
  { id: 'ecmwf', label: 'ECMWF', sub: 'IFS · 0.1°' },
  { id: 'icon', label: 'ICON שמ״ט', sub: 'ICON-IL' },
  { id: 'cosmo', label: 'COSMO שמ״ט', sub: 'COSMO-IL' },
];

/** Best IMS map type for an ECMWF param/level, used when switching model. */
export function imsTypeFor(p: ParamId, l: LevelId): string {
  if (l === 'sfc') {
    return ({ temp: 'temp2wind10', wind: 'temp2wind10', gust: 'temp2wind10', precip: 'precip3hr', clouds: 'wxclouds', rh: 'rh2wind10', pressure: 'slp', cape: 'capeWater' } as Record<ParamId, string>)[p];
  }
  if (p === 'rh') return l === '850' || l === '925' ? 'map850rhWind' : l === '700' ? 'map700rhWind' : 'map500rhWind';
  if (p === 'wind' && (l === '300' || l === '250')) return 'map200';
  if (l === '925' || l === '850') return 'map850tempWind';
  if (l === '700') return 'map700hgt';
  if (l === '500') return '500temp';
  return 'map200';
}

/** Reverse mapping, used when switching from an IMS model back to ECMWF. */
export function ecmwfFor(type: string): { p: ParamId; l: LevelId } {
  const m: Record<string, { p: ParamId; l: LevelId }> = {
    temp2wind10: { p: 'temp', l: 'sfc' }, rh2wind10: { p: 'rh', l: 'sfc' }, slp: { p: 'pressure', l: 'sfc' },
    slpThickness: { p: 'pressure', l: 'sfc' }, precip3hr: { p: 'precip', l: 'sfc' }, prProb05: { p: 'precip', l: 'sfc' },
    prProb2: { p: 'precip', l: 'sfc' }, prProb5: { p: 'precip', l: 'sfc' }, map850tempWind: { p: 'temp', l: '850' },
    map850rhWind: { p: 'rh', l: '850' }, map700omega: { p: 'rh', l: '700' }, map700hgt: { p: 'temp', l: '700' },
    map700rhWind: { p: 'rh', l: '700' }, '500temp': { p: 'temp', l: '500' }, '500vort': { p: 'pressure', l: '500' },
    map500rhWind: { p: 'rh', l: '500' }, map200: { p: 'wind', l: '250' }, capeWater: { p: 'cape', l: 'sfc' },
    ecclouds: { p: 'clouds', l: 'sfc' }, wxclouds: { p: 'clouds', l: 'sfc' },
  };
  return m[type] ?? { p: 'temp', l: 'sfc' };
}
