import { BASE } from './ecmwf';

export interface ImsManifest {
  generated: string;
  source: string;
  site: string;
  template: string;
  models: Record<string, { name: string; runs: string[]; areas: string[] }>;
  areas: Record<string, string>;
  categories: Record<string, string>;
  mapTypes: Record<string, { title: string; category: string; text: string }>;
  maps: Record<string, Record<string, Record<string, Record<string, (number | [number, string])[]>>>>;
}

export async function loadImsManifest(): Promise<ImsManifest> {
  const r = await fetch(`${BASE}/data/ims/manifest.json`, { cache: 'no-cache' });
  if (!r.ok) throw new Error(`ims manifest ${r.status}`);
  return r.json();
}

const stamp = (ms: number) => {
  const d = new Date(ms);
  const p = (n: number) => String(n).padStart(2, '0');
  return `${d.getUTCFullYear()}${p(d.getUTCMonth() + 1)}${p(d.getUTCDate())}${p(d.getUTCHours())}${p(d.getUTCMinutes())}`;
};

export interface Frame {
  lead: number;
  url: string;
}

export function framesFor(m: ImsManifest, model: string, run: string, area: string, type: string): Frame[] {
  const list = m.maps[model]?.[run]?.[area]?.[type];
  if (!list) return [];
  const runMs = Date.parse(run.replace('Z', ':00Z'));
  return list.map((x) => {
    if (Array.isArray(x)) return { lead: x[0], url: x[1] };
    const path = m.template
      .replace(/\{model\}/g, model)
      .replace('{code}', type)
      .replace('{area}', area)
      .replace('{run}', stamp(runMs))
      .replace('{valid}', stamp(runMs + x * 3600e3))
      .replace('{lead}', String(x));
    return { lead: x, url: m.site + path };
  });
}

/** Map types available for model/run/area, grouped by IMS category in catalogue order. */
export function typesFor(m: ImsManifest, model: string, run: string, area: string) {
  const avail = m.maps[model]?.[run]?.[area] ?? {};
  const groups: { cat: string; label: string; types: { code: string; title: string }[] }[] = [];
  for (const [cat, label] of Object.entries(m.categories)) {
    const types = Object.entries(m.mapTypes)
      .filter(([code, t]) => t.category === cat && avail[code])
      .map(([code, t]) => ({ code, title: t.title }));
    if (types.length) groups.push({ cat, label, types });
  }
  return groups;
}
