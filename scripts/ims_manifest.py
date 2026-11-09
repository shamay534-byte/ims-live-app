#!/usr/bin/env python3
"""Fetch the IMS model-maps catalogue (ICON-IL, COSMO-IL, IFS images) and save a compact copy.

The IMS site (ims.gov.il/he/ModelMaps) loads its catalogue from /he/model_maps, which is not
CORS-enabled, so the browser cannot read it directly. The map images themselves can be
displayed from any site, so we only need to re-publish the index.

Output: public/data/ims/manifest.json
{
  "generated": "...",
  "models": {"icon": {"name": "ICON - IL", "runs": ["2026-11-09T00:00Z", ...]}, ...},
  "areas": {"ilplus": "ישראל וסביבתה", ...},
  "categories": {"Surface": "קרקע", ...},
  "mapTypes": {"slp": {"title": "...", "category": "Surface", "text": "..."}},
  "template": "/sites/.../modelMap_{model}_{code}_{area}_{run}_{valid}_{lead}.png",
  "maps": {"icon": {"2026-11-09T00:00Z": {"ilplus": {"slp": [3, 6, ..., [lead, "url"]]}}}}
  A bare number is a lead hour whose image follows the template; a pair carries its own URL.
}
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import sys
import urllib.request

URL = "https://ims.gov.il/he/model_maps"
SITE = "https://ims.gov.il"
UA = "Mozilla/5.0 (ims-live-app model viewer)"
# Image path pattern used by the IMS site; frames that follow it are stored as a bare lead hour.
TEMPLATE = ("/sites/default/files/ims_data/map_images/model_maps/{model}/"
            "modelMap_{model}_{code}_{area}_{run}_{valid}_{lead}.png")


def iso(s: str) -> str:
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%dT%H:%MZ")


def strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="public/data/ims/manifest.json")
    args = ap.parse_args()

    req = urllib.request.Request(URL, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)["data"]

    models = {}
    for code, m in data["models"].items():
        runs = sorted((iso(x) for x in data.get("all_run_times", {}).get(code, [])), reverse=True)
        models[code] = {"name": m["name"], "runs": runs, "areas": list(m.get("areas", {}))}

    map_types = {}
    for cat, types in data.get("map_types", {}).items():
        for code, t in types.items():
            map_types[code] = {"title": t.get("title", code), "category": cat, "text": strip_html(t.get("text", ""))}

    maps: dict = {}
    for model, runs in data.get("model_map", {}).items():
        for run_s, areas in runs.items():
            run = dt.datetime.strptime(run_s, "%Y-%m-%d %H:%M:%S")
            r_iso = iso(run_s)
            for area, cats in areas.items():
                for cat, types in cats.items():
                    for code, levels in types.items():
                        frames = []
                        for lvl_key, times in levels.items():
                            if lvl_key == "coordinates" or not isinstance(times, dict):
                                continue
                            for vt, path in times.items():
                                valid = dt.datetime.strptime(vt, "%Y-%m-%d %H:%M:%S")
                                lead = int((valid - run).total_seconds() // 3600)
                                expect = TEMPLATE.format(model=model, code=code, area=area,
                                                         run=run.strftime("%Y%m%d%H%M"),
                                                         valid=valid.strftime("%Y%m%d%H%M"), lead=lead)
                                frames.append(lead if path == expect else [lead, SITE + path])
                        frames.sort(key=lambda x: x if isinstance(x, int) else x[0])
                        if frames:
                            maps.setdefault(model, {}).setdefault(r_iso, {}).setdefault(area, {})[code] = frames
                            map_types.setdefault(code, {"title": code, "category": cat, "text": ""})

    out = {
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "source": "השירות המטאורולוגי הישראלי – ims.gov.il/he/ModelMaps",
        "site": SITE,
        "template": TEMPLATE,
        "models": models,
        "areas": data.get("areas", {}),
        "categories": data.get("categories", {}),
        "mapTypes": map_types,
        "maps": maps,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    n = sum(len(f) for m in maps.values() for r in m.values() for a in r.values() for f in a.values())
    print(f"wrote {args.out}: {len(maps)} models, {n} frames")


if __name__ == "__main__":
    sys.exit(main())
