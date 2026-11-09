#!/usr/bin/env python3
"""Download the latest ECMWF IFS 0.1° open-data run and cut it to an Israel/East-Med box.

Only the GRIB messages we need are downloaded, using HTTP range requests driven by the
`.index` files that ECMWF publishes next to every GRIB2 file. Each message is decoded with
ecCodes (the open data uses CCSDS packing, so a GRIB library with libaec is required),
cropped to the box, converted to friendly units and quantised to int16.

Output (default: public/data/ecmwf):
  manifest.json            run time, grid, steps, and one entry per field
  <runId>/<field>.bin      int16 little-endian, steps x nlat x nlon, value = q*scale+offset,
                           -32768 = missing

Usage:
  python3 scripts/ecmwf_fetch.py [--out public/data/ecmwf] [--max-step 360] [--run 20261109_00]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import math
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

import numpy as np

try:
    import eccodes
except ImportError:  # pragma: no cover
    sys.exit("ecCodes is required: pip install eccodes")

BASE = "https://data.ecmwf.int/forecasts"
UA = "ims-live-app model viewer (github.com/shamay534-byte/ims-live-app)"

# Crop box (degrees). The 0.1° grid is aligned to whole tenths.
LAT_N, LAT_S = 37.5, 27.0
LON_W, LON_E = 27.0, 42.0
LEVELS = ["925", "850", "700", "500", "300", "250"]

# field id -> (levtype, param, level or None)
SFC = ["2t", "2d", "10u", "10v", "10fg", "tp", "tcc", "msl", "mucape"]
PL = ["t", "u", "v", "r", "gh"]

# Quantisation per output field: (scale, offset). Range covered = offset +/- 32767*scale.
QUANT = {
    "t2m": (0.01, 0), "rh2m": (0.01, 0), "u10": (0.01, 0), "v10": (0.01, 0),
    "gust": (0.01, 0), "tp": (0.01, 0), "tcc": (0.01, 0), "msl": (0.01, 1000),
    "cape": (0.5, 0), "t": (0.01, 0), "u": (0.01, 0), "v": (0.01, 0), "r": (0.01, 0),
    "gh": (0.1, 0),
}
UNITS = {
    "t2m": "°C", "rh2m": "%", "u10": "m/s", "v10": "m/s", "gust": "m/s", "tp": "mm",
    "tcc": "%", "msl": "hPa", "cape": "J/kg", "t": "°C", "u": "m/s", "v": "m/s", "r": "%",
    "gh": "dam",
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def http_get(url: str, rng: tuple[int, int] | None = None, tries: int = 5) -> bytes:
    headers = {"User-Agent": UA}
    if rng:
        headers["Range"] = f"bytes={rng[0]}-{rng[1]}"
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            err = e
        except Exception as e:  # network hiccup
            err = e
        time.sleep(2 ** i)
    raise RuntimeError(f"GET failed {url}: {err}")


def steps_for(run_hour: int, max_step: int) -> list[int]:
    if run_hour in (0, 12):
        s = list(range(0, 145, 3)) + list(range(150, 361, 6))
    else:  # 06/18 runs are shorter
        s = list(range(0, 91, 3))
    return [x for x in s if x <= max_step]


def file_url(run: dt.datetime, step: int, kind: str) -> str:
    d = run.strftime("%Y%m%d")
    hh = run.strftime("%H")
    return f"{BASE}/{d}/{hh}z/ifs/0p1/oper/{d}{hh}0000-{step}h-oper-fc-{kind}"


def run_complete(run: dt.datetime, max_step: int) -> bool:
    last = steps_for(run.hour, max_step)[-1]
    try:
        http_get(file_url(run, last, "pl") + ".index")
        http_get(file_url(run, last, "sfc") + ".index")
        return True
    except Exception:
        return False


def find_latest_run(max_step: int) -> dt.datetime:
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    cand = now.replace(hour=12 if now.hour >= 12 else 0)
    for _ in range(8):
        if run_complete(cand, max_step):
            return cand
        cand -= dt.timedelta(hours=12)
    raise SystemExit("No complete ECMWF 0.1° run found in the last 4 days")


def read_index(url: str) -> list[dict]:
    txt = http_get(url).decode()
    return [json.loads(l) for l in txt.splitlines() if l.strip()]


_GRID = {}


def decode_crop(buf: bytes) -> np.ndarray:
    g = eccodes.codes_new_from_message(buf)
    try:
        ni = eccodes.codes_get(g, "Ni")
        nj = eccodes.codes_get(g, "Nj")
        la1 = eccodes.codes_get(g, "latitudeOfFirstGridPointInDegrees")
        lo1 = eccodes.codes_get(g, "longitudeOfFirstGridPointInDegrees")
        dlon = eccodes.codes_get(g, "iDirectionIncrementInDegrees")
        dlat = eccodes.codes_get(g, "jDirectionIncrementInDegrees")
        vals = eccodes.codes_get_values(g).reshape(nj, ni)
    finally:
        eccodes.codes_release(g)
    # Rows go north -> south starting at la1; columns start at lo1 (180 in the open data)
    r0 = int(round((la1 - LAT_N) / dlat))
    r1 = int(round((la1 - LAT_S) / dlat))
    cols = [int(round(((lon - lo1) % 360) / dlon)) % ni for lon in lon_axis()]
    return vals[r0:r1 + 1][:, cols].astype(np.float32)


def lon_axis():
    n = int(round((LON_E - LON_W) / 0.1)) + 1
    return [LON_W + i * 0.1 for i in range(n)]


def fetch_step(run: dt.datetime, step: int) -> dict[str, np.ndarray]:
    """Return {(param, level): array} for one forecast step."""
    out: dict[str, np.ndarray] = {}
    for kind, wanted in (("sfc", SFC), ("pl", PL)):
        base = file_url(run, step, kind)
        idx = read_index(base + ".index")
        for e in idx:
            p = e.get("param")
            if p not in wanted:
                continue
            lev = e.get("levelist")
            if kind == "pl" and lev not in LEVELS:
                continue
            o, n = int(e["_offset"]), int(e["_length"])
            buf = http_get(base + ".grib2", (o, o + n - 1))
            key = p if kind == "sfc" else f"{p}_{lev}"
            out[key] = decode_crop(buf)
    return out


def rh_from(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    a, b = 17.625, 243.04
    return np.clip(100 * np.exp(a * td_c / (b + td_c) - a * t_c / (b + t_c)), 0, 100)


def convert(raw: dict[str, np.ndarray], prev_tp: np.ndarray | None):
    """Map raw GRIB fields to output fields in display units."""
    f: dict[str, np.ndarray] = {}
    if "2t" in raw:
        f["t2m"] = raw["2t"] - 273.15
        if "2d" in raw:
            f["rh2m"] = rh_from(f["t2m"], raw["2d"] - 273.15)
    for a, b in (("10u", "u10"), ("10v", "v10"), ("10fg", "gust"), ("mucape", "cape")):
        if a in raw:
            f[b] = raw[a]
    if "tcc" in raw:
        f["tcc"] = np.clip(raw["tcc"] * 100, 0, 100)
    if "msl" in raw:
        f["msl"] = raw["msl"] / 100
    tp_acc = raw.get("tp")
    if tp_acc is not None:
        acc_mm = tp_acc * 1000
        f["tp"] = np.clip(acc_mm - (prev_tp if prev_tp is not None else 0), 0, None)
    for lev in LEVELS:
        if f"t_{lev}" in raw:
            f[f"t_{lev}"] = raw[f"t_{lev}"] - 273.15
        for p in ("u", "v", "r"):
            if f"{p}_{lev}" in raw:
                f[f"{p}_{lev}"] = raw[f"{p}_{lev}"]
        if f"gh_{lev}" in raw:
            f[f"gh_{lev}"] = raw[f"gh_{lev}"] / 10
    return f, (tp_acc * 1000 if tp_acc is not None else prev_tp)


def quant_key(field: str) -> str:
    return field.split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="public/data/ecmwf")
    ap.add_argument("--max-step", type=int, default=int(os.environ.get("ECMWF_MAX_STEP", 360)))
    ap.add_argument("--run", help="YYYYMMDD_HH; default: latest complete run")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.run:
        run = dt.datetime.strptime(args.run, "%Y%m%d_%H").replace(tzinfo=dt.timezone.utc)
    else:
        run = find_latest_run(args.max_step)
    run_id = run.strftime("%Y%m%d_%H")
    steps = steps_for(run.hour, args.max_step)
    log(f"run {run_id}: {len(steps)} steps up to +{steps[-1]}h")

    man_path = os.path.join(args.out, "manifest.json")
    if os.path.exists(man_path):
        try:
            old = json.load(open(man_path))
            if old.get("runId") == run_id and old.get("steps") == steps:
                log("already up to date")
                return
        except Exception:
            pass

    nlat = int(round((LAT_N - LAT_S) / 0.1)) + 1
    nlon = len(lon_axis())
    raws: dict[int, dict[str, np.ndarray]] = {}
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch_step, run, s): s for s in steps}
        for i, fu in enumerate(cf.as_completed(futs), 1):
            s = futs[fu]
            raws[s] = fu.result()
            log(f"  step +{s}h done ({i}/{len(steps)}, {len(raws[s])} fields)")

    # Convert in step order (precipitation is de-accumulated against the previous step)
    fields: dict[str, np.ndarray] = {}
    prev_tp = None
    for si, s in enumerate(steps):
        conv, prev_tp = convert(raws.pop(s), prev_tp)
        for k, arr in conv.items():
            if k not in fields:
                fields[k] = np.full((len(steps), nlat, nlon), np.nan, np.float32)
            fields[k][si] = arr

    out_dir = os.path.join(args.out, run_id)
    tmp_dir = out_dir + ".tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    entries = {}
    for k, arr in sorted(fields.items()):
        scale, offset = QUANT[quant_key(k)]
        q = np.round((arr - offset) / scale)
        q = np.where(np.isnan(arr), -32768, np.clip(q, -32767, 32767)).astype("<i2")
        q.tofile(os.path.join(tmp_dir, f"{k}.bin"))
        valid = ~np.isnan(arr)
        entries[k] = {
            "file": f"{run_id}/{k}.bin",
            "scale": scale,
            "offset": offset,
            "units": UNITS[quant_key(k)],
            "min": float(np.nanmin(arr)) if valid.any() else None,
            "max": float(np.nanmax(arr)) if valid.any() else None,
        }
    shutil.rmtree(out_dir, ignore_errors=True)
    os.rename(tmp_dir, out_dir)
    # Remove older runs
    for d in os.listdir(args.out):
        p = os.path.join(args.out, d)
        if os.path.isdir(p) and d != run_id:
            shutil.rmtree(p, ignore_errors=True)

    manifest = {
        "model": "ECMWF IFS HRES",
        "resolution": "0.1°",
        "source": "ECMWF open data (CC-BY-4.0)",
        "run": run.strftime("%Y-%m-%dT%H:%MZ"),
        "runId": run_id,
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "grid": {"lat0": LAT_N, "lon0": LON_W, "dlat": -0.1, "dlon": 0.1, "nlat": nlat, "nlon": nlon},
        "steps": steps,
        "levels": LEVELS,
        "fields": entries,
    }
    with open(man_path, "w") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    log(f"wrote {len(entries)} fields to {out_dir}")


if __name__ == "__main__":
    main()
