# -*- coding: utf-8 -*-
"""Measurement sources.

Three ways in, in order of preference:

    huihifi   the evaluation site the user named.  Its pages are client-rendered,
              so the curve only exists at the JSON API; scraping the DOM returns
              nothing.  Endpoints used (all public, no auth):
                  POST /api/v1/evaluations                       search by keyword
                  GET  /api/v1/evaluations/<uuid>                full record
                  GET  /api/v1/global-configs/unique             site target curves
    autoeq    jaakkopasanen/AutoEq GitHub results: a second, independent
              measurement of the same model, plus its own target curves.
    file      a curve the user already has (CSV/TSV, or a URL to one).

Every fetch returns the same shape so the rest of the toolkit never branches on
source:

    {"title", "source", "variants": [{"name", "freq", "level", "note"}], "meta"}
"""
import io
import json
import os
import re
import time
import urllib.error
import urllib.request

from . import curves

HUIHIFI = "https://huihifi.com"
AUTOEQ_API = "https://api.github.com/repos/jaakkopasanen/AutoEq/contents"
AUTOEQ_RAW = "https://raw.githubusercontent.com/jaakkopasanen/AutoEq/master"
UA = "Mozilla/5.0 (compatible; eqkit/1.0)"

# The site's own target curves live under a name we cannot rely on; the digitised
# blue reference curve is shipped with the skill instead (see data/blue-dashed.csv).
SITE_TARGETS = "global-configs/unique"


def _request(url, data=None, method=None, timeout=30, proxy=None):
    body = None
    headers = {"User-Agent": UA, "Accept": "application/json"}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method or ("POST" if body else "GET"))
    if proxy:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    else:
        opener = urllib.request.build_opener()
    with opener.open(req, timeout=timeout) as fh:
        return fh.read().decode("utf-8", "replace")


def _json(url, **kw):
    return json.loads(_request(url, **kw))


# ------------------------------------------------------------------- huihifi
def huihifi_search(keyword, proxy=None, page_size=20):
    """Search the site.  Returns the list records (uuid, title, brand, modes).

    The list endpoint demands category/orderBy/direction and rejects a *page*
    parameter outright, so this asks for one page and filters locally.
    """
    body = {
        "category": "EARBUDS",
        "orderBy": "updateTime",
        "direction": "DESC",
        "keyword": keyword,
        "pageSize": page_size,
    }
    payload = _json(HUIHIFI + "/api/v1/evaluations", data=body, proxy=proxy)
    items = ((payload or {}).get("data") or {}).get("list") or []
    out = []
    for it in items:
        out.append({
            "uuid": it.get("uuid"),
            "title": it.get("title") or "",
            "brand": (it.get("brand") or {}).get("title") or "",
            "modes": it.get("dataGroups") or [],
            "update_time": it.get("updateTime"),
            "url": "%s/evaluation/%s" % (HUIHIFI, it.get("uuid")),
        })
    return out


def _spec_labels(proxy=None):
    """Map specification uuid -> readable label (tuning mode / wearing / driver)."""
    try:
        specs = _json(HUIHIFI + "/api/v1/specifications/all", proxy=proxy).get("data") or []
    except Exception:
        return {}
    label = {}
    for s in specs:
        uuid = s.get("uuid")
        if not uuid:
            continue
        mapping = {}
        extra = s.get("extra") or {}
        for opt in (extra.get("mapOptions") or []):
            if isinstance(opt, dict) and "value" in opt and "label" in opt:
                mapping[opt["value"]] = opt["label"]
        label[uuid] = {"name": s.get("name"), "map": mapping}
    return label


def huihifi_record(uuid, proxy=None):
    """Fetch one evaluation and pull out every raw frequency-response variant."""
    rec = _json("%s/api/v1/evaluations/%s" % (HUIHIFI, uuid), proxy=proxy)
    data = (rec or {}).get("data") or {}
    specs = _spec_labels(proxy=proxy) if data.get("specifications") else {}

    detail = []
    for s in data.get("specifications") or []:
        info = specs.get(s.get("specificationUuid")) or {}
        raw = (s.get("value") or "").strip(";")
        if not raw:
            continue
        vals = [v for v in raw.split(";") if v]
        shown = "、".join((info.get("map") or {}).get(v, v) for v in vals)
        detail.append({"name": info.get("name") or s.get("specificationUuid"), "value": shown, "raw": vals})

    variants = []
    for ds in data.get("datas") or []:
        project = ds.get("project") or {}
        if project.get("code") != "RAW_FR":
            continue
        for child in ds.get("children") or []:
            rows = child.get("dataSet") or []
            if len(rows) < 3:
                continue
            pairs = []
            for row in rows[1:]:
                try:
                    pairs.append((float(row[0]), float(row[1])))
                except (TypeError, ValueError, IndexError):
                    continue
            if len(pairs) < 10:
                continue
            freq, level = curves.from_xy(pairs)
            child_name = child.get("name") or "频响"
            # The site labels a plot as "<series name>:<y axis>", e.g.
            # "2.4G模式频响:SPL(dB)".  Keep that exact string so a config can be
            # matched back to the line the user pointed at, character for character.
            y_label = ""
            if isinstance(rows[0], (list, tuple)) and len(rows[0]) > 1:
                y_label = str(rows[0][1])
            variants.append({
                "name": child_name,
                "label": ("%s:%s" % (child_name, y_label)) if y_label else child_name,
                "freq": freq,
                "level": level,
                "note": "raw SPL, uncompensated (%d points, %.0f-%.0f Hz)"
                        % (len(freq), float(min(freq)), float(max(freq))),
            })
    return {
        "title": data.get("title") or "",
        "brand": (data.get("brand") or {}).get("title") or "",
        "source": "huihifi",
        "url": "%s/evaluation/%s" % (HUIHIFI, uuid),
        "uuid": uuid,
        "modes": data.get("dataGroups") or [],
        "specifications": detail,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "variants": variants,
        "meta": {"price": data.get("price"), "category": data.get("category")},
    }


def huihifi_site_targets(proxy=None):
    """The site's own target curves, as {name: (freq, level)}."""
    payload = _json(HUIHIFI + "/api/v1/" + SITE_TARGETS, proxy=proxy)
    out = {}
    for t in ((payload or {}).get("data") or {}).get("targets") or []:
        rows = t.get("dataSet") or []
        pairs = []
        for row in rows:
            try:
                pairs.append((float(row[0]), float(row[1])))
            except (TypeError, ValueError, IndexError):
                continue
        if len(pairs) > 10:
            out[t.get("name")] = curves.from_xy(pairs)
    return out


# --------------------------------------------------------------------- AutoEq
def autoeq_find(model, proxy=None, measurement="oratory1990"):
    """Search AutoEq's results tree for a model name.  Returns candidate folders."""
    listings = {}
    out = []
    for group in ("over-ear", "in-ear", "earbud"):
        try:
            items = _json("%s/results/%s/%s" % (AUTOEQ_API, measurement, group), proxy=proxy)
        except Exception:
            continue
        for it in items:
            name = it.get("name") or ""
            if it.get("type") != "dir":
                if name.endswith(".csv"):
                    listings.setdefault(group, []).append(it)
                continue
            listings.setdefault(group, []).append(it)
            if _name_matches(model, name):
                out.append({"group": group, "name": name, "path": it.get("path"), "measurement": measurement})
    return out, listings


def _tokens(text):
    return [t for t in re.split(r"[^0-9a-zA-Z\u4e00-\u9fff]+", text.lower()) if t]


def _name_matches(query, candidate):
    q = _tokens(query)
    c = _tokens(candidate)
    if not q:
        return False
    joined_c = "".join(c)
    if "".join(q) in joined_c:
        return True
    hits = sum(1 for t in q if t in c or t in joined_c)
    return hits == len(q)


def autoeq_curve(path, proxy=None):
    """Read an AutoEq result CSV (frequency,raw,...) -> (freq, level)."""
    url = "%s/%s/%s.csv" % (AUTOEQ_RAW, path.strip("/"), path.strip("/").split("/")[-1])
    text = _request(url, proxy=proxy)
    freq, level, header = [], [], None
    for line in text.splitlines():
        if header is None:
            header = line.strip().split(",")
            idx = 1
            if "smoothed" in header:
                idx = header.index("smoothed")
            continue
        parts = line.strip().split(",")
        if len(parts) <= idx:
            continue
        try:
            freq.append(float(parts[0]))
            level.append(float(parts[idx]))
        except ValueError:
            continue
    return curves.from_xy(list(zip(freq, level)))


def autoeq_target(name, proxy=None):
    """Read one of AutoEq's target curves from targets/<name>.csv."""
    url = "%s/targets/%s.csv" % (AUTOEQ_RAW, urllib.parse.quote(name))
    text = _request(url, proxy=proxy)
    freq, level = [], []
    for i, line in enumerate(text.splitlines()):
        if i == 0:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            freq.append(float(parts[0]))
            level.append(float(parts[1]))
        except ValueError:
            continue
    return curves.from_xy(list(zip(freq, level)))


# ----------------------------------------------------------------------- file
def from_file(path):
    freq, level = curves.read_csv(path)
    return {
        "title": os.path.splitext(os.path.basename(path))[0],
        "brand": "",
        "source": "file",
        "url": path,
        "variants": [{"name": "curve", "freq": freq, "level": level, "note": "user-supplied file"}],
        "meta": {},
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def resolve_proxy(explicit=None):
    """Honour an explicit proxy, else the usual environment variables."""
    if explicit:
        return explicit
    return os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTP_PROXY")
