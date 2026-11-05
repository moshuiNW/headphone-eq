# -*- coding: utf-8 -*-
"""Where things live on disk, and what a scraped measurement looks like on disk.

Layout under the skill root (created on demand):

    data/targets/         target curves: <name>.csv with frequency,db
    data/headphones/      one folder per headphone:
        <slug>/meta.json          provenance, tuning modes, variants
        <slug>/<variant>.csv      the raw curve, un-normalised
        <slug>/<variant>.png      the curve as the site drew it (optional)
    data/blue-dashed.csv  the digitised game reference curve (bundled)
    out/<slug>/           generated configs, plots and reports
    fixtures/             regression curves + expected scores

Provenance is data, not prose: meta.json is what a later run reads to know which
fixture, which tuning mode and which grab date a config came from.
"""
import io
import json
import os
import re
import time

SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(SKILL_ROOT, "data")
HEADPHONES = os.path.join(DATA, "headphones")
OUT = os.path.join(SKILL_ROOT, "out")
FIXTURES = os.path.join(SKILL_ROOT, "fixtures")


def ensure_dirs():
    for d in (DATA, HEADPHONES, OUT, FIXTURES):
        os.makedirs(d, exist_ok=True)


def slugify(text):
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "-", str(text).strip(), flags=re.UNICODE)
    s = re.sub(r"-{2,}", "-", s).strip("-").lower()
    return s or "unnamed"


def headphone_dir(name):
    return os.path.join(HEADPHONES, slugify(name))


def out_dir(name):
    d = os.path.join(OUT, slugify(name))
    os.makedirs(d, exist_ok=True)
    return d


def write_meta(name, meta):
    d = headphone_dir(name)
    os.makedirs(d, exist_ok=True)
    meta = dict(meta)
    meta.setdefault("saved_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    path = os.path.join(d, "meta.json")
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    return path


def read_meta(name):
    path = os.path.join(headphone_dir(name), "meta.json")
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def variant_path(name, variant):
    return os.path.join(headphone_dir(name), slugify(variant) + ".csv")


def write_json(path, obj):
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    return path
