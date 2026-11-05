# -*- coding: utf-8 -*-
"""Judgement: which measurement to fit, and when to refuse to ship a config.

This module is the part a person would otherwise have to re-decide every run,
so it is written down as code with reasons attached.  Three decisions live here:

  * which of several measurements of the same model to fit (variants, re-tests,
    borrowed units, different ear pads or tuning modes);
  * whether the available data supports any config at all;
  * what to tell the user this fit *cannot* fix.

Refusing is a first-class outcome.  A config fitted to a curve that is really a
different headphone, or to a copy of the target itself, is worse than no config:
it spends the user's time and teaches them the EQ does not work.
"""
import re

import numpy as np

from .grid import GRID
from . import curves

# Wording that marks a measurement as not being a plain retail unit.
PENALTY_PATTERNS = (
    (r"\u501f\u6d4b|borrowed|sent by|loaner|user unit", 25, "borrowed unit, not a retail sample"),
    (r"\u91cd\u6d4b|retest|re-test", -10, "a re-test (usually the better sample)"),
    (r"\u56fa\u4ef6|firmware", 8, "firmware-specific result"),
    (r"\u5382\u5bb6\u9001\u6d4b|sample provided|golden sample", 30, "vendor-supplied sample"),
    (r"\u5c55\u4f1a|expo|show", 12, "expo unit"),
)

# Tuning modes that are not the plain analog/wired presentation.
MODE_PENALTIES = (
    (r"7\.1|\u865a\u62df\u73af\u7ed5|virtual", 20, "virtual surround mode colours the response"),
    (r"usb|\u58f0\u5361|dongle", 8, "USB/dongle path, not the analog path"),
    (r"\u6e38\u620f|game mode", 6, "game mode may apply its own DSP"),
    (r"\u964d\u566a|anc|\u900f\u4f20|transparen", 10, "ANC/transparency changes the response"),
    (r"\u4f4e\u9891.*(\u5f00|\u5173)|bass (on|off)|switch", 10, "hardware bass switch position"),
)

DEFAULT_MODE = re.compile(r"\u9ed8\u8ba4|default", re.I)


def name_matches(query, title):
    """Does *title* plausibly denote *query*?

    Token-subset matching, not substring: a user typing "HD 600" should match
    "Sennheiser HD 600", and a user typing "G2Pro" should not match "G28 Pro".
    """
    q = [t for t in re.split(r"[^0-9a-zA-Z\u4e00-\u9fff]+", query.lower()) if t]
    if not q:
        return False
    flat = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]", "", title.lower())
    return all(t in flat for t in q)


def _penalty(text):
    hits = []
    score = 0
    for pattern, weight, why in PENALTY_PATTERNS + MODE_PENALTIES:
        if re.search(pattern, text, re.I):
            score += weight
            hits.append(why)
    return score, hits


def rank_candidates(records):
    """Order search hits: exact-model matches first, retail units before borrowed ones.

    A search for "G2Pro" returns the model, its siblings, and often unrelated
    products that share a token.  Ranking is explicit so the choice is
    reviewable rather than "whatever came back first".
    """
    out = []
    for r in records:
        text = " ".join([r.get("title", ""), r.get("brand", "")])
        penalty, reasons = _penalty(text)
        modes = r.get("modes") or []
        default_mode = [m for m in modes if DEFAULT_MODE.search(m)]
        if modes and not default_mode:
            penalty += 5
            reasons.append("no default/stock tuning mode offered")
        out.append({
            "record": r,
            "penalty": penalty,
            "reasons": reasons,
            "modes": modes,
            "preferred_mode": (default_mode or modes or ["default"])[0],
        })
    return out


def rank_for_query(records, query):
    """Order hits by how well each title matches *query*, then by unit quality.

    Match quality has to come first.  Anything else puts a long unrelated name
    above the model the user asked for, and the runner-up then looks like the
    answer because it sits further down a list nobody reads.  Match first, then
    penalties, then the plainest name.
    """
    ranked = rank_candidates(records)

    def key(entry):
        title = entry["record"].get("title", "")
        return (0 if name_matches(query, title) else 1, entry["penalty"], len(title))

    ranked.sort(key=key)
    return ranked


def pick_variant(record, mode=None):
    """Choose which frequency-response variant of one evaluation to fit.

    The site publishes one RAW_FR child per tuning mode / pad set / connection.
    Fitting the wrong child is the single easiest way to produce a config that
    models a different headphone, so the pick and its reason are returned.
    """
    variants = record.get("variants") or []
    if not variants:
        return None, "the evaluation contains no frequency-response data"
    if len(variants) == 1:
        return variants[0], "only one response variant published (%s)" % variants[0]["name"]
    if mode:
        wanted = mode.strip().lower()
        for v in variants:
            hay = ("%s %s" % (v.get("name", ""), v.get("label", ""))).lower()
            if wanted in hay:
                return v, "matched the requested curve %r" % v.get("label", v["name"])
        return None, ("requested curve %r is not one of: %s"
                      % (mode, "; ".join(v.get("label", v["name"]) for v in variants)))
    # More than one curve and nothing chosen: the user must choose.  Two curves
    # of the same model can differ more than the correction itself (a 2.4G path
    # and a wired path, or two pad sets), so guessing would silently model a
    # different headphone.
    return None, "choosing between %d published curves" % len(variants)


def similarity(a, b, grid=GRID):
    """Level-free similarity of two curves: RMS gap in dB over 20 Hz-16 kHz."""
    m = (grid >= 20) & (grid < 16000)
    d = (a - b)[m]
    return float(np.sqrt(np.mean((d - np.mean(d)) ** 2)))


def eligibility(measured, target, measured_freq=None, min_points=60):
    """Can a config be fitted from this pair at all?  Returns (ok, reasons)."""
    reasons = []
    if measured_freq is None:
        reasons.append("no raw measurement frequencies recorded")
    else:
        f = np.asarray(measured_freq, float)
        if len(f) < min_points:
            reasons.append("only %d measurement points, too few to resolve peaks" % len(f))
        span_oct = float(np.log2(f.max() / f.min())) if len(f) else 0.0
        if f.max() < 10000.0:
            reasons.append("measurement stops at %.0f Hz, so the treble cannot be fitted" % f.max())
        if span_oct < 7.0:
            reasons.append("measurement spans only %.1f octaves" % span_oct)
    gap = similarity(measured, target)
    if gap < 0.8:
        reasons.append("the measurement is already within %.2f dB of the target; an EQ would be all cost and no benefit" % gap)
    flat = float(np.max(curves.smooth(measured, 1 / 3)) - np.min(curves.smooth(measured, 1 / 3)))
    if flat < 1.0:
        reasons.append("the measured curve is almost flat (%.1f dB spread); there is nothing meaningful to correct" % flat)
    return (not reasons), reasons, gap


def caveats(record, variant, measured_freq, stats, stability=None):
    """The honest limitations paragraph, derived from the data rather than boilerplate."""
    out = []
    if measured_freq is not None and len(measured_freq):
        f = np.asarray(measured_freq, float)
        out.append("Single measurement from %s, covering %.0f-%.0f Hz; pad seal and placement move 1-4 kHz by up to 3 dB on the same unit."
                   % (record.get("source", "unknown source"), f.min(), f.max()))
    conf = None
    for b in stats["bands"]:
        if b["confidence"] != "measured" and b["rms"] > 1.0:
            conf = b
    if conf is not None:
        out.append("Above 5 kHz this is a trend correction, not a fit: the %s band residuals (%.2f dB RMS) are dominated by fixture comb filtering."
                   % (conf["name"], conf["rms"]))
    if stability:
        fake = [s for s in stability if not s["real"] and abs(s["excess"]) >= 3.0]
        if fake:
            out.append("Features at %s did not survive wide-window smoothing, so they were treated as measurement artefacts and were not corrected."
                       % ", ".join("%.0f Hz" % s["freq"] for s in fake[:4]))
    if variant and variant.get("name"):
        out.append("Fitted to the %r response; other tuning modes or pad sets of the same model need their own config."
                   % variant["name"])
    out.append("Predicted residual assumes the EQ is exactly the model in this file; real units vary by unit-to-unit spread of 1-2 dB.")
    return out
