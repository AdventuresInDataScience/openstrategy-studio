"""Pattern research (plan phase 3): what happened after each pattern?

A signal on bar t fills at bar t + 1's open, so forward returns here run from
the next open to the close `horizon` bars after t: exactly what a trade on the
signal would have earned before costs. Bearish signals (-1) are judged as
shorts and everything else as longs. Each pattern is compared with the same
trades taken on every bar (the baseline), so a pattern that only rides the
market's drift does not look clever.

Read the numbers with care. Hits closer together than the horizon overlap, so
the t-statistic is optimistic; and scanning 300 detectors will find some that
look good by chance. Use the optimiser's held-back test before trusting one.
"""

import math

import numpy as np
import pandas as pd

from . import indicators

# Detectors whose cost grows roughly with the square of the pivot count (plan annex D).
SLOW_PATTERNS = ("abc_correction", "measured_move_up", "measured_move_down", "hs_top", "hs_bottom")
SLOW_PREFIXES = ("busted_hs_", "complex_hs_")


def is_slow(name):
    return name in SLOW_PATTERNS or name.startswith(SLOW_PREFIXES)


def forward_returns(data, horizon):
    """Return from the next bar's open to the close `horizon` bars later, one column per symbol."""
    return data["close"].shift(-horizon) / data["open"].shift(-1) - 1


def pattern_signals(data, module, name, params=None):
    """The detector's int8 output (+1 bullish, -1 bearish, 0 none) for every symbol."""
    return indicators.run_pattern(module, name, dict(params or {}), data["open"], data["high"], data["low"],
                                  data["close"], data["volume"])


def pattern_rows():
    """Every pattern in the catalogue, as the study wants them: module, name, direction, family, label."""
    rows = []
    for item in indicators.catalogue().values():
        if item["source"] == "pattern":
            module, name = item["function"].split(":")
            rows.append({"module": module, "name": name, "direction": item.get("direction", ""),
                         "family": item["menu"][1], "label": item["label"], "id": item["id"]})
    return rows


def _signed(values, forward):
    """Forward returns at the hits, signed by direction (bearish judged as a short)."""
    hits = (values != 0) & np.isfinite(forward)
    sign = np.where(values == -1, -1.0, 1.0)
    return (forward * sign)[hits], sign[hits]


def study(data, patterns, horizons=(1, 5, 10, 20), min_hits=10):
    """One row per pattern: hits, and per horizon the mean signed return, hit rate,
    excess over the baseline and a t-statistic.

    patterns: rows of pattern_rows() (module, name, direction ...).
    """
    forwards = {horizon: forward_returns(data, horizon).to_numpy() for horizon in horizons}
    baseline = {horizon: float(np.nanmean(forward)) for horizon, forward in forwards.items()}
    rows = []
    for pattern in patterns:
        values = pattern_signals(data, pattern["module"], pattern["name"]).to_numpy()
        row = {"module": pattern["module"], "pattern": pattern["name"], "direction": pattern.get("direction", ""),
               "hits": int((values != 0).sum()), "bullish": int((values == 1).sum()), "bearish": int((values == -1).sum())}
        for horizon in horizons:
            signed, sign = _signed(values, forwards[horizon])
            if len(signed) < max(2, min_hits):
                continue
            spread = signed.std(ddof=1)
            row[f"mean {horizon}"] = float(signed.mean())
            row[f"hit rate {horizon}"] = float((signed > 0).mean())
            # The same trades taken on every bar earn baseline * sign on average.
            row[f"excess {horizon}"] = float(signed.mean() - (baseline[horizon] * sign).mean())
            row[f"t {horizon}"] = float(signed.mean() / (spread / math.sqrt(len(signed)))) if spread > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def event_path(data, module, name, horizon=20, params=None):
    """The average path after the pattern: signed return from the next open to each of
    the following `horizon` closes, with quartiles and the number of hits still in range."""
    values = pattern_signals(data, module, name, params).to_numpy()
    rows = []
    for step in range(1, horizon + 1):
        signed, _ = _signed(values, forward_returns(data, step).to_numpy())
        if len(signed) == 0:
            rows.append({"bars": step, "mean": np.nan, "median": np.nan, "p25": np.nan, "p75": np.nan, "n": 0})
            continue
        rows.append({"bars": step, "mean": float(signed.mean()), "median": float(np.median(signed)),
                     "p25": float(np.percentile(signed, 25)), "p75": float(np.percentile(signed, 75)),
                     "n": int(len(signed))})
    return pd.DataFrame(rows).set_index("bars")


def add_pattern_to_rule(spec, slot, module, name, side):
    """Add the pattern as a feature, and a condition on it to a rule slot, joined with "and".

    The feature is named after the pattern (hammer, hammer_2 ...); the condition takes
    the next free letter and joins the rule's logic line with "and".
    """
    from .expressions import append_letter, next_letter

    features = spec.setdefault("features", {})
    feature_name, number = name, 2
    while feature_name in features and features[feature_name].get("indicator") != f"pattern:{module}:{name}":
        feature_name, number = f"{name}_{number}", number + 1
    features.setdefault(feature_name, indicators.new_feature(f"pattern:{module}:{name}"))
    rules = spec.setdefault("rules", {})
    rule = rules.get(slot) or {"conditions": {}, "logic": ""}
    letter = next_letter(set(rule["conditions"]))
    rule["conditions"][letter] = {"pattern": feature_name, "side": side}
    rule["logic"] = append_letter(rule.get("logic", ""), letter)
    rules[slot] = rule
    return spec
