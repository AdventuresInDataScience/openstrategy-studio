"""Save golden numbers from vectorbtX's examples/strategies/01_ma_cross_spreadbet_fx.py.

Run once, with network access, from the Studio's folder:

    uv run python -I tests/fixtures/golden/make_golden.py ../vectorbtx

It runs the example file itself, unmodified, on real Yahoo data, with the one change the
example's own comment offers for running without a FRED key: the static rate table
(EUR 4%, USD 5.3%) in place of FRED. It then saves the prices the example downloaded, and
its pf.stats(), pf.trades.stats() and pf.returns_stats(), so that
tests/test_engine.py::test_golden_numbers_from_the_vectorbtx_example checks the Studio's
MA-cross template against them offline (plan section 12, test 1; docs/VERIFY.md section 4).

Re-run it only on purpose: the saved prices pin the comparison, and Yahoo revises history.
"""

import importlib.metadata
import json
import math
import os
import runpy
import sys
from datetime import date
from pathlib import Path

# uv's CPython on Windows aborts the whole process when SSLKEYLOGFILE is set and an HTTPS
# context is created (CLAUDE.md); the Yahoo download creates one.
os.environ.pop("SSLKEYLOGFILE", None)

import pandas as pd            # noqa: E402
import vectorbtx as vbt        # noqa: E402

HERE = Path(__file__).parent
STATIC_RATES = {"EUR": 0.04, "USD": 0.053}          # the example's own static line


def static_rates(currencies, start=None, end=None, **_ignored):
    # A daily calendar over the whole span: the front door reindexes it onto the bars.
    index = pd.date_range(start, end, freq="D")
    return vbt.x.static_rate_table(list(currencies), index, {code: STATIC_RATES[code] for code in currencies})


def jsonable(value):
    """A float stays a float (None if not finite); anything else is kept as its text."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return number if math.isfinite(number) else None


def main(vectorbtx_dir):
    example = Path(vectorbtx_dir) / "examples" / "strategies" / "01_ma_cross_spreadbet_fx.py"
    vbt.x.FREDData.download_rates = static_rates
    namespace = runpy.run_path(str(example), run_name="golden")
    pf = namespace["pf"]
    prices = pd.DataFrame({"open": namespace["open_"], "high": namespace["high"], "low": namespace["low"],
                           "close": namespace["close"]})
    prices.index.name = "date"
    prices.to_csv(HERE / "eurusd_2019_2024_yahoo.csv")
    source = json.loads(importlib.metadata.distribution("vectorbtx").read_text("direct_url.json") or "{}")
    golden = {
        "made_on": date.today().isoformat(),
        "example": "examples/strategies/01_ma_cross_spreadbet_fx.py",
        "rates": STATIC_RATES,
        "vectorbtx": source.get("vcs_info", {}).get("commit_id") or importlib.metadata.version("vectorbtx"),
        "vectorbt": importlib.metadata.version("vectorbt"),
        "stats": {str(key): jsonable(value) for key, value in pf.stats().items()},
        "trades": {str(key): jsonable(value) for key, value in pf.trades.stats().items()},
        "returns": {str(key): jsonable(value) for key, value in pf.returns_stats().items()},
    }
    (HERE / "ma_cross_01.json").write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(prices)} bars and {sum(len(golden[k]) for k in ('stats', 'trades', 'returns'))} numbers.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../vectorbtx")
