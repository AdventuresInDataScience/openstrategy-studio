"""Results, saved runs, stats tables, figures and themes.

A run is reduced to derived data straight away. The Portfolio is never pickled:
it is large and tied to library versions. Everything a figure needs is saved
under runs/<run_id>/, so a past run redraws from files; to get a live
Portfolio again, re-run the spec on the same data.

Every figure is fig_<name>(results, theme, **options) returning a plotly
Figure, and every figure ends with finish(fig, theme). Nothing else in the
code mentions a theme.
"""

import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly_themes      # registers its themes in plotly's template registry on import

from . import engine, store
from .compile import compile_features
from .optimise import periods_per_year
from .spec import spec_hash

# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------

PLAIN_THEMES = ["plotly_white", "plotly_dark"]
SOBER_THEMES = ["financial_times", "tableau", "nord", "dracula", "nasa", "blueprint", "powerbi"]
GLOW_THEMES = ["star_wars", "cyberpunk", "synthwave"]


def theme_groups():
    """The selector's three groups, read from the registry at runtime."""
    registered = list(plotly_themes.registered_themes())
    return {"Plain": PLAIN_THEMES,
            "Sober": [name for name in SOBER_THEMES if name in registered],
            "Playful": [name for name in registered if name not in SOBER_THEMES]}


def font_css():
    """@font-face rules for the bundled fonts, injected into the Streamlit page once.

    plotly_themes.enable_fonts() is for notebooks. Static image export (kaleido)
    needs the .ttf files installed on the operating system instead.
    """
    return plotly_themes.font_face_css()


def finish(fig, theme, glow=False):
    """The one place a theme is applied. Glow suits star_wars, cyberpunk and synthwave."""
    fig.update_layout(template=theme, margin={"l": 50, "r": 20, "t": 60, "b": 40},
                      legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0})
    if glow:
        plotly_themes.add_glow(fig)
    return fig


def _diverging(theme):
    # Some themes (nike, for one) set no colourscale, so fall back to a neutral one.
    template = pio.templates[theme] if theme in pio.templates else None
    scale = template.layout.colorscale.diverging if template is not None else None
    return scale if scale else "RdBu"


# ---------------------------------------------------------------------------
# Extracting and saving a run
# ---------------------------------------------------------------------------

def _total(value):
    """A portfolio-level Series from a result that may be one column per asset."""
    if isinstance(value, pd.DataFrame):
        return value.iloc[:, 0] if value.shape[1] == 1 else value.sum(axis=1)
    return value


def _optional(fn):
    """Mode-dependent extras (margin level, holding costs ...): None when the mode has none.

    This is the one place exceptions are swallowed, because "this mode has no such
    data" is an expected answer, and the figure is then hidden.
    """
    try:
        value = fn()
    except Exception:   # noqa: BLE001
        return None
    if value is None or (hasattr(value, "empty") and value.empty):
        return None
    return value


def _jsonable(value):
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value)


def stats_tables(pf):
    """{"portfolio", "trades", "returns"} x {"all", "mt5", "vectorbtx"} as {label: value} dicts.

    ⚠️ absolute_drawdown and recovery_factor need a RESOLVED init_cash: pass
    pf.get_init_cash(), not the raw argument. Benchmark-relative statistics (beta,
    alpha, M2, Treynor) use returns_stats' own default, pf.benchmark_returns(): the
    market, held unlevered. Phase 0 checked they come out finite (VERIFY 2.7).

    ⚠️ Several symbols that each have their own cash are several accounts, and vectorbt's
    stats() on ungrouped columns AVERAGES them, with only a warning. That matched
    neither the equity curve (their sum) nor the optimiser (group_by=True). Regrouped
    without sharing cash, the tables describe the accounts taken together, as one book
    of N x init_cash; per_symbol_table shows each account on its own.
    """
    if len(pf.wrapper.columns) > 1 and not pf.wrapper.grouper.is_grouped():
        pf = pf.regroup(True)
    settings = {"init_cash": pf.get_init_cash()}
    out = {}
    for tag in ("all", "mt5", "vectorbtx"):
        keywords = {} if tag == "all" else {"tags": tag}
        tables = {"portfolio": pf.stats(settings=settings, **keywords),
                  "trades": pf.trades.stats(**keywords),
                  "returns": pf.returns_stats(**keywords)}
        for name, table in tables.items():
            # A tag that no metric carries gives None, not an empty table: vectorbt 1.1.0's
            # returns_stats has no MT5 metrics, so returns_stats(tags="mt5") is None.
            values = {}
            if table is not None:
                values = {str(label): _jsonable(value) for label, value in table.items()}
            out.setdefault(name, {})[tag] = values
    return out


def per_symbol_table(pf, accounts):
    """One row per symbol. Trades and holding costs always; each account's own results when they are separate.

    In a shared account a symbol has no equity of its own, so only what it traded is
    shown. Separate accounts are independent: each row equals that symbol run on its
    own (tests/test_engine.py checks it).
    """
    columns = list(pf.wrapper.columns)
    trades = pf.trades
    frame = pd.DataFrame(index=pd.Index(columns, name="Symbol"))
    frame["Trades"] = trades.count(group_by=False).reindex(columns).to_numpy()
    frame["Win rate [%]"] = (trades.win_rate(group_by=False) * 100).reindex(columns).to_numpy()
    frame["Trade P&L"] = trades.pnl.sum(group_by=False).reindex(columns).to_numpy()
    holding = _optional(lambda: pd.DataFrame(pf.holding_costs))
    if holding is not None:
        frame["Holding costs"] = holding.sum().reindex(columns).to_numpy()
    if accounts["model"] == "separate":
        # ⚠️ vectorbtX 9f4cf54: final_value and total_return leave holding costs out, while
        # value() and returns() include them (docs/UPSTREAM.md issue 15). These are the
        # library's own numbers, the same as the tear sheet's tables, until that is fixed.
        frame["End value"] = pf.final_value(group_by=False).reindex(columns).to_numpy()
        frame["Total return [%]"] = (pf.total_return(group_by=False) * 100).reindex(columns).to_numpy()
        # vectorbt reports a drawdown as a negative fraction; the stats table shows it positive.
        frame["Max drawdown [%]"] = (-pf.max_drawdown(group_by=False) * 100).reindex(columns).to_numpy()
        frame["Sharpe ratio"] = pf.sharpe_ratio(group_by=False).reindex(columns).to_numpy()
    return frame


def extract_results(run, benchmark=False):
    """Everything the tear sheet draws, as plain frames and dicts. `run` is engine.run_backtest's result.

    benchmark=True adds the buy-and-hold line: vectorbt's pf.benchmark_value(), the
    same instruments held unlevered and equal-weighted from the first bar, before
    costs, in the same currency. See the note above engine.run_backtest for why it
    is not a second front-door call.
    """
    pf = run["pf"]
    data = run["data"]
    spec = run["spec"]
    value = _total(pf.value()).rename("value")
    init_cash = float(np.asarray(pf.get_init_cash(), dtype=float).sum())
    returns = value.pct_change()
    returns.iloc[0] = value.iloc[0] / init_cash - 1
    frame = pd.DataFrame({"value": value, "returns": returns, "drawdown": value / value.cummax() - 1})
    if getattr(pf, "high", None) is not None and getattr(pf, "low", None) is not None:
        # vectorbtX's own curve (16293a2): shared cash counted once, each side bracketed
        # by its own extremes (a long is worst at the low, a short at the high). Before
        # that fix (docs/UPSTREAM.md issue 11) the Studio computed it here. vectorbtX
        # reports a positive fraction; the tear sheet draws drawdowns as negatives.
        # Never shallower than the close-to-close drawdown, and routinely deeper.
        frame["intrabar_drawdown"] = (-pf.x.intrabar_drawdown_series()).reindex(frame.index)
    if benchmark:
        bench_value = _total(pf.benchmark_value()).reindex(frame.index)
        frame["benchmark"] = bench_value / bench_value.iloc[0] * value.iloc[0]

    # How the run was funded, read off the portfolio itself so it describes what ran.
    # "separate" is one account per symbol, each starting with the spec's init_cash;
    # "shared" is one account across every symbol (a shared signal account, or any
    # weights portfolio); "single" is one symbol.
    symbols = list(data["close"].columns)
    if len(symbols) == 1:
        accounts = {"model": "single", "count": 1}
    elif pf.cash_sharing:
        accounts = {"model": "shared", "count": 1}
    else:
        accounts = {"model": "separate", "count": len(symbols)}
    accounts["each"] = init_cash / accounts["count"]

    results = {
        "series": frame,
        "trades": pf.trades.records_readable,
        "orders": pf.orders.records_readable,
        "stats": stats_tables(pf),
        "meta": {"name": spec["name"], "kind": run["plan"]["kind"], "freq": spec["account"]["freq"],
                 "symbols": list(spec["data"]["symbols"]), "init_cash": init_cash, "accounts": accounts,
                 "front_door": run["plan"]["front_door"], "has_benchmark": bool(benchmark)},
    }
    if len(symbols) > 1:
        results["per_symbol"] = per_symbol_table(pf, accounts)
    # Prices, long format, for the candlestick view and MAE/MFE.
    prices = []
    for symbol in data["close"].columns:
        prices.append(pd.DataFrame({"symbol": symbol, "open": data["open"][symbol], "high": data["high"][symbol],
                                    "low": data["low"][symbol], "close": data["close"][symbol]}))
    results["prices"] = pd.concat(prices).rename_axis("time").reset_index()
    # Features, long format, for overlays and pattern hits on the price chart.
    features = compile_features(spec.get("features") or {}, data)
    long = []
    for name, outputs in features.items():
        for output, values in outputs.items():
            stacked = values.stack().rename("value").reset_index()
            stacked.columns = ["time", "symbol", "value"]
            stacked["feature"] = f"{name}.{output}"
            long.append(stacked)
    if long:
        results["features"] = pd.concat(long, ignore_index=True)

    exposure = {}
    for name in ("user_size", "margin_level", "borrowed_portion"):
        found = _optional(lambda name=name: getattr(pf.x, name)())
        if found is not None:
            exposure[name] = _total(found)
    if exposure:
        results["exposure"] = pd.DataFrame(exposure)
    holding = _optional(lambda: pf.holding_costs)
    if holding is not None:
        results["holding_costs"] = _total(holding).rename("holding_costs").to_frame()
    if len(data["close"].columns) > 1:
        asset_value = _optional(lambda: pf.asset_value(group_by=False))
        if isinstance(asset_value, pd.DataFrame):
            results["asset_value"] = asset_value
        column_returns = _optional(lambda: pf.returns(group_by=False))
        if isinstance(column_returns, pd.DataFrame):
            results["column_returns"] = column_returns
    drift = _optional(lambda: pf.x.drift())
    if isinstance(drift, (pd.Series, pd.DataFrame)):
        results["drift"] = drift.to_frame("drift") if isinstance(drift, pd.Series) else drift
    if "schedule" in run["compiled"]:
        results["schedule"] = run["compiled"]["schedule"].rename("schedule").to_frame()
    return results


def save_run(data_dir, results, spec, settings, fingerprint):
    """Write results under runs/<run_id>/ and index them. Returns the run folder."""
    stamp = datetime.now(timezone.utc)
    run_id = stamp.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_dir = Path(data_dir) / "runs" / run_id
    run_dir.mkdir(parents=True)
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            # Parquet needs string column names (a drift report may be keyed by integers).
            frame = value.copy()
            frame.columns = [str(column) for column in frame.columns]
            frame.to_parquet(run_dir / f"{key}.parquet")
        else:
            (run_dir / f"{key}.json").write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    (run_dir / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    (run_dir / "versions.json").write_text(json.dumps(engine.versions(), indent=2), encoding="utf-8")
    (run_dir / "fingerprint.json").write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    portfolio = results["stats"]["portfolio"]["all"]
    summary = {key: portfolio.get(key) for key in ("Total Return [%]", "Sharpe Ratio", "Max Drawdown [%]")}
    store.add_run(data_dir, run_id, spec["name"], spec_hash(spec), stamp.isoformat(timespec="seconds"),
                  json.dumps(summary), run_dir)
    return run_dir


def load_results(run_dir):
    run_dir = Path(run_dir)
    results = {}
    for path in run_dir.iterdir():
        if path.suffix == ".parquet":
            results[path.stem] = pd.read_parquet(path)
        elif path.suffix == ".json":
            results[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return results


def stats_frame(results, table, tag="all"):
    """One stats table as a two-column frame of strings, ready to display."""
    rows = []
    for label, value in results["stats"][table][tag].items():
        if isinstance(value, float):
            text = f"{value:,.4g}" if abs(value) < 1e6 else f"{value:,.0f}"
        else:
            text = "" if value is None else str(value)
        rows.append({"Metric": label, "Value": text})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trade records
# ---------------------------------------------------------------------------

def trade_column(trades, *words):
    """A records_readable column found by its words, so names are read at runtime
    rather than hard-coded from memory ("Entry Timestamp", "PnL" ...)."""
    for column in trades.columns:
        name = str(column).lower()
        if all(word in name for word in words):
            return column
    return None


def mae_mfe(results):
    """Worst and best price excursion of each trade between entry and exit, from high and low.

    vectorbt 1.1.0 has no native MAE/MFE (confirmed in phase 0), so the Studio computes it.
    Returns a frame with mae, mfe (as fractions of the entry price) and the trade's return.
    """
    trades = results["trades"]
    prices = results["prices"].set_index("time")
    entry_t = trade_column(trades, "entry", "timestamp") or trade_column(trades, "entry", "idx")
    exit_t = trade_column(trades, "exit", "timestamp") or trade_column(trades, "exit", "idx")
    entry_p = trade_column(trades, "entry", "price")
    direction = trade_column(trades, "direction")
    column = trade_column(trades, "column")
    ret = trade_column(trades, "return")
    rows = []
    for _, trade in trades.iterrows():
        symbol = trade[column] if column else results["meta"]["symbols"][0]
        bars = prices[prices["symbol"] == symbol]
        end = trade[exit_t] if pd.notna(trade[exit_t]) else bars.index[-1]
        window = bars.loc[trade[entry_t]:end]
        if window.empty:
            continue
        entry = float(trade[entry_p])
        long = "short" not in str(trade[direction]).lower()
        low = (window["low"].min() - entry) / entry
        high = (window["high"].max() - entry) / entry
        rows.append({"mae": low if long else -high, "mfe": high if long else -low,
                     "return": float(trade[ret]), "direction": "Long" if long else "Short"})
    return pd.DataFrame(rows)


def monte_carlo(trade_returns, paths=500, seed=0):
    """Percentile bands of compounded equity from trade returns resampled with replacement."""
    returns = np.asarray(pd.Series(trade_returns).dropna(), dtype=float)
    if returns.size < 2:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.choice(returns, size=(paths, returns.size), replace=True)
    curves = np.cumprod(1 + draws, axis=1)
    bands = np.percentile(curves, [5, 25, 50, 75, 95], axis=0)
    return pd.DataFrame(bands.T, columns=["p5", "p25", "p50", "p75", "p95"])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_equity(results, theme, glow=False):
    series = results["series"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series["value"], name="Strategy", mode="lines"))
    if "benchmark" in series:
        fig.add_trace(go.Scatter(x=series.index, y=series["benchmark"], name="Buy and hold", mode="lines"))
    fig.update_layout(title="Equity", yaxis_title="Account value")
    return finish(fig, theme, glow)


def fig_drawdown(results, theme, glow=False):
    series = results["series"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series["drawdown"], name="Close to close", fill="tozeroy",
                             mode="lines"))
    if "intrabar_drawdown" in series:
        fig.add_trace(go.Scatter(x=series.index, y=series["intrabar_drawdown"], name="Intrabar", mode="lines"))
    fig.update_layout(title="Underwater drawdown", yaxis_tickformat=".0%")
    return finish(fig, theme, glow)


def fig_monthly(results, theme, glow=False):
    returns = results["series"]["returns"].fillna(0.0)
    monthly = (1 + returns).resample("MS").prod() - 1
    grid = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "value": monthly.to_numpy()})
    table = grid.pivot(index="year", columns="month", values="value").reindex(columns=range(1, 13))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(go.Heatmap(z=table.to_numpy(), x=months, y=[str(year) for year in table.index],
                               colorscale=_diverging(theme), zmid=0, texttemplate="%{z:.1%}",
                               hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>"))
    fig.update_layout(title="Monthly returns", yaxis_autorange="reversed")
    return finish(fig, theme)


def fig_rolling(results, theme, glow=False, window=63):
    """Rolling Sharpe and volatility, annualised the way vectorbt does (365-day year).

    ⚠️ Computed in pandas: swap in vectorbtX's rolling_* methods (API.md section 11)
    once their names are confirmed.
    """
    returns = results["series"]["returns"]
    factor = periods_per_year(results["meta"]["freq"])
    rolling = returns.rolling(window)
    sharpe = rolling.mean() / rolling.std() * math.sqrt(factor)
    volatility = rolling.std() * math.sqrt(factor)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=returns.index, y=sharpe, name=f"Sharpe ({window} bars)", mode="lines"))
    fig.add_trace(go.Scatter(x=returns.index, y=volatility, name="Volatility", mode="lines", yaxis="y2"))
    fig.update_layout(title="Rolling metrics", yaxis2={"overlaying": "y", "side": "right", "tickformat": ".0%"})
    return finish(fig, theme, glow)


def fig_trades(results, theme, glow=False):
    trades = results["trades"]
    pnl = trade_column(trades, "pnl")
    exit_t = trade_column(trades, "exit", "timestamp")
    direction = trade_column(trades, "direction")
    fig = go.Figure()
    for side, frame in trades.groupby(trades[direction].astype(str)):
        fig.add_trace(go.Bar(x=frame[exit_t], y=frame[pnl], name=side))
    fig.update_layout(title="Trade PnL by exit date", barmode="overlay")
    return finish(fig, theme)


def fig_distribution(results, theme, glow=False):
    returns = results["trades"][trade_column(results["trades"], "return")].dropna()
    fig = go.Figure(go.Histogram(x=returns, nbinsx=40, name="Trades"))
    fig.add_vline(x=returns.mean(), line_dash="dash", annotation_text="mean")
    fig.add_vline(x=returns.median(), line_dash="dot", annotation_text="median")
    fig.update_layout(title="Trade return distribution", xaxis_tickformat=".1%")
    return finish(fig, theme)


def fig_mae_mfe(results, theme, glow=False):
    table = mae_mfe(results)
    fig = go.Figure()
    for side, frame in table.groupby("direction"):
        fig.add_trace(go.Scatter(x=frame["mae"], y=frame["mfe"], mode="markers", name=side,
                                 marker={"color": frame["return"], "colorscale": _diverging(theme), "cmid": 0},
                                 text=[f"return {value:.2%}" for value in frame["return"]]))
    fig.update_layout(title="Adverse (MAE) and favourable (MFE) excursion", xaxis_title="MAE",
                      yaxis_title="MFE", xaxis_tickformat=".1%", yaxis_tickformat=".1%")
    return finish(fig, theme)


def fig_price(results, theme, glow=False, symbol=None, overlays=()):
    """Candles, chosen feature overlays, pattern hits and trade markers: "why did it trade here"."""
    prices = results["prices"]
    symbol = symbol or results["meta"]["symbols"][0]
    bars = prices[prices["symbol"] == symbol]
    fig = go.Figure(go.Candlestick(x=bars["time"], open=bars["open"], high=bars["high"], low=bars["low"],
                                   close=bars["close"], name=symbol))
    features = results.get("features")
    for name in overlays:
        if features is None:
            break
        line = features[(features["feature"] == name) & (features["symbol"] == symbol)]
        if name.endswith(".signal"):
            # Pattern hits: mark the bar that completed the pattern.
            hits = line[line["value"] != 0].merge(bars[["time", "low"]], on="time")
            fig.add_trace(go.Scatter(x=hits["time"], y=hits["low"] * 0.998, mode="markers", name=name,
                                     marker={"symbol": "diamond", "size": 7}))
        else:
            fig.add_trace(go.Scatter(x=line["time"], y=line["value"], mode="lines", name=name))
    trades = results["trades"]
    column = trade_column(trades, "column")
    if column is not None:
        trades = trades[trades[column].astype(str) == str(symbol)]
    entry_t, exit_t = trade_column(trades, "entry", "timestamp"), trade_column(trades, "exit", "timestamp")
    entry_p, exit_p = trade_column(trades, "entry", "price"), trade_column(trades, "exit", "price")
    direction = trade_column(trades, "direction")
    if entry_t is not None and not trades.empty:
        long = ~trades[direction].astype(str).str.lower().str.contains("short")
        fig.add_trace(go.Scatter(x=trades.loc[long, entry_t], y=trades.loc[long, entry_p], mode="markers",
                                 name="Long entry", marker={"symbol": "triangle-up", "size": 11}))
        fig.add_trace(go.Scatter(x=trades.loc[~long, entry_t], y=trades.loc[~long, entry_p], mode="markers",
                                 name="Short entry", marker={"symbol": "triangle-down", "size": 11}))
        fig.add_trace(go.Scatter(x=trades[exit_t], y=trades[exit_p], mode="markers", name="Exit",
                                 marker={"symbol": "x", "size": 9}))
    fig.update_layout(title=f"{symbol} with trades", xaxis_rangeslider_visible=False)
    return finish(fig, theme)


def fig_exposure(results, theme, glow=False):
    exposure = results["exposure"]
    fig = go.Figure()
    for column in exposure.columns:
        fig.add_trace(go.Scatter(x=exposure.index, y=exposure[column], name=column.replace("_", " "), mode="lines"))
    fig.update_layout(title="Exposure and leverage")
    return finish(fig, theme, glow)


def fig_holding_costs(results, theme, glow=False):
    costs = results["holding_costs"]["holding_costs"].fillna(0.0).cumsum()
    fig = go.Figure(go.Scatter(x=costs.index, y=costs, name="Cumulative holding costs", mode="lines"))
    fig.update_layout(title="Holding costs (applied after the simulation, so no cash feedback)")
    return finish(fig, theme, glow)


def fig_weights(results, theme, glow=False):
    values = results["asset_value"]
    total = results["series"]["value"].reindex(values.index)
    weights = values.div(total, axis=0)
    fig = go.Figure()
    for column in weights.columns:
        fig.add_trace(go.Scatter(x=weights.index, y=weights[column], name=str(column), stackgroup="one",
                                 mode="lines"))
    if "schedule" in results:
        dates = results["schedule"].index[results["schedule"]["schedule"].astype(bool)]
        fig.add_trace(go.Scatter(x=dates, y=[1.02] * len(dates), mode="markers", name="Rebalance",
                                 marker={"symbol": "line-ns-open", "size": 8}))
    fig.update_layout(title="Portfolio weights over time", yaxis_tickformat=".0%")
    return finish(fig, theme)


def fig_contribution(results, theme, glow=False):
    returns = results["column_returns"].fillna(0.0)
    compounded = (1 + returns).prod() - 1
    fig = go.Figure(go.Bar(x=[str(column) for column in compounded.index], y=compounded, name="Return"))
    fig.update_layout(title="Return per asset", yaxis_tickformat=".0%")
    correlation = returns.corr()
    heat = go.Figure(go.Heatmap(z=correlation.to_numpy(), x=list(map(str, correlation.columns)),
                                y=list(map(str, correlation.index)), zmin=-1, zmax=1, colorscale=_diverging(theme)))
    heat.update_layout(title="Correlation of asset returns")
    return finish(fig, theme), finish(heat, theme)


def fig_drift(results, theme, glow=False):
    drift = results["drift"]
    fig = go.Figure()
    for column in drift.columns:
        fig.add_trace(go.Scatter(x=drift.index, y=drift[column], name=str(column), mode="lines"))
    fig.update_layout(title="Drift: what quantisation and min_trade suppression cost")
    return finish(fig, theme, glow)


def fig_monte_carlo(results, theme, glow=False, paths=500):
    bands = monte_carlo(results["trades"][trade_column(results["trades"], "return")], paths=paths)
    fig = go.Figure()
    x = list(range(1, len(bands) + 1))
    fig.add_trace(go.Scatter(x=x, y=bands["p95"], mode="lines", line={"width": 0}, showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=bands["p5"], mode="lines", fill="tonexty", line={"width": 0}, name="5-95%"))
    fig.add_trace(go.Scatter(x=x, y=bands["p75"], mode="lines", line={"width": 0}, showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=bands["p25"], mode="lines", fill="tonexty", line={"width": 0}, name="25-75%"))
    fig.add_trace(go.Scatter(x=x, y=bands["p50"], mode="lines", name="Median"))
    fig.update_layout(title=f"Monte Carlo: {paths} reshuffles of the trade returns", xaxis_title="Trade number",
                      yaxis_title="Growth of 1")
    return finish(fig, theme)


def _has_trades(results):
    return len(results.get("trades", [])) > 0


# name: label, function, whether it is a line figure (glow applies), and when it is available.
FIGURES = {
    "equity": {"label": "Equity vs benchmark", "fn": fig_equity, "lines": True, "when": lambda r: True},
    "drawdown": {"label": "Underwater drawdown", "fn": fig_drawdown, "lines": True, "when": lambda r: True},
    "monthly": {"label": "Monthly returns", "fn": fig_monthly, "lines": False, "when": lambda r: True},
    "rolling": {"label": "Rolling metrics", "fn": fig_rolling, "lines": True, "when": lambda r: True},
    "price": {"label": "Price chart with trades", "fn": fig_price, "lines": False, "when": lambda r: True},
    "trades": {"label": "Trade PnL", "fn": fig_trades, "lines": False, "when": _has_trades},
    "distribution": {"label": "Return distribution", "fn": fig_distribution, "lines": False, "when": _has_trades},
    "mae_mfe": {"label": "MAE and MFE", "fn": fig_mae_mfe, "lines": False, "when": _has_trades},
    "monte_carlo": {"label": "Monte Carlo fan", "fn": fig_monte_carlo, "lines": False,
                    "when": lambda r: len(r.get("trades", [])) > 1},
    "exposure": {"label": "Exposure and leverage", "fn": fig_exposure, "lines": True,
                 "when": lambda r: "exposure" in r},
    "holding_costs": {"label": "Holding costs", "fn": fig_holding_costs, "lines": True,
                      "when": lambda r: "holding_costs" in r},
    "weights": {"label": "Portfolio weights", "fn": fig_weights, "lines": False,
                "when": lambda r: "asset_value" in r},
    "contribution": {"label": "Per-asset return and correlation", "fn": fig_contribution, "lines": False,
                     "when": lambda r: "column_returns" in r},
    "drift": {"label": "Drift report", "fn": fig_drift, "lines": True, "when": lambda r: "drift" in r},
}


def available_figures(results):
    return [name for name, figure in FIGURES.items() if figure["when"](results)]


def fig_preview(data, features, signals, symbol, theme, overlays=()):
    """The Build page's live preview: candles, feature overlays and signal markers (signal bars)."""
    close = data["close"][symbol]
    fig = go.Figure(go.Candlestick(x=close.index, open=data["open"][symbol], high=data["high"][symbol],
                                   low=data["low"][symbol], close=close, name=symbol))
    for name in overlays:
        feature, output = name.split(".", 1)
        values = features[feature][output][symbol]
        if output == "signal":
            hits = values[values != 0].index
            fig.add_trace(go.Scatter(x=hits, y=data["low"][symbol].loc[hits] * 0.998, mode="markers",
                                     name=name, marker={"symbol": "diamond", "size": 7}))
        else:
            fig.add_trace(go.Scatter(x=values.index, y=values, mode="lines", name=name))
    markers = {"entries": ("Long entry", "triangle-up", "low", 0.996),
               "exits": ("Long exit", "x", "high", 1.004),
               "short_entries": ("Short entry", "triangle-down", "high", 1.004),
               "short_exits": ("Short exit", "x", "low", 0.996)}
    for key, (label, symbol_name, field, factor) in markers.items():
        flags = signals[key][symbol]
        when = flags[flags].index
        if len(when):
            fig.add_trace(go.Scatter(x=when, y=data[field][symbol].loc[when] * factor, mode="markers", name=label,
                                     marker={"symbol": symbol_name, "size": 10}))
    fig.update_layout(title=f"{symbol}: signals on the bar they fire (fills happen at the next open)",
                      xaxis_rangeslider_visible=False, height=520)
    return finish(fig, theme)


def fig_weight_preview(weights, schedule, theme):
    """The Build page's preview for portfolios: target weights, with rebalance dates marked."""
    fig = go.Figure()
    for column in weights.columns:
        fig.add_trace(go.Scatter(x=weights.index, y=weights[column], name=str(column), stackgroup="one", mode="lines"))
    dates = schedule.index[schedule.astype(bool)]
    fig.add_trace(go.Scatter(x=dates, y=[1.02] * len(dates), mode="markers", name="Rebalance",
                             marker={"symbol": "line-ns-open", "size": 8}))
    fig.update_layout(title="Target weights (applied only on rebalance dates)", yaxis_tickformat=".0%", height=480)
    return finish(fig, theme)


# ---------------------------------------------------------------------------
# Optimisation results and run comparison
# ---------------------------------------------------------------------------

def param_columns(trials):
    return [column for column in trials.columns if str(column).startswith("param:")]


def fig_convergence(trials, direction, theme):
    search = trials[trials["phase"] == "search"].sort_values("trial")
    values = search["objective"].astype(float)
    best = values.cummax() if direction == "maximize" else values.cummin()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=search["trial"], y=values, mode="markers", name="Trial", marker={"size": 5}))
    fig.add_trace(go.Scatter(x=search["trial"], y=best.ffill(), mode="lines", name="Best so far"))
    fig.update_layout(title="Convergence", xaxis_title="Trial", yaxis_title="Objective")
    return finish(fig, theme)


def fig_pair_heatmap(trials, x, y, theme, bins=12):
    """Mean objective over a pair of parameters. Non-grid runs are binned first."""
    search = trials[(trials["phase"] == "search") & np.isfinite(trials["objective"].astype(float))]

    def binned(column):
        values = search[column]
        if pd.api.types.is_numeric_dtype(values) and values.nunique() > bins:
            return pd.cut(values, bins).map(lambda interval: round(interval.mid, 4)).astype(float)
        return values.astype(str) if not pd.api.types.is_numeric_dtype(values) else values

    table = search.assign(_x=binned(x), _y=binned(y)).pivot_table(index="_y", columns="_x", values="objective",
                                                                    aggfunc="mean", observed=True)
    fig = go.Figure(go.Heatmap(z=table.to_numpy(), x=[str(v) for v in table.columns], y=[str(v) for v in table.index],
                               colorscale=_diverging(theme)))
    fig.update_layout(title="Mean objective by parameter pair", xaxis_title=x.removeprefix("param:"),
                      yaxis_title=y.removeprefix("param:"))
    return finish(fig, theme)


def fig_parallel(trials, theme):
    search = trials[(trials["phase"] == "search") & np.isfinite(trials["objective"].astype(float))]
    dimensions = []
    for column in param_columns(search):
        values = search[column]
        if pd.api.types.is_numeric_dtype(values):
            dimensions.append({"label": column.removeprefix("param:"), "values": values})
        else:
            codes, labels = pd.factorize(values.astype(str))
            dimensions.append({"label": column.removeprefix("param:"), "values": codes,
                               "tickvals": list(range(len(labels))), "ticktext": list(labels)})
    dimensions.append({"label": "objective", "values": search["objective"]})
    fig = go.Figure(go.Parcoords(line={"color": search["objective"], "colorscale": _diverging(theme)},
                                 dimensions=dimensions))
    fig.update_layout(title="Parallel coordinates")
    return finish(fig, theme)


def fig_param_scatter(trials, column, theme):
    search = trials[trials["phase"] == "search"]
    fig = go.Figure(go.Scatter(x=search[column], y=search["objective"], mode="markers", marker={"size": 6}))
    fig.update_layout(title=f"Objective against {column.removeprefix('param:')}", xaxis_title=column.removeprefix("param:"),
                      yaxis_title="Objective")
    return finish(fig, theme)


def fig_importance(importances, theme):
    names = list(importances)
    fig = go.Figure(go.Bar(x=[importances[name] for name in names], y=names, orientation="h"))
    fig.update_layout(title="Parameter importance (Optuna)", yaxis_autorange="reversed")
    return finish(fig, theme)


def fig_walk_forward(returns, theme, glow=False):
    equity = (1 + returns.fillna(0.0)).cumprod()
    fig = go.Figure(go.Scatter(x=equity.index, y=equity, mode="lines", name="Out of sample"))
    fig.update_layout(title="Stitched out-of-sample equity (growth of 1)")
    return finish(fig, theme, glow)


def fig_compare(curves, theme, glow=False):
    """Equity curves of several saved runs, each rebased to 1."""
    fig = go.Figure()
    for name, value in curves:
        fig.add_trace(go.Scatter(x=value.index, y=value / value.iloc[0], mode="lines", name=name))
    fig.update_layout(title="Saved runs compared (growth of 1)")
    return finish(fig, theme, glow)


def fig_compare_backends(trials, direction, theme):
    """Median best-so-far curve per backend over its seeds, with the range of the seeds."""
    from .optimise import best_so_far

    search = trials[trials["phase"] == "search"]
    fig = go.Figure()
    for backend, runs in search.groupby("backend", sort=False):
        curves = [best_so_far(run.sort_values("run_trial")["objective"], direction).reset_index(drop=True)
                  for _, run in runs.groupby("seed")]
        table = pd.concat(curves, axis=1)
        x = list(range(1, len(table) + 1))
        fig.add_trace(go.Scatter(x=x, y=table.max(axis=1), mode="lines", line={"width": 0}, showlegend=False,
                                 legendgroup=backend))
        fig.add_trace(go.Scatter(x=x, y=table.min(axis=1), mode="lines", line={"width": 0}, fill="tonexty",
                                 opacity=0.2, showlegend=False, legendgroup=backend))
        fig.add_trace(go.Scatter(x=x, y=table.median(axis=1), mode="lines", name=f"{backend} (median of {table.shape[1]})",
                                 legendgroup=backend))
    fig.update_layout(title="Best value so far, by backend", xaxis_title="Trial", yaxis_title="Best objective")
    return finish(fig, theme)


def fig_event_path(path, label, theme, glow=False):
    """research.event_path as a chart: the mean and median path with the middle half shaded."""
    x = list(path.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=path["p75"], mode="lines", line={"width": 0}, showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=path["p25"], mode="lines", line={"width": 0}, fill="tonexty", name="Middle half"))
    fig.add_trace(go.Scatter(x=x, y=path["mean"], mode="lines+markers", name="Mean"))
    fig.add_trace(go.Scatter(x=x, y=path["median"], mode="lines", name="Median", line={"dash": "dot"}))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(title=f"After {label}: return from the next open (bearish signals as shorts)",
                      xaxis_title="Bars after the signal", yaxis_tickformat=".2%")
    return finish(fig, theme, glow)


# ---------------------------------------------------------------------------
# Pipeline diagram
# ---------------------------------------------------------------------------

def _dot(text):
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _rule_references(node, found):
    """Every feature, price, pattern and filter a rule tree reads."""
    if not isinstance(node, dict):
        return found
    for key in ("all", "any"):
        for child in node.get(key, []):
            _rule_references(child, found)
    if "not" in node:
        _rule_references(node["not"], found)
    if "pattern" in node:
        found.add(("pattern", f"{node['pattern']} ({node.get('side', 'either')})"))
    if "filter" in node:
        found.add(("filter", node["filter"].replace("_", " ")))
    for side in ("left", "right", "lower", "upper"):
        ref = node.get(side)
        if isinstance(ref, dict):
            if "feature" in ref:
                found.add(("feature", ref["feature"]))
            elif "price" in ref:
                found.add(("price", ref["price"]))
    return found


def pipeline_dot(spec):
    """The strategy as a Graphviz diagram: data, features, rules or portfolio, risk, the front door.

    Alternatives (structure search) are drawn as they are currently chosen.
    """
    from .blocks import BLOCKS
    from .engine import call_plan
    from .spec import apply_params, collect_choices, strategy_kind

    choices = collect_choices(spec)
    resolved = apply_params(spec)
    kind = strategy_kind(resolved)
    data = resolved["data"]
    lines = ["digraph pipeline {", "  rankdir=LR;", '  node [shape=box, style="rounded,filled", fillcolor="white", '
             'fontname="Helvetica", fontsize=11];', '  edge [color="#666666"];']
    lines.append(f"  data [label={_dot('Data' + chr(10) + ', '.join(data['symbols']) + chr(10) + data['timeframe'] + ' bars')}, "
                 "shape=cylinder];")
    for name, feature in (resolved.get("features") or {}).items():
        block = BLOCKS[feature["block"]]
        settings = ", ".join(f"{key}={value}" for key, value in feature.items()
                             if key not in ("block", "source", "code", "params"))
        if feature["block"] == "pattern":
            settings = f"{feature.get('name')} ({feature.get('module', 'candle')})"
        lines.append(f"  feature_{name} [label={_dot(name + chr(10) + block['label'] + (chr(10) + settings if settings else ''))}];")
        source = feature.get("source")
        upstream = f"feature_{source['feature']}" if isinstance(source, dict) else "data"
        lines.append(f"  {upstream} -> feature_{name};")
    plan = call_plan(resolved)
    door_label = "vbt.x." + plan["front_door"] + chr(10) + "fills at the next open"
    if "cash" in plan["args"]:
        door_label += chr(10) + ("one shared account" if plan["args"]["cash"][1] == "shared"
                                 else f"{len(data['symbols'])} accounts, one per symbol")
    door = f"door [label={_dot(door_label)}, shape=box3d];"
    lines.append("  " + door)
    if plan["args"].get("entry_priority") == ("var", "entry_priority"):
        score = resolved["account"]["entry_priority"]["score"]
        source = f"feature_{score['feature']}" if "feature" in score else "data"
        lines.append(f"  {source} -> door [label={_dot('entry priority')}, fontsize=9, style=dashed];")
    extras = []
    if kind == "signals":
        labels = {"long_entry": "Long entry", "long_exit": "Long exit", "short_entry": "Short entry",
                  "short_exit": "Short exit"}
        for slot, label in labels.items():
            rule = resolved["rules"].get(slot)
            if not rule:
                continue
            searched = " (alternatives searched)" if f"rules.{slot}" in choices else ""
            lines.append(f"  rule_{slot} [label={_dot(label + searched)}, fillcolor=\"#eef3ff\"];")
            for kind_name, name in sorted(_rule_references(rule, set())):
                if kind_name == "feature":
                    lines.append(f"  feature_{name} -> rule_{slot};")
                elif kind_name == "price":
                    lines.append(f"  data -> rule_{slot} [label={_dot(name)}, fontsize=9];")
                else:
                    node = f"{kind_name}_{abs(hash(name)) % 10_000_000}"
                    extras.append(f"  {node} [label={_dot(kind_name.capitalize() + chr(10) + name)}, shape=note];")
                    extras.append(f"  data -> {node};")
                    extras.append(f"  {node} -> rule_{slot};")
            lines.append(f"  rule_{slot} -> door;")
    elif kind == "portfolio":
        previous = "data"
        selection = resolved.get("selection")
        if selection:
            lines.append(f"  selection [label={_dot('Selection' + chr(10) + str(selection.get('mode', 'top')) + ' ' + str(selection.get('top_n', '')))}];")
            score = selection.get("score") or {}
            lines.append(f"  {'feature_' + score['feature'] if 'feature' in score else 'data'} -> selection;")
            previous = "selection"
        weighting = resolved["weighting"]
        lines.append(f"  weighting [label={_dot('Weighting' + chr(10) + str(weighting['scheme']))}];")
        lines.append(f"  {previous} -> weighting;")
        schedule = resolved["schedule"]
        calendar = schedule.get("calendar")
        text = "every bar" if not calendar else f"{calendar['freq']} ({calendar.get('on', 'first')})"
        if schedule.get("threshold") is not None:
            text += f" {schedule.get('combine', 'and')} drift > {schedule['threshold']}"
        lines.append(f"  schedule [label={_dot('Schedule' + chr(10) + text)}];")
        lines += ["  weighting -> schedule;", "  schedule -> door;"]
    else:
        lines.append("  data -> door [label=\"buy and hold\", fontsize=9];")
    risk = resolved.get("risk") or {}
    risk_lines = []
    sizing = risk.get("sizing") or {}
    if sizing:
        key, value = next(iter(sizing.items()))
        risk_lines.append(f"{key} {value}")
    stop = risk.get("stop") or {}
    if stop.get("kind") in ("fixed", "trailing"):
        risk_lines.append(f"{stop['kind']} stop {stop['points']} pts")
    if risk.get("target"):
        risk_lines.append(f"target {risk['target']['points']} pts")
    for key, value in (risk.get("exits") or {}).items():
        if value is not None:
            risk_lines.append(f"{key.replace('_', ' ')} {value}")
    if risk_lines:
        lines.append(f"  risk [label={_dot('Risk' + chr(10) + chr(10).join(risk_lines))}, fillcolor=\"#fff6e8\"];")
        lines.append("  risk -> door;")
    account = resolved.get("account") or {}
    account_lines = [f"cash {account.get('init_cash')}", f"freq {account.get('freq')}"]
    if account.get("base_currency"):
        account_lines.append(f"report in {account['base_currency']}")
    if account.get("rates"):
        account_lines.append(f"rates: {account['rates']['source']}")
    lines.append(f"  account [label={_dot('Account' + chr(10) + chr(10).join(account_lines))}, fillcolor=\"#f2f2f2\"];")
    lines.append("  account -> door;")
    lines += sorted(set(extras))
    lines.append("}")
    return "\n".join(lines)
