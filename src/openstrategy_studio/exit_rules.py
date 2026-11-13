"""Extra exits as fixed numba templates (plan phase 3): time stop, breakeven, step trailing.

vectorbt 1.1.0 has no time stop, breakeven or step trailing, but from_signals
takes two numba callbacks that may move the stop and the target on every bar:
adjust_sl_func_nb and adjust_tp_func_nb. These are a small set of fixed
templates, never generated from user graphs. The spec turns them on with:

    "risk": {"stop": {"kind": "fixed", "points": 60},
             "exits": {"time_bars": 20, "breakeven_points": 40, "step_points": 25}}

* time_bars: close the position at the open of the bar `time_bars` bars after entry.
* breakeven_points: once the trade has moved this far in favour, the stop moves to the entry price.
* step_points: each time the trade advances another step, the stop moves up by one step.

Two vectorbt facts shape the code:

1. A stop is a FRACTION of a reference price and may not be negative, so a stop
   above the entry (a long locked into profit) cannot be expressed against the
   entry. The templates therefore switch the stop to "trailing" and express
   their own stop price P against vectorbt's trailing reference (curr_price):
   fraction = 1 - P / curr_price. They remember P per column, so vectorbt's own
   trailing never loosens or drifts it.
2. A time-stop exit "at the open" is a stop placed at or beyond the open. On the
   losing side of the reference the stop can do it; on the winning side the
   target can, so a time stop also adjusts the take profit (the engine enables a
   far-away target when the strategy has none).

Favourable excursion is measured on COMPLETED bars only (entry bar to i - 1),
because the adjustment runs before bar i's stop check: using bar i's own high
would be lookahead.

Verified in phase 0 against vectorbt 1.1.0 (the phase 3 gate passes):
* The contexts are AdjustSLContext / AdjustTPContext with the 0.2x field names used
  here (vectorbt adds val_price_now and curr_i, which the templates do not need).
* The callbacks exist only in simulate_from_signal_func_nb; from_signals routes there
  whenever an adjust function is given (can_use_static is False).
* Order on each bar: adjust_sl, adjust_tp, then the stop check, then the trailing
  reference is updated from that bar's high (long) or low (short).
* init_price is the order price, the fill open before the spread, because vectorbtX's
  front door sets stop_entry_price="price" (vectorbtX 16293a2).

vectorbtX checks a stop on the bar it was entered by simulating each bar as its open plus
the rest of the bar. Since 9f4cf54 it wraps these callbacks in a bar view: they are called
once per bar, at the open, with c.i, c.init_i and c.curr_i as BAR indices and their
arguments untouched, exactly as on whole bars, so they keep indexing their per-bar arrays
by c.i. ⚠️ Keep them that way: no template may assume it is called on every simulation
row. Called on the raw split index, c.i would run to twice the bar count and write past
the end of state_* (numba does not bounds-check). docs/UPSTREAM.md, issue 14.
tests/test_exit_rules.py::test_gate_matches_the_reference_simulator pins all of it.
"""

import inspect

import numpy as np

from .spec import EXIT_KEYS

try:
    from numba import njit
except ImportError:      # numba comes with vectorbt; without it the arithmetic still runs as Python (tests)
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda function: function

# A take-profit this many points away never fills; it only exists so the time
# stop's target adjustment is called.
FAR_TARGET_POINTS = 1e9


@njit(cache=True)
def favourable_price(high, low, init_i, i, col, is_long, entry):
    """The best price reached on completed bars since entry (bars init_i to i - 1)."""
    best = entry
    for bar in range(init_i, i):
        if is_long:
            if high[bar, col] > best:
                best = high[bar, col]
        elif low[bar, col] < best:
            best = low[bar, col]
    return best


@njit(cache=True)
def protective_price(entry, best, initial_distance, point, breakeven_points, step_points, is_long):
    """The stop price the breakeven and step templates want (before 'never loosen')."""
    gain = best - entry if is_long else entry - best
    price = entry - initial_distance if is_long else entry + initial_distance
    if not np.isnan(step_points) and step_points > 0:
        steps = np.floor(gain / (step_points * point))
        if steps >= 1:
            move = steps * step_points * point
            price = price + move if is_long else price - move
    if not np.isnan(breakeven_points) and gain >= breakeven_points * point:
        price = max(price, entry) if is_long else min(price, entry)
    return price


@njit(cache=True)
def stop_fraction(price, reference, is_long):
    """vectorbt's stop for a stop price: long reference * (1 - f), short reference * (1 + f); never negative."""
    fraction = 1.0 - price / reference if is_long else price / reference - 1.0
    return max(fraction, 0.0)


@njit(cache=True)
def time_stop_due(i, init_i, time_bars):
    return (not np.isnan(time_bars)) and (i - init_i) >= time_bars


@njit(cache=True)
def adjust_sl_nb(c, open_, high, low, point, stop_points, time_bars, breakeven_points, step_points,
                 state_init_i, state_price):
    """adjust_sl_func_nb: returns (stop fraction, trailing flag) for bar c.i."""
    col = c.col
    if c.position_now == 0 or np.isnan(c.init_price):
        return c.curr_stop, c.curr_trail
    is_long = c.position_now > 0
    entry = c.init_price
    reference = c.curr_price if c.curr_trail else entry
    if state_init_i[col] != c.init_i:
        # A new position: start from the stop vectorbt placed at entry.
        state_init_i[col] = c.init_i
        state_price[col] = reference * (1.0 - c.curr_stop) if is_long else reference * (1.0 + c.curr_stop)
    if time_stop_due(c.i, c.init_i, time_bars):
        # Exit at this bar's open. A stop at the reference is at or beyond the open
        # when the open is on the losing side of it; otherwise adjust_tp_nb exits.
        price = open_[c.i, col]
        if (is_long and price <= reference) or (not is_long and price >= reference):
            return 0.0, True
    best = favourable_price(high, low, c.init_i, c.i, col, is_long, entry)
    wanted = protective_price(entry, best, stop_points[col] * point[col], point[col], breakeven_points,
                              step_points, is_long)
    # Never loosen.
    if is_long:
        wanted = max(wanted, state_price[col])
    else:
        wanted = min(wanted, state_price[col])
    state_price[col] = wanted
    return stop_fraction(wanted, reference, is_long), True


@njit(cache=True)
def adjust_tp_nb(c, open_, time_bars):
    """adjust_tp_func_nb: a time stop on the winning side, as a target at or below the open."""
    if c.position_now == 0 or np.isnan(c.init_price):
        return c.curr_stop
    if time_stop_due(c.i, c.init_i, time_bars):
        price = open_[c.i, c.col]
        entry = c.init_price
        if c.position_now > 0 and price >= entry:
            return max(price / entry - 1.0, 0.0)
        if c.position_now < 0 and price <= entry:
            return max(1.0 - price / entry, 0.0)
    return c.curr_stop


# ---------------------------------------------------------------------------
# Spec and runtime glue
# ---------------------------------------------------------------------------

def exit_settings(spec):
    """{key: float or NaN} for the three templates. `spec` has its params applied."""
    exits = ((spec.get("risk") or {}).get("exits")) or {}
    return {key: float(exits[key]) if exits.get(key) is not None else float("nan") for key in EXIT_KEYS}


def active(spec):
    return any(not np.isnan(value) for value in exit_settings(spec).values())


def _matrix(value):
    """A price Series or DataFrame as the 2-D float array vectorbt's callbacks index by [bar, column]."""
    array = np.asarray(value, dtype=np.float64)
    return np.ascontiguousarray(array.reshape(len(array), -1))


def runtime_values(spec, runtime, point_sizes):
    """The callbacks and their argument tuples for one backtest (fresh state every run).

    point_sizes: one point size per column, in column order.
    """
    settings = exit_settings(spec)
    open_, high, low = _matrix(runtime["open_"]), _matrix(runtime["high"]), _matrix(runtime["low"])
    columns = open_.shape[1]
    stop_points = float(spec["risk"]["stop"]["points"])
    out = {
        "adjust_sl_func_nb": adjust_sl_nb,
        "adjust_sl_args": (open_, high, low, np.asarray(point_sizes, dtype=np.float64),
                           np.full(columns, stop_points), settings["time_bars"], settings["breakeven_points"],
                           settings["step_points"], np.full(columns, -1, dtype=np.int64), np.full(columns, np.nan)),
    }
    if not np.isnan(settings["time_bars"]):
        out["adjust_tp_func_nb"] = adjust_tp_nb
        out["adjust_tp_args"] = (open_, settings["time_bars"])
    return out


def export_source():
    """The templates' source, for export_script (cache off: exec'd code has no file to cache against)."""
    parts = []
    for function in (favourable_price, protective_price, stop_fraction, time_stop_due, adjust_sl_nb, adjust_tp_nb):
        function = getattr(function, "py_func", function)
        parts.append(inspect.getsource(function).replace("@njit(cache=True)", "@njit(cache=False)"))
    return "\n\n".join(parts)
