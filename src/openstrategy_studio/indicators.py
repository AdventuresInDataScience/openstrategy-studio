"""The feature catalogue: every indicator, pattern and transform the Build page offers.

Where they come from
* TA-Lib: 176 functions, every one but its 61 candlestick functions.
* pandas-ta-classic: the indicators it has that TA-Lib does not (Chandelier exit,
  Squeeze, QQE, Jurik MA, TD Sequential ...). One that duplicates a TA-Lib
  function is left out, so each indicator is offered once.
* ta_patterns: every candlestick and chart pattern. Patterns come from here only,
  never from TA-Lib's or pandas-ta's own candle functions.
* The Studio's own (STUDIO below): MA envelopes, the previous day, week or month,
  percentage offsets and bands, distance between lines, bars ago, bars since, and
  Python code.

Two layers
* The decisions are dict literals in this file: where each indicator sits in the
  menu (TALIB_MENU, PANDAS_TA_MENU), what is left out and why (EXCLUDED), the
  parameters never shown (HIDDEN_PARAMS), what each setting does (PARAM_HELP, for
  its ⓘ) and the Studio's own entries (STUDIO).
* The facts are a generated snapshot, indicator_catalogue.json: each entry's
  label, inputs, parameters with defaults (and the library's own description of a
  parameter, where its docstring gives one), and outputs. pandas-ta only reveals its
  outputs by running, which takes up to 2.5 s an indicator, so the picker reads
  the snapshot. `python -m openstrategy_studio.indicators` rewrites it from the
  installed libraries; tests/test_indicators.py regenerates it and compares, and
  runs every entry through the lookahead check.

⚠️ No lookahead. Every entry must give the same value on a bar whether or not
later bars exist. tests/test_indicators.py proves it for every entry, by
computing on the full data and on truncated copies. What failed is in EXCLUDED
(pandas-ta's centred DPO, Ichimoku's chikou span, TOS_STDEVALL's whole-sample
fit), and pandas-ta's `offset` parameter, which can shift values backwards in
time, is never shown.

Runtime: compute() calls the library once per symbol, with plain numpy arrays
(TA-Lib) or Series (pandas-ta), and rebuilds one frame per output on the close
frame's index and columns. export_source() gives the same functions' source for
the exported script, so the two paths cannot drift apart.
"""

import functools
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

CATALOGUE_FILE = Path(__file__).with_name("indicator_catalogue.json")
PRICE_FIELDS = ["open", "high", "low", "close", "volume"]

# TA-Lib's moving-average types, by the integer its matype parameters take (TA-Lib 0.8
# has 16; tests/test_indicators.py checks this list against talib.MA_Type). NONE is no
# smoothing at all; DEFAULT is the function's own choice.
TALIB_MA_TYPES = ["SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "MAMA", "T3", "HMA", "NONE", "DEFAULT",
                  "ZLEMA", "RMA", "VIDYA", "ALMA"]
# The single-type averages that TA-Lib's Moving average (talib:MA) reproduces exactly at
# that matype: the same input, the same one setting (timeperiod) and the same numbers
# (tests/test_indicators.py checks each). A feature using one can be swapped for talib:MA so
# the optimiser can pick the type (the feature card's "Let the optimiser pick the type").
# Left out: T3 and ALMA (they match only with their extra settings at the defaults, which
# the swap would silently drop), VIDYA (its own cmoperiod differs from MA's) and MAMA
# (two outputs and no timeperiod).
SAME_AS_MA = {"talib:SMA": "SMA", "talib:EMA": "EMA", "talib:WMA": "WMA", "talib:DEMA": "DEMA",
              "talib:TEMA": "TEMA", "talib:TRIMA": "TRIMA", "talib:KAMA": "KAMA", "talib:HMA": "HMA",
              "talib:ZLEMA": "ZLEMA", "talib:RMA": "RMA"}
# pandas-ta's moving averages, by the name its mamode parameters take.
# ⚠️ VERIFY on a pandas-ta-classic bump: read from pandas_ta_classic.ma's own list.
PANDAS_TA_MA_TYPES = ["sma", "ema", "dema", "tema", "wma", "hma", "rma", "fwma", "pwma", "sinwma", "swma", "t3",
                      "trima", "vidya", "linreg", "midpoint", "zlma"]

# The menu: (category, family) for every TA-Lib function outside Pattern Recognition.
# tests/test_indicators.py checks that each such function appears exactly once.
TALIB_MENU = {
    ("Indicator", "Moving averages"): [
        "SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "MAMA", "T3", "MA", "HMA", "ALMA", "FRAMA", "VIDYA",
        "VWMA", "ZLEMA", "RMA", "MCGD", "HT_TRENDLINE", "MAVP", "MIDPOINT", "MIDPRICE", "SWAK_BUTTER", "SWAK_GAUSS"],
    ("Indicator", "Bands and channels"): ["BBANDS", "KC", "DONCHIAN", "ACCBANDS", "PERCENTB", "BBW"],
    ("Indicator", "Trend and stops"): ["SAR", "SAREXT", "SUPERTREND", "CKSP", "ICHIMOKU"],
    ("Indicator", "Trend strength"): [
        "ADX", "ADXR", "DX", "PLUS_DI", "MINUS_DI", "PLUS_DM", "MINUS_DM", "AROON", "AROONOSC", "VHF", "CHOP",
        "CHOPTR", "VORTEX", "ER"],
    ("Indicator", "Oscillators"): [
        "RSI", "STOCH", "STOCHF", "STOCHRSI", "WILLR", "CCI", "CMO", "CMOU", "ULTOSC", "CRSI", "CTI", "FISHER", "IMI",
        "KDJ", "PSO", "SMI", "STC", "WAVETREND"],
    ("Indicator", "Momentum"): [
        "MOM", "ROC", "ROCP", "ROCR", "ROCR100", "MACD", "MACDEXT", "MACDFIX", "PPO", "APO", "TRIX", "TSI", "KST",
        "KSTEXT", "COPPOCK", "AO", "AC", "BOP", "QSTICK", "SQZMOM", "DOSC", "DPO", "ERI", "ASI", "SI", "IBS", "CG",
        "FOSC"],
    ("Indicator", "Volatility"): ["ATR", "NATR", "TRANGE", "ADR", "CVI", "MASSI", "RVI", "RVIR", "ROGERSSATCHELL"],
    ("Indicator", "Volume"): [
        "AD", "ADOSC", "CMF", "EFI", "EMV", "KVO", "MARKETFI", "NVI", "OBV", "PVI", "PVO", "PVT", "RVOL", "VWAP",
        "MFI", "WAD"],
    ("Indicator", "Cycles"): [
        "HT_DCPERIOD", "HT_DCPHASE", "HT_PHASOR", "HT_SINE", "HT_TRENDMODE", "SWAK_2PHP", "SWAK_BP", "SWAK_HP"],
    ("Statistical", "Regression"): ["LINEARREG", "LINEARREG_ANGLE", "LINEARREG_INTERCEPT", "LINEARREG_SLOPE", "TSF"],
    ("Statistical", "Spread and distribution"): [
        "STDDEV", "VAR", "AVGDEV", "KURTOSIS", "MEDIAN", "PERCENTILE", "PERCENTRANK"],
    ("Statistical", "Correlation"): ["BETA", "CORREL"],
    ("Price levels", "Highs and lows over n bars"): ["MAX", "MIN", "MINMAX", "MAXINDEX", "MININDEX", "MINMAXINDEX"],
    ("Price levels", "Swing points"): ["FRACTAL", "ZIGZAG"],
    ("Price levels", "Price types"): ["AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE", "HA"],
    ("Transform", "Arithmetic"): ["ADD", "SUB", "MULT", "DIV", "SUM", "CUMSUM"],
    ("Transform", "Functions"): [
        "LN", "LOG10", "EXP", "SQRT", "CEIL", "FLOOR", "SIN", "COS", "TAN", "ASIN", "ACOS", "ATAN", "SINH", "COSH",
        "TANH"],
}

# The same for pandas-ta's indicators that TA-Lib does not have (vectorbt's names, upper case).
PANDAS_TA_MENU = {
    ("Indicator", "Moving averages"): [
        "FWMA", "HWMA", "JMA", "PWMA", "SINWMA", "SSF", "SWMA", "MMAR", "RAINBOW"],
    ("Indicator", "Bands and channels"): ["ABERRATION", "HWC"],
    ("Indicator", "Trend and stops"): ["CE", "PMAX", "HILO", "AMAT"],
    ("Indicator", "Oscillators"): ["LRSI", "RSX", "QQE", "PSL", "INERTIA", "RVGI"],
    ("Indicator", "Momentum"): ["BIAS", "BRAR", "CFO", "PGO", "PO", "TRIXH", "SQUEEZE", "SQUEEZE_PRO"],
    ("Indicator", "Volatility"): ["HVOL", "AVOLUME", "UI", "THERMO", "PDIST"],
    ("Indicator", "Volume"): ["AOBV", "PVOL", "PVR", "VFI", "VOSC", "VWMACD"],
    ("Indicator", "Cycles"): ["DSP", "EBSW", "MSW"],
    ("Statistical", "Regression"): ["SLOPE", "STDERR"],
    ("Statistical", "Spread and distribution"): ["MAD", "MD", "QUANTILE", "SKEW", "ENTROPY", "ZSCORE", "CDL_Z"],
    ("Statistical", "Returns"): ["LOG_RETURN", "PERCENT_RETURN", "DRAWDOWN"],
    ("Price levels", "Swing points"): ["SMC_SWEEP"],
    ("Price levels", "Pivots and previous periods"): ["CPR"],
    ("Price levels", "Runs and counts"): ["INCREASING", "DECREASING", "LONG_RUN", "SHORT_RUN", "TD_SEQ"],
    ("Transform", "Arithmetic"): ["DECAY", "EDECAY"],
    ("Transform", "Functions"): ["NPABS", "NPROUND", "TRUNC", "TODEG", "TORAD"],
}

# Left out, and why. A pandas-ta indicator with the same name as a TA-Lib function is
# left out without being listed: TA-Lib's is offered instead.
EXCLUDED = {
    "pandas_ta:CDL_DOJI": "candlestick patterns come from ta_patterns",
    "pandas_ta:CDL_INSIDE": "candlestick patterns come from ta_patterns",
    "pandas_ta:CDL_PATTERN": "candlestick patterns come from ta_patterns",
    "pandas_ta:TOS_STDEVALL": "⚠️ looks ahead: it fits one regression over the whole history",
    "pandas_ta:TSIGNALS": "turns a trend series into trades, which is what the rules do",
    "pandas_ta:TTM_TREND": "does not run through vectorbt's pandas-ta wrapper (a 3-D output)",
    "pandas_ta:DM": "the same as TA-Lib's PLUS_DM and MINUS_DM",
    "pandas_ta:EOM": "the same as TA-Lib's EMV",
    "pandas_ta:HL2": "the same as TA-Lib's MEDPRICE",
    "pandas_ta:HLC3": "the same as TA-Lib's TYPPRICE",
    "pandas_ta:OHLC4": "the same as TA-Lib's AVGPRICE",
    "pandas_ta:WCP": "the same as TA-Lib's WCLPRICE",
    "pandas_ta:LINREG": "the same as TA-Lib's LINEARREG",
    "pandas_ta:LINREGANGLE": "the same as TA-Lib's LINEARREG_ANGLE",
    "pandas_ta:LINREGINTERCEPT": "the same as TA-Lib's LINEARREG_INTERCEPT",
    "pandas_ta:LINREGSLOPE": "the same as TA-Lib's LINEARREG_SLOPE",
    "pandas_ta:STDEV": "the same as TA-Lib's STDDEV",
    "pandas_ta:VARIANCE": "the same as TA-Lib's VAR",
    "pandas_ta:ROLLING_MAX": "the same as TA-Lib's MAX",
    "pandas_ta:ROLLING_MIN": "the same as TA-Lib's MIN",
    "pandas_ta:ROLLING_SUM": "the same as TA-Lib's SUM",
    "pandas_ta:TRUE_RANGE": "the same as TA-Lib's TRANGE",
    "pandas_ta:UO": "the same as TA-Lib's ULTOSC",
    "pandas_ta:PSAR": "the same as TA-Lib's SAR",
    "pandas_ta:ZLMA": "the same as TA-Lib's ZLEMA",
    # The flag and pennant detectors were left out while ta_patterns 1.2.1's _flagpole read c[-1],
    # the last close, on each detector's first bar (docs/UPSTREAM.md, P1). Fixed in 1.2.2, which
    # uv.lock pins, so they are back; test_no_entry_looks_ahead checks them with the rest.
}

# Parameters never shown, so never set. ⚠️ pandas-ta's `offset` shifts the output by
# that many bars, and a negative one moves the future into the present. `talib` makes
# pandas-ta hand the work to TA-Lib; the default is kept.
HIDDEN_PARAMS = {"offset", "talib"}

# What a setting does, for the ⓘ beside it on the Build page (param_help below), by
# library and parameter name. A name missing here falls back to the description the
# library's own docstring gives (captured in the snapshot as the declaration's "help"),
# and every tooltip also says which function takes the value and its default. The
# Studio's own entries carry their help in STUDIO. Plain words; no claim here that the
# library's code does not bear out.
PARAM_HELP = {
    "talib": {
        "timeperiod": "How many bars it looks back over.",
        "fastperiod": "Bars in the fast (shorter) average.",
        "slowperiod": "Bars in the slow (longer) average.",
        "signalperiod": "Bars in the signal line's average.",
        "matype": "The kind of moving average: SMA simple, EMA exponential, WMA weighted, and so on. NONE applies "
                  "no smoothing.",
        "fastmatype": "The kind of moving average for the fast line.",
        "slowmatype": "The kind of moving average for the slow line.",
        "signalmatype": "The kind of moving average for the signal line.",
        "slowk_matype": "The kind of moving average that smooths %K.",
        "slowd_matype": "The kind of moving average that smooths %D.",
        "fastd_matype": "The kind of moving average that smooths fast %D.",
        "rocmatype": "The kind of moving average that smooths each rate of change.",
        "nbdevup": "How many standard deviations above the middle line the upper band sits.",
        "nbdevdn": "How many standard deviations below the middle line the lower band sits.",
        "fastk_period": "Bars for %K: where the close sits in the high-low range of this many bars.",
        "slowk_period": "Bars that smooth %K into slow %K.",
        "slowd_period": "Bars that smooth slow %K into %D.",
        "fastd_period": "Bars that smooth %K into fast %D.",
        "acceleration": "The step the acceleration factor grows by at each new extreme: higher pulls the stop "
                        "towards price faster.",
        "maximum": "The largest the acceleration factor can grow to.",
        "startvalue": "The starting value and direction: 0 finds them itself, above 0 starts long, below 0 starts "
                      "short.",
        "offsetonreverse": "A percentage added to the stop when it flips direction.",
        "accelerationinitlong": "The acceleration factor's starting value for longs.",
        "accelerationlong": "The step the acceleration factor grows by for longs.",
        "accelerationmaxlong": "The largest acceleration factor for longs.",
        "accelerationinitshort": "The acceleration factor's starting value for shorts.",
        "accelerationshort": "The step the acceleration factor grows by for shorts.",
        "accelerationmaxshort": "The largest acceleration factor for shorts.",
        "fastlimit": "The upper limit on MAMA's adaptive smoothing factor.",
        "slowlimit": "The lower limit on MAMA's adaptive smoothing factor.",
        "vfactor": "The volume factor, 0 to 1: 0 behaves like an EMA, 1 like a DEMA. 0.7 is usual.",
        "timeperiod1": "Bars in the shortest of the three periods.",
        "timeperiod2": "Bars in the middle period.",
        "timeperiod3": "Bars in the longest period.",
        "minperiod": "The shortest period the variable average may use.",
        "maxperiod": "The longest period the variable average may use.",
        "sigma": "How spread out the Gaussian weights are: larger is smoother.",
        "offset": "Where in the window the weights peak, 0 to 1: higher follows price more closely, lower is "
                  "smoother.",
        "tenkanperiod": "Bars for the conversion line (tenkan-sen).",
        "kijunperiod": "Bars for the base line (kijun-sen).",
        "senkoubperiod": "Bars for leading span B (senkou span B).",
        "atrperiod": "Bars in the Average True Range it uses.",
    },
    "pandas_ta": {
        "length": "How many bars it looks back over.",
        "mamode": "The kind of moving average it uses: sma simple, ema exponential, wma weighted, and so on.",
        "fast": "Bars in the fast (shorter) average.",
        "slow": "Bars in the slow (longer) average.",
        "signal": "Bars in the signal line's average.",
        "drift": "The difference period: each bar is compared with the one this many bars before.",
        "scalar": "A multiplier on the result (100 turns a fraction into a percentage).",
        # Two whose docstrings repeat another parameter's words (pandas-ta-classic 0.9).
        "atr_length": "Bars in the Average True Range it uses.",
        "std": "A multiplier on the standard deviation it divides by.",
    },
    "pattern": {
        "mode": "confirmed fires when the pattern completes, such as a close through its neckline. forming fires "
                "earlier, as the shape falls into place and before the break: riskier as a trigger.",
        "window": "How many bars back it looks for the pattern's shape.",
        "pivot_n": "Bars on each side that confirm a swing high or low. Smaller finds more, smaller swings; larger "
                   "only broad ones. A swing is only known this many bars after it, so the signal comes no "
                   "earlier.",
        "pivot_pct": "The smallest move for a swing to count, as a fraction (0.02 = 2%). Blank: any swing counts.",
        "tol": "How close counts as equal, as a fraction: of the bar's range for a candlestick (a shadow under "
               "5% of it counts as none), of price for a chart pattern (two lows within 3% are level).",
        "doji_thr": "A bar is a doji when its body is smaller than this share of its range (0.1 = 10%).",
        "body_thr": "The body's size limit, as a share of the bar's range.",
        "shadow_thr": "The shadow's size limit, as a share of the bar's range.",
        "shadow_factor": "How many times the body the long shadow must be (2 = twice).",
        "require_trend": "Fire only after the trend the textbook pattern needs. Off: the shape alone.",
        "trend_n": "Bars used to judge that prior trend.",
        "penetration": "How far into the first bar's body the second must close, as a fraction (0.5 = past its "
                       "midpoint).",
        "min_separation": "The fewest bars between the pattern's peaks or troughs.",
        "fib_tol": "How far each leg may stray from its Fibonacci ratio, as a fraction (0.05 = 5%).",
        "reversal_bars": "Bars within which the failed breakout must reverse.",
        "reversal_pct": "How far the failed breakout must reverse, as a fraction (0.05 = 5%).",
        "flat_tol": "How level the flat side must be, as a fraction of price (0.02 = 2%).",
    },
}
SOURCE_NAMES = {"talib": "TA-Lib", "pandas_ta": "pandas-ta", "pattern": "ta_patterns"}

# Labels and descriptions written for the Studio, where the library's own say too little: TA-Lib
# gives a display name and its group ("Price Transform", "Math Operators") and nothing else.
# Each label carries its formula or meaning in brackets, so the picker explains itself.
# Applied when the snapshot is built (build_catalogue); checked against the libraries' numbers
# in tests/test_indicators.py.
ENTRY_TEXT = {
    "talib:AVGPRICE": ("Average price ((open + high + low + close) / 4)",
                       "The mean of the bar's four prices: a smoother stand-in for the close, often fed to other "
                       "indicators."),
    "talib:MEDPRICE": ("Median price ((high + low) / 2)",
                       "The middle of the bar's range, halfway between its high and low. Not a median over many "
                       "bars: that is Median, under Statistical."),
    "talib:TYPPRICE": ("Typical price ((high + low + close) / 3)",
                       "The average of the high, the low and the close: the price CCI and MFI are built on, and the "
                       "pivot of a floor-trader pivot."),
    "talib:WCLPRICE": ("Weighted close ((high + low + 2 x close) / 4)",
                       "The typical price with the close counted twice, so it leans towards where the bar closed."),
    "talib:HA": ("Heikin-Ashi candles (smoothed open, high, low, close)",
                 "Heikin-Ashi bars: close = (open + high + low + close) / 4; open = halfway between the previous "
                 "Heikin-Ashi open and close; high and low also take in those two. They smooth a trend into runs of "
                 "one colour. ⚠️ They are not prices you could trade at: compare them with each other, not with the "
                 "real prices."),
    "talib:MAX": ("Highest over n bars (this bar included)",
                  "The highest value of its input over the last timeperiod bars, the current one included. Apply it "
                  "to high for the highest high. For a breakout above the bars before, compare with it 1 bar ago, or "
                  "use Highest high and lowest low over n bars, which can leave the current bar out."),
    "talib:MIN": ("Lowest over n bars (this bar included)",
                  "The lowest value of its input over the last timeperiod bars, the current one included. Apply it "
                  "to low for the lowest low. For a break below the bars before, compare with it 1 bar ago, or use "
                  "Highest high and lowest low over n bars, which can leave the current bar out."),
    "talib:MINMAX": ("Lowest and highest over n bars (this bar included)",
                     "Both of Lowest over n bars and Highest over n bars, from one input: min and max."),
    "talib:MAXINDEX": ("Bar number of the highest over n bars",
                       "The position, counted from the first bar of the data, of the highest value in the last "
                       "timeperiod bars. To ask how many bars ago the high was, use Highest high and lowest low over "
                       "n bars (its bars_since_high output)."),
    "talib:MININDEX": ("Bar number of the lowest over n bars",
                       "The position, counted from the first bar of the data, of the lowest value in the last "
                       "timeperiod bars. To ask how many bars ago the low was, use Highest high and lowest low over "
                       "n bars (its bars_since_low output)."),
    "talib:MINMAXINDEX": ("Bar numbers of the lowest and highest over n bars",
                          "Both of the bar-number entries above, from one input."),
    "talib:FRACTAL": ("Williams fractal (a swing high or low, marked once confirmed)",
                      "100 on the bar that confirms a swing: a high above the leftbars bars before it and the "
                      "rightbars bars after it (a low below them, for swinglow). The mark comes rightbars bars after "
                      "the swing itself, when it is known; 0 elsewhere."),
    "talib:ZIGZAG": ("Zig zag (the latest swing of at least a set size)",
                     "Follows swings at least as large as its sensitivity setting. zigzag is the price of the latest "
                     "swing high or low, which moves while that swing is still extending; trend is its direction (1 "
                     "up, -1 down); pivotidx is its bar number, counted from the first bar of the data."),
    "talib:MEDIAN": ("Median over n bars",
                     "The middle value of its input over the last timeperiod bars: half were above it, half below. "
                     "Less pulled about by one wild bar than a moving average."),
}

# The candlestick and chart pattern families: (module path in ta_patterns, family label).
PATTERN_FAMILIES = {
    "candle": [("single", "Single bar"), ("two_bar", "Two bars"), ("three_bar", "Three bars"),
               ("multi_bar", "Four or more bars")],
    "chart": [("classic", "Classic"), ("double_multi", "Double and multiple tops and bottoms"),
              ("harmonic", "Harmonic"), ("busted", "Busted patterns"), ("short", "Short-term"),
              ("volume", "Volume patterns")],
}

# The Studio's own entries. Their compute functions are below; export_source() copies them.
STUDIO = {
    "studio:envelope": {
        "label": "Moving average envelope", "menu": ["Indicator", "Bands and channels"],
        "inputs": ["close"],
        "params": {"timeperiod": {"type": "int", "default": 20, "help": "Bars in the moving average."},
                   "matype": {"type": "choice", "default": "SMA", "options": TALIB_MA_TYPES,
                              "help": "The kind of moving average: SMA simple, EMA exponential, WMA weighted, and "
                                      "so on."},
                   "percent": {"type": "float", "default": 2.5,
                               "help": "How far above and below the average the bands sit, in percent (2.5 = "
                                       "2.5%)."}},
        "outputs": ["upper", "middle", "lower"],
        "description": "A moving average with bands a fixed percentage above and below it.",
    },
    "studio:previous_period": {
        "label": "Previous day, week or month", "menu": ["Price levels", "Pivots and previous periods"],
        "inputs": [],
        "params": {"period": {"type": "choice", "default": "day", "options": ["day", "week", "month"],
                              "help": "Which period to show: the last completed day, week or calendar month."}},
        "outputs": ["open", "high", "low", "close"],
        "description": "The last completed day's (week's, month's) open, high, low and close, held until the next "
                       "one completes. It never shows a period that is still forming.",
    },
    "studio:highest_lowest": {
        "label": "Highest high and lowest low over n bars", "menu": ["Price levels", "Highs and lows over n bars"],
        "inputs": ["high", "low"], "defaults": {"high": "high", "low": "low"},
        "params": {"bars": {"type": "int", "default": 20, "help": "How many bars the highest and lowest are taken "
                                                                   "over."},
                   "include_current": {"type": "bool", "default": False,
                                       "help": "Off: the n bars before this one, the level a breakout must clear "
                                               "(close above upper is a 20-bar breakout). On: this bar and the n - 1 "
                                               "before it, as a Donchian channel draws it, which the bar's own high "
                                               "can never be above."}},
        "outputs": ["upper", "lower", "middle", "bars_since_high", "bars_since_low"],
        "description": "The highest high (upper) and lowest low (lower) of n bars, the midpoint between them, and how "
                       "many bars ago each was set (0 is this bar). Apply it to close for the highest and lowest "
                       "close.",
    },
    "studio:pivots": {
        "label": "Pivot points (from the previous day, week or month)",
        "menu": ["Price levels", "Pivots and previous periods"],
        "inputs": [],
        "params": {"period": {"type": "choice", "default": "day", "options": ["day", "week", "month"],
                              "help": "Which completed period the levels are worked out from: yesterday, last week "
                                      "or last month. They hold until the next one completes."},
                   "method": {"type": "choice", "default": "classic",
                              "options": ["classic", "fibonacci", "camarilla", "woodie"],
                              "help": "classic: pivot (high + low + close) / 3, R1 = 2 x pivot - low, S1 = 2 x pivot "
                                      "- high, R2 and S2 one range away, R3 and S3 further. fibonacci: the pivot "
                                      "plus or minus 0.382, 0.618 and 1 times the range. camarilla: the close plus "
                                      "or minus 1.1 / 12, 1.1 / 6 and 1.1 / 4 times the range. woodie: classic with "
                                      "the pivot (high + low + 2 x close) / 4."}},
        "outputs": ["pivot", "r1", "r2", "r3", "s1", "s2", "s3"],
        "description": "Floor-trader support and resistance from the last completed period's high, low and close: "
                       "the pivot, three resistance levels above it (r1 to r3) and three supports below (s1 to s3). A "
                       "period still forming is never used.",
    },
    "studio:period_to_date": {
        "label": "This day, week or month so far (open, high, low)",
        "menu": ["Price levels", "Pivots and previous periods"],
        "inputs": [],
        "params": {"period": {"type": "choice", "default": "day", "options": ["day", "week", "month", "year"],
                              "help": "The period that is still forming: today, this week, this month or this year "
                                      "(calendar, by the bars' dates)."}},
        "outputs": ["open", "high", "low"],
        "description": "The current period's open, and its highest high and lowest low so far, this bar included. "
                       "On intraday bars, today so far; on daily bars, this week or month so far. With Previous day, "
                       "week or month it gives breakouts of yesterday's range.",
    },
    "studio:offset": {
        "label": "Offset a line by a percentage", "menu": ["Transform", "Offsets and bands"],
        "inputs": ["source"],
        "params": {"percent": {"type": "float", "default": 5.0,
                               "help": "How far to move the line, in percent: 5 is 5% above it, -5 is 5% below."}},
        "outputs": ["value"],
        "description": "The line times (1 + percent / 100). Use a negative percent to go below it.",
    },
    "studio:percent_band": {
        "label": "Band a percentage around a line", "menu": ["Transform", "Offsets and bands"],
        "inputs": ["source"],
        "params": {"percent": {"type": "float", "default": 5.0,
                               "help": "How far above and below the line the two lines sit, in percent (5 = 5%)."}},
        "outputs": ["upper", "lower"],
        "description": "Two lines, percent above and percent below the input. 'Within 5% of the EMA' is a price "
                       "between the lower and upper lines of a 5% band around the EMA.",
    },
    "studio:distance": {
        "label": "Distance between two lines (%)", "menu": ["Transform", "Offsets and bands"],
        "inputs": ["a", "b"],
        "params": {},
        "outputs": ["value"],
        "description": "How far line a is above line b, as a percentage of b: (a / b - 1) x 100.",
    },
    "studio:bars_ago": {
        "label": "A line some bars ago", "menu": ["Transform", "Offsets and bands"],
        "inputs": ["source"],
        "params": {"bars": {"type": "int", "default": 1,
                            "help": "How many bars back to read the input. 1 is the bar before this one."}},
        "outputs": ["value"],
        "description": "The input's value that many bars back. History only: it never moves a value forward.",
    },
    "studio:bars_since": {
        "label": "Bars since a signal", "menu": ["Transform", "Offsets and bands"],
        "inputs": ["source"],
        "params": {},
        "outputs": ["value"],
        "description": "How many bars ago the input was last non-zero, such as a pattern firing. Blank before the "
                       "first time.",
    },
    "studio:python": {
        "label": "Python code", "menu": ["Custom", "Python"],
        "inputs": [],
        "params": {"code": {"type": "code", "default": "return close.rolling(10).mean()",
                            "help": "The body of a function given open, high, low, close and volume (frames with "
                                    "one column per symbol), np and pd. Return a frame shaped like close. ⚠️ Use "
                                    "only past bars: a shift(-1) or a centred window looks ahead."}},
        "outputs": ["value"],
        "description": "Trusted local code. The function body receives open, high, low, close, volume, np and pd, "
                       "and returns a frame shaped like close.",
    },
}


# ---------------------------------------------------------------------------
# Reading the catalogue
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def catalogue():
    """{indicator id: entry} from the snapshot. An entry is
    {"id", "source", "function", "label", "menu": [category, family], "inputs": [names],
     "defaults": {input: price name or None}, "params": {name: {"type", "default", "options"?}},
     "outputs": [names], "description", "direction" (patterns)}."""
    entries = json.loads(CATALOGUE_FILE.read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in entries}


def entry(indicator_id):
    return catalogue()[indicator_id]


def menu():
    """{category: {family: [indicator ids, by label]}}, categories and families in menu order."""
    order = ["Indicator", "Statistical", "Price levels", "Candlestick", "Chart pattern", "Transform", "Custom"]
    out = {category: {} for category in order}
    for item in catalogue().values():
        category, family = item["menu"]
        out.setdefault(category, {}).setdefault(family, []).append(item["id"])
    for families in out.values():
        for family, ids in families.items():
            ids.sort(key=lambda indicator_id: catalogue()[indicator_id]["label"].lower())
    return {category: families for category, families in out.items() if families}


def outputs(indicator_id):
    return list(entry(indicator_id)["outputs"])


def new_feature(indicator_id):
    """A new feature dict for the spec: the indicator, its default inputs and its default parameters."""
    item = entry(indicator_id)
    feature = {"indicator": indicator_id, "inputs": {}, "params": {}}
    for name in item["inputs"]:
        feature["inputs"][name] = item["defaults"].get(name) or "close"
    for name, declaration in item["params"].items():
        if declaration.get("default") is not None:
            feature["params"][name] = declaration["default"]
    return feature


def param_help(indicator_id, name):
    """The ⓘ text for one of an entry's settings: what it does, which function takes it, its default.

    What it does comes from PARAM_HELP, else the library's own words (the declaration's
    "help", from its docstring or STUDIO). A setting neither describes still says where
    the value goes, so the library's documentation can be looked up.
    """
    item = entry(indicator_id)
    declaration = item["params"][name]
    parts = [PARAM_HELP.get(item["source"], {}).get(name) or declaration.get("help") or ""]
    if item["source"] in SOURCE_NAMES:
        function = item["function"].split(":")[-1]
        parts.append(f"{function} ({SOURCE_NAMES[item['source']]}) takes it as `{name}`.")
    default = declaration.get("default")
    if declaration["type"] == "code":
        pass
    elif default is None:
        parts.append("Blank uses the library's own default.")
    else:
        parts.append(f"Default: {default}.")
    return " ".join(part for part in parts if part)


def describe(feature):
    """A short text form, for summaries and the pipeline diagram: 'EMA(close, 50)'."""
    item = catalogue().get(feature.get("indicator"))
    if item is None:
        return str(feature.get("indicator"))
    parts = []
    for name in item["inputs"]:
        value = (feature.get("inputs") or {}).get(name, item["defaults"].get(name))
        if isinstance(value, dict):
            value = value["feature"] + (f".{value['output']}" if value.get("output") else "")
        if value is not None:
            parts.append(str(value))
    for name, value in (feature.get("params") or {}).items():
        if name != "code":
            parts.append(f"{value:g}" if isinstance(value, float) else str(value))
    short = item["function"] if item["source"] in ("talib", "pandas_ta") else item["label"]
    return f"{short}({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Computing one feature
# ---------------------------------------------------------------------------
# The library calls are plain functions with their imports inside, so export_source()
# can copy them into the exported script word for word.

def run_library_indicator(source, function, input_names, inputs, params, output_names):
    """A TA-Lib or pandas-ta indicator on every column: {output name: DataFrame}.

    `inputs` are frames shaped like close, in the function's input order. The call is
    made once per column with that column's values, because both libraries take one
    series at a time.
    """
    like = inputs[0]
    columns = {name: {} for name in output_names}
    for column in like.columns:
        arrays = [np.ascontiguousarray(frame[column].to_numpy(dtype=float)) for frame in inputs]
        if source == "talib":
            import talib
            result = getattr(talib, function)(*arrays, **params)
            values = list(result) if isinstance(result, (tuple, list)) else [result]
        else:
            import pandas_ta_classic
            series = {name: pd.Series(array, index=like.index) for name, array in zip(input_names, arrays)}
            result = getattr(pandas_ta_classic, function.lower())(**series, **params)
            if isinstance(result, pd.Series):
                values = [result]
            else:
                # ⚠️ Outputs are matched by position with the catalogue's names: pandas-ta
                # names its columns after the parameters (CEl_22_3.0). Non-numeric
                # columns were left out of the catalogue, and are dropped here too.
                numeric = [result[name] for name in result.columns if pd.api.types.is_numeric_dtype(result[name])]
                values = numeric
        if len(values) != len(output_names):
            raise ValueError(f"{function} gave {len(values)} outputs where the catalogue lists {len(output_names)}; "
                             "the library has changed. Run `python -m openstrategy_studio.indicators`.")
        for name, value in zip(output_names, values):
            columns[name][column] = np.asarray(value, dtype=float)
    return {name: pd.DataFrame(values, index=like.index, columns=like.columns) for name, values in columns.items()}


def run_pattern(module, name, params, open_, high, low, close, volume):
    """A ta_patterns detector on every column: an int8 frame, +1 bullish, -1 bearish, 0 none.

    Signals sit on the bar that COMPLETES the pattern and use only past bars, so the
    front door's next-open fill is correct execution.
    """
    import inspect as _inspect
    import ta_patterns
    import ta_patterns.chart_patterns
    detector = getattr(ta_patterns.chart_patterns if module == "chart" else ta_patterns, name)
    # Each detector names the prices it reads (black_candle takes only o and c), and five
    # volume patterns run only when v is given: pass exactly the ones it names.
    wanted = [key for key in _inspect.signature(detector).parameters if key in ("o", "h", "l", "c", "v")]
    frames = {"o": open_, "h": high, "l": low, "c": close, "v": volume}
    out = pd.DataFrame(0, index=close.index, columns=close.columns, dtype="int8")
    for symbol in close.columns:
        prices = {key: frames[key][symbol].to_numpy(dtype=float) for key in wanted}
        out[symbol] = np.asarray(detector(**prices, **params), dtype="int8")
    return out


def run_studio(function, params, inputs, data):
    """One of the Studio's own entries: {output name: DataFrame}."""
    close = data["close"]
    if function == "envelope":
        import talib
        ma_type = TALIB_MA_TYPES.index(params["matype"])
        source = inputs[0]
        middle = pd.DataFrame({column: talib.MA(np.ascontiguousarray(source[column].to_numpy(dtype=float)),
                                                timeperiod=int(params["timeperiod"]), matype=ma_type)
                               for column in source.columns}, index=source.index)
        share = float(params["percent"]) / 100.0
        return {"upper": middle * (1 + share), "middle": middle, "lower": middle * (1 - share)}
    if function == "previous_period":
        # ⚠️ No higher-timeframe leakage: bins labelled and closed on the left, then
        # shifted one period, so a bar only ever sees periods that have finished.
        rule = {"day": "1D", "week": "W-SUN", "month": "MS"}[params["period"]]
        out = {}
        for field, how in (("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")):
            binned = getattr(data[field].resample(rule, label="left", closed="left"), how)()
            binned = binned[binned.notna().any(axis=1)].shift(1)
            out[field] = binned.reindex(close.index, method="ffill")
        return out
    if function == "highest_lowest":
        bars = int(params["bars"])
        high, low = inputs[0], inputs[1]
        out = {"upper": high.rolling(bars, min_periods=bars).max(), "lower": low.rolling(bars, min_periods=bars).min()}
        for name, frame, pick in (("bars_since_high", high, np.argmax), ("bars_since_low", low, np.argmin)):
            # How far back in each window its extreme sits, the latest one when two are equal:
            # pick runs on the window reversed (newest first), so it finds the newest first.
            values = frame.to_numpy(dtype=float)
            since = np.full(values.shape, np.nan)
            if len(values) >= bars:
                windows = np.lib.stride_tricks.sliding_window_view(values, bars, axis=0)[..., ::-1]
                filled = np.where(np.isnan(windows), -np.inf if pick is np.argmax else np.inf, windows)
                since[bars - 1:] = pick(filled, axis=-1)
                since[bars - 1:][np.isnan(windows).any(axis=-1)] = np.nan
            out[name] = pd.DataFrame(since, index=frame.index, columns=frame.columns)
        if not params.get("include_current"):
            # The n bars BEFORE this one: shifted by a bar, so each bar-since counts one more.
            out = {name: frame.shift(1) + (1 if name.startswith("bars_since") else 0) for name, frame in out.items()}
        out["middle"] = (out["upper"] + out["lower"]) / 2
        return out
    if function == "pivots":
        # ⚠️ No leakage: the previous period's own levels, which only ever show completed periods.
        last = run_studio("previous_period", {"period": params["period"]}, [], data)
        high, low, close = last["high"], last["low"], last["close"]
        span = high - low
        method = params["method"]
        pivot = (high + low + 2 * close) / 4 if method == "woodie" else (high + low + close) / 3
        if method == "fibonacci":
            levels = {f"r{n}": pivot + k * span for n, k in ((1, 0.382), (2, 0.618), (3, 1.0))}
            levels.update({f"s{n}": pivot - k * span for n, k in ((1, 0.382), (2, 0.618), (3, 1.0))})
        elif method == "camarilla":
            levels = {f"r{n}": close + span * 1.1 / k for n, k in ((1, 12), (2, 6), (3, 4))}
            levels.update({f"s{n}": close - span * 1.1 / k for n, k in ((1, 12), (2, 6), (3, 4))})
        else:
            levels = {"r1": 2 * pivot - low, "s1": 2 * pivot - high, "r2": pivot + span, "s2": pivot - span,
                      "r3": high + 2 * (pivot - low), "s3": low - 2 * (high - pivot)}
        return {"pivot": pivot, **levels}
    if function == "period_to_date":
        # Grouped by the bars' own calendar period, and cumulative within it: a bar sees its
        # period's bars up to itself, never a later one.
        rule = {"day": "D", "week": "W-SUN", "month": "M", "year": "Y"}[params["period"]]
        key = close.index.to_period(rule)
        return {"open": data["open"].groupby(key).transform("first"), "high": data["high"].groupby(key).cummax(),
                "low": data["low"].groupby(key).cummin()}
    if function == "offset":
        return {"value": inputs[0] * (1 + float(params["percent"]) / 100.0)}
    if function == "percent_band":
        share = float(params["percent"]) / 100.0
        return {"upper": inputs[0] * (1 + share), "lower": inputs[0] * (1 - share)}
    if function == "distance":
        return {"value": (inputs[0] / inputs[1] - 1.0) * 100.0}
    if function == "bars_ago":
        return {"value": inputs[0].shift(int(params["bars"]))}
    if function == "bars_since":
        fired = inputs[0].fillna(0.0) != 0
        position = pd.DataFrame(np.arange(len(fired))[:, None].repeat(fired.shape[1], axis=1),
                                index=fired.index, columns=fired.columns, dtype=float)
        last = position.where(fired).ffill()
        return {"value": position - last}
    if function == "python":
        # ⚠️ Runs arbitrary code in-process. The UI labels it as trusted local code and
        # warns before running an imported spec that contains one.
        body = "\n".join("    " + line for line in params["code"].splitlines()) or "    pass"
        namespace = {}
        exec(f"def block(open, high, low, close, volume, np, pd):\n{body}\n", namespace)
        result = namespace["block"](data["open"], data["high"], data["low"], close, data["volume"], np, pd)
        if isinstance(result, pd.Series):
            result = result.to_frame(close.columns[0]) if close.shape[1] == 1 else None
        if not isinstance(result, pd.DataFrame) or result.shape != close.shape:
            raise ValueError("Python code must return a DataFrame shaped like `close` "
                             "(one column per symbol, same index).")
        return {"value": result.astype(float)}
    raise ValueError(f"Unknown Studio entry {function}.")


def call_params(feature):
    """The keyword arguments for the library: defaults filled in, choices turned into the
    library's own values, and parameters left blank (None) not passed at all."""
    item = entry(feature["indicator"])
    given = feature.get("params") or {}
    out = {}
    for name, declaration in item["params"].items():
        value = given.get(name, declaration.get("default"))
        if value is None:
            continue
        kind = declaration["type"]
        if kind == "choice" and item["source"] == "talib":
            value = TALIB_MA_TYPES.index(value)          # TA-Lib's matype parameters take an integer
        elif kind == "int":
            value = int(value)
        elif kind == "float":
            value = float(value)
        elif kind == "bool":
            value = bool(value)
        out[name] = value
    return out


def compute(feature, inputs, data):
    """{output name: DataFrame shaped like close} for one feature.

    `inputs` holds a frame for each of the entry's inputs, in order (price fields or
    other features' outputs, resolved by the compiler).
    """
    item = entry(feature["indicator"])
    params = call_params(feature)
    if item["source"] in ("talib", "pandas_ta"):
        return run_library_indicator(item["source"], item["function"], item["inputs"], inputs, params,
                                     item["outputs"])
    if item["source"] == "pattern":
        module, name = item["function"].split(":")
        return {"signal": run_pattern(module, name, params, data["open"], data["high"], data["low"], data["close"],
                                      data["volume"])}
    return run_studio(item["function"], params, inputs, data)


def export_source():
    """The compute functions' own source, for the exported script."""
    return "\n\n".join([f"TALIB_MA_TYPES = {TALIB_MA_TYPES!r}",
                        inspect.getsource(run_library_indicator), inspect.getsource(run_pattern),
                        inspect.getsource(run_studio)])


# ---------------------------------------------------------------------------
# Building the snapshot (offline: `python -m openstrategy_studio.indicators`)
# ---------------------------------------------------------------------------

def _talib_entries():
    import talib
    from talib import abstract

    placed = {name: place for place, names in TALIB_MENU.items() for name in names}
    rows = []
    for function in sorted(talib.get_functions()):
        info = abstract.Function(function).info
        if info["group"] == "Pattern Recognition":
            continue            # candlesticks come from ta_patterns
        inputs = []
        for value in info["input_names"].values():
            inputs += list(value) if isinstance(value, (list, tuple)) else [value]
        params = {}
        for name, default in info["parameters"].items():
            if "matype" in name:
                params[name] = {"type": "choice", "default": TALIB_MA_TYPES[int(default)], "options": TALIB_MA_TYPES}
            elif isinstance(default, int):
                params[name] = {"type": "int", "default": default}
            else:
                params[name] = {"type": "float", "default": float(default)}
        category, family = placed[function]       # KeyError: a new TA-Lib function needs a menu place
        rows.append({"id": f"talib:{function}", "source": "talib", "function": function,
                     "label": info["display_name"], "menu": [category, family], "inputs": inputs,
                     "defaults": {name: name if name in PRICE_FIELDS else None for name in inputs},
                     "params": params, "outputs": list(info["output_names"]),
                     "description": info["group"]})
    return rows


def _pandas_ta_default(function, name, doc, source):
    """A pandas-ta parameter's (type, default, description). Most signatures say None and
    set the real default in the body; the docstring's `name (type): text. Default: x` line
    gives all three."""
    import ast
    match = re.search(rf"^\s*{re.escape(name)}\s*\(([^)]*)\):(.*?)(?=^\s*\w+\s*\(|\Z)", doc, re.M | re.S)
    kind, default, words = None, None, ""
    if match:
        # The description: the text before "Default", in its own paragraph (the last
        # argument's match runs on into the Kwargs: heading).
        words = " ".join(re.split(r"Default", match.group(2).split("\n\n")[0])[0].split()).strip()
        kind = match.group(1).split(",")[0].strip().lower()
        found = re.search(r"Default:?\s*(-?[\w.'\"]+)", match.group(2))
        if found:
            text = found.group(1).rstrip(".")
            try:
                default = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                default = text.strip("'\"")
    if default is None:
        found = re.search(rf"^\s*{re.escape(name)}\s*=.*?\belse\s+([^\n#]+)", source, re.M)
        if found:
            try:
                default = ast.literal_eval(found.group(1).strip().rstrip(")"))
            except (ValueError, SyntaxError):
                default = None
    return kind, default, words


def _pandas_ta_entries(talib_names):
    import warnings

    import pandas_ta_classic
    import vectorbt

    placed = {name: place for place, names in PANDAS_TA_MENU.items() for name in names}
    sample = _sample_data()
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        names = sorted(vectorbt.IndicatorFactory.get_pandas_ta_indicators())
        for name in names:
            if name in talib_names or f"pandas_ta:{name}" in EXCLUDED:
                continue
            factory = vectorbt.IndicatorFactory.from_pandas_ta(name)
            function = getattr(pandas_ta_classic, name.lower())
            doc = inspect.getdoc(function) or ""
            try:
                code = inspect.getsource(function)
            except (OSError, TypeError):
                code = ""
            params = {}
            for param in factory.param_names:
                if param in HIDDEN_PARAMS:
                    continue
                kind, default, words = _pandas_ta_default(name, param, doc, code)
                if param == "mamode":
                    params[param] = {"type": "choice", "default": default if default in PANDAS_TA_MA_TYPES else None,
                                     "options": PANDAS_TA_MA_TYPES}
                elif isinstance(default, bool) or kind == "bool":
                    params[param] = {"type": "bool", "default": default if isinstance(default, bool) else None}
                elif isinstance(default, int) or kind == "int":
                    params[param] = {"type": "int", "default": default if isinstance(default, int) else None}
                elif isinstance(default, str) or kind == "str":
                    params[param] = {"type": "str", "default": default if isinstance(default, str) else None}
                else:
                    # A number, or no default and no type in the docs (Holt-Winter's na, nb ...):
                    # a number left blank, which means "the library's own default".
                    params[param] = {"type": "float", "default": float(default) if isinstance(default, (int, float))
                                     else None}
                if words:
                    params[param]["help"] = words
            # Which outputs are numbers: run once on the sample, positionally like compute().
            inputs = list(factory.input_names)
            series = {key: sample[_price_for(key)]["X"] for key in inputs}
            result = function(**series)
            if isinstance(result, pd.Series):
                numeric = [True]
            else:
                numeric = [pd.api.types.is_numeric_dtype(result[column]) for column in result.columns]
            output_names = [out for out, keep in zip(factory.output_names, numeric) if keep]
            first = doc.splitlines()[0].strip() if doc else name
            category, family = placed[name]      # KeyError: a new pandas-ta indicator needs a place or a reason
            rows.append({"id": f"pandas_ta:{name}", "source": "pandas_ta", "function": name,
                         "label": first.rstrip("."), "menu": [category, family], "inputs": inputs,
                         "defaults": {key: _price_for(key) for key in inputs}, "params": params,
                         "outputs": output_names, "description": " ".join(doc.split("\n\n")[1:2]).strip()[:300]})
    return rows


def _price_for(input_name):
    key = input_name.lower().rstrip("_")
    return key if key in PRICE_FIELDS else "close"


def _pattern_entries():
    import ta_patterns
    import ta_patterns.chart_patterns as chart
    import ta_patterns.scanner as scanner

    rows = []
    for module, families in PATTERN_FAMILIES.items():
        package = chart if module == "chart" else ta_patterns
        classes = chart if module == "chart" else scanner
        labels = ["BULLISH", "BEARISH", "DIRECTIONAL", "BIDIRECTIONAL", "NON_DIRECTIONAL"]
        known = set()
        for label in labels:
            known |= set(getattr(classes, label, ()))
        for submodule, family in families:
            source = getattr(package, submodule)
            for name in sorted(vars(source)):
                function = vars(source)[name]
                if name.startswith("_") or not callable(function) or getattr(function, "__module__", "") != source.__name__:
                    continue
                if name not in known or not hasattr(package, name) or f"pattern:{module}:{name}" in EXCLUDED:
                    continue
                direction = next(label.lower() for label in labels if name in getattr(classes, label, ()))
                params = {}
                for key, parameter in inspect.signature(function).parameters.items():
                    if key in ("o", "h", "l", "c", "v") or parameter.default is inspect.Parameter.empty:
                        continue
                    default = parameter.default
                    if key == "mode":
                        params[key] = {"type": "choice", "default": default, "options": ["confirmed", "forming"]}
                    elif isinstance(default, bool):
                        params[key] = {"type": "bool", "default": default}
                    elif isinstance(default, int):
                        params[key] = {"type": "int", "default": default}
                    elif isinstance(default, float) or default is None:
                        params[key] = {"type": "float", "default": default}
                    else:
                        params[key] = {"type": "str", "default": default}
                    # Some detectors document their settings numpy-style: "name : type", then
                    # indented lines saying what it does.
                    described = re.search(rf"^{re.escape(key)}\s*:[^\n]*\n((?:[ \t]+\S[^\n]*\n?)+)",
                                          inspect.getdoc(function) or "", re.M)
                    if described:
                        params[key]["help"] = " ".join(described.group(1).split())
                doc = (inspect.getdoc(function) or "").replace("**", "")
                rows.append({"id": f"pattern:{module}:{name}", "source": "pattern", "function": f"{module}:{name}",
                             "label": name.replace("_", " ").capitalize(),
                             "menu": ["Candlestick" if module == "candle" else "Chart pattern", family],
                             "inputs": [], "defaults": {}, "params": params, "outputs": ["signal"],
                             "direction": direction, "description": " ".join(doc.split())[:300]})
    return rows


def _studio_entries():
    rows = []
    for indicator_id, item in STUDIO.items():
        # An entry reads the close unless it names its own defaults (highest_lowest reads high and low).
        defaults = {name: (item.get("defaults") or {}).get(name, "close") for name in item["inputs"]}
        rows.append(dict(item, id=indicator_id, source="studio", function=indicator_id.split(":", 1)[1],
                         defaults=defaults))
    return rows


def _sample_data(n=400, seed=3):
    """A small synthetic OHLCV set for the snapshot and the lookahead check."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    volume = rng.integers(1_000, 10_000, n).astype(float)
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    return {name: pd.DataFrame({"X": values}, index=index) for name, values in
            (("open", open_), ("high", high), ("low", low), ("close", close), ("volume", volume))}


def build_catalogue():
    """Every entry, from the installed libraries and the decisions above, sorted by id."""
    talib_rows = _talib_entries()
    talib_names = {row["function"] for row in talib_rows}
    rows = talib_rows + _pandas_ta_entries(talib_names) + _pattern_entries() + _studio_entries()
    for row in rows:
        if row["id"] in ENTRY_TEXT:
            row["label"], row["description"] = ENTRY_TEXT[row["id"]]
    return sorted(rows, key=lambda row: row["id"])


def write_catalogue():
    rows = build_catalogue()
    CATALOGUE_FILE.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    catalogue.cache_clear()
    return len(rows)


if __name__ == "__main__":
    print(f"Wrote {write_catalogue()} entries to {CATALOGUE_FILE}")
