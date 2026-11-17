All data is bars, from one minute to weekly.

- **Download from Yahoo** for symbols such as `SPY`, `VOD.L` or `EURUSD=X`. Intraday history on Yahoo only goes back a limited window.
- **Import a file** (CSV or Parquet, MT4 and MT5 exports included).
- **Refresh, rename or delete** stored data, and store **universe files** that describe many instruments at once.

**Names.** Every dataset is stored under a name, which is what a strategy picks: its symbol unless you change it in **Store as** before storing. A name holds one series at several timeframes (SPY at 1d and 1w), and a strategy reads its own timeframe or the finest stored one, combined up. Give a second source of the same symbol a name of its own (SPY_csv beside SPY): storing under a name and timeframe already held from another source asks first, because it replaces those bars.

**The check.** Every import is checked, and nothing is stored until you accept it. *Problems* would give a backtest wrong numbers: a high below the close, a spike that comes straight back (a bad price), an unadjusted split, bars that do not fit the timeframe. *Notes* are worth knowing and often fine: a gap over a holiday, a big move that held, no volume. Each says what it means, with sample rows. A move counts as unusual against the 100 bars before it, so a calm year and a wild one are each judged by their own standard; how unusual is set on the Settings page.
