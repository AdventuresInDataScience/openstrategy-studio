**1. Choose the variables.** Add each value you want searched: a feature's setting (the EMA's period), a number in a rule (RSI below 30), a risk dial (the stop distance, the take profit, the bars a trade is held, the size), a portfolio setting, or which version of a rule to use.

For each, give **a range** (start, step, stop) or **a list of values**. The grid tries every step; the other searchers pick any value in the range, whole numbers for a whole-number setting. The count under each, and the total, show how big a full grid is.

**2. Choose how to search.** One search, walk-forward, or a comparison of backends. Optuna is the usual choice; BUTChC and Optuna can also search between versions of a rule.

**3. Set the objective and the guard rails.** The metric to maximise or minimise, a minimum number of trades (without one, optimisers find settings that trade twice and win both), and optional drawdown and profit-factor limits.

**4. Hold some history back.** The optimiser never sees the held-back part; the best result is then run on it once. Walk-forward does this repeatedly.

Jobs run in the background and keep going if you close the page. Results are on the **Results** page, where **Apply to the strategy** writes the best values back.
