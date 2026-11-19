One optimisation job at a time, with the answer first.

**Best trial.** The best settings, each named as the strategy names it (*jma7 › length*), beside the value the job started from and the range it searched. Above them, the objective **in sample** (the history the optimiser searched, where the best trial is the best score) and **out of sample** (the history held back from it, run once with the best settings: the honest test), with trades and return for each. A big drop from in sample to out of sample means the settings fit the past rather than found an edge. Nothing is out of sample unless the job held some history back (set on the Optimise page).

- **Apply to the strategy** writes the best values into the strategy and opens it on the Build page; the variables and their ranges stay.
- **Show its backtest** runs the best settings over the whole history and draws it, with a dashed line where the held-back history starts and that part shaded. Pick the equity curve against buy and hold, the drawdown, or the price chart with its trades. The table under it gives the numbers in sample, out of sample and over the whole period; each part is run from the starting cash on its own, as the optimiser scored it, so its numbers match the job's.
- **Check it reproduces** re-runs the best trial to check the stored result still comes out the same, and works out the deflated Sharpe ratio: the chance the true Sharpe is above zero once the number of trials is allowed for.

**Walk-forward** jobs show each fold instead: its in-sample and out-of-sample dates, scores and best settings, the out-of-sample to in-sample ratio, and the stitched out-of-sample equity.

**How the search went.** How many settings were tried and how many were valid, how the neighbours of the best did (the stability check), how the search converged, and how the score varies with the settings (a pair heatmap, parallel coordinates, or one setting at a time).

**Every trial**, best first, with each setting named in words and the reason any trial is not valid.

Saved backtests are compared on the **Backtest results** page.
