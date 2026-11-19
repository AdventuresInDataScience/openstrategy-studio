Work through the steps left to right. The **At a glance** panel on the right shows the whole strategy as it stands, and lists anything to fix before it can run.

The Studio starts with a blank strategy each time. **Start from** at the top gives a blank one again, or a ready-made template to edit.

**1. Setup.** Name the strategy, pick the symbols, timeframe and dates, then answer two questions:

- *What are you building?* **Trade on signals** (rules decide when to buy and sell) or **hold a portfolio** (target weights, rebalanced on a schedule or never; one symbol held from the start is buy and hold).
- *How are you trading it?* Buying outright, CFDs, spread bets or FX spot. This sets the costs, how a short is charged, and how positions are sized, for every symbol. The instrument grid then shows only the fields that way of trading uses. Its **leverage** column fills in with the usual limit: on spread bets and CFDs 30x for FX, 10x for indices and commodities and 5x for single stocks; none (1x) bought outright, on FX spot and in every portfolio. Type your own to change it.
- **Points.** Stops, targets, spreads and a spread bet's stake are counted in points, each instrument's point size of price. On a share a point is 1 cent, so a 1.00 move is 100 points (on a share priced in pence, 1p); on FX it is a pip (0.0001, or 0.01 on a yen pair); on an index, 1.0. The point_size column fills in with the usual one when you pick how you trade, an asset class or a quote currency; a commodity's varies by broker, so check yours. The line under the grid says what a point is for each symbol, and the line under a stop or target box what its distance is in price.

**2. Features.** The lines and signals your rules compare. **Add a feature** in three steps: a category (Indicator, Statistical, Price levels, Candlestick, Chart pattern, Transform, Custom), a family (Moving averages, Momentum ...), then the item. Its outputs, the prices it reads and its settings (with their defaults) are listed before you add it; each output becomes a line you can use, such as `bb.upperband`.

- *Price levels* holds the highest and lowest over n bars (with or without the current bar, and how many bars ago each was set), pivot points from the previous day, week or month, the previous period's open, high, low and close, and this day, week or month so far. *Price types* explain themselves: Typical price ((high + low + close) / 3) and so on.

- Rename a feature in its **Name** box at any time; the rules, other features and optimise variables that use it follow.
- Every input can read a price or another feature: an EMA of an RSI, or bands around an EMA.
- *Transform* holds the adjustments: offset a line by a percentage, a band either side of it, the distance between two lines, a line some bars ago, bars since a signal. "Crosses within 5% of the EMA" is a 5% band around the EMA, compared in a rule.
- On an SMA, EMA or other single-type average, **Let the optimiser pick the type** swaps it for the *Moving average* entry at the same type (the same numbers), so the Optimise page can try SMA against EMA and others.

**3. Rules** (signals). There are four: long entry, long exit, short entry, short exit. **Add a … rule** starts one and **Delete the … rule** removes it with everything in it. Each rule sits in its own panel, titled with its logic line, and is a list of named conditions and a **logic line** that combines them:

- **Add condition**, then pick its **Kind**: compare two lines (*fast crosses above slow*), a pattern appears, the day, month or time of day, or a comparison on a higher timeframe. Every rule takes every kind, exits included: *A or Friday* is a time exit.
- Each condition has its own panel, titled in words, so it can be closed once it is right.
- **Must hold** says when: on this bar, on each of the last N bars (*RSI above 50 for 3 bars*), on any of the last N bars (*a cross within the last 5 bars*), or N bars ago (*a cross 1 bar ago, then a close below the EMA now*). On a higher timeframe, N counts its bars.
- New conditions are named A, B, C ...; type a name of your own in a condition's **Name** box (letters, digits and underscores, such as `trend` or `rsi_low`) and the logic line follows.
- The logic line uses `and`, `or`, `xor`, `nand`, `nor`, `xnor` and `not`, with brackets: `(A and (B or C)) and not D`, or `trend and not overbought`. Brackets are needed wherever two different words meet, so the line always says exactly what you mean. Names match exactly, capitals included.
- **Optimise**, at the foot of each condition, lists the values in it the optimiser can search: the comparison, each number, each side's bars ago (0 included), the N of **Must hold**, the bars of rising and falling, a pattern's side, a higher timeframe. Pick one and it is added to the Optimise page with a starting range, which you change there. To search how many bars a condition must hold, pick *on each of* or *on any of the last N bars* first: N = 1 is the same as on this bar, so a search from 1 includes it.
- **Delete condition** in a condition's panel removes it, from the logic line too.
- **Make versions of this rule** keeps several versions; the optimiser can choose between them.

**3. Portfolio** (portfolios). Optionally pick assets by a score, choose the weighting, and how often to go back to the target weights: never (buy and hold), every bar, or on a calendar, optionally only when the weights have drifted.

**4. Risk.** How big each position is (labelled for the way you trade): a risk per trade, a cash amount, the instrument's own units, or a position worth a share of the account. A risk of 0.01 is 1%, and a share can be of the starting cash (the same amount on every trade), the balance or the equity (read at each signal bar, so trades grow with profits and shrink after losses); the line under the box says what it is in money. A risk is sized from whichever stop you set, and with no stop from the leverage (the trade's margin is the risk, and no stop is placed). Then a stop and a take profit, each in points, in % of the price, as a multiple of a line (3 × an ATR) or at a line's level (an EMA, the signal bar's low); the line is read on the bar the signal fires on, and an entry whose level is on the wrong side of the fill is not placed. A trailing stop in points stays that far behind the best price; at a level, it moves to each completed bar's value and never loosens. The extra exits need a fixed stop in points. Then:

- **Leverage**: each symbol's leverage and what it funds. It is set in the instrument grid on the Setup tab.
- **When a trade is bigger than the account can fund**: fill as much as it funds, or skip the trade.
- **Trade size check**: the largest trade your entries place, with its stop in price (implied by the leverage when there is no stop), against what the account funds. A warning says what to change before you run, including a stop narrower than a usual bar. A share of the balance or equity is estimated here from the starting cash.
- Extra exits (after a number of bars, breakeven, a stepped stop), and for intraday bars the order timing: entries good for the day only, and closing before the end of the day.

**5. Account.** Starting cash, how several symbols share it, the reporting currency, and interest rates for holding costs.

**Preview** marks every signal on the chart. **Pipeline** draws the strategy. **Strategy file** shows it as JSON and as a standalone vectorbtX script.

Optimisation ranges are not set here: add the values to search on the **Optimise** page (or, for a rule's values, in the condition's **Optimise** list), and set their ranges there.
