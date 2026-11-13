Work through the steps left to right. The **At a glance** panel on the right shows the whole strategy as it stands, and lists anything to fix before it can run.

**1. Setup.** Name the strategy, pick the symbols, timeframe and dates, then answer two questions:

- *What are you building?* **Trade on signals** (rules decide when to buy and sell) or **hold a portfolio** (target weights, rebalanced on a schedule or never; one symbol held from the start is buy and hold).
- *How are you trading it?* Buying outright, CFDs, spread bets or FX spot. This sets the costs, how a short is charged, and how positions are sized, for every symbol. The instrument grid then shows only the fields that way of trading uses.

**2. Features.** The lines and signals your rules compare. **Add a feature** in three steps: a category (Indicator, Statistical, Price levels, Candlestick, Chart pattern, Transform, Custom), a family (Moving averages, Momentum ...), then the item. Its outputs are listed before you add it; each becomes a line you can use, such as `bb.upperband`.

- Every input can read a price or another feature: an EMA of an RSI, or bands around an EMA.
- *Transform* holds the adjustments: offset a line by a percentage, a band either side of it, the distance between two lines, a line some bars ago, bars since a signal. "Crosses within 5% of the EMA" is a 5% band around the EMA, compared in a rule.

**3. Rules** (signals). Each rule is a list of lettered conditions and a **logic line** that combines them:

- A condition compares two lines (*fast crosses above slow*), checks that a pattern appears, or (entries only) filters by weekday, month, session or a higher timeframe.
- The logic line uses `and`, `or`, `xor`, `nand`, `nor`, `xnor` and `not`, with brackets: `(A and (B or C)) and not D`. Brackets are needed wherever two different words meet, so the line always says exactly what you mean.
- **Make versions of this rule** keeps several versions; the optimiser can choose between them.

**3. Portfolio** (portfolios). Optionally pick assets by a score, choose the weighting, and how often to go back to the target weights: never (buy and hold), every bar, or on a calendar, optionally only when the weights have drifted.

**4. Risk.** How big each position is (labelled for the way you trade), a stop, a take profit, extra exits (after a number of bars, breakeven, a stepped stop), and for intraday bars the order timing: entries good for the day only, and closing before the end of the day.

**5. Account.** Starting cash, how several symbols share it, the reporting currency, and interest rates for holding costs.

**Preview** marks every signal on the chart. **Pipeline** draws the strategy. **Strategy file** shows it as JSON and as a standalone vectorbtX script.

Optimisation ranges are not set here: add the values to search on the **Optimise** page.
