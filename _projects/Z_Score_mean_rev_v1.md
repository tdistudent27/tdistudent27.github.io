---
layout: page
title: "Trading Strategy: Mean Reversion Z-Score version 1"
description: cTrader backtest interactive report
category: Personal
importance: 1
nav: false
permalink: /projects/z-score-mean-rev-v1/

---
## General Info
This is the result of the backtest computed in cTrader of a strategy based on custom indicators built by me in the platform.
The [Version 2 of the strategy](/projects/z-score-mean-rev-v2/) offers a different approach to the classic `risk–reward` trade-off that every investor faces.

While Strategy 1 achieves a significantly higher cumulative return (+376.24%), it does so at the cost of substantial risk exposure — with a maximum equity drawdown of 41.92%, it shows periods of considerable underperformance that could challenge an investor's risk tolerance. The balance drawdown of 32.29% further confirms that even realized profits were at risk during adverse market phases, potentially undermining portfolio liquidity or investor confidence.

Strategy 2, in contrast, delivers a lower return (+298.66%), but exhibits markedly lower drawdowns: 19.87% on balance and 25.99% on equity. This suggests a smoother performance profile, more controlled downside risk, and potentially better behavior under market stress.

An important consideration is that both strategies were calibrated using only the first half of the backtest period, with the remaining data used for out-of-sample validation. This methodological choice was made to avoid look-ahead bias and to test the robustness and consistency of each model beyond the optimization window. The fact that both strategies maintained strong performance in the out-of-sample segment reinforces the credibility of the results.

Interestingly, Strategy 2 sacrifices only about 21% of the total return compared to Strategy 1, but achieves a 38% reduction in maximum equity drawdown. From a risk-adjusted performance perspective, this represents a meaningful efficiency gain. Such a profile could be more appealing for capital-constrained investors or in portfolio contexts where drawdown limits are binding.

Ultimately, the comparison reflects two strategic design choices:

- Strategy 1 favors return maximization through higher exposure, accepting deeper drawdowns as a trade-off.

- Strategy 2 emphasizes stability and risk control, potentially making it more suitable for leveraged applications, multi-strategy integration, or investors with tighter risk constraints.

The choice between the two depends not only on return expectations, but also on the risk philosophy, investment horizon, and capital discipline of the investor.

> NOTE:
>
> If you need more info, stats or metricts feel free to contact me at my email 
demar.tommaso@gmail.com

### Balance Drawdown vs Equity Drawdown

- Balance Drawdown refers to the decline from the highest historical account balance, considering only closed trades. It measures losses that are **realized**, i.e., after positions have been exited. It's a more conservative metric because it only reflects actual, booked losses.

- Equity Drawdown, on the other hand, includes both closed and open positions. It reflects the worst-case scenario that occurred in real time, even if the loss was later recovered. This makes it more volatile but also more informative about temporary risk exposure.

### Example
Imagine your account grows to €10,000, then drops to €8,000:

- If the €2,000 loss came from a still-open trade that eventually recovers, the equity drawdown registers it, but the balance drawdown does not.

- If the trade is closed at €8,000, both drawdowns will record the loss.



<iframe src="/assets/proj/report_zscore_v1.html"
        style="width: 100%; height: 800px; border: none;"
        loading="lazy">
</iframe>
