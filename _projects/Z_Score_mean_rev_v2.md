---
layout: page
title: "Trading Strategy: Mean Reversion Z-Score version 2"
description: cTrader backtest interactive report
category: Personal
importance: 2
nav: false
permalink: /projects/z-score-mean-rev-v2/
---

### General Info
This strategy is a direct evolution of the version 1 of the strategy 1, built on the same core principles but with key adjustments to improve risk control. For background on the main idea, please refer to the [General Info section of version 1 of the strategy](/projects/z-score-mean-rev-v1/).

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


<iframe src="/assets/proj/report_zscore_v2.html"
        style="width: 100%; height: 800px; border: none;"
        loading="lazy">
</iframe>