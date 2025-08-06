---
layout: page
title: "Trading Strategy Optimization"
description: This is the procedure I adopt to optimize a strategy and avoid Overfitting
category: Personal
importance: 3
nav: false
---




## Libraries


```python
import pandas as pd
import vectorbt as vbt
import ta
from ta.trend import SMAIndicator, ADXIndicator
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from itertools import product
import time  

```
## Data
To **backtest a strategy in python** I use 2 version of the same dataset.

1. The `timerframe` the strategy is based on

2. A lower `timeframe` to have a more precise indications of the development of the candle

> Example:
>
> In this case the strategy is based on H4 candles. Using this timeframe I will compute the signals. Suppose a position is taken at `9am` and a **Take Profit (TP)** and a **Stop Loss (SL)** are set at some levels. The next candle reaches **both the TP and the SL**, so is it a loss or a profit?
>
> I will need the lower timeframe (e.g 1 minute candles in this case) to understand if the Take Proftit or the Stop Loss came first. 
> This is the the closer approach to a real word environment to backtest a strategy.
>
> Obviously, the lower the `timeframe`, the more precision I will have

This is my dataset

```python
print(f'H4 dataset:\n {df_4h}\n 1m dataset\n{df_1m}')
```

    H4 dataset:
                             open     high      low    close  tickvol
    datetime                                                        
    2017-07-21 16:00:00  1.16400  1.16702  1.16358  1.16660    31730
    2017-07-21 20:00:00  1.16662  1.16827  1.16612  1.16634    20497
    2017-07-24 00:00:00  1.16599  1.16842  1.16599  1.16709    12678
    2017-07-24 04:00:00  1.16708  1.16753  1.16631  1.16708     9686
    2017-07-24 08:00:00  1.16709  1.16751  1.16302  1.16529    35881
    ...                      ...      ...      ...      ...      ...
    2025-07-08 08:00:00  1.17416  1.17653  1.17319  1.17490    15925
    2025-07-08 12:00:00  1.17489  1.17506  1.17140  1.17211    15208
    2025-07-08 16:00:00  1.17210  1.17222  1.16827  1.17088    23138
    2025-07-08 20:00:00  1.17088  1.17298  1.17075  1.17244    11899
    2025-07-09 00:00:00  1.17199  1.17293  1.17142  1.17246     5627
    
    [12382 rows x 5 columns]
    
    1m dataset
                            open     high      low    close  tickvol
    datetime                                                        
    2017-07-21 12:45:00  1.16470  1.16480  1.16460  1.16476      146
    2017-07-21 12:46:00  1.16475  1.16480  1.16469  1.16472      162
    2017-07-21 12:47:00  1.16472  1.16486  1.16468  1.16485      150
    2017-07-21 12:48:00  1.16484  1.16484  1.16469  1.16479      174
    2017-07-21 12:49:00  1.16480  1.16481  1.16456  1.16463      208
    ...                      ...      ...      ...      ...      ...
    2025-07-08 23:55:00  1.17253  1.17257  1.17251  1.17257       19
    2025-07-08 23:56:00  1.17257  1.17268  1.17253  1.17267       28
    2025-07-08 23:57:00  1.17267  1.17267  1.17254  1.17257       23
    2025-07-08 23:58:00  1.17257  1.17257  1.17243  1.17245       22
    2025-07-08 23:59:00  1.17245  1.17247  1.17239  1.17244       20
    
    [2966785 rows x 5 columns]


## Parameters
This strategy open positions based on some parameters:

- Simple moving Average 1 (SMA1)

- SMA 2

- Average Directional Index (ADX)

- ADX Max: a threshold that filters positions

> WARNING:
> For obvious reasons, I will not disclose the exact signal-generation logic of my strategy.
>
> However, I will provide the parameter grid used for the optimization process.

Simply generate a code that adds a `signal` column to the main timeframe of the strategy (4H my case), which takes the value `1` for `long` positions, `-1` for `short` positions, and `0` otherwise.

Additionally, I will apply a time filter, as the original study was conducted over this specific period:

- Start: `2022-05-19`

- End: `2025-07-09`

This is the result:

```python
print(df)
```

                            open     high      low    close  tickvol      sma1  \
    datetime                                                                     
    2022-05-19 00:00:00  1.04683  1.05006  1.04612  1.04884    11808  1.049522   
    2022-05-19 04:00:00  1.04884  1.05068  1.04789  1.04963    16846  1.049998   
    2022-05-19 08:00:00  1.04963  1.05090  1.04649  1.04964    28109  1.050408   
    2022-05-19 12:00:00  1.04964  1.05451  1.04872  1.05377    36015  1.051011   
    2022-05-19 16:00:00  1.05377  1.05986  1.05366  1.05934    56746  1.052029   
    ...                      ...      ...      ...      ...      ...       ...   
    2025-07-08 08:00:00  1.17416  1.17653  1.17319  1.17490    15925  1.175075   
    2025-07-08 12:00:00  1.17489  1.17506  1.17140  1.17211    15208  1.174769   
    2025-07-08 16:00:00  1.17210  1.17222  1.16827  1.17088    23138  1.174295   
    2025-07-08 20:00:00  1.17088  1.17298  1.17075  1.17244    11899  1.173959   
    2025-07-09 00:00:00  1.17199  1.17293  1.17142  1.17246     5627  1.173679   
    
                             sma2        adx  signal  
    datetime                                          
    2022-05-19 00:00:00  1.051903  13.630782       0  
    2022-05-19 04:00:00  1.051437  13.577530       0  
    2022-05-19 08:00:00  1.050971  13.551828       0  
    2022-05-19 12:00:00  1.050879  13.442523       0  
    2022-05-19 16:00:00  1.051263  13.220664       0  
    ...                       ...        ...     ...  
    2025-07-08 08:00:00  1.173984  19.680101       0  
    2025-07-08 12:00:00  1.173503  19.458087       0  
    2025-07-08 16:00:00  1.172791  19.161909       0  
    2025-07-08 20:00:00  1.172373  18.888370       0  
    2025-07-09 00:00:00  1.172375  18.619805       0  
    
    [4880 rows x 9 columns]


## Backtest
In addition to the signal-generation parameters, the strategy also includes risk management parameters:

`risk_pct` = Percentage of the equity to risk on each trade if the stop-loss is hit.

`extra_pips` = Additional “breathing room” in pips given to the stop-loss.

`tp_multiplier` = Take-profit target expressed as a multiple of the stop-loss distance.

After the best parameters over the grid have been found, the backes function will be something like this (some values have been included due to my personal experience in strategies like this):

```python
param_grid = {
    'sma1_window'   : [15, 30, 60],
    'sma2_window'   : [10, 20, 40],
    'adx_window'    : [15, 35, 55],
    'adx_max'       : [5, 13.5, 30],
    'risk_pct'      : [5, 10, 15],
    'extra_pips'    : [4],
    'tp_multiplier' : [1.3, 2, 5, 10]
}

# 1) GENERATE ALL COMBINATIONS
param_combos = list(product(*param_grid.values()))
TOTAL_COMBOS = len(param_combos)
print(f"Total combinations to test: {TOTAL_COMBOS:,}")
```

    Total combinations to test: 972

The optimization function is straightforward: it simply loops through each parameter combination, runs the backtest, and returns the combination associated with the chosen optimization metric. In my case, I selected the final equity.

> NOTE:
> 
> The backtest already accounts for swaps and commisions

```python
# ── DATA ───────────────────────────────────────────────────────────────────
h4 = df.copy()           # already contains the 'signal' column (+1 / –1 / 0)
m1 = df_1m.copy()

# ── RUNTIME STATE ──────────────────────────────────────────────────────────
equity            = init_equity
open_trade        = None
blocked_until     = pd.Timestamp.min
trades, curve     = [], []

def round_lot(x):
    return max(min_lot, np.floor(x / min_lot) * min_lot)

def nightly_swap_cash(entry_ts, exit_ts, lots, dir_):
    """Total overnight swap (€) between entry and exit."""
    first_night = entry_ts.normalize() + pd.Timedelta(days=1)
    last_night  = exit_ts.normalize()
    if exit_ts.time() == pd.Timestamp.min.time():
        last_night -= pd.Timedelta(days=1)
    if first_night > last_night:
        return 0.0

    nights = pd.date_range(first_night, last_night,
                           freq='D', inclusive='left')
    if nights.empty:
        return 0.0

    nightly_eur = swap_long_eur if dir_ == 1 else swap_short_eur
    total = 0.0
    for d in nights:
        total += nightly_eur
        if d.dayofweek == 2:            # Wednesday ⇒ +2 extra nights
            total += 2 * nightly_eur
    return total * lots

# ── 4-HOUR LOOP ────────────────────────────────────────────────────────────
for i in range(2, len(h4)):
    row, prev1, prev2 = h4.iloc[i], h4.iloc[i-1], h4.iloc[i-2]
    if (row.name < blocked_until) or (open_trade is not None) or (row['signal'] == 0):
        continue

    dir_     = int(row['signal'])        # +1 BUY, –1 SELL
    entry_ts = row.name
    entry_px = row['open']

    # Stop-loss price: previous 2 lows for long, previous 2 highs for short
    sl_px = (min(prev1['low'], prev2['low']) - extra_pips * pip_size
             if dir_ == 1 else
             max(prev1['high'], prev2['high']) + extra_pips * pip_size)
    sl_pips = abs(entry_px - sl_px) / pip_size
    if sl_pips == 0:
        continue

    # —— Adaptive lot sizing (risk + margin) ————————————————
    # Desired lot size based on risk management
    lots_risk = round_lot((equity * risk_pct / 100) / (sl_pips * pip_value))

    # Maximum lot size allowed by available margin
    lots_max_margin = round_lot((equity * leverage) / (entry_px * 100_000))

    lots = min(lots_risk, lots_max_margin, max_lot)
    if lots < min_lot:
        continue            # No trade possible → skip

    tp_px = entry_px + dir_ * sl_pips * pip_size * tp_multiplier
    open_trade = dict(lots=lots, dir=dir_,
                      entry_ts=entry_ts, entry=entry_px,
                      sl_px=sl_px, tp_px=tp_px)

    # —— Scan minute bars until TP or SL is hit ————————————————
    for ts, bar in m1.loc[entry_ts:].iloc[1:].iterrows():
        hit_tp = (dir_ == 1 and bar['high'] >= tp_px) or \
                 (dir_ ==-1 and bar['low']  <= tp_px)
        hit_sl = (dir_ == 1 and bar['low']  <= sl_px) or \
                 (dir_ ==-1 and bar['high'] >= sl_px)
        if not (hit_tp or hit_sl):
            continue

        exit_px  = tp_px if hit_tp else sl_px
        pips     = (exit_px - entry_px) / pip_size * dir_
        gross_pl = pips * lots * pip_value

        commission = lots * commission_per_lot
        swap_cash  = nightly_swap_cash(entry_ts, ts, lots, dir_)

        pl_cash = gross_pl - commission + swap_cash
        equity += pl_cash

        trades.append({**open_trade,
                       'exit_ts': ts, 'exit': exit_px,
                       'pips': pips, 'pl': pl_cash,
                       'equity_after': equity})
        curve.append((ts, equity))

        open_trade    = None
        blocked_until = ts               # Don't open a new trade until this candle closes
        break

    if equity <= 0:
        print("Equity depleted – stop back-test")
        break

# ── RESULTS ────────────────────────────────────────────────────────────────
trades_df = pd.DataFrame(trades)
equity_curve = (pd.DataFrame(curve, columns=['ts', 'equity'])
                .set_index('ts')
                .sort_index())

print(trades_df.tail())
print("Initial Equity:" 100,)
print("Final equity:", equity)

```

         lots  dir            entry_ts    entry    sl_px     tp_px  \
    325  50.0   -1 2025-06-06 00:00:00  1.14429  1.14991  1.136984   
    326  50.0   -1 2025-06-12 00:00:00  1.14825  1.15035  1.145520   
    327  50.0   -1 2025-06-12 08:00:00  1.15151  1.15332  1.149157   
    328  50.0   -1 2025-06-12 20:00:00  1.15761  1.16356  1.149875   
    329  50.0   -1 2025-06-13 16:00:00  1.15162  1.15646  1.145328   
    
                    exit_ts      exit   pips       pl  equity_after  
    325 2025-06-11 21:02:00  1.149910 -56.20 -27500.0   698672.6221  
    326 2025-06-12 01:57:00  1.150350 -21.00 -10800.0   687872.6221  
    327 2025-06-12 11:15:00  1.153320 -18.10  -9350.0   678522.6221  
    328 2025-06-13 13:47:00  1.149875  77.35  38600.0   717122.6221  
    329 2025-06-13 18:31:00  1.156460 -48.40 -24500.0   692622.6221
    Initial Equity: 100
    Final Equity: 692622.6220999856


    
<img src="/assets/proj/strategy_opt_files/equity_1.png" alt="eq_1" style="max-width: 100%; height: auto;">      



## Stats


```python


# ────────────────────── DRAW-DOWN ──────────────────────
def _max_drawdown(equity_ser: pd.Series):
    """
    Returns (absolute drawdown in €, positive % drawdown).
    Calculation identical to the one used in the book / cTrader:
        DD% = equity / max(equity_to_date) - 1
    """
    running_max = equity_ser.cummax()
    dd_pct_series = equity_ser / running_max - 1.0      # negative or zero
    dd_abs_series = running_max - equity_ser            # €
    dd_abs = round(dd_abs_series.max(), 2)
    dd_pct = round(-dd_pct_series.min()*100, 2)
    return dd_abs, dd_pct

# ───────────────────────── HELPER ──────────────────────────
def _stats(trades, label, commission_per_lot, swap_long, swap_short):
    if trades.empty:
        return {'label': label, 'Trades': 0}

    # commissions (if missing)
    if 'commission' not in trades.columns:
        trades = trades.assign(commission=trades.lots * commission_per_lot)

    # swap (if missing → quick recalculation without 1-min loop)
    if 'swap' not in trades.columns:
        pt = np.where(trades.dir == 1, swap_long, swap_short)  # €/lot/night
        swap_cash = (
            ((trades.exit_ts.dt.normalize()            # last day included?
              - (trades.entry_ts.dt.normalize() + pd.Timedelta(days=1)))
             .dt.days.clip(lower=0))                   # full nights
            .add(                                      # triple Wednesday
                ((trades.exit_ts.dt.dayofweek == 2)   # exit on Wednesday?
                 & (trades.exit_ts.dt.normalize()
                    > trades.entry_ts.dt.normalize()+pd.Timedelta(days=1))
                 ).astype(int)*2)
            * pt * trades.lots
        )
        trades = trades.assign(swap=swap_cash)

    gross_prof = trades.pl[trades.pl > 0].sum()
    gross_loss = trades.pl[trades.pl < 0].sum()
    pf = np.inf if gross_loss == 0 else gross_prof / abs(gross_loss)

    wins   = trades.pl.gt(0).sum()
    losses = trades.pl.lt(0).sum()
    avg_tr = trades.pl.mean()

    consec_w = trades.pl.gt(0).astype(int).groupby(
               (trades.pl.gt(0) != trades.pl.gt(0).shift()).cumsum()).cumsum()
    consec_l = trades.pl.lt(0).astype(int).groupby(
               (trades.pl.lt(0) != trades.pl.lt(0).shift()).cumsum()).cumsum()

    return {
        'label'           : label,
        'Trades'          : len(trades),
        'Net profit €'    : round(trades.pl.sum(), 2),
        'Profit factor'   : round(pf, 2),
        'Commission €'    : round(trades['commission'].sum(), 2),
        'Swap €'          : round(trades['swap'].sum(), 2),
        'Winners'         : wins,
        'Losers'          : losses,
        'Max consec win'  : int(consec_w.max()),
        'Max consec loss' : int(consec_l.max()),
        'Largest win €'   : round(trades.pl.max(), 2),
        'Largest loss €'  : round(trades.pl.min(), 2),
        'Avg trade €'     : round(avg_tr, 2)
    }

# ───────────────────── MAIN FUNCTION ────────────────────────
def full_stats(trades_df, equity_curve,
               commission_per_lot=6.0,
               swap_long_eur=-9.71,
               swap_short_eur=4.50):

    long_tr  = trades_df[trades_df.dir == 1]
    short_tr = trades_df[trades_df.dir == -1]

    rows = [
        _stats(long_tr,  'Long',  commission_per_lot, swap_long_eur, swap_short_eur),
        _stats(short_tr, 'Short', commission_per_lot, swap_long_eur, swap_short_eur),
        _stats(trades_df,'Total', commission_per_lot, swap_long_eur, swap_short_eur)
    ]

    # drawdown
    dd_abs, dd_pct = _max_drawdown(equity_curve['equity'])
    print(f"\nMax absolute drawdown : {dd_abs:,.2f} €")
    print(f"Max percentage drawdown: {dd_pct:.2f} %")

    cols_order = ['Trades','Net profit €','Profit factor',
                  'Commission €','Swap €',
                  'Winners','Losers',
                  'Max consec win','Max consec loss',
                  'Largest win €','Largest loss €','Avg trade €']

    return (pd.DataFrame(rows)
              .set_index('label')
              [cols_order])

# --------------------------- USAGE ----------------------------------
stats_df = full_stats(trades_df, equity_curve,
                      commission_per_lot=6.0,
                      swap_long_eur=-9.71,
                      swap_short_eur=4.50)

print(stats_df)

```

    
    Max absolute drawdown : 126,557.00 €
    Max percentage drawdown  : 68.53 %
           Trades  Net profit €  Profit factor  Commission €    Swap €  Winners  \
    label                                                                         
    Long      145     421206.77           2.43      18090.36 -17320.21       91   
    Short     185     271315.85           1.43      24816.24  14836.95      103   
    Total     330     692522.62           1.75      42906.60  -2483.26      194   
    
           Losers  Max consec win  Max consec loss  Largest win €  Largest loss €  \
    label                                                                           
    Long       54               5                4        54992.5        -34692.0   
    Short      82               8                7        73175.0        -38775.0   
    Total     136               7                7        73175.0        -38775.0   
    
           Avg trade €  
    label               
    Long       2904.87  
    Short      1466.57  
    Total      2098.55  
     

```python
# ─────────────────── Functions ─────────────────────────────

def sharpe_function(returns, timeframe=252):
    mean = returns.mean() * timeframe
    std = returns.std() * np.sqrt(timeframe)
    return mean / std

def sortino_function(returns, timeframe=252):
    downside = returns[returns < 0]
    mean = returns.mean() * timeframe
    std = downside.std() * np.sqrt(timeframe)
    return mean / std

def drawdown_function(returns):
    cum_rets = (returns + 1).cumprod()
    running_max = np.maximum.accumulate(cum_rets.dropna())
    running_max[running_max < 1] = 1
    drawdown = (cum_rets / running_max) - 1
    return drawdown

def VaR_function(theta, mu, sigma):
    n = 100000
    t = int(n * theta)
    sims = pd.DataFrame(np.random.normal(mu, sigma, size=n), columns=["Simulations"])
    return sims.sort_values(by="Simulations").iloc[t].values[0]

def cVaR_function(theta, mu, sigma):
    n = 100000
    t = int(n * theta)
    sims = pd.DataFrame(np.random.normal(mu, sigma, size=n), columns=["Simulations"])
    return sims.sort_values(by="Simulations").iloc[:t].mean().values[0]

# ─────────────── Metrics ─────────────────────────

returns = portfolio['returns']
mu = returns.mean()
sigma = returns.std()

```

### Cumulative returns

<img src="/assets/proj/strategy_opt_files/cum_ret_1.png" alt="cumret_1" style="max-width: 100%; height: auto;"> 


### Sharpe and Sortino Ratio

```python
print(f"Sharpe Ratio : {sharpe_function(returns):.4f}")
print(f"Sortino Ratio: {sortino_function(returns):.4f}")
```

    Sharpe Ratio : 2.6897
    Sortino Ratio: 3.4151



### Drawdown
    
<img src="/assets/proj/strategy_opt_files/drawdown.png" alt="drawdown" style="max-width: 100%; height: auto;"> 


### VAR and CVAR

```python
# ─────────────── VAR and CVAR ───────────────────────────────

print(f"Daily VaR 5%: {VaR_function(0.05, mu, sigma) * 100:.2f}%")
print(f"Monthly VaR 5%: {VaR_function(0.05, mu*20, sigma*np.sqrt(20)) * 100:.2f}%")
print(f"Annual VaR 5%: {VaR_function(0.05, mu*252, sigma*np.sqrt(252)) * 100:.2f}%")

print(f"Daily CVaR 5%: {cVaR_function(0.05, mu, sigma) * 100:.2f}%")
print(f"Monthly CVaR 5%: {cVaR_function(0.05, mu*20, sigma*np.sqrt(20)) * 100:.2f}%")
print(f"Annual CVaR 5%: {cVaR_function(0.05, mu*252, sigma*np.sqrt(252)) * 100:.2f}%")
```

    Daily VaR 5%: -12.86%
    Monthly VaR 5%: -34.28%
    Annual VaR 5%: 143.55%
    Daily CVaR 5%: -16.39%
    Monthly CVaR 5%: -50.91%
    Annual CVaR 5%: 85.91%

The results appear to be extremely good — could this indicate potential overfitting? To verify, we can perform a permutation test.


## Permutation Test

When a trading strategy achieves **exceptionally high returns** on historical data, it is important to determine whether the result is **genuine** or simply a consequence of **overfitting** or **data-mining bias**.  
One robust method for this validation is the **Permutation Test**.

#### 1. What is a Permutation Test?

A permutation test evaluates the **statistical significance** of a strategy’s performance by comparing it against results obtained on **randomized market data**.  
The procedure is:

1. **Compute the real performance** of the strategy (e.g., final equity after parameter optimization).
2. **Generate thousands of synthetic price series** by randomly **shuffling historical returns**, which destroys any true temporal patterns but preserves the statistical distribution of the returns.
3. **Run the same optimization on each permuted series**, recording the **best equity** obtained in each case.
4. **Build a distribution of “best equity by chance”**, representing what could be achieved without any real market edge.


#### 2. What is the p-value?

After generating all the permutations:

$$
p\text{-value} = \frac{N_{\text{equity} \geq \text{real}}}{N_{\text{total}}}
$$

Where $N_{\text{equity} \geq \text{real}}$ is the number of permutations whose equity exceeds or matches the real strategy's equity.




- A **low p-value** (e.g., < 5%) indicates that the strategy’s performance is **unlikely to occur by chance**, suggesting a **robust edge**.
- A **high p-value** implies that similar or better results can often be obtained on **randomized data**, pointing to **overfitting** or **lack of true predictive power**.


#### 3. Why It Matters

- Protects against **over-optimistic results** caused by backtest overfitting.
- Quantifies the **probability that the performance is due to luck**.
- Complements **walk-forward testing** and **out-of-sample validation** for a rigorous strategy evaluation.


By combining **Permutation Tests** with **risk-adjusted metrics**, traders can better assess the **credibility and robustness** of their algorithmic strategies.


To use this type of test we must create a function that will backtest only on the H4 timeframe. The results will be slightly different but this is not so important since we just wanna know if our strategy is due to **noise optimization**


```python
# Backtest with only h4
def round_lot(x):
    """Round the lot size down to the nearest allowed lot size."""
    return max(min_lot, np.floor(x / min_lot) * min_lot)

def nightly_swap_cash(entry_ts, exit_ts, lots, dir_):
    """Total overnight swap (€) between entry and exit (H4 only)."""
    first_night = entry_ts.normalize() + pd.Timedelta(days=1)
    last_night  = exit_ts.normalize()
    if exit_ts.time() == pd.Timestamp.min.time():
        last_night -= pd.Timedelta(days=1)
    if first_night > last_night:
        return 0.0

    # Generate all the intermediate nights
    nights = pd.date_range(first_night, last_night,
                           freq='D', inclusive='left')
    if nights.empty:
        return 0.0

    nightly_eur = swap_long_eur if dir_ == 1 else swap_short_eur
    total = 0.0
    for d in nights:
        total += nightly_eur
        if d.dayofweek == 2:  # Wednesday ⇒ +2 extra nights
            total += 2 * nightly_eur
    return total * lots

# ── MAIN LOOP (H4 only) ──────────────────────────────────────────────
equity            = init_equity
open_trade        = None
blocked_until     = pd.Timestamp.min
trades, curve     = [], []

for i in range(2, len(df)):
    row, prev1, prev2 = df.iloc[i], df.iloc[i-1], df.iloc[i-2]
    # Skip if trade is blocked, one is already open, or no signal
    if (row.name < blocked_until) or (open_trade is not None) or (row['signal'] == 0):
        continue

    dir_     = int(row['signal'])  # +1 long, -1 short
    entry_ts = row.name
    entry_px = row['open']

    # SL based on previous 2 bars
    sl_px = (min(prev1['low'], prev2['low']) - extra_pips * pip_size
             if dir_ == 1 else
             max(prev1['high'], prev2['high']) + extra_pips * pip_size)
    sl_pips = abs(entry_px - sl_px) / pip_size
    if sl_pips == 0:
        continue

    # Lot size calculation (risk-based and margin-limited)
    lots_risk = round_lot((equity * risk_pct / 100) / (sl_pips * pip_value))
    lots_max_margin = round_lot((equity * leverage) / (entry_px * 100_000))
    lots = min(lots_risk, lots_max_margin, max_lot)
    if lots < min_lot:
        continue  # No trade possible → skip

    tp_px = entry_px + dir_ * sl_pips * pip_size * tp_multiplier
    open_trade = dict(lots=lots, dir=dir_,
                      entry_ts=entry_ts, entry=entry_px,
                      sl_px=sl_px, tp_px=tp_px)

    # Scan future H4 bars to check TP/SL
    for j in range(i+1, len(df)):
        bar = df.iloc[j]
        ts = bar.name
        hit_tp = (dir_ == 1 and bar['high'] >= tp_px) or (dir_ == -1 and bar['low'] <= tp_px)
        hit_sl = (dir_ == 1 and bar['low'] <= sl_px) or (dir_ == -1 and bar['high'] >= sl_px)
        if not (hit_tp or hit_sl):
            continue

        exit_px  = tp_px if hit_tp else sl_px
        pips     = (exit_px - entry_px) / pip_size * dir_
        gross_pl = pips * lots * pip_value

        commission = lots * commission_per_lot
        swap_cash  = nightly_swap_cash(entry_ts, ts, lots, dir_)

        pl_cash = gross_pl - commission + swap_cash
        equity += pl_cash

        trades.append({**open_trade,
                       'exit_ts': ts, 'exit': exit_px,
                       'pips': pips, 'pl': pl_cash,
                       'equity_after': equity})
        curve.append((ts, equity))

        open_trade    = None
        blocked_until = ts  # Prevent opening a new trade until this bar closes
        break

    if equity <= 0:
        print("Equity depleted – stop back-test")
        break

# ── RESULTS ─────────────────────────────────────────────────────────
trades_df = pd.DataFrame(trades)
equity_curve = (pd.DataFrame(curve, columns=['ts', 'equity'])
                .set_index('ts')
                .sort_index())

print("Final equity:", equity)


```
 
    Final equity: 776237.4760999826

    
<img src="/assets/proj/strategy_opt_files/equity_h4.png" alt="eq_h4" style="max-width: 100%; height: auto;">     


### Permutation Code

```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt, time
from itertools  import product
from joblib     import Parallel, delayed, cpu_count, parallel
from tqdm       import tqdm
from contextlib import contextmanager

# ════════════════════════  CONFIGURATION  ═══════════════════════════
N_PERMUTATIONS   = 1_000        # number of permutations
MAX_PLOT_CURVES  = 1_000        # how many equity curves to display
BATCH_SIZE_PAR   = 5            # batch size for joblib

pip_size, pip_value       = 0.0001, 10
min_lot, max_lot          = 0.01, 50.0
leverage, init_equity     = "your leverage", 100
commission_per_lot        = 6.0
swap_long_eur, swap_short_eur = -9.71, 4.50   # BUY / SELL

# optimization grid (used by each permutation)
param_grid = {
    "sma1_window"  : [15, 30, 60],
    "sma2_window"  : [10, 20, 40],
    "adx_window"   : [15, 35, 55],
    "adx_max"      : [5, 13.5, 30],
    "risk_pct"     : [5, 10, 15],
    "extra_pips"   : [4],
    "tp_multiplier": [1.3, 2, 5, 10],
}
param_combos = list(product(*param_grid.values()))
print(f"Combinations per permutation: {len(param_combos)}")

# ════════════════════════  FUNCTIONS  ═════════════════════════
def round_lot(x):  # lot rounding
    return max(min_lot, np.floor(x / min_lot) * min_lot)

def nightly_swap_cash(entry_ts, exit_ts, lots, dir_):
    entry_ts, exit_ts = map(pd.Timestamp, (entry_ts, exit_ts))
    fst = entry_ts.normalize() + pd.Timedelta(days=1)
    lst = exit_ts.normalize() - (pd.Timedelta(days=1)
                                 if exit_ts.time()==pd.Timestamp.min.time() else pd.Timedelta(0))
    if fst > lst: return 0.0
    nights = pd.date_range(fst, lst, freq="D", inclusive="left")
    nightly = swap_long_eur if dir_==1 else swap_short_eur
    extra   = (nights.dayofweek == 2).sum() * nightly * 2     # Wed → +2 extra nights
    return (nights.size * nightly + extra) * lots

# backtest (with optional equity curve)
def backtest_h4_numpy(df, sma1_w, sma2_w, adx_w,
                      adx_max, risk_pct, extra_pips, tp_mult,
                      return_curve=False):

    open_, high_, low_, close_, adx_ = [df[c].values for c in
                                        ("open","high","low","close","adx")]
    n = len(df)
    sma1 = pd.Series(close_).rolling(sma1_w).mean().to_numpy()
    sma2 = pd.Series(close_).rolling(sma2_w).mean().to_numpy()
    adx  = pd.Series(adx_ ).rolling(adx_w ).mean().to_numpy()

    p1 = np.maximum(np.arange(n)-1, 0)
    p2 = np.maximum(np.arange(n)-2, 0)

    ok_time = (df.index.hour > 7) | ((df.index.hour==0)&(df.index.minute==0))
    long_sig  = generate_your_custom_LONG_signals(df)
    short_sig = generate_your_custom_SHORT_signals(df)
    signal = np.where(long_sig,1,np.where(short_sig,-1,0))

    equity, block = init_equity, pd.Timestamp.min
    if return_curve:
        curve_idx, curve_val = [df.index[0]], [equity]

    for i in range(2, n):
        ts = df.index[i]
        if ts < block or signal[i]==0: continue
        dir_ = int(signal[i]); ent_px=open_[i]
        sl_px = (min(low_[p1[i]],low_[p2[i]]) - extra_pips*pip_size
                 if dir_==1 else
                 max(high_[p1[i]],high_[p2[i]]) + extra_pips*pip_size)
        sl_pips = abs(ent_px-sl_px)/pip_size
        if sl_pips==0: continue
        lots = min(round_lot((equity*risk_pct/100)/(sl_pips*pip_value)),
                   round_lot((equity*leverage)/(ent_px*100_000)),
                   max_lot)
        if lots<min_lot: continue
        tp_px = ent_px + dir_*sl_pips*pip_size*tp_mult

        for j in range(i+1, n):
            hit_tp = ((dir_==1 and high_[j]>=tp_px) or
                      (dir_==-1 and low_[j] <=tp_px))
            hit_sl = ((dir_==1 and low_[j] <=sl_px) or
                      (dir_==-1 and high_[j]>=sl_px))
            if not (hit_tp or hit_sl): continue
            exit_px = tp_px if hit_tp else sl_px
            pips    = (exit_px-ent_px)/pip_size*dir_
            equity += (pips*lots*pip_value -
                       lots*commission_per_lot +
                       nightly_swap_cash(ts, df.index[j], lots, dir_))
            block = df.index[j]
            if return_curve:
                curve_idx.append(block); curve_val.append(equity)
            break
        if equity<=0: break

    if return_curve:
        return equity, pd.Series(curve_val, index=curve_idx)
    return equity

# generate a permutation
def get_permutation(df, seed=None):
    np.random.seed(seed)
    n=len(df); log=np.log(df[["open","high","low","close"]])
    r_o=(log["open"]-log["close"].shift()).to_numpy()
    r_h=(log["high"]-log["open"]).to_numpy()
    r_l=(log["low" ]-log["open"]).to_numpy()
    r_c=(log["close"]-log["open"]).to_numpy()
    idx=np.arange(1,n); p1,p2=np.random.permutation(idx),np.random.permutation(idx)
    perm=np.zeros((n,4)); perm[0]=log.iloc[0]
    for i in range(1,n):
        perm[i,0]=perm[i-1,3]+r_o[p2[i-1]]
        perm[i,1]=perm[i,0]+r_h[p1[i-1]]
        perm[i,2]=perm[i,0]+r_l[p1[i-1]]
        perm[i,3]=perm[i,0]+r_c[p1[i-1]]
    out=pd.DataFrame(np.exp(perm), index=df.index,
                     columns=["open","high","low","close"])
    out["adx"]=df["adx"].values
    return out

# bridge tqdm ↔ joblib
@contextmanager
def tqdm_joblib(tqdm_obj):
    class _Batch(parallel.BatchCompletionCallBack):
        def __call__(self,*a,**k):
            tqdm_obj.update(n=self.batch_size)
            return super().__call__(*a,**k)
    old = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = _Batch
    try:   yield
    finally: parallel.BatchCompletionCallBack = old; tqdm_obj.close()

# ═════════════════════  PERMUTATION TEST  ══════════════════════
best_equity_real = equity     # already computed outside this cell
print(f"Real equity (already computed): {best_equity_real:,.2f} €")

def run_perm(seed, save_curve):
    df_p = get_permutation(df_4h, seed)
    if save_curve:
        best_eq, best_curve = -np.inf, None
        for cmb in param_combos:
            eq, cv = backtest_h4_numpy(df_p, *cmb, return_curve=True)
            if eq > best_eq:
                best_eq, best_curve = eq, cv
        return best_eq, best_curve
    else:
        return max(backtest_h4_numpy(df_p,*cmb) for cmb in param_combos)

perm_equities, perm_curves = [], []
with tqdm_joblib(tqdm(total=N_PERMUTATIONS,
                      desc="Permutations", ncols=110)) as _:
    results = Parallel(n_jobs=max(cpu_count()-1,1),
                       backend="loky",
                       batch_size=BATCH_SIZE_PAR)(
        delayed(run_perm)(s, len(perm_curves)<MAX_PLOT_CURVES)
        for s in range(1, N_PERMUTATIONS+1)
    )

for r in results:
    if isinstance(r, tuple):
        eq, cv = r
        perm_curves.append(cv)
    else:
        eq = r
    perm_equities.append(eq)

p_value = np.mean([eq >= best_equity_real for eq in perm_equities])
print(f"P-value ≈ {p_value*100:.3f}%")

```

    Combinations per permutation: 972
    Real equity (already computed): 776,237.48 €


    Permutations: 100%|█████████████████████████████████████████████████████| 1000/1000 [1:00:29<00:00,  3.63s/it]

    P-value ≈ 1.100%


#### Equity Cyrves on Permutations
    
<img src="/assets/proj/strategy_opt_files/equity_perm.png" alt="eq_perm" style="max-width: 100%; height: auto;">     
    


#### Distribution of Best Equities on Permutations
    
<img src="/assets/proj/strategy_opt_files/distr_eq.png" alt="distr_eq" style="max-width: 100%; height: auto;">     
    

#### Log of Cumulative Returns on Pemutations
    
<img src="/assets/proj/strategy_opt_files/log_cum_re_perm.png" alt="log_cum_re_perm" style="max-width: 100%; height: auto;">     
    

## Conclusion
The permutation test yielded a p-value of approximately 1.1%, indicating that only about 11 out of 1,000 randomly permuted datasets produced an out-of-sample equity equal to or greater than the one obtained using the original, non-permuted data.

This low p-value suggests that the observed performance is unlikely to be the result of random chance alone. Under the null hypothesis—that the strategy captures no meaningful structure and any apparent performance arises from data mining or noise—such a result would be expected in roughly 1.1% of cases.

> While this does not conclusively prove the presence of a true signal, it provides statistical evidence against the null hypothesis, supporting the idea that the strategy may be exploiting a genuine structure in the data.

Nonetheless, it is important to emphasize that:

- A low p-value does not guarantee out-of-sample robustness.

- The results are contingent on the specific optimization procedure and parameter space.