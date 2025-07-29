---
layout: page
title: Volatility Forecasting
description: A comparative analysis of volatility forecasting models applied to the Euro Stoxx 50
importance: 2
category: Academic
nav: false
---

## Introduction

This project was carried out by Valentina Sanna (Sapienza University of Rome), Tommaso de Martino (Sapienza University of Rome), and Omar Tronelli (Sapienza University of Rome).
Each of us focused on a specific Machine Learning model:

- Long Short-Term Memory (LSTM)

- Multilayer Perceptron (MLP)

- Random Forest (RF)

The goal of the project is to compare the accuracy of different machine learning methods in forecasting volatility. As a benchmark, we use the GARCH model.
The evaluation metrics applied to assess model performance are MAE, MSE, and RMSE.
The analysis is conducted on daily data from the European stock index Euro Stoxx 50.

## Libraries


```python
# === Standard libraries
import os
import sys
import time
import random
import logging
import warnings
import math
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timedelta

# === Numerical computing and DataFrames
import numpy as np
import pandas as pd

# === Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# === Statistics
from scipy import stats
from scipy.stats import (
    norm,
    skew,
    kurtosis,
    jarque_bera,
    shapiro  # Added as per your request
)

# === Econometrics
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from arch import arch_model  # for GARCH models

# === Machine Learning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# === Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
    LSTM
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# === TensorFlow Probability
import tensorflow_probability as tfp
tfpl = tfp.layers
tfd  = tfp.distributions

# === Optimization and Hyperparameter Tuning
import optuna
from optuna import create_study
from optuna.integration import TFKerasPruningCallback
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.logging  import set_verbosity, WARNING

# === Miscellaneous
from threadpoolctl import threadpool_limits

# === Reproducibility (set random seeds)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# TensorFlow Probability seed generator
tfp_seed_generator = tf.random.Generator.from_seed(SEED)
# Optuna sampler with fixed seed
optuna_sampler = TPESampler(seed=SEED)
set_verbosity(WARNING)  # silence Optuna warnings

# Force TensorFlow to use only CPU
tf.config.set_visible_devices([], 'GPU')

# Silence TensorFlow and Python warnings
tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

```

## Our Volatility Measure

The log return is defined as:

$$
\text{Log Return}_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
$$

Where:
- $\ P_t$ is the closing price at time $\ t$,
- $\ P_{t-1}$ is the closing price at time $\ t-1$.

$$\text{Volatility}_t = |\text{Log Return}_t| = | \ln\left(\frac{P_t}{P_{t-1}}\right) |$$


## GARCH(p,q) Model

$p$ = order of the GARCH → how many past variance terms ($\sigma^2$) are used.

$q$ = order of the ARCH → how many past shock/innovation squared terms ($\varepsilon^2$) are used.


### Data


```python
print(eurostoxx50_garch)
```

                       Open         High          Low        Close    Volume
    Date                                                                    
    2010-01-05  3016.639893  3025.600098  3006.469971  3012.360107         0
    2010-01-06  3010.889893  3016.830078  2997.050049  3009.659912         0
    2010-01-07  3000.139893  3013.899902  2979.800049  3007.340088         0
    2010-01-08  3012.969971  3024.580078  2993.729980  3017.850098         0
    2010-01-11  3030.419922  3044.370117  3007.340088  3010.239990         0
    ...                 ...          ...          ...          ...       ...
    2024-12-20  4861.589844  4872.660156  4803.200195  4862.279785  52536500
    2024-12-23  4859.040039  4862.509766  4832.149902  4852.930176  13206800
    2024-12-27  4854.160156  4898.879883  4847.890137  4898.879883  17347300
    2024-12-30  4890.520020  4904.080078  4862.160156  4869.279785  14296800
    2025-01-03  4915.540039  4917.339844  4861.120117  4871.450195  17720300
    
    [3763 rows x 5 columns]
    Missing values per column:
     Open      0
    High      0
    Low       0
    Close     0
    Volume    0
    dtype: int64


    
<img src="/assets/proj/Volatility_files/close.png" alt="Close Image" style="max-width: 100%; height: auto;">



### Log-Returns


    
<img src="/assets/proj/Volatility_files/log_ret.png" alt="Log-Ret" style="max-width: 100%; height: auto;">
    


### Optimal $\text{p}$ and $\text{q}$


```python
p_range_garch = range(1, 6)  # from 1 to 5
q_range_garch = range(1, 6)  # from 1 to 5

best_aic_garch = np.inf
best_bic_garch = np.inf
best_aic_order_garch = None
best_bic_order_garch = None

for p_garch in p_range_garch:
    for q_garch in q_range_garch:
        # Define a GARCH(p, q) model with constant mean and Normal residuals
        model_garch = arch_model(eurostoxx50_garch['Log_Returns'], vol='GARCH', rescale=False,
                                 p=p_garch, q=q_garch,
                                 mean='constant',
                                 dist='normal')
        # Fit the model
        res_garch = model_garch.fit(disp='off')

        # Update best AIC
        if res_garch.aic < best_aic_garch:
            best_aic_garch = res_garch.aic
            best_aic_order_garch = (p_garch, q_garch)

        # Update best BIC
        if res_garch.bic < best_bic_garch:
            best_bic_garch = res_garch.bic
            best_bic_order_garch = (p_garch, q_garch)

print("\n=== (p, q) Selection Results ===")
print(f"Best (p, q) based on AIC: {best_aic_order_garch}, AIC = {best_aic_garch:.3f}")
print(f"Best (p, q) based on BIC: {best_bic_order_garch}, BIC = {best_bic_garch:.3f}")

```

    
    === (p, q) Selection Results ===
    Best (p, q) based on AIC: (1, 1), AIC = -23111.520
    Best (p, q) based on BIC: (1, 1), BIC = -23086.589


### Train and Test Data

    TRAIN SET range: from 2010-01-06 to 2021-11-30
    Number of observations in TRAIN: 2985
    
    TEST SET range: from 2021-12-01 to 2025-01-03
    Number of observations in TEST: 777


### Fit GARCH(1,1) Constant Mean and Normal Distribution on TRAIN SET



    
    === GARCH(1,1) train fit summary ===
                         Constant Mean - GARCH Model Results                      
    ==============================================================================
    Dep. Variable:            Log_Returns   R-squared:                       0.000
    Mean Model:             Constant Mean   Adj. R-squared:                  0.000
    Vol Model:                      GARCH   Log-Likelihood:                9124.03
    Distribution:                  Normal   AIC:                          -18240.1
    Method:            Maximum Likelihood   BIC:                          -18216.1
                                            No. Observations:                 2985
    Date:                Thu, May 08 2025   Df Residuals:                     2984
    Time:                        16:45:18   Df Model:                            1
                                     Mean Model                                 
    ============================================================================
                     coef    std err          t      P>|t|      95.0% Conf. Int.
    ----------------------------------------------------------------------------
    mu         4.9415e-04  1.032e-04      4.789  1.674e-06 [2.919e-04,6.964e-04]
                                  Volatility Model                              
    ============================================================================
                     coef    std err          t      P>|t|      95.0% Conf. Int.
    ----------------------------------------------------------------------------
    omega      3.4725e-06  9.230e-12  3.762e+05      0.000 [3.473e-06,3.473e-06]
    alpha[1]       0.1000  1.907e-02      5.245  1.566e-07   [6.263e-02,  0.137]
    beta[1]        0.8800  1.548e-02     56.848      0.000     [  0.850,  0.910]
    ============================================================================
    
    Covariance estimator: robust


### One-Step Forecast Ahead


```python
full_returns_garch = pd.concat([train_returns_garch, test_returns_garch])
test_index_garch = test_returns_garch.index

def rolling_forecast_1step_garch(train_series_garch, full_series_garch, test_idx_garch):
    """
    Performs rolling day-by-day (1-step ahead) forecasts.
    For each date in test_idx:
      - fits a GARCH(1,1) model with constant mean and normal residuals,
        using data up to the previous day
      - obtains the forecasted variance for that date (out-of-sample)
    Returns a Series with the predicted variance, indexed by test_idx.
    """
    predictions_garch = []

    for date_garch in test_idx_garch:
        # Subset: all data up to the previous day (exclude 'date' itself)
        subset_garch = full_series_garch.loc[:date_garch].iloc[:-1]

        # Define the GARCH(1,1) model with constant mean and normal distribution
        am_garch = arch_model(subset_garch, vol='GARCH', p=1, q=1, mean='constant', dist='normal', rescale=False)

        # Fit the model on data up to the previous day
        res_garch = am_garch.fit(disp='off')

        # 1-step ahead forecast
        fcst_garch = res_garch.forecast(horizon=1)

        # Extract the predicted variance (last row, column 'h.1')
        var_pred_garch = fcst_garch.variance.iloc[-1, 0]
        predictions_garch.append(var_pred_garch)

    # Return a Series with the same index as the test set
    return pd.Series(predictions_garch, index=test_idx_garch, name='Predicted_Variance')

# Obtain the predicted variances using rolling forecast
pred_var_series_garch = rolling_forecast_1step_garch(train_returns_garch, full_returns_garch, test_index_garch)

# Convert variance to volatility (standard deviation)
pred_vol_series_garch = np.sqrt(pred_var_series_garch)

# Realized volatility = absolute value of log-return in the test set
realized_vol_garch = test_returns_garch.abs()

# Insert everything into the test_data for convenience
test_data_garch['Predicted_Vol'] = pred_vol_series_garch
test_data_garch['Realized_Vol']  = realized_vol_garch
```

### GARCH(1,1) Results
    
<img src="/assets/proj/Volatility_files/garch.png" alt="GARCH" style="max-width: 100%; height: auto;">
    


    MAE: 0.0064304569
    MSE: 0.0000658292
    RMSE: 0.0081135219


## Long Short-Term Memory (LSTM) by **Omar Tronelli**

### Data
Additional regressors were included in this model.


            Date         Open         High          Low        Close  Volume  \
    0 2010-01-07  3000.139893  3013.899902  2979.800049  3007.340088       0   
    1 2010-01-08  3012.969971  3024.580078  2993.729980  3017.850098       0   
    2 2010-01-11  3030.419922  3044.370117  3007.340088  3010.239990       0   
    3 2010-01-12  3010.580078  3019.169922  2966.149902  2976.889893       0   
    4 2010-01-13  2967.429932  2986.219971  2964.110107  2978.409912       0   
    
       Log_Returns  Volatility_1d  Log_Trading_Range  Volatility_1d_lag1  \
    0    -0.000771       0.000771           0.011379            0.000897   
    1     0.003489       0.003489           0.010252            0.000771   
    2    -0.002525       0.002525           0.012238            0.003489   
    3    -0.011141       0.011141           0.017717            0.002525   
    4     0.000510       0.000510           0.007432            0.011141   
    
       Log_Trading_Range_lag1  day_0  day_1  day_2  day_3  day_4  
    0                0.006578  False  False  False   True  False  
    1                0.011379  False  False  False  False   True  
    2                0.010252   True  False  False  False  False  
    3                0.012238  False   True  False  False  False  
    4                0.017717  False  False   True  False  False  



```python
# Convert 'Date' to index to facilitate splitting
df_lstm = df_lstm.set_index("Date")

# Split by date
train_df_lstm = df_lstm.loc[:'2021-11-30']
test_df_lstm = df_lstm.loc['2021-12-01':]

```

### Model


```python
# === 0. Set time window (window)
window_size_lstm = 17  # You can change this to 5, 15, 20 etc. to test other options
lr_lstm = 0.0005

features_lstm = [
    "Log_Trading_Range_lag1", "Volatility_1d_lag1", "day_0", "day_1", "day_2", "day_3", "day_4"
]

target_col_lstm = "Volatility_1d"

# === 2. Normalization
scaler_lstm = MinMaxScaler()
scaler_lstm.fit(train_df_lstm[features_lstm])
train_scaled_lstm = pd.DataFrame(scaler_lstm.transform(train_df_lstm[features_lstm]), columns=features_lstm, index=train_df_lstm.index)
test_scaled_lstm = pd.DataFrame(scaler_lstm.transform(test_df_lstm[features_lstm]), columns=features_lstm, index=test_df_lstm.index)

# === 3. Sequence creation
def create_sequences_multifeat_lstm(data_lstm, target_lstm, window_lstm):
    X_lstm, y_lstm = [], []
    for i in range(window_lstm, len(data_lstm)):
        X_lstm.append(data_lstm.iloc[i-window_lstm:i].values)
        y_lstm.append(target_lstm.iloc[i])  # target goes beyond the sequence
    return np.array(X_lstm), np.array(y_lstm)

X_train_lstm, y_train_lstm = create_sequences_multifeat_lstm(train_scaled_lstm, train_df_lstm[target_col_lstm], window_lstm=window_size_lstm)
X_test_lstm, y_test_lstm = create_sequences_multifeat_lstm(test_scaled_lstm, test_df_lstm[target_col_lstm], window_lstm=window_size_lstm)

# === 4. Build simple model
model_lstm = Sequential()
model_lstm.add(LSTM(32, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), dropout=0.2, recurrent_dropout=0.2))
model_lstm.add(Dense(1))

```


```python
def hmae_lstm(y_true_lstm, y_pred_lstm):
    epsilon = 1e-6  # avoid division by zero or very small values
    denominator = tf.maximum(tf.abs(y_true_lstm), epsilon)
    return tf.reduce_mean(tf.abs((y_pred_lstm - y_true_lstm) / denominator))

def hmse_lstm(y_true_lstm, y_pred_lstm):
    epsilon = 1e-6
    denominator = tf.maximum(tf.abs(y_true_lstm), epsilon)
    return tf.reduce_mean(tf.square((y_pred_lstm - y_true_lstm) / denominator))

model_lstm.compile(
    optimizer=Adam(learning_rate=lr_lstm),
    loss="mse",  # or any other loss
    metrics=["mae", hmae_lstm, hmse_lstm]
)

# === 5. Training
early_stop_lstm = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

val_size_lstm = int(len(X_train_lstm) * 0.1)
X_val_lstm = X_train_lstm[-val_size_lstm:]
y_val_lstm = y_train_lstm[-val_size_lstm:]
X_train_lstm = X_train_lstm[:-val_size_lstm]
y_train_lstm = y_train_lstm[:-val_size_lstm]

history_lstm = model_lstm.fit(
    X_train_lstm, y_train_lstm,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_lstm, y_val_lstm),
    callbacks=[early_stop_lstm],
    verbose=0
)
```

### One-Step Ahead Forecast


```python
# === 6. Simulated one-step ahead forecast (real-life scenario)
y_pred_lstm = []
for i in range(window_size_lstm, len(test_scaled_lstm)):
    x_input_lstm = test_scaled_lstm.iloc[i-window_size_lstm:i].values.reshape(1, window_size_lstm, len(features_lstm))
    pred_lstm = model_lstm.predict(x_input_lstm, verbose=0)[0][0]
    y_pred_lstm.append(pred_lstm)

y_test_aligned_lstm = y_test_lstm[:len(y_pred_lstm)]
```

### LSTM Results

    
<img src="/assets/proj/Volatility_files/lstm.png" alt="LSTM" style="max-width: 100%; height: auto;">
    


    
    LSTM - One-step ahead forecast
    MAE:  0.0051580811
    MSE:  0.0000550418
    RMSE: 0.0074190130


## Multilayer Perceptron (MLP) by **Tommaso de Martino**

### Data

Here I used 2 datasets:

- `df_mlp` = the closing price of both the target asset and the predictors

- `eur50_mlp` = the OHLC prices for Euro Stoxx 50, it will be used later to do some features engineering
```
    df_mlp:
                Date  eurostoxx50    EURUSD   EURCHF     EXY     DXY  XDS_CHF  \
    0     2010-01-05  3012.360107  1.436596  1.48460  105.82   77.62    96.70   
    1     2010-01-06  3009.659912  1.440403  1.48000  106.11   77.49    97.31   
    2     2010-01-07  3007.340088  1.431803  1.47970  105.84   77.91    96.67   
    3     2010-01-08  3017.850098  1.441109  1.47470  106.18   77.47    97.65   
    4     2010-01-11  3010.239990  1.451126  1.47530  106.64   77.00    98.43   
    ...          ...          ...       ...      ...     ...     ...      ...   
    3758  2024-12-20  4862.279785  1.036495  0.93172  126.20  107.62   111.90   
    3759  2024-12-23  4852.930176  1.043308  0.93169  126.33  108.04   111.26   
    3760  2024-12-27  4898.879883  1.042318  0.93683  126.67  108.00   110.83   
    3761  2024-12-30  4869.279785  1.042938  0.94052  126.77  108.13   110.60   
    3762  2025-01-03  4871.450195  1.026821  0.93641  125.62  108.95   110.08   
    
          BTP_Yield_10y  BUND_Yield_10y  BTP_BUND_SPREAD        cac40  \
    0             4.095          3.3690           0.7260  4012.909912   
    1             4.116          3.3910           0.7250  4017.669922   
    2             4.089          3.3710           0.7180  4024.800049   
    3             4.100          3.3790           0.7210  4045.139893   
    4             4.054          3.3450           0.7090  4043.090088   
    ...             ...             ...              ...          ...   
    3758          3.447          2.2865           1.1605  7274.479980   
    3759          3.499          2.3270           1.1720  7272.319824   
    3760          3.535          2.3895           1.1455  7355.370117   
    3761          3.522          2.3590           1.1630  7313.560059   
    3762          3.592          2.4245           1.1675  7282.220215   
    
                   dax  ftsemib        sp500      brent        wti         gold  
    0      6031.859863  23556.0  1136.520020  80.589996  81.769997  1118.099976  
    1      6034.330078  23622.0  1137.140015  81.889999  83.180000  1135.900024  
    2      6019.359863  23709.0  1141.689941  81.510002  82.660004  1133.099976  
    3      6037.609863  23811.0  1144.979980  81.370003  82.750000  1138.199951  
    4      6040.500000  23775.0  1146.979980  80.970001  82.519997  1150.699951  
    ...            ...      ...          ...        ...        ...          ...  
    3758  19884.750000  33766.0  5930.850098  72.940002  69.459999  2628.699951  
    3759  19848.769531  33740.0  5974.069824  72.629997  69.239998  2612.300049  
    3760  19984.320312  34161.0  5970.839844  74.169998  70.599998  2617.199951  
    3761  19909.140625  34186.0  5906.939941  74.389999  70.989998  2606.100098  
    3762  19906.080078  34128.0  5942.470215  76.510002  73.959999  2645.000000  
    
    [3763 rows x 17 columns]
    eur50_mlp:
                Date         Open         High          Low        Close    Volume
    0     2010-01-05  3016.639893  3025.600098  3006.469971  3012.360107         0
    1     2010-01-06  3010.889893  3016.830078  2997.050049  3009.659912         0
    2     2010-01-07  3000.139893  3013.899902  2979.800049  3007.340088         0
    3     2010-01-08  3012.969971  3024.580078  2993.729980  3017.850098         0
    4     2010-01-11  3030.419922  3044.370117  3007.340088  3010.239990         0
    ...          ...          ...          ...          ...          ...       ...
    3758  2024-12-20  4861.589844  4872.660156  4803.200195  4862.279785  52536500
    3759  2024-12-23  4859.040039  4862.509766  4832.149902  4852.930176  13206800
    3760  2024-12-27  4854.160156  4898.879883  4847.890137  4898.879883  17347300
    3761  2024-12-30  4890.520020  4904.080078  4862.160156  4869.279785  14296800
    3762  2025-01-03  4915.540039  4917.339844  4861.120117  4871.450195  17720300
    
    [3763 rows x 6 columns]
```

### Absolute Log-Returns as Daily Volatility


```python
# Log returns computation
df_mlp['log_return_eur50'] = np.log(df_mlp['eurostoxx50'] / df_mlp['eurostoxx50'].shift(1))

# Daily volatility computation
df_mlp['vol_eur50'] = df_mlp['log_return_eur50'].abs()

# Creation of the variable 'vol'
vol_mlp = df_mlp[['Date', 'vol_eur50']].copy()

```

### Choosing the Optimal Number of Lags for Volatility


```python
vol_mlp['Date'] = pd.to_datetime(vol_mlp['Date'])
vol_mlp.set_index('Date', inplace=True)

# Maximum lag parameter to test
P_max_mlp = 30

aic_vals_mlp = []
bic_vals_mlp = []

for p_mlp in range(1, P_max_mlp + 1):
    # 1) Create the matrix of regressors with p lags
    df_lags_mlp = pd.concat(
        [vol_mlp['vol_eur50'].shift(i) for i in range(1, p_mlp + 1)],
        axis=1
    )
    df_lags_mlp.columns = [f'vol_lag{i}' for i in range(1, p_mlp + 1)]

    # Join with the target (drop NaN)
    data_mlp = pd.concat([vol_mlp['vol_eur50'], df_lags_mlp], axis=1).dropna()
    y_mlp = data_mlp['vol_eur50']
    X_mlp = sm.add_constant(data_mlp.drop(columns='vol_eur50'))

    # 2) Estimate OLS
    model_mlp = sm.OLS(y_mlp, X_mlp).fit()

    # 3) Extract AIC and BIC
    aic_vals_mlp.append(model_mlp.aic)
    bic_vals_mlp.append(model_mlp.bic)

# Create a DataFrame to store results
crit_mlp = pd.DataFrame({
    'p': np.arange(1, P_max_mlp + 1),
    'AIC': aic_vals_mlp,
    'BIC': bic_vals_mlp
})

print(crit_mlp)

# 4) Optimal lag
best_p_aic_mlp = crit_mlp.loc[crit_mlp['AIC'].idxmin(), 'p']
best_p_bic_mlp = crit_mlp.loc[crit_mlp['BIC'].idxmin(), 'p']
print(f"p* (AIC) = {best_p_aic_mlp},   p* (BIC) = {best_p_bic_mlp}")

```

         p           AIC           BIC
    0    1 -24698.451879 -24685.986999
    1    2 -24843.250779 -24824.554256
    2    3 -24988.678176 -24963.750543
    3    4 -25046.216967 -25015.058757
    4    5 -25051.693611 -25014.305355
    5    6 -25086.851282 -25043.233514
    6    7 -25088.074963 -25038.228215
    7    8 -25108.182050 -25052.106855
    8    9 -25113.158244 -25050.855136
    9   10 -25110.691492 -25042.161004
    10  11 -25107.091561 -25032.334228
    11  12 -25103.200034 -25022.216389
    12  13 -25093.556848 -25006.347426
    13  14 -25084.136774 -24990.702109
    14  15 -25076.299100 -24976.639727
    15  16 -25067.078175 -24961.194629
    16  17 -25059.752118 -24947.644934
    17  18 -25051.900097 -24933.569810
    18  19 -25042.776205 -24918.223351
    19  20 -25033.445852 -24902.670967
    20  21 -25024.756271 -24887.759889
    21  22 -25027.268577 -24884.051236
    22  23 -25026.494638 -24877.056874
    23  24 -25024.244340 -24868.586690
    24  25 -25017.375051 -24855.498051
    25  26 -25007.811241 -24839.715430
    26  27 -24998.999698 -24824.685611
    27  28 -24990.444466 -24809.912642
    28  29 -24981.598888 -24794.849863
    29  30 -24978.215862 -24785.250176
    p* (AIC) = 9,   p* (BIC) = 8



```python
vol_mlp = vol_mlp.reset_index()
vol_mlp['Date'] = vol_mlp['Date'].dt.strftime('%Y-%m-%d')  # now it is an object

```

### Features Engineering

Now I build the variable `features` which will contain all the necessary features (or predictors)


```python
# Columns to keep
columns_to_keep_mlp = [col for col in df_mlp.columns if col not in ['eurostoxx50', 'log_return_eur50', 'vol_eur50']]

# Creation of 'features' variable
features_mlp = df_mlp[columns_to_keep_mlp].copy()
```

```python
# 1. Intraday volatility proxy: use the log-range High/Low
eur50_mlp["range_log"] = np.log(eur50_mlp["High"] / eur50_mlp["Low"])

# 2. Open-Close spread percentage (absolute value, only magnitude)
eur50_mlp["open_close_pct"] = (eur50_mlp["Open"] - eur50_mlp["Close"]).abs() / eur50_mlp["Close"]

# 3. Keep only the two new columns + Date
spreads_mlp = eur50_mlp[["Date", "range_log", "open_close_pct"]]

# 4. Merge with the 'features' DataFrame
features_mlp = features_mlp.merge(spreads_mlp, on="Date", how="left")
```


```python
# 1. Create the variable ret_sign (−1, 0, +1; 0 for initial NaNs)
df_mlp['ret_sign'] = np.sign(df_mlp['log_return_eur50']).fillna(0).astype(np.int8)

# 2. Add it to the 'features' DataFrame, aligning by index/Date
features_mlp = features_mlp.assign(ret_sign=df_mlp['ret_sign'])
```


```python
# 1. Calculate rolling std over windows 5 and 10
for w_mlp in (5, 10):
    col_mlp = f"vol_std_{w_mlp}"
    vol_mlp[col_mlp] = (
        vol_mlp["vol_eur50"]
        .rolling(window=w_mlp, min_periods=w_mlp)
        .std()
    )

# 2. Merge the rolling stds into features
vol_stds_mlp = vol_mlp[["Date", "vol_std_5", "vol_std_10"]]
features_mlp = features_mlp.merge(vol_stds_mlp, on="Date", how="left")
```


```python
# 1) log-return of EuroStoxx 50
features_mlp['log_return_eur50'] = df_mlp['log_return_eur50']
```

### Lagging

We want to predict volatility at time `t` and to do so we must use only the informations up to `t-1` since otherwise we will be using variables that we don't really know at time `t` (it will be like cheating, i.e. using infos not available at `t`).

To do so we use the `lag` of the variable, i.e. its previous observation


```python
# New variable to store lagged features
features_lag_mlp = features_mlp[['Date']].copy()
```


```python
# Desired lags
lags_mlp = (1, 2, 3)

# List to accumulate lagged series
parts_mlp = [features_mlp[["Date"]]]  # always keep the Date

for col_mlp in features_mlp.columns:
    if col_mlp in ["Date"]:  # NO lag for dummies or Date
        continue
    for k_mlp in lags_mlp:  # lags 1-3
        parts_mlp.append(features_mlp[col_mlp].shift(k_mlp).rename(f"{col_mlp}_lag{k_mlp}"))

# Combine all lagged features
features_lag_mlp = pd.concat(parts_mlp, axis=1)
```


```python
# Lag of vol_eur50 (target) 1-9
for k_mlp in (1, 2, 3, 4, 5, 6, 7, 8, 9):
    vol_mlp[f"vol_eur50_lag{k_mlp}"] = vol_mlp["vol_eur50"].shift(k_mlp)

vol_lags_mlp = vol_mlp[["Date", "vol_eur50_lag1", "vol_eur50_lag2", "vol_eur50_lag3", "vol_eur50_lag4",
                        "vol_eur50_lag5", "vol_eur50_lag6", "vol_eur50_lag7", "vol_eur50_lag8", "vol_eur50_lag9"]]

# Merge with other lagged features
features_lag_mlp = features_lag_mlp.merge(vol_lags_mlp, on="Date", how="left")
```


```python
# (a) Create a boolean mask for rows where all lagged features are not null
mask_mlp = features_lag_mlp.notnull().all(axis=1)
# mask_mlp[i] == True only if ALL columns in features_lag_mlp.iloc[i] are NOT NaN

# (b) Apply the mask and reset index
features_lag_mlp = features_lag_mlp[mask_mlp].reset_index(drop=True)
```

> Ultimately, the model includes 72 predictors, each appropriately lagged to prevent look-ahead bias.

### Adjusting the `vol_mlp` variable as well

In computing some new features we lost come initial rows in the dataset, se we need to remove them also in our `Y`, i.e. the variable `vol_mlp`


```python
vol_mlp = vol_mlp[['Date', 'vol_eur50']].copy()
```


```python
# (c) Apply the same mask to vol and reset the index
vol_mlp = vol_mlp[mask_mlp].reset_index(drop=True)
```


```python
print(vol_mlp)
```

                Date  vol_eur50
    0     2010-01-22   0.009281
    1     2010-01-25   0.010453
    2     2010-01-26   0.007255
    3     2010-01-27   0.014369
    4     2010-01-28   0.018126
    ...          ...        ...
    3745  2024-12-20   0.003433
    3746  2024-12-23   0.001925
    3747  2024-12-27   0.009424
    3748  2024-12-30   0.006061
    3749  2025-01-03   0.000446
    
    [3750 rows x 2 columns]


### Train and Test

Now we must split our `Y` (the `vol_mlp` variable) and the `X` matrix (the `features_lag_mlp` variable) into `train_mlp` and `test_mlp` data.
Obviously the split is the same as the GARCH.


```python
# Y
Y_mlp = vol_mlp.copy()
Y_mlp.set_index('Date', inplace=True)  # date as index
Y_mlp.index = pd.to_datetime(Y_mlp.index)  # converting the index to datetime

Y_train_mlp = Y_mlp[:'2021-11-30']  # Y_train
Y_test_mlp = Y_mlp['2021-12-01':]   # Y_test

# X
X_mlp = features_lag_mlp.copy()
X_mlp.set_index('Date', inplace=True)  # date as index
X_mlp.index = pd.to_datetime(X_mlp.index)  # converting the index to datetime

X_train_mlp = X_mlp[:'2021-11-30']  # X_train
X_test_mlp = X_mlp['2021-12-01':]   # X_test
```

### Variables Selection with Lasso


```python
# 1) Time series splitter as before
tscv_mlp = TimeSeriesSplit(n_splits=5)

# 2) Pipeline with scaler + LassoCV
pipe_mlp = make_pipeline(
    StandardScaler(),
    LassoCV(
        cv=tscv_mlp,
        n_alphas=100,      # explore more α values
        max_iter=100000,    # more iterations for convergence
        random_state=SEED
    )
)

# 3) Fit the pipeline
pipe_mlp.fit(X_train_mlp, Y_train_mlp['vol_eur50'])

# 4) Extract the Lasso model from the pipeline
lasso_mlp = pipe_mlp.named_steps['lassocv']

# 5) Mask of non-zero features
mask_mlp = lasso_mlp.coef_ != 0
selected_features_mlp = X_train_mlp.columns[mask_mlp].tolist()

print(f"Selected features ({len(selected_features_mlp)}):\n", selected_features_mlp)
```

    Selected features (14):
     ['cac40_lag1', 'range_log_lag1', 'range_log_lag2', 'range_log_lag3', 'open_close_pct_lag1', 'vol_std_10_lag3', 'log_return_eur50_lag1', 'log_return_eur50_lag2', 'log_return_eur50_lag3', 'vol_eur50_lag3', 'vol_eur50_lag4', 'vol_eur50_lag6', 'vol_eur50_lag8', 'vol_eur50_lag9']



### Scaling


```python
# ==== SCALING ====

# Initialize scalers for X and Y
scaler_X_mlp = MinMaxScaler()
scaler_Y_mlp = MinMaxScaler()

# Fit & transform on the training set, transform on the test set
X_train_scaled_mlp = scaler_X_mlp.fit_transform(X_train_mlp)
X_test_scaled_mlp = scaler_X_mlp.transform(X_test_mlp)

Y_train_scaled_mlp = scaler_Y_mlp.fit_transform(Y_train_mlp)
Y_test_scaled_mlp = scaler_Y_mlp.transform(Y_test_mlp)

```

### MLP

> **Theorem**: A neural network with a single hidden layer with any (continuous and non-constant) activation function can approximate any continuous function on a closed and bounded interval, provided it has a sufficient number of neurons in its hidden layer.

### Optimal Neurons

To choose the optimal number of hidden nodes, the training MSE was evaluated as the number of nodes increased from 1 up to the number of input features.
    
    → Optimal number of neurons = 1  (train-MSE = 1.4373e-02)


    
<img src="/assets/proj/Volatility_files/mse_neurons.png" alt="opt_nodes" style="max-width: 100%; height: auto;">


### Model

The model uses a **Bayesian loss function** that combines two components:

1. **Mean Squared Error (MSE)** – This is the standard loss for regression tasks. It measures the average squared difference between predicted values and actual targets. It ensures the model fits the training data well.

2. **KL Divergence (regularization term)** – This term acts as a Bayesian regularizer. For each layer with weights, the model assumes:
   - a **prior distribution** over weights: standard normal $ \mathcal{N}(0, 1) $,
   - and a **posterior distribution**: normal with mean equal to the actual weights and a fixed standard deviation (0.1).
   
   The KL divergence between the posterior and prior measures how much the learned weights deviate from the prior belief. Minimizing it prevents overfitting by penalizing overly complex weight configurations.

The **total loss** is:

$$
\text{Loss} = \text{MSE} + \frac{1}{N} \cdot \text{KL divergence}
$$

where $N$ is the number of training samples, used here as a scaling factor ($\lambda = 1/N$) to balance the regularization term.

This loss encourages the model not only to fit the data but also to keep weights close to a prior distribution.


```python
# Build the final model
inp_mlp = Input(shape=(input_dim_mlp,))
x_mlp = Dense(best_h_mlp, activation="relu", kernel_initializer="glorot_uniform")(inp_mlp)
out_mlp = Dense(1, activation="linear")(x_mlp)
model_mlp = Model(inputs=inp_mlp, outputs=out_mlp)

# Bayesian loss function with regularization
def bayesian_loss_mlp(y_true_mlp, y_pred_mlp):
    # Error term (MSE)
    mse_loss_mlp = tf.reduce_mean(tf.square(y_true_mlp - y_pred_mlp))
    
    # Regularization term (KL divergence)
    kl_loss_mlp = 0
    for layer_mlp in model_mlp.layers:
        if hasattr(layer_mlp, "kernel"):  # Check if the layer has weights
            kernel_mlp = layer_mlp.kernel
            prior_mlp = tfp.distributions.Normal(0, scale=1)  # Prior: Normal(0, 1)
            posterior_mlp = tfp.distributions.Normal(kernel_mlp, scale=0.1)  # Posterior: Normal(kernel, 0.1)
            kl_loss_mlp += tf.reduce_sum(tfp.distributions.kl_divergence(posterior_mlp, prior_mlp))
    
    # Total loss with regularization term
    return mse_loss_mlp + (1 / N_mlp) * kl_loss_mlp  # Using λ = 1/N

# Compile the model
model_mlp.compile(optimizer=Adam(), loss=bayesian_loss_mlp)

# Train the final model
hist_mlp = model_mlp.fit(
    X_train_values_mlp, Y_train_values_mlp,
    epochs=500,      # Use the same number of epochs
    batch_size=32,
    verbose=0
)
```

### One-Step Ahead Forecast


```python
# ════════════════════════════════════════════════════════════════
# 2. Static OSA loop on the test period (no fine-tuning)
# ════════════════════════════════════════════════════════════════
y_pred_scaled_mlp, y_true_scaled_mlp = [], []

for i_mlp in range(X_test_scaled_mlp.shape[0]):
    x_i_mlp = X_test_scaled_mlp[i_mlp:i_mlp+1]          # Features known up to t-1
    y_i_true_mlp = Y_test_scaled_mlp[i_mlp]             # True target (scaled 0-1)
    y_i_pred_mlp = model_mlp.predict(x_i_mlp, verbose=0)[0, 0]  # Model prediction

    y_pred_scaled_mlp.append(y_i_pred_mlp)
    y_true_scaled_mlp.append(y_i_true_mlp)

y_pred_scaled_mlp = np.array(y_pred_scaled_mlp)
y_true_scaled_mlp = np.array(y_true_scaled_mlp)

# ════════════════════════════════════════════════════════════════
# 3. Invert scaling to return to the original scale
# ════════════════════════════════════════════════════════════════
y_pred_mlp = scaler_Y_mlp.inverse_transform(y_pred_scaled_mlp.reshape(-1, 1)).ravel()
y_true_mlp = scaler_Y_mlp.inverse_transform(y_true_scaled_mlp.reshape(-1, 1)).ravel()
```

### MLP Results


    
<img src="/assets/proj/Volatility_files/mlp.png" alt="mlp" style="max-width: 100%; height: auto;">
    


    MAE: 0.0050319649
    MSE: 0.0000540227
    RMSE: 0.0073500164


## Random Forest by **Valentina Sanna**

### Raw Data



            Date  EUROSTOXX50        SP500          DAX    EURUSD         GOLD  \
    0 2010-01-05  3012.360107  1136.520020  6031.859863  1.436596  1118.099976   
    1 2010-01-06  3009.659912  1137.140015  6034.330078  1.440403  1135.900024   
    2 2010-01-07  3007.340088  1141.689941  6019.359863  1.431803  1133.099976   
    3 2010-01-08  3017.850098  1144.979980  6037.609863  1.441109  1138.199951   
    4 2010-01-11  3010.239990  1146.979980  6040.500000  1.451126  1150.699951   
    
        BRENTOIL  US10Y_Yield  US10Y_Change  BTP10Y  BUND10Y  BTP_BUND_SPREAD  \
    0  80.589996        3.763       -0.0157   4.095    3.369            0.726   
    1  81.889999        3.829        0.0175   4.116    3.391            0.725   
    2  81.510002        3.827       -0.0005   4.089    3.371            0.718   
    3  81.370003        3.836        0.0024   4.100    3.379            0.721   
    4  80.970001        3.824       -0.0031   4.054    3.345            0.709   
    
       ECB_POLICYRATE  
    0             1.0  
    1             1.0  
    2             1.0  
    3             1.0  
    4             1.0  


### Target Variable


```python
df_rf['LOG_RET_EUROSTOXX50'] = np.log(df_rf['EUROSTOXX50'] / df_rf['EUROSTOXX50'].shift(1))
df_rf['REALIZED_VOLATILITY'] = df_rf['LOG_RET_EUROSTOXX50'].abs()
```

### Feature Engineering 

#### EUROSTOXX50


```python
# Memory of the *target* itself
target_ret_rf = df_rf['LOG_RET_EUROSTOXX50']

# --- Target memory: 5- and 10-day lags of log-returns -----------------------
for w_rf in (5, 10):
    df_rf[f'ROLL_STD_{w_rf}D_EUROSTOXX50'] = target_ret_rf.rolling(w_rf).std()
    df_rf[f'MOV_AVE_{w_rf}D_EUROSTOXX50']  = df_rf['REALIZED_VOLATILITY'].rolling(w_rf).mean()
    df_rf[f'VOL_STD_{w_rf}D_EUROSTOXX50']  = df_rf['REALIZED_VOLATILITY'].rolling(w_rf).std()

```

#### Macro-Asset


```python
macro_assets_rf = ['SP500', 'DAX', 'EURUSD', 'GOLD', 'BRENTOIL']
logret_cols_rf  = []

for name_rf in macro_assets_rf:
    lr_rf = np.log(df_rf[name_rf] / df_rf[name_rf].shift(1))
    col_lr_rf = f'LOGRET_{name_rf}'
    df_rf[col_lr_rf] = lr_rf
    logret_cols_rf.append(col_lr_rf)

    for w_rf in (5, 10):
        df_rf[f'ROLL_STD_{w_rf}D_{name_rf}'] = lr_rf.rolling(w_rf).std()
        df_rf[f'MOV_AVE_{w_rf}D_{name_rf}']  = lr_rf.rolling(w_rf).mean()
        df_rf[f'VOL_STD_{w_rf}D_{name_rf}']  = lr_rf.rolling(w_rf).std()

```

### Price lags & extra memory


```python
for lag_rf in (1, 2, 3):
    # --- Eurostoxx price lags ------------------------------------
    df_rf[f'LAG{lag_rf}_EUROSTOXX50'] = df_rf['EUROSTOXX50'].shift(lag_rf)
    # --- Log-return macro asset lags ------------------------------
    for col_rf in logret_cols_rf:
        df_rf[f'{col_rf}_L{lag_rf}'] = df_rf[col_rf].shift(lag_rf)

print(f"Total engineered columns: {df_rf.shape[1] - 2:d}")

```

    Total engineered columns: 72


### Build X and y


```python
y_rf = df_rf['REALIZED_VOLATILITY']

cols_to_drop_rf = ['Date', 'EUROSTOXX50', 'REALIZED_VOLATILITY', 'LOG_RET_EUROSTOXX50']
X_rf = df_rf.drop(columns=cols_to_drop_rf)

# --- Identify already lagged features --------------------------
lag_flags_rf = X_rf.columns.str.startswith('LAG') | X_rf.columns.str.contains('_L[123]$')
cols_raw_rf  = X_rf.columns[~lag_flags_rf]          # to be shifted by +1

# --- Apply 1-day shift ONLY to raw features ---------------------
X_lagged_rf = X_rf.copy()
X_lagged_rf[cols_raw_rf] = X_lagged_rf[cols_raw_rf].shift(1)

# --- Remove rows with NaN in X or y ---------------------------
mask_valid_rf = ~(X_lagged_rf.isna().any(axis=1) | y_rf.isna())
X_lagged_rf = X_lagged_rf.loc[mask_valid_rf]
y_rf        = y_rf.loc[mask_valid_rf]

# --- Re-index with the date -----------------------------------
dates_rf = df_rf.loc[mask_valid_rf, 'Date']
X_lagged_rf.index = dates_rf
y_rf.index        = dates_rf

```

### Train and Test split

Obviously the split is the same as the GARCH.

```python
TRAIN_END_rf  = '2021-11-30'
TEST_START_rf = '2021-12-01'

train_X_rf = X_lagged_rf.loc[:TRAIN_END_rf]
train_y_rf = y_rf.loc[:TRAIN_END_rf]
test_X_rf  = X_lagged_rf.loc[TEST_START_rf:]
test_y_rf  = y_rf.loc[TEST_START_rf:]
```


> ### READ ME
> The hyperparameter and features optimization was accelerated using `n_jobs = -1`, which enables parallel processing across all available CPU cores.
> While this significantly reduces computation time, parallel execution can lead to slight variability between runs because different core/thread scheduling can affect the order and precision of floating-point operations. 
> This variability may also differ across machines or operating systems, regardless the `SEED` set at the beginning.
> To ensure reproducibility, I re-trained the final model using `n_jobs = 1` and applied the exact hyperparameters and features identified during the best optimization.

### Cross Validated permutation importance


```python
tscv_rf = TimeSeriesSplit(n_splits=5, gap=5)  # gap avoids overlap
importance_accum_rf = pd.Series(0.0, index=train_X_rf.columns)

t0_rf = time.time()
for fold_rf, (idx_tr_rf, idx_val_rf) in enumerate(tscv_rf.split(train_X_rf), 1):
    X_tr_rf, X_val_rf = train_X_rf.iloc[idx_tr_rf], train_X_rf.iloc[idx_val_rf]
    y_tr_rf, y_val_rf = train_y_rf.iloc[idx_tr_rf], train_y_rf.iloc[idx_val_rf]

    rf_fold_rf = RandomForestRegressor(
        n_estimators=300,
        random_state=SEED,
        n_jobs=1
    ).fit(X_tr_rf, y_tr_rf)

    with threadpool_limits(1):  # no nested threads
        imp_rf = permutation_importance(
            rf_fold_rf, X_val_rf, y_val_rf,
            n_repeats=10,
            random_state=SEED,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

    importance_accum_rf += pd.Series(imp_rf.importances_mean, index=X_val_rf.columns)
    ping_rf(fold_rf, tscv_rf.n_splits, t0_rf, every=1, msg='perm‑imp folds')

importances_rf = (importance_accum_rf / tscv_rf.n_splits).sort_values(ascending=False)

```


### ELBOW curve


```python
mse_curve_cv_rf = []

t0_rf = time.time()
for k_rf in range(1, len(importances_rf) + 1):
    subset_rf = importances_rf.index[:k_rf]  # top k features by perm‑imp

    # ---------- CV MSE with the same tscv (gap=5) ---------------
    cv_mse_rf = []
    for idx_tr_rf, idx_val_rf in tscv_rf.split(train_X_rf):
        X_tr_rf, X_val_rf = train_X_rf.iloc[idx_tr_rf][subset_rf], train_X_rf.iloc[idx_val_rf][subset_rf]
        y_tr_rf, y_val_rf = train_y_rf.iloc[idx_tr_rf], train_y_rf.iloc[idx_val_rf]

        rf_tmp_rf = RandomForestRegressor(
            n_estimators=300, random_state=SEED, n_jobs=-1
        ).fit(X_tr_rf, y_tr_rf)

        cv_mse_rf.append(mean_squared_error(y_val_rf, rf_tmp_rf.predict(X_val_rf)))

    mse_curve_cv_rf.append(np.mean(cv_mse_rf))
    ping_rf(k_rf, len(importances_rf), t0_rf, every=10, msg='elbow‑CV')

```
 
<img src="/assets/proj/Volatility_files/elbow.png" alt="elbow" style="max-width: 100%; height: auto;">
    



```python
# ---------------------------------------------------------
# arg-min on the MSE curve (CV-safe)
# ---------------------------------------------------------
k_best_rf  = int(np.argmin(mse_curve_cv_rf) + 1)  # +1 because Python indexing starts at 0
mse_min_rf = float(min(mse_curve_cv_rf))

print(f"Minimum CV‑MSE = {mse_min_rf:.6f} reached at k = {k_best_rf} features")

# The top k_best features from the permutation importance ranking
top_k_features_rf = importances_rf.index[:k_best_rf].tolist()
```

    Minimum CV‑MSE = 0.000087 reached at k = 15 features



```python
print(top_k_features_rf)
```

    ['MOV_AVE_10D_EUROSTOXX50', 'ROLL_STD_10D_SP500', 'LAG1_EUROSTOXX50', 'MOV_AVE_5D_DAX', 'VOL_STD_10D_SP500', 'BRENTOIL', 'VOL_STD_10D_EUROSTOXX50', 'MOV_AVE_10D_DAX', 'MOV_AVE_5D_SP500', 'MOV_AVE_10D_SP500', 'LOGRET_SP500_L2', 'LOGRET_SP500_L1', 'LOGRET_BRENTOIL_L3', 'LOGRET_SP500', 'ROLL_STD_10D_EUROSTOXX50']


### Hyperparameter Tuning


```python
N_TRIALS_rf  = 200
LOG_EVERY_rf = 50

def objective_rf(trial_rf):
    params_rf = {
        "n_estimators":      trial_rf.suggest_int("n_estimators", 300, 900, step=100),
        "max_depth":         trial_rf.suggest_categorical("max_depth",
                                                         [None, 4, 6, 8, 12, 16, 20]),
        "min_samples_split": trial_rf.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial_rf.suggest_int("min_samples_leaf", 1, 15),
        "max_features":      trial_rf.suggest_float("max_features", 0.10, 1.0),
        "bootstrap":         trial_rf.suggest_categorical("bootstrap", [True, False]),
        "random_state":      SEED,
        "n_jobs":            - 1,
    }

    cv_mse_rf = []
    for tr_idx_rf, val_idx_rf in tscv_rf.split(train_X_rf):
        X_tr_rf = train_X_rf.iloc[tr_idx_rf][top_k_features_rf]
        X_val_rf = train_X_rf.iloc[val_idx_rf][top_k_features_rf]
        y_tr_rf = train_y_rf.iloc[tr_idx_rf]
        y_val_rf = train_y_rf.iloc[val_idx_rf]

        model_rf = RandomForestRegressor(**params_rf).fit(X_tr_rf, y_tr_rf)
        cv_mse_rf.append(mean_squared_error(y_val_rf, model_rf.predict(X_val_rf)))

    return float(np.mean(cv_mse_rf))

def progress_cb_rf(study_rf, trial_rf):
    done_rf = trial_rf.number + 1
    if done_rf == 1 or done_rf % LOG_EVERY_rf == 0 or done_rf == N_TRIALS_rf:
        best_rf = study_rf.best_trial
        elapsed_rf = time.time() - t0_rf
        eta_rf = elapsed_rf / done_rf * (N_TRIALS_rf - done_rf)
        stamp_rf = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[I {stamp_rf}] trial {done_rf:4}/{N_TRIALS_rf}  "
              f"value {trial_rf.value:.6f}  best {best_rf.value:.6f}  "
              f"ETA {eta_hms_rf(eta_rf)}", file=sys.stderr, flush=True)

study_rf = optuna.create_study(direction="minimize",
                               sampler=TPESampler(seed=SEED))

t0_rf = time.time()
study_rf.optimize(objective_rf,
                  n_trials=N_TRIALS_rf,
                  callbacks=[progress_cb_rf],
                  show_progress_bar=False,
                  n_jobs=1)  # one trial at a time to avoid race conditions

best_params_rf = {**study_rf.best_params,
                  "random_state": SEED,
                  "n_jobs": - 1}
print("\nBest MSE (CV):", study_rf.best_value, "\nparams:", best_params_rf)

```
 
    Best MSE (CV): 6.987525013856155e-05 
    params: {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 18, 'min_samples_leaf': 8, 'max_features': 0.1912960433663199, 'bootstrap': True, 'random_state': 42, 'n_jobs': -1}


### Best Optimization Model

Here I use the the exact hyperparameters and features identified during the best optimization applying `n_jobs = 1` ensuring reproducibility


```python
final_features_rf = [
    'MOV_AVE_10D_EUROSTOXX50', 'ROLL_STD_10D_SP500', 'LAG1_EUROSTOXX50',
    'MOV_AVE_5D_DAX', 'VOL_STD_10D_SP500', 'BRENTOIL', 'MOV_AVE_10D_DAX',
    'MOV_AVE_10D_EUROSTOXX50', 'LOGRET_SP500_L2', 'MOV_AVE_5D_SP500',
    'MOV_AVE_10D_SP500', 'LOGRET_SP500', 'ROLL_STD_5D_EUROSTOXX50',
    'LOGRET_SP500_L1', 'LOGRET_BRENTOIL_L3', 'EURUSD', 'MOV_AVE_5D_EUROSTOXX50',
    'ROLL_STD_10D_EUROSTOXX50', 'VOL_STD_5D_EUROSTOXX50', 'ROLL_STD_10D_DAX',
    'LOGRET_DAX_L2', 'LOGRET_DAX_L1', 'ROLL_STD_5D_DAX', 'VOL_STD_10D_DAX',
    'DAX', 'LOGRET_DAX_L3', 'VOL_STD_5D_DAX', 'LOGRET_BRENTOIL',
    'VOL_STD_10D_BRENTOIL', 'MOV_AVE_5D_GOLD', 'ROLL_STD_5D_BRENTOIL'
]

final_params_rf = {
    'n_estimators': 800,
    'max_depth': 8,
    'min_samples_split': 12,
    'min_samples_leaf': 15,
    'max_features': 0.1166744064508401,
    'bootstrap': True,
    'random_state': SEED,
    'n_jobs': 1   # for determinism
}
```

### Best Optimization One-Step Ahead Forecast


```python
final_preds_rf, final_truth_rf = [], []
t0_rf = time.time()
dates_test_rf = test_X_rf.index

for i_rf, pred_date_rf in enumerate(dates_test_rf, 1):
    X_tr_final_rf = X_lagged_rf.loc[:pred_date_rf - pd.Timedelta(days=1), final_features_rf]
    y_tr_final_rf = y_rf.loc[:pred_date_rf - pd.Timedelta(days=1)]
    X_pd_final_rf = test_X_rf.loc[[pred_date_rf], final_features_rf]

    final_model_rf = RandomForestRegressor(**final_params_rf).fit(X_tr_final_rf, y_tr_final_rf)

    final_preds_rf.append(final_model_rf.predict(X_pd_final_rf)[0])
    final_truth_rf.append(test_y_rf.loc[pred_date_rf])

    ping_rf(i_rf, len(dates_test_rf), t0_rf, every=100, msg='final rolling forecast')

# Series with the final predictions
final_pred_series_rf = pd.Series(final_preds_rf, index=dates_test_rf)
```

### Best Optimization RF Results



    
<img src="/assets/proj/Volatility_files/rf.png" alt="rf" style="max-width: 100%; height: auto;">
    


    MAE: 0.0052309866
    MSE: 0.0000530746
    RMSE: 0.0072852287


## Conclusion
<img src="/assets/proj/Volatility_files/comparison.png" alt="comparison" style="max-width: 100%; height: auto;">

> The MLP is the model with the lowest prediction errors, although all machine learning methods outperform the GARCH benchmark.