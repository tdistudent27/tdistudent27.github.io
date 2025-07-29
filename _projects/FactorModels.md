---
layout: page
title: Factor Models
description: Using Factor Models to Predict Macro Variables
importance: 1
category: Academic
nav: false
---

## **Factor Models to Predict Macroeconomics Variables**

**Replication of "FRED-MD: A Monthly Database for Macroeconomic Research"**  
**Academic Year 2024/2025**  
**Tommaso de Martino**

## Libraries


```python
import pandas as pd
import numpy as np
import requests
import io
import certifi
from scipy import linalg
from numpy.linalg import svd
import matplotlib.pyplot as plt
from IPython.display import display, HTML, Image
import re
import statsmodels.api as sm
from itertools import product
import time
```

## Data

The dataset used in this project was originally downloaded from the **FRED-MD Monthly Databases for Macroeconomic Research**, available on the Federal Reserve Bank of St. Louis website [here](https://www.stlouisfed.org/research/economists/mccracken/fred-databases).

To ensure reproducibility and consistent access to the data, a copy of the CSV file has been uploaded to a public GitHub [repository](https://github.com/tdistudent27/fred-md-data).

The corresponding file on the FRED-MD website is named **2025-05**, reflecting the dataset version used in this project.

```python
print(df)
```

            sasdate        RPI  W875RX1  DPCERA3M086SBEA     CMRMTSPLx  \
    0    Transform:      5.000      5.0            5.000  5.000000e+00   
    1      1/1/1959   2583.560   2426.0           15.188  2.766768e+05   
    2      2/1/1959   2593.596   2434.8           15.346  2.787140e+05   
    3      3/1/1959   2610.396   2452.7           15.491  2.777753e+05   
    4      4/1/1959   2627.446   2470.0           15.435  2.833627e+05   
    ..          ...        ...      ...              ...           ...   
    791   11/1/2024  20091.169  16376.8          122.396  1.545040e+06   
    792   12/1/2024  20101.629  16387.7          123.077  1.558008e+06   
    793    1/1/2025  20148.969  16391.2          122.614  1.543178e+06   
    794    2/1/2025  20209.351  16389.5          122.742  1.556553e+06   
    795    3/1/2025  20311.260  16500.4          123.601           NaN   
    
              RETAILx    INDPRO   IPFPNSS   IPFINAL   IPCONGD  ...  \
    0         5.00000    5.0000    5.0000    5.0000    5.0000  ...   
    1     17689.23968   21.9616   23.3868   22.2620   31.6664  ...   
    2     17819.01912   22.3917   23.7024   22.4549   31.8987  ...   
    3     17967.91336   22.7142   23.8459   22.5651   31.8987  ...   
    4     17978.97983   23.1981   24.1903   22.8957   32.4019  ...   
    ..            ...       ...       ...       ...       ...  ...   
    791  712145.00000  101.9619   99.3808   98.8609  100.8691  ...   
    792  717662.00000  103.1177  100.4976   99.9719  101.6868  ...   
    793  711461.00000  103.3418  101.0766  100.6319  102.1879  ...   
    794  711680.00000  104.2202  101.8233  101.4377  102.7245  ...   
    795  722025.00000  103.8892  101.6665  101.1465  101.7332  ...   
    
         DNDGRG3M086SBEA  DSERRG3M086SBEA  CES0600000008  CES2000000008  \
    0              6.000            6.000           6.00           6.00   
    1             18.294           10.152           2.13           2.45   
    2             18.302           10.167           2.14           2.46   
    3             18.289           10.185           2.15           2.45   
    4             18.300           10.221           2.16           2.47   
    ..               ...              ...            ...            ...   
    791          119.230          129.380          31.59          36.26   
    792          119.746          129.875          31.72          36.43   
    793          120.457          130.281          31.91          36.56   
    794          120.615          130.990          32.00          36.66   
    795          119.760          131.192          32.21          36.79   
    
         CES3000000008  UMCSENTx  DTCOLNVHFNM   DTCTHFNM     INVEST  VIXCLSx  
    0             6.00       2.0         6.00       6.00     6.0000   1.0000  
    1             2.04       NaN      6476.00   12298.00    84.2043      NaN  
    2             2.05       NaN      6476.00   12298.00    83.5280      NaN  
    3             2.07       NaN      6508.00   12349.00    81.6405      NaN  
    4             2.08       NaN      6620.00   12484.00    81.8099      NaN  
    ..             ...       ...          ...        ...        ...      ...  
    791          28.22      71.8    556011.41  938335.20  5381.4576  15.9822  
    792          28.33      74.0    559364.75  943484.76  5366.6686  15.6997  
    793          28.58      71.7    559087.09  944167.06  5350.2541  16.8122  
    794          28.68      64.7    556142.06  941199.49  5367.9408  17.0705  
    795          28.92      57.0          NaN        NaN  5406.5887  21.6579  
    
    [796 rows x 127 columns]


### Extracting `TCODE` variables


I extract the transformation codes from the first row of the dataset, since they show how each variable must be pre-processed. I drop the date column because it's not a real variable. In the end, I get a one-row table with the transformation code for each variable.


```python
print(tcode_df)
```

           RPI  W875RX1  DPCERA3M086SBEA  CMRMTSPLx  RETAILx  INDPRO  IPFPNSS  \
    TCODE  5.0      5.0              5.0        5.0      5.0     5.0      5.0   
    
           IPFINAL  IPCONGD  IPDCONGD  ...  DNDGRG3M086SBEA  DSERRG3M086SBEA  \
    TCODE      5.0      5.0       5.0  ...              6.0              6.0   
    
           CES0600000008  CES2000000008  CES3000000008  UMCSENTx  DTCOLNVHFNM  \
    TCODE            6.0            6.0            6.0       2.0          6.0   
    
           DTCTHFNM  INVEST  VIXCLSx  
    TCODE       6.0     6.0      1.0  
    
    [1 rows x 126 columns]


### Adjusting the dataset

I remove the first row, since it only contains transformation codes. Then, I rename the date column to "Date" and convert it to proper datetime format. Finally, I make sure all other columns contain numeric values so they’re ready for analysis.


```python
# Check
print(df)
```

              Date        RPI  W875RX1  DPCERA3M086SBEA     CMRMTSPLx  \
    0   1959-01-01   2583.560   2426.0           15.188  2.766768e+05   
    1   1959-02-01   2593.596   2434.8           15.346  2.787140e+05   
    2   1959-03-01   2610.396   2452.7           15.491  2.777753e+05   
    3   1959-04-01   2627.446   2470.0           15.435  2.833627e+05   
    4   1959-05-01   2642.720   2486.4           15.622  2.853072e+05   
    ..         ...        ...      ...              ...           ...   
    790 2024-11-01  20091.169  16376.8          122.396  1.545040e+06   
    791 2024-12-01  20101.629  16387.7          123.077  1.558008e+06   
    792 2025-01-01  20148.969  16391.2          122.614  1.543178e+06   
    793 2025-02-01  20209.351  16389.5          122.742  1.556553e+06   
    794 2025-03-01  20311.260  16500.4          123.601           NaN   
    
              RETAILx    INDPRO   IPFPNSS   IPFINAL   IPCONGD  ...  \
    0     17689.23968   21.9616   23.3868   22.2620   31.6664  ...   
    1     17819.01912   22.3917   23.7024   22.4549   31.8987  ...   
    2     17967.91336   22.7142   23.8459   22.5651   31.8987  ...   
    3     17978.97983   23.1981   24.1903   22.8957   32.4019  ...   
    4     18119.82573   23.5476   24.3911   23.1161   32.5567  ...   
    ..            ...       ...       ...       ...       ...  ...   
    790  712145.00000  101.9619   99.3808   98.8609  100.8691  ...   
    791  717662.00000  103.1177  100.4976   99.9719  101.6868  ...   
    792  711461.00000  103.3418  101.0766  100.6319  102.1879  ...   
    793  711680.00000  104.2202  101.8233  101.4377  102.7245  ...   
    794  722025.00000  103.8892  101.6665  101.1465  101.7332  ...   
    
         DNDGRG3M086SBEA  DSERRG3M086SBEA  CES0600000008  CES2000000008  \
    0             18.294           10.152           2.13           2.45   
    1             18.302           10.167           2.14           2.46   
    2             18.289           10.185           2.15           2.45   
    3             18.300           10.221           2.16           2.47   
    4             18.280           10.238           2.17           2.48   
    ..               ...              ...            ...            ...   
    790          119.230          129.380          31.59          36.26   
    791          119.746          129.875          31.72          36.43   
    792          120.457          130.281          31.91          36.56   
    793          120.615          130.990          32.00          36.66   
    794          119.760          131.192          32.21          36.79   
    
         CES3000000008  UMCSENTx  DTCOLNVHFNM   DTCTHFNM     INVEST  VIXCLSx  
    0             2.04       NaN      6476.00   12298.00    84.2043      NaN  
    1             2.05       NaN      6476.00   12298.00    83.5280      NaN  
    2             2.07       NaN      6508.00   12349.00    81.6405      NaN  
    3             2.08       NaN      6620.00   12484.00    81.8099      NaN  
    4             2.08      95.3      6753.00   12646.00    80.7315      NaN  
    ..             ...       ...          ...        ...        ...      ...  
    790          28.22      71.8    556011.41  938335.20  5381.4576  15.9822  
    791          28.33      74.0    559364.75  943484.76  5366.6686  15.6997  
    792          28.58      71.7    559087.09  944167.06  5350.2541  16.8122  
    793          28.68      64.7    556142.06  941199.49  5367.9408  17.0705  
    794          28.92      57.0          NaN        NaN  5406.5887  21.6579  
    
    [795 rows x 127 columns]


### Applying the correct `TCODE` to each variable

To apply the appropriate transformation to each variable, I built a function that reads the transformation code (TCODE) associated with each series and applies the corresponding operation. 

These codes are provided in the first row of the original dataset and were stored separately for clarity.

The function loops through all columns (except the Date), looks up the TCODE for each one, and applies the correct transformation using a dictionary that maps each code to a specific operation. This approach ensures that each variable is treated consistently and according to the definitions used in the FRED-MD dataset.


The column `TCODE` denotes the following data transformation for a series $x$:  

1. $\text{no transformation}$;  

2. $\Delta x_t$;  

3. $\Delta^2 x_t$;  

4. $log(x_t)$;  

5. $\Delta log(x_t)$;  

6. $\Delta^2 log(x_t)$;  

7. $\Delta(x_t / x_{t-1} - 1.0)$.  



```python
tcode_functions = {
    1: lambda x: x,                               # (1) Level
    2: lambda x: x.diff(),                        # (2) Δx
    3: lambda x: x.diff().diff(),                 # (3) Δ²x
    4: lambda x: np.log(x),                       # (4) log(x)
    5: lambda x: np.log(x).diff(),                # (5) Δlog(x)
    6: lambda x: np.log(x).diff().diff(),         # (6) Δ²log(x)
    7: lambda x: (x/x.shift(1)-1).diff()          # (7) Δ(x/x₋₁ - 1)
}
```


```python
def apply_transformations(df, tcode_df):
    df_transformed = df.copy()
    for col in df.columns:
        if col == 'Date':
            continue
        tcode = int(tcode_df[col])
        func = tcode_functions.get(tcode)
        if func:
            df_transformed[col] = func(df[col])
        else:
            print(f"Warning: No transformation defined for TCODE {tcode} in column {col}")
    return df_transformed

```


```python
# Apply the transformations
df_transformed = apply_transformations(df, tcode_df.loc['TCODE'])
# Now remove the first 2 rows
df_transformed = df_transformed.iloc[2:].reset_index(drop=True)
```

Another important thing to do is check the `-inf` with `NaN`, since they may be the result of `log(0)`. 


```python
# Count +inf values per column
pos_inf_count = (df_transformed == np.inf).sum()

# Count -inf values per column
neg_inf_count = (df_transformed == -np.inf).sum()

# Display only columns where +inf or -inf values are present
print("Columns with +inf values:\n", pos_inf_count[pos_inf_count > 0])
print("\nColumns with -inf values:\n", neg_inf_count[neg_inf_count > 0])
```

    Columns with +inf values:
     Series([], dtype: int64)
    
    Columns with -inf values:
     Series([], dtype: int64)


So there are no `inf` values


### Outliers

Now let's replace the outliers with `NaN`

**An outlier is defined as an observation that deviates from the sample median by more than ten interquartile ranges.**


```python
print(f'Outliers found and set NaN:\n{tot_outliers}')
```

    Outliers found and set NaN:
    159


## Iterative Expectation-Maximization algorithm

Estimation of the static factors by PCA adapted to allow for missing values. This is essentially the EM algorithm given in Stock and Watson (2002).

Observations that are missing are initialized to the unconditional mean based on the non-missing values (which is zero since the data are demeaned and standardized) so that the panel is re-balanced

Now let's check for `NaN` values


```python
nan_count = df_transformed.isna().sum()
print("NaN values per column:\n", nan_count[nan_count > 0])
```

    NaN values per column:
     RPI                  7
    W875RX1              2
    DPCERA3M086SBEA      3
    CMRMTSPLx            2
    RETAILx              2
                      ... 
    CUSR0000SAS          1
    UMCSENTx           227
    DTCOLNVHFNM          9
    DTCTHFNM             7
    VIXCLSx             40
    Length: 72, dtype: int64


In this section, I estimate the latent static factors from the transformed FRED-MD dataset using an Expectation-Maximization Principal Component Analysis (EM-PCA) procedure. This method follows the approach described in McCracken and Ng (2015). 

The estimation technique is based on the algorithm proposed by Stock and Watson (2002), which allows principal component analysis to be performed in the presence of `NaN` data by iteratively imputing missing values and re-estimating factors until convergence.

### Computing the mean for each variable


```python
# Now let's compute the unconditional mean for each variable excluding the NaN values
nan_mean = df_transformed.mean(skipna=True)
```


### Filling the `NaN` with the mean


```python
df_nan = df_transformed.copy()

# Let's substitute the NaN with the mean of the variable
for col in nan_mean.index:
    df_nan[col] = df_nan[col].fillna(nan_mean[col])

```

### Standardizing the Data

- Standardize each variable using:

  $$
  z_{it} = \frac{x_{it} - \mu_i}{\sigma_i}
  $$

- This ensures that each variable has mean 0 and standard deviation 1.


```python
# Now compute standard deviation and the mean on the filled version
std = df_nan.std()
mean = df_nan.mean()

# Create a new standardized DataFrame
df_standardized = df_nan.copy()

# Standardize each column (subtract mean and divide by std)
for col in mean.index:
    df_standardized[col] = (df_nan[col] - mean[col]) / std[col]
```


```python
nan_count_standardized = df_standardized.isna().sum()
print("NaN values per column:\n", nan_count_standardized[nan_count_standardized > 0])
```

    NaN values per column:
     Series([], dtype: int64)


### Factors

Starting from the data panel (with missing values initialized to zero), I estimate:

- A matrix of factors:  
  $$
  F = (f_1, \dots, f_T)' \in \mathbb{R}^{T \times r}
  $$

- A matrix of loadings:  
  $$
  \lambda = (\lambda_1, \dots, \lambda_N)' \in \mathbb{R}^{N \times r}
  $$

These are estimated using the normalization:
$$
\lambda' \lambda / N = I_r
$$

Where:
- $T$ = number of time periods (rows),
- $N$ = number of series (columns),
- $r$ = number of factors.

The missing value for series i at time t is updated from zero to $\lambda'_i f_t$.

This is multiplied by the standard deviation of the series and the mean is re-added.

The resulting value is treated as an observation for series i at time t, and the mean and variance of the complete sample are re-calculated. The data are demeaned and standardized again, and the factors and loadings are re-estimated from the updated panel. The iteration stops when the factor estimates do not change.

$PC_p$ criteria developed in Bai and Ng (2002) is used, which is a generalization of Mallow’s $C_p$ criteria for large dimensional panels.

The penalty used is:

$$
(N+T)/N T \log(\text{min}(N,T))
$$


```python
# 1. PCA with normalization λ'λ / N = I_r

def pca_stock_watson(X_std: np.ndarray, r: int):
    """
    Parameters
    ----------
    X_std : ndarray (T × N) – columns already demeaned and standardized
    r     : number of factors

    Returns
    -------
    F_hat      : (T × r)     – estimated factors
    Lambda_hat : (N × r)     – estimated loadings
    X_hat      : reconstruction F̂ Λ̂' (T × N)
    sing_vals  : singular values of X'X
    """
    T, N = X_std.shape
    U, s, _ = svd(X_std.T @ X_std, full_matrices=False)
    Lambda_hat = np.sqrt(N) * U[:, :r]
    F_hat      = (X_std @ Lambda_hat) / N
    X_hat      = F_hat @ Lambda_hat.T
    return F_hat, Lambda_hat, X_hat, s

```


```python
# 2. Bai & Ng 2002 Criterion 

def pc_p2_criterion(X_std: np.ndarray, kmax: int = 15) -> int:
    """
    Returns r* ∈ {0, …, kmax} that minimizes the PC_p2 criterion.
    """
    T, N  = X_std.shape
    NT, NT1 = T * N, N + T
    log_pen = np.log(min(N, T))

    crit = np.empty(kmax + 1)
    for r in range(kmax + 1):
        if r == 0:
            X_hat = np.zeros_like(X_std)
        else:
            X_hat = pca_stock_watson(X_std, r)[2]

        sigma2 = np.sum((X_std - X_hat) ** 2) / NT
        pen    = 0 if r == 0 else (NT1 / NT) * log_pen * r
        crit[r] = np.log(sigma2) + pen

    return int(np.argmin(crit))

```


```python
# 3. EM algorithm 

def em_factors(
    df_transformed:   pd.DataFrame,   # original NaNs
    df_missing:       pd.DataFrame,   # NaNs → unconditional mean (not standardized)
    df_standardized:  pd.DataFrame,   # z-score (demeaned + std), no NaNs
    *,
    kmax:         int   = 15,
    tol:          float = 1e-6,
    max_iter:     int   = 200,
    print_every:  int   = 10,
):
    """
    Python port of the MATLAB function `factors_em` (only DEMEAN=2).
    Returns:
        factors_df, loadings_df, r_star,
        X_filled_unstd_df, X_filled_std_df, X_hat_df
    """

    # setup 
    idx, cols = df_transformed.index, df_transformed.columns
    mask_nan  = df_transformed.isna().to_numpy()          # True on original NaNs

    X_std      = df_standardized.to_numpy(float)          # (T × N)
    X_hat_prev = np.zeros_like(X_std)
    err, it    = np.inf, 0

    #  EM loop 
    while err > tol and it < max_iter:
        it += 1

        # 1. select r* with PC_p2
        r_star = pc_p2_criterion(X_std, kmax)

        # 2. Stock-Watson PCA
        F_hat, Lambda_hat, X_hat, _ = pca_stock_watson(X_std, r_star)

        # 3. convergence criterion
        if it > 1:
            err = np.sum((X_hat - X_hat_prev) ** 2) / np.sum(X_hat_prev ** 2)
        X_hat_prev = X_hat.copy()

        if it == 1 or it % print_every == 0:
            print(f"Iter {it:3d} | r*={r_star:2d} | err={err:9.2e}")

        # 4. BACK-TRANSFORM with CURRENT mean/std 
        mean_curr = X_std.mean(axis=0)
        std_curr  = X_std.std(axis=0, ddof=0)
        std_curr[std_curr == 0] = 1.0

        X_unstd = X_std * std_curr + mean_curr          # original scale
        updates = X_hat * std_curr + mean_curr          # predictions in original scale
        X_unstd[mask_nan] = updates[mask_nan]           # replace only NaNs

        # 5. re-standardize for next iteration 
        mean_next = X_unstd.mean(axis=0)
        std_next  = X_unstd.std(axis=0, ddof=0)
        std_next[std_next == 0] = 1.0
        X_std = (X_unstd - mean_next) / std_next

    #  output 
    factors_df = pd.DataFrame(
        F_hat, index=idx, columns=[f"F{i+1}" for i in range(r_star)]
    )
    loadings_df = pd.DataFrame(
        Lambda_hat, index=cols, columns=[f"F{i+1}" for i in range(r_star)]
    )
    X_filled_unstd_df = pd.DataFrame(X_unstd, index=idx, columns=cols)
    X_filled_std_df   = pd.DataFrame(X_std,   index=idx, columns=cols)
    X_hat_df          = pd.DataFrame(X_hat,   index=idx, columns=cols)

    if it == max_iter:
        print(f" EM: maximum number of iterations reached ({max_iter}), err = {err:.2e}")
    else:
        print(f"Converged in {it} iterations (err = {err:.2e})")

    return (
        factors_df,
        loadings_df,
        r_star,
        X_filled_unstd_df,
        X_filled_std_df,
        X_hat_df,
    )

```


```python
factors, loadings, r_opt, X_f, X_f_std, xhat = em_factors(
    df_transformed = df_transformed,   # original NaNs
    df_missing     = df_nan,           # NaNs → unconditional mean
    df_standardized= df_standardized,  # z-score
    kmax           = 15,               # same as in the MATLAB script
    tol            = 1e-6,
    max_iter       = 200,
    print_every    = 10                # log every 10 iterations
)

```
The optimal number of factors is 7

```python
print("Optimal number of factors (r*) =", r_opt, "\n")
```

    Optimal number of factors (r*) = 7 



    
<img src="/assets/proj/FactorModels_files/factors.png" alt="Factors Plot" style="max-width: 100%; height: auto;">
    


### `NaN` have been filled

Also `NaN` values, now, have been filled thanks to this process

Now let's check if the procedure has succeed, running the line below should show `False`


```python
X_f.isna().any().any()
```




    False



And we also have its standardized version `X_std`


```python
X_f_std.isna().any().any()
```




    False



## Regression, $R^2$, and $mR^2$

**Cumulative regression for each series**

For each series $x_i$ (i.e., each column of the dataset, such as inflation, employment, etc.), do the following:

1. **Regression on $F1$**

$$
x_{t,i} = \alpha_i + \beta_{i,1} F_{1,t} + \varepsilon_{i,t}
$$

→ Let's save $R_i(1)^{2}$

2. **Regression on $F1 + F2$**

$$
x_{t,i} = \alpha_i + \beta_{i,1} F_{1,t} + \beta_{i,2} F_{2,t} + \varepsilon_{i,t}
$$

→ Let's save $R_i(2)^{2}$

3. **Regression on $F1 + F2 + F3$**

$$
x_{t,i} = \alpha_i + \beta_{i,1} F_{1,t} + \beta_{i,2} F_{2,t} + \beta_{i,t} F_{3,t} + \varepsilon_{i,t}
$$

→ Let's save $R_i(3)^{2}$

4. we continue up to $r$, the **optimal number of factors**


> NOTE:
> 
> `lstsq` solves the least squares problem but it is numerically stable and optimized and handles multiple dependent variables in a single call.
> 
> It avoids manual matrix inversion.


```python
T, N = X_f_std.shape
r = factors.shape[1]

X_np_std = X_f_std.to_numpy()
F_np = factors.to_numpy()
columns = X_f_std.columns

y_mean = np.mean(X_np_std, axis=0)
SST = np.sum((X_np_std - y_mean)**2, axis=0)

# Initialize DataFrame with index = variable names
R2_matrix = pd.DataFrame(index=columns)

# Loop over k factors
for k in range(1, r + 1):
    F_k = F_np[:, :k]  # first k factors
    beta_hat, *_ = np.linalg.lstsq(F_k, X_np_std, rcond=None)
    y_hat = F_k @ beta_hat
    SSR = np.sum((y_hat - y_mean)**2, axis=0)
    R2 = SSR / SST

    # Add column for cumulative R² with k factors
    R2_matrix[f'R2_F{k}'] = R2

```

Now we have a matrix `R2_matrix` containing the $R^2$ (computed as stated before) on the columns and the variables on the rows


```python
print(R2_matrix.head())
```

                        R2_F1     R2_F2     R2_F3     R2_F4     R2_F5     R2_F6  \
    RPI              0.110971  0.131306  0.155288  0.165859  0.170515  0.170852   
    W875RX1          0.299966  0.321856  0.367886  0.382022  0.389687  0.390745   
    DPCERA3M086SBEA  0.297816  0.308254  0.343691  0.368542  0.375204  0.448385   
    CMRMTSPLx        0.470080  0.475465  0.526062  0.539802  0.554758  0.577185   
    RETAILx          0.319538  0.365299  0.367627  0.379809  0.400234  0.452580   
    
                        R2_F7  
    RPI              0.171356  
    W875RX1          0.393956  
    DPCERA3M086SBEA  0.454303  
    CMRMTSPLx        0.578868  
    RETAILx          0.463431  


**Compute marginals for each series**

After obtaining all $R_i(k)^{2}$, compute for each $k$ and each series:

- $mR_i(1)^{2} = R_i(1)^{2}$
- $mR_i(2)^{2} = R_i(2)^{2} - R_i(1)^{2}$
- $mR_i(3)^{2} = R_i(3)^{2} - R_i(2)^{2}$
- ... up to $mR_i(r)^{2}$



```python
mR2_matrix = R2_matrix.copy()

# Marginals R²
mR2_matrix.iloc[:, 1:] = R2_matrix.iloc[:, 1:].values - R2_matrix.iloc[:, :-1].values
```


```python
print(mR2_matrix.head())
```

                        R2_F1     R2_F2     R2_F3     R2_F4     R2_F5     R2_F6  \
    RPI              0.110971  0.020335  0.023983  0.010571  0.004656  0.000337   
    W875RX1          0.299966  0.021890  0.046030  0.014137  0.007664  0.001058   
    DPCERA3M086SBEA  0.297816  0.010438  0.035437  0.024850  0.006662  0.073181   
    CMRMTSPLx        0.470080  0.005385  0.050597  0.013741  0.014956  0.022427   
    RETAILx          0.319538  0.045761  0.002328  0.012182  0.020425  0.052346   
    
                        R2_F7  
    RPI              0.000504  
    W875RX1          0.003211  
    DPCERA3M086SBEA  0.005918  
    CMRMTSPLx        0.001684  
    RETAILx          0.010851  


**Average per factor**

Finally, for each factor $k$, compute:

$$
mR(k)^{2} = \frac{1}{N} \sum_{i=1}^{N} mR_i(k)^{2}
$$


```python
# Mean for each factor (i.e., by column)
mR2_average = mR2_matrix.mean(axis=0)

mR2_average.name = 'mean_mR2'
```


```python
print(mR2_average)
```

    R2_F1    0.171063
    R2_F2    0.074508
    R2_F3    0.067727
    R2_F4    0.053311
    R2_F5    0.047765
    R2_F6    0.032216
    R2_F7    0.027055
    Name: mean_mR2, dtype: float64


**Series that “load the most” on each factor**

For each factor $k$:

- sort the series by $mR_i(k)^{2}$
- take the top 10


```python
top10_by_factor = {}

for col in mR2_matrix.columns:
    top10_series = mR2_matrix[col].sort_values(ascending=False).head(10)
    top10_by_factor[col] = top10_series


```


```python
print("Top 10 series for F1:")
print(top10_by_factor['R2_F1'])
```

    Top 10 series for F1:
    IPMANSICS    0.802027
    PAYEMS       0.783759
    INDPRO       0.761174
    IPFPNSS      0.756845
    CUMFNS       0.749975
    USGOOD       0.742252
    MANEMP       0.690483
    IPFINAL      0.688075
    DMANEMP      0.645203
    IPDMAT       0.625748
    Name: R2_F1, dtype: float64


Total Variation Explained by the factors


```python
xhat_np = xhat.to_numpy(float)      

# Total Variation Explained
SSE = np.square(X_np_std - xhat_np).sum()   # Resigual sum of squares
SST = np.square(X_np_std).sum()             # Total square sum

TVE = 1 - SSE / SST
```



<img src="/assets/proj/FactorModels_files/Table_ADV_ECON.png" alt="Table1" style="max-width: 100%; height: auto;">


### $R(7)^2$ ordered by groups

The figure shows the explanatory power of the seven factors in all the series organized into eight groups as given in the original paper. Group 1 is output and income, Group 2 is labor market, Group 3 is consumption and housing, Group 4 is orders and inventories, Group 5 is money and credit, Group 6 is interest rate and exchange rates, Group 7 is price, and Group 8 is stock market.

<img src="/assets/proj/FactorModels_files/r2_by_groups.png" alt="r2bygroup" style="max-width: 100%; height: auto;">



## Forecast

### 1. Six - Month Forecast


```python
print("NaN in CPIAUCSL:", df["CPIAUCSL"].isna().sum())
print("NaN in INDPRO:", df["INDPRO"].isna().sum())
```

    NaN in CPIAUCSL: 0
    NaN in INDPRO: 0


Since the original series don't have `NaN` we can avoid re-contructing the series. We can use the original ones

Then, I compute the first difference fo `CPIAUCSL` and remove also first row

As in the paper I use the `BIC` to choose best lag both for dependent variable `Y` and independent variables `X's`

### `INDPRO`

```python
print("Best Models according to BIC")
```

    Best Models according to BIC
     lag_Y  lag_F         BIC       R2
         3      1 2375.467375 0.998480
         6      1 2375.729966 0.998489
         1      1 2378.052052 0.998467
         4      1 2378.771488 0.998477
         2      1 2379.500400 0.998468


### `diff_CPIAUCSL`

```python
bic_6m_CPI = results_CPI.sort_values(by='BIC').head(5)
print("Best Models according to BIC")
print(bic_6m_CPI[['lag_Y', 'lag_F', 'BIC', 'R2']].to_string(index=False))
```

    Best Models according to BIC
     lag_Y  lag_F         BIC       R2
         6      1 1083.646067 0.178863
         5      1 1097.017813 0.158363
         4      1 1098.018519 0.150947
         3      1 1099.174212 0.143283
         1      1 1112.052241 0.115923


> ### Assumption: Persistence of Factors
> I assume that the values of the factors remain constant over the forecast horizon.  
> In other words, the lagged values of $F1–F7$ at time **$T$** are reused for all future periods **$T+1$ to $T+6$**.
> - Fast and simple: no need to estimate additional models.
> - Not professional for extremely accuracy forecast

## Forecast `INDPRO`


The **optimal lags** are used as independent variables for an OLS regression.
After estimating the model, the coefficients will be applied to the most recent available data to generate the forecast.


```python
print("\nINDPRO forecasts for the next 6 months:")
print(forecast_6m_IND.round(3))

```

    
    INDPRO forecasts for the next 6 months:
    2025-04-01    104.136
    2025-05-01    103.669
    2025-06-01    104.109
    2025-07-01    103.536
    2025-08-01    104.109
    2025-09-01    103.371
    Name: INDPRO_forecast, dtype: float64



    
<img src="/assets/proj/FactorModels_files/indpro_for.png" alt="indpro_forecast" style="max-width: 100%; height: auto;">
    


## Forecast `CPI`

The same procedure is applied to CPI


```python
print("\nCPI forecasts for the next 6 months:")
print(forecast_6m_CPI.round(3))
```

    
    CPI forecasts for the next 6 months:
    2025-04-01    0.539
    2025-05-01    0.285
    2025-06-01    0.593
    2025-07-01    0.309
    2025-08-01    0.457
    2025-09-01    0.294
    Name: diff_CPIAUCSL_forecast, dtype: float64

    
<img src="/assets/proj/FactorModels_files/cpi_for.png" alt="cpi_forecast" style="max-width: 100%; height: auto;">
    


## 2. Real Time Evaluation

The dataset is split into 80% as `train data` and 20% as `test data`

I act as if I knew data up to `t`, I estimate the factors and use their lags, in this way I simulate the reality.

E.G. Today we can use just the data we know $\rightarrow$ no future informations

So each 'month' I estimate the model and then make a forecast about next month. When the real info is available I include it in my data and estimate the model again to predict next month observation and so on.

> NOTE:
>
> For this `real time evaluation` I'm going to skip the **outliers** procedure
>
> The model will be re-estimated each month, i.e. each observation


### Loop for Real Time Evaluation

Let's create a silent version of the `em_factors` function such that I'm not gonna have a confusing output


```python
def em_factors_silent(
    df_transformed:   pd.DataFrame,      # NaN “veri”
    df_missing:       pd.DataFrame,      # NaN → μ
    df_standardized:  pd.DataFrame,      # z-score
    *,
    kmax:  int = 15,
    tol:   float = 1e-6,
    max_iter: int = 200,
    print_every: int | None = 0          # 0/None → no log
):
    idx, cols = df_transformed.index, df_transformed.columns
    mask_nan  = df_transformed.isna().to_numpy()

    X_std      = df_standardized.to_numpy(float)
    X_hat_prev = np.zeros_like(X_std)
    err, it    = np.inf, 0

    while err > tol and it < max_iter:
        it += 1
        r_star = pc_p2_criterion(X_std, kmax)
        F_hat, Λ_hat, X_hat, _ = pca_stock_watson(X_std, r_star)

        if it > 1:
            err = ((X_hat - X_hat_prev)**2).sum() / (X_hat_prev**2).sum()
        X_hat_prev = X_hat

        if print_every and (it == 1 or it % print_every == 0):
            print(f"Iter {it:3d} | r*={r_star:2d} | err={err:9.2e}")

        μ, σ = X_std.mean(0), X_std.std(0, ddof=0)
        σ[σ == 0] = 1.0
        X_unstd = X_std*σ + μ
        X_unstd[mask_nan] = (X_hat*σ + μ)[mask_nan]

        μn, σn = X_unstd.mean(0), X_unstd.std(0, ddof=0)
        σn[σn == 0] = 1.0
        X_std = (X_unstd - μn) / σn

    return (pd.DataFrame(F_hat, idx, [f"F{i+1}" for i in range(r_star)]),
            pd.DataFrame(Λ_hat, cols, [f"F{i+1}" for i in range(r_star)]),
            r_star,
            pd.DataFrame(X_unstd, idx, cols),
            pd.DataFrame(X_std,    idx, cols),
            pd.DataFrame(X_hat,    idx, cols))
```

### `INDPRO` (Level)

- Lags of `INDPRO` = $3$ 

- Lags of `factors` = $1$


```python

# PARAMETERS (same as your original block)
p_IND        = 3          # lags of INDPRO
m_IND        = 1          # lags of factors (can be 1, 2, 3, ...)
kmax_em      = 15
tol_em       = 1e-6
maxiter_em   = 200
progress_mod = 16         # print ETA every 16 iterations

# DATA SPLIT
df_train = df_transformed.loc[:'2011-11-01'].copy()
df_test  = df_transformed.loc['2011-12-01':].copy()

fcst_dates, fcst_values = [], []
tic, steps_total, step_count = time.time(), len(df_test), 0

# ROLLING-WINDOW LOOP
while not df_test.empty:

    # 1) fill NaNs + standardize
    df_nan = df_train.fillna(df_train.mean())
    df_std = (df_nan - df_nan.mean()) / df_nan.std()

    # 2) extract factors via EM (silent)
    factors_train, _, r_opt, *_ = em_factors_silent(
        df_transformed  = df_train,
        df_missing      = df_nan,
        df_standardized = df_std,
        kmax            = kmax_em,
        tol             = tol_em,
        max_iter        = maxiter_em,
        print_every     = 0
    )

    # 3) regressors = lags 1…m_IND of factors + lags 1…p_IND of INDPRO
    lag_F = [factors_train.shift(l).add_suffix(f"_lag{l}")
             for l in range(1, m_IND + 1)]
    lag_Y = [df.loc[:, "INDPRO"].shift(l).to_frame(f"INDPRO_lag{l}")
             for l in range(1, p_IND + 1)]
    X_train = pd.concat(lag_F + lag_Y, axis=1).dropna()

    # 4) target Y
    Y_train = df.loc[X_train.index, "INDPRO"]

    # 5) OLS
    Xc   = np.hstack([np.ones((len(X_train), 1)), X_train.values])
    beta = np.linalg.inv(Xc.T @ Xc) @ (Xc.T @ Y_train.values.reshape(-1, 1))

    # 6) regressor vector for t+1
    #    • factor blocks: F(t), F(t-1)…F(t-(m_IND-1))
    F_blocks = [factors_train.shift(l).iloc[-1].values
                for l in range(0, m_IND)]          # 0 = current
    #    • INDPRO scalars: lag 1…p_IND
    Y_scalars = [df.loc[df_train.index[-1-l], "INDPRO"]
                 for l in range(p_IND)]

    x_new = np.hstack([1.0, *F_blocks, *Y_scalars])
    y_hat = float((x_new @ beta).squeeze())        # no warnings

    # 7) save forecast
    next_date = df_test.index[0]
    fcst_dates.append(next_date)
    fcst_values.append(y_hat)

    # 8) roll-forward
    df_train = pd.concat([df_train, df_test.iloc[[0]]])
    df_test  = df_test.iloc[1:]

    # 9) progress bar
    step_count += 1
    if step_count % progress_mod == 0 or step_count == steps_total:
        elapsed = time.time() - tic
        eta_sec = (elapsed / step_count) * (steps_total - step_count)
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_sec))
        print(f"[{step_count:>3}/{steps_total}]  r*={r_opt:2d}  ETA ≈ {eta_str}")

# OUT-OF-SAMPLE FORECAST SERIES
forecast_INDPRO_oos = pd.Series(fcst_values,
                                index=pd.to_datetime(fcst_dates),
                                name="INDPRO_forecast")

```



```python
# 1. Extract the actual INDPRO series corresponding to the OOS forecast dates
y_true = df.loc[forecast_INDPRO_oos.index, "INDPRO"].astype(float)

# 2. Forecast series (already in correct order and with same index)
y_pred = forecast_INDPRO_oos.astype(float)

# 3. MSE and RMSE
mse  = np.mean((y_true - y_pred) ** 2)
rmse = np.sqrt(mse)

print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")

```

    MSE  : 1.8641
    RMSE : 1.3653


    
<img src="/assets/proj/FactorModels_files/indpro_osa.png" alt="indpro_osa" style="max-width: 100%; height: auto;">


### `CPIAUSCL` (First Difference)

**The same procedure is applied to CPI**

- Lags of `CPIAUSCL` = $6$ 

- Lags of `factors` = $1$



    MSE  (CPI) : 0.360018
    RMSE (CPI) : 0.600015


<img src="/assets/proj/FactorModels_files/cpi_osa.png" alt="cpi_osa" style="max-width: 100%; height: auto;">
    

