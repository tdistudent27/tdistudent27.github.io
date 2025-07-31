---
layout: page
title: Portfolio Construction
description: Markowitz Approach to Portfolio Construction
importance: 3
category: Academic
nav: false
---

## Introduction
This project was carried out by **Tommaso de Martino** (Sapienza University of Rome), Health Sector by [**Sahar Shirazi**](https://www.linkedin.com/in/sahar-shirazi-906b3a167/)
(Sapienza University of Rome), **Alberto Pio Stigliani** (Sapienza University of Rome), and [**Robert Roman**](https://www.linkedin.com/in/robert-roman-998b99213/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app) (Sapienza University of Rome).
Each of us focused on a specific Machine Learning model:

## 1. Markowitz Portfolio Optimization

The **Markowitz approach** for portfolio optimization, known as the **efficient frontier theory**, is a theoretical model that helps investors build a portfolio that maximizes expected return for each level of risk or, conversely, minimizes risk for a given level of return.

### 1. Assumptions of Markowitz’s Theory
1) Investors want to maximize return and minimize risk

2) The risk of a portfolio can be reduced through **diversification**

### 2. Expected Return and Volatility
Markowitz takes into account two main metrics for evaluating the portfolio

**1) Expected Return:** The weighted average of the expected returns of the assets in the portfolio. It represents how much can be earned on average by investing in that portfolio.

**2) Risk or Volatility:** The standard deviation of the portfolio's returns, which measures the variability or uncertainty associated with that return

### 3. Correlation and the Covariance Matrix
The risk of a portfolio is not just the sum of the risks of the individual assets, but also depends on the **correlation** between them. The **covariance matrix** reflects how assets move together, allowing the calculation of the total risk of the portfolio accurately.

**Diversification and correlation:** A portfolio can have lower risk than the individual assets thanks to diversification since if some assets go up while others go down, the combined effect can reduce the overall volatility.

### 4. Efficient Frontier
Markowitz demonstrates that, by combining assets in various proportions, it’s possible to trace a **curve of optimal portfolios** called the **efficient frontier**.
- Each portfolio represents the maximum possible return for a given level of risk.
- Any portfolio outside the frontier is less efficient since it offers a lower return for the same level of risk.


### 5. The Portfolio with the Highest Sharpe Ratio (Tangency Portfolio)
A special point on the efficient frontier is the tangency portfolio, which has the highest Sharpe Ratio.

This portfolio offers the best possible return for each unit of risk, making it an ideal portfolio for those who want the best risk-return trade-off.

### 6. Optimal Portfolio
According to Markowitz’s theory, if looking for an optimal trade-off between risk and return, the tangency portfolio with the highest Sharpe Ratio is theoretically the best choice.

## 2. Our Approach

We are going to proceed as follows:

**APPROACH 1:** We will use a **Monte Carlo simulation** approach to calculate the **efficient frontier** and then find the portfolio with the maximum **Sharpe Ratio** between those generated.

**APPROACH 2:** We will solve an **Optimization Problem** to find the **Optimal Portfolio**

Once we have the 2 portfolios found by the 2 approaches for each sector we can compare the results

## 3. Variables legend

It's important to say that to avoid overwriting of variables we are going to define the variables with the following logic:

**(name of the variable)_(initial letters of the sector)**

So, for example, the data for the financial sector will be called:

- **data_fs**

where **fs** means **financial sector**

## 4. From Sector-Portfolio to General-Portfolio

We are going to perform the 2 approaches for 4 sectors in the Economy and we are going to pick the 3 stocks (if available) with more weight in each sector.

The 4 sectors are:

1. Financial Sector - **Tommaso de Martino**

2. Energy Sector - **Alberto Pio Stigliani**

3. Technology Sector - [**Robert Roman**](https://www.linkedin.com/in/robert-roman-998b99213/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)

4. Health Sector - [**Sahar Shirazi**](https://www.linkedin.com/in/sahar-shirazi-906b3a167/)

We will end up with maximum 3 stocks for each sector which will be used in the final optimization problem.

## 5. Libraries


```python
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from datetime import datetime
from scipy import optimize
from scipy.optimize import minimize

```

## 6. Financial Sector by **Tommaso de Martino**

Finance companies facilitate different financial transactions, which are crucial for an economy to function well. Additionally, they provide many services to businesses and individuals alike, allowing for more control over financial decisions for the average person.

Some of the standard financial companies that exist today include:

1) Banks

2) Credit unions

3) Insurance companies

4) Investment companies

5) Brokerage firms

6) Sales finance and consumer finance companies

### How do we choose these companies?

#### **MSCI World Financials Index**

The [**MSCI World Financials Index**](https://www.msci.com/indexes/index/106802) captures large and mid cap representation across Developed Markets countries. All securities in the index are classified in the Financials sector as per the **Global Industry Classification Standard (GICS®)**.


[**Global Industry Classification Standard (GICS®):**](https://www.msci.com/our-solutions/indexes/gics) is an industry analysis framework that helps investors understand the key business activities for companies around the world. MSCI and S&P Dow Jones Indices developed this classification standard to provide investors with consistent and exhaustive industry definitions.

The top 10 constituents of the MSI World Financial Index are:
1) JPMORGAN CHASE & CO

2) BERKSHIRE HATHAWAY B

3) VISA A

4) MASTERCARD A

5) BANK OF AMERICA CORP

6) WELLS FARGO & CO

7) ROYAL BANK OF CANADA

8) HSBC HOLDINGS (GB)

9) GOLDMAN SACHS GROUP

10) COMMONWEALTH BANK OF AUS

### Data preprocessing

First of all let's define the ticker and import daily data for this portfolio.

It's important to specify that we will work on the adjusted close data and from them we will derive the log-returns


```python
# Tickers list
tickers_fs = ["JPM", "BRK-B", "V", "MA", "BAC", "WFC", "RY", "HSBC", "GS", "CBA.AX"]
# Dictionary to store stocks' name
ticker_name_fs = {
    "JPM": "JPMORGAN CHASE & CO",
    "BRK-B": "BERKSHIRE HATHAWAY B",
    "V": "VISA A",
    "MA": "MASTERCARD A",
    "BAC": "BANK OF AMERICA CORP",
    "WFC": "WELLS FARGO & CO",
    "RY": "ROYAL BANK OF CANADA",
    "HSBC": "HSBC HOLDINGS (GB)",
    "GS": "GOLDMAN SACHS GROUP",
    "CBA.AX": "COMMONWEALTH BANK OF AUS"
}


# Setting the dates
start_date = "2020-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Importing the data
data_fs = yf.download(tickers_fs, start= start_date , end= end_date )['Adj Close']
print(data_fs)
```

    [*********************100%***********************]  10 of 10 completed

    Ticker                           BAC       BRK-B      CBA.AX          GS  \
    Date                                                                       
    2020-01-02 00:00:00+00:00  31.647984  228.389999   65.844452  208.356293   
    2020-01-03 00:00:00+00:00  30.990875  226.179993   66.198891  205.919891   
    2020-01-06 00:00:00+00:00  30.946470  226.990005   65.753777  208.027298   
    2020-01-07 00:00:00+00:00  30.742239  225.919998   66.932518  209.396652   
    2020-01-08 00:00:00+00:00  31.053032  225.990005   66.685242  211.415146   
    ...                              ...         ...         ...         ...   
    2024-11-19 00:00:00+00:00  46.410000  468.859985  155.610001  581.380005   
    2024-11-20 00:00:00+00:00  46.060001  468.829987  156.380005  581.929993   
    2024-11-21 00:00:00+00:00  46.459999  472.059998  156.240005  596.109985   
    2024-11-22 00:00:00+00:00  47.000000  476.570007  159.029999  602.780029   
    2024-11-25 00:00:00+00:00  47.500000  477.429993  160.139999  603.030029   
    
    Ticker                          HSBC         JPM          MA          RY  \
    Date                                                                       
    2020-01-02 00:00:00+00:00  29.879021  122.104614  295.023865   65.949982   
    2020-01-03 00:00:00+00:00  29.416080  120.493271  292.145569   65.718315   
    2020-01-06 00:00:00+00:00  29.317415  120.397453  292.923431   65.858978   
    2020-01-07 00:00:00+00:00  29.097328  118.350624  291.931641   65.676918   
    2020-01-08 00:00:00+00:00  29.029024  119.273888  297.082550   66.074120   
    ...                              ...         ...         ...         ...   
    2024-11-19 00:00:00+00:00  46.279999  243.089996  519.460022  122.900002   
    2024-11-20 00:00:00+00:00  46.230000  240.779999  512.539978  121.790001   
    2024-11-21 00:00:00+00:00  46.320000  244.759995  515.099976  125.089996   
    2024-11-22 00:00:00+00:00  45.939999  248.550003  520.859985  125.080002   
    2024-11-25 00:00:00+00:00  46.490002  250.289993  526.599976  124.320000   
    
    Ticker                              V        WFC  
    Date                                              
    2020-01-02 00:00:00+00:00  184.487366  47.053131  
    2020-01-03 00:00:00+00:00  183.020096  46.764248  
    2020-01-06 00:00:00+00:00  182.624329  46.484104  
    2020-01-07 00:00:00+00:00  182.141678  46.098942  
    2020-01-08 00:00:00+00:00  185.259567  46.239006  
    ...                               ...        ...  
    2024-11-19 00:00:00+00:00  311.850006  73.430000  
    2024-11-20 00:00:00+00:00  307.390015  73.580002  
    2024-11-21 00:00:00+00:00  309.899994  74.830002  
    2024-11-22 00:00:00+00:00  309.920013  75.959999  
    2024-11-25 00:00:00+00:00  313.190002  76.900002  
    
    [1267 rows x 10 columns]


## Daily Log Returns

First of all, we need to calculate the **daily logarithmic returns** based on Adjusted Close prices. Logarithmic returns are preferred to simple returns since they have 2 useful properties:

1) Simmetry

2) Time - Additivity


```python
# Log Returns
log_returns_fs = np.log(data_fs / data_fs.shift(1))
log_returns_fs = log_returns_fs.dropna()

```

Now we are going to plot the daily logarithmic returns


```python
log_returns_df_fs = log_returns_fs.reset_index()  # index reset to include date column
log_returns_df_fs = log_returns_df_fs.melt(id_vars="Date", var_name="Company", value_name="Daily LogReturn")

fig = px.line(log_returns_df_fs, x="Date", y="Daily LogReturn", color="Company",
              title="Daily LogReturns of Stocks")
fig.show()
```


## Approach 1 - Monte Carlo

### Formulas that we will use

1. **Annualized expected return**

2. **Annualized expected volatility**

3. **Sharpe Ratio**

### 1. Annualized expected return

$$
\text{Annualized expected return} = ( \mathbf{w}^T \cdot \mu ) \cdot 252
$$

where:
- $\mathbf{w}$ is the weight vector of the portfolio.
- $\mu$ is the vector of the average daily log returns of each asset.
- $252$ is the annualization factor based on the approximate number of trading days in a year.


or, in expanded form:

$$
\text{Annualized expected return} = \left( \sum (\text{log\_returns} \times \text{weights}) \right) \times 252
$$



**In the code it will be**

- first of all let us assume that in a year there are **252 trading days**

- **'log_returns.mean()'** computes the average daily return for each security in the **'log_returns'** dataset, which contains the daily logarithmic returns of securities

- **'log_returns.mean().dot(weights)'** computes the scalar product between the average daily return of each security and the portfolio weights

- **'log_returns.mean().dot(weights) * 252'** here the weighted average of daily returns is multiplied by **252** to obtain the portfolio's annualized expected return.

### 2. Annualized expected volatility

We are going to write a code that computes the **annualized expected volatility** in a **matrix form**


$$
\text{Annualized Expected Volatility} = \sqrt{w^T \cdot (252 \cdot \Sigma) \cdot w}
$$

where:
- $w$ is the **weight vector** of the portfolio, representing the proportion of each asset.
- $\Sigma$ is the **daily covariance matrix** of asset returns.
- $252$ is the **annualization factor**, based on the approximate number of trading days in a year.

**In the code it will be**

- **'log_returns.cov()'** computes the covariance matrix of daily (logarithmic) returns for all securities in the portfolio

- **'252 * log_returns.cov()'** covariance annualized

- **'log_returns.cov().dot(weights)'** we multiply the covariance matrix by the weight vector to reflect the risk contribution of each security

- **'weights.T.dot'** multiplication of the transposed weights obtaining the weighted sum of the covariances, which represents the total variance of the portfolio

### 3. Sharpe Ratio

The Sharpe ratio compares the return of an investment with its risk

$$ \text{Sharpe Ratio} = \frac{E[R_p] - R_f}{\sigma_p} $$

Where:

* $E[R_p]$ is the $\textbf{expected return of the portfolio}$,
* $R_f$ is the $\textbf{risk-free rate}$,
* $\sigma_p$ is the $\textbf{volatility of the portfolio (standard deviation of the portfolio's returns)}$.

### Risk-free rate

We can use the **3-Month Treasury Bill**


```python
ticker_rf = "^IRX"
rf_data = yf.download(ticker_rf, start=start_date, end=end_date)
print(rf_data)
rf = rf_data['Adj Close'].iloc[-1].values[0] / 100
print(f"Risk-free rate (rf) in decimal: {rf}")
```

    
[*********************100%***********************]  1 of 1 completed

    Price                     Adj Close  Close   High    Low   Open Volume
    Ticker                         ^IRX   ^IRX   ^IRX   ^IRX   ^IRX   ^IRX
    Date                                                                  
    2020-01-02 00:00:00+00:00     1.495  1.495  1.510  1.495  1.510      0
    2020-01-03 00:00:00+00:00     1.473  1.473  1.490  1.460  1.490      0
    2020-01-06 00:00:00+00:00     1.488  1.488  1.490  1.475  1.478      0
    2020-01-07 00:00:00+00:00     1.500  1.500  1.505  1.500  1.505      0
    2020-01-08 00:00:00+00:00     1.493  1.493  1.493  1.485  1.493      0
    ...                             ...    ...    ...    ...    ...    ...
    2024-11-19 00:00:00+00:00     4.408  4.408  4.410  4.403  4.408      0
    2024-11-20 00:00:00+00:00     4.410  4.410  4.415  4.410  4.410      0
    2024-11-21 00:00:00+00:00     4.413  4.413  4.413  4.400  4.403      0
    2024-11-22 00:00:00+00:00     4.415  4.415  4.420  4.408  4.408      0
    2024-11-25 00:00:00+00:00     4.405  4.405  4.413  4.403  4.413      0
    
    [1234 rows x 6 columns]
    Risk-free rate (rf) in decimal: 0.0440500020980835


    


Now we are ready to compute the **Benchmark-Sharpe Ratio**

### Monte Carlo simulation

This for loop simulates 100,000 random portfolios and calculates for each:

- Normalized random weights,
- Annualized expected return,
- Annualized volatility,
- Sharpe Ratio.


```python
np.random.seed(1) # set seed for reproducibility

# number of simulations
n = 100_000 # it could take some minutes

port_weights_fs = np.zeros(shape=(n,len(data_fs.columns)))
port_volatility_fs = np.zeros(n)
port_sr_fs = np.zeros(n)
port_return_fs = np.zeros(n)
nume_securities_fs = len(data_fs.columns)

# generation of random portfolios with random normalized weights (sum to 1)
for i in range(n):
    # Weight each security
    weights_fs = np.random.random(len(data_fs.columns)) # np.random.random only produces non-negative values
    # normalize it, so that sum is one
    weights_fs /= np.sum(weights_fs)
    port_weights_fs[i,:] = weights_fs # 'i' indicates the specific iteration (i.e. the current portfolio in the simulation).
                                # ':' indicates that we are saving all the weights generated for this portfolio (i.e. for each security)

    # Calculation of the annualized expected return
    exp_ret_fs = log_returns_fs.mean().dot(weights_fs) * 252
    port_return_fs[i] = exp_ret_fs # save this value in the port_return array at position i, corresponding to the current simulation.

    # Annualized expected volatility
    exp_vol_fs = np.sqrt(weights_fs.T.dot(252 * log_returns_fs.cov().dot(weights_fs)))
    port_volatility_fs[i] = exp_vol_fs

    # Sharpe Ratio
    sr_fs = (exp_ret_fs - rf) / exp_vol_fs
    port_sr_fs[i] = sr_fs
```

### Identification of best portfolio

Now we are going to manually call the best portfolio (not oprimal since it is not an optimization problem), i.e. the portfolio with the maximum **Sharpe Ratio** between those generated


**Note that:**
It is not necessarily true that this portfolio will be the same as the one resulting from solving the optimization problem, i.e. directly maximizing the **Sharpe Ratio**


```python
# Index of max Sharpe Ratio
max_sr_fs = port_sr_fs.max()
ind_fs = port_sr_fs.argmax()
# Return and Volatility at Max SR
max_sr_ret_fs = port_return_fs[ind_fs]
max_sr_vol_fs = port_volatility_fs[ind_fs]
```

### Monte Carlo Portfolio Weights


```python
for weight, stock in sorted(zip(port_weights_fs[ind_fs], data_fs.columns), key=lambda x: x[0], reverse=True):
    print(f'{round(weight * 100, 2)} % of {stock} should be bought.')

print(f'\nMonte Carlo portfolio return is : {round(max_sr_ret_fs * 100, 2)}% with volatility {max_sr_vol_fs} and Sharpe Ratio {max_sr_fs}')
```

    37.08 % of COMMONWEALTH BANK OF AUS should be bought.
    33.2 % of GOLDMAN SACHS GROUP should be bought.
    11.09 % of WELLS FARGO & CO should be bought.
    6.34 % of ROYAL BANK OF CANADA should be bought.
    4.79 % of BERKSHIRE HATHAWAY B should be bought.
    3.03 % of VISA A should be bought.
    2.02 % of JPMORGAN CHASE & CO should be bought.
    1.36 % of MASTERCARD A should be bought.
    0.84 % of BANK OF AMERICA CORP should be bought.
    0.26 % of HSBC HOLDINGS (GB) should be bought.
    
    Monte Carlo portfolio return is : 17.84% with volatility 0.2350763868947315 and Sharpe Ratio 0.571580612208514


- **zip(port_weights[ind], stocks)**: Matches each weight of the optimal portfolio (found by the index 'ind') with the corresponding asset in the data list

- **sort** is just to have the results in a decreasing way

- The **round** function is used to round a number to a specified number of decimal places.

### Monte Carlo Efficient Frontier


```python
# function to minimize volatility
def minimize_volatility(weights_fs):
    return np.sqrt(weights_fs.T.dot(252 * log_returns_fs.cov().dot(weights_fs)))

# range of target_returns to efficient frontier
target_returns_fs = np.linspace(port_return_fs.min(), port_return_fs.max() * 1.1, 100)

# Comuputation of efficient frontier
efficient_volatilities_fs = []
for target_return_fs in target_returns_fs:
    # Constraints: sum of weights = 1 and return = target_return
    constraints_fs = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # sum of weights = 1
        {'type': 'eq', 'fun': lambda w: log_returns_fs.mean().dot(w) * 252 - target_return_fs}  # target return
    ]
    bounds_fs = [(0, 1) for _ in range(len(data_fs.columns))]  # every weights between 0 and 1
    initial_guess_fs = np.ones(len(data_fs.columns)) / len(data_fs.columns)  # unifrom weight

    # minimize volatility for target return
    result_fs = minimize(minimize_volatility, initial_guess_fs, method='SLSQP', bounds=bounds_fs, constraints=constraints_fs)
    efficient_volatilities_fs.append(result_fs.fun)
```


    
<img src="/assets/proj/Portfolio_construction_files/fs_1.png" alt="fs1" style="max-width: 100%; height: auto;">    


## Approach 2 - Optimization Problem

### Returns and Covariance


```python
# Annualized mean log return
log_returns_mean_fs = log_returns_fs.mean()*252

# Annualized covariance matrix
cov_fs = log_returns_fs.cov()*252
```

### Functions


```python
# Function to get return, volatility and Sharpe Ratio (Benchmark-Based Sharpe Ratio)
def get_ret_vol_sr(weights_fs):
    weights_fs = np.array(weights_fs)  # weights to a NumPy array
    ret_fs = log_returns_mean_fs.dot(weights_fs)  # expected portfolio return
    vol_fs = np.sqrt(weights_fs.T.dot(cov_fs.dot(weights_fs)))  # portfolio volatility
    sr_fs = (ret_fs - rf) / vol_fs  # Benchmark-Based Sharpe Ratio
    return np.array([ret_fs, vol_fs, sr_fs])  # Returns return, volatility, and the Sharpe Ratio


# Negative Sharpe ratio as we need to max it but Scipy minimize the given function
def neg_sr(weights_fs):
    return get_ret_vol_sr(weights_fs)[-1] * -1  # Returns the negative of the Sharpe Ratio


# check sum of weights
def check_sum(weights_fs):
    return np.sum(weights_fs) - 1  # Returns the difference between the sum of weights and 1 (Obviously it should be 0)


# Constraints for the optimization problem
cons_fs = {'type':'eq','fun':check_sum} # type = type of constraint
                                     # eq = equality
                                     # fun = function

# Bounds on weights (NON-NEGATIVITY)
bounds_fs = [(0, 1) for _ in range(10)]

# Initial guess for optimization to start with
init_guess_fs = [1/10 for _ in range(10)]  # it starts with 10% to each asset


# Call minimizer
opt_results_fs = optimize.minimize(neg_sr, init_guess_fs, constraints=cons_fs, bounds=bounds_fs, method='SLSQP') # SLSQP
                                                                                                    # (Sequential Least Squares Programming)
```

### Optimal Weights


```python
optimal_weights_fs = opt_results_fs.x
# optimal_weights
for st, i in zip(data_fs,optimal_weights_fs):
    print(f'Stock {st} has weight {np.round(i*100,2)} %')
```

    Stock BANK OF AMERICA CORP has weight 0.0 %
    Stock BERKSHIRE HATHAWAY B has weight 19.05 %
    Stock COMMONWEALTH BANK OF AUS has weight 53.71 %
    Stock GOLDMAN SACHS GROUP has weight 27.24 %
    Stock HSBC HOLDINGS (GB) has weight 0.0 %
    Stock JPMORGAN CHASE & CO has weight 0.0 %
    Stock MASTERCARD A has weight 0.0 %
    Stock ROYAL BANK OF CANADA has weight 0.0 %
    Stock VISA A has weight 0.0 %
    Stock WELLS FARGO & CO has weight 0.0 %


### Monte Carlo weights


```python
monte_c_weights_fs = port_weights_fs[ind_fs]
for st, i in zip(data_fs,monte_c_weights_fs):
    print(f'Stock {st} has weight {np.round(i*100,2)} %')
```

    Stock BANK OF AMERICA CORP has weight 0.84 %
    Stock BERKSHIRE HATHAWAY B has weight 4.79 %
    Stock COMMONWEALTH BANK OF AUS has weight 37.08 %
    Stock GOLDMAN SACHS GROUP has weight 33.2 %
    Stock HSBC HOLDINGS (GB) has weight 0.26 %
    Stock JPMORGAN CHASE & CO has weight 2.02 %
    Stock MASTERCARD A has weight 1.36 %
    Stock ROYAL BANK OF CANADA has weight 6.34 %
    Stock VISA A has weight 3.03 %
    Stock WELLS FARGO & CO has weight 11.09 %


### Comparison


```python
# Weights
(optimal_weights_fs - monte_c_weights_fs)
```




    array([-0.00837525,  0.14258728,  0.16630205, -0.05954855, -0.00256049,
           -0.02016752, -0.01360042, -0.06338842, -0.03032042, -0.11092826])




```python
# Metrics of the portfolios
get_ret_vol_sr(optimal_weights_fs), get_ret_vol_sr(monte_c_weights_fs)

print('For a given portfolio we have: (Using Markowitz - SciPy)\n \n')
for i, j in enumerate('Return Volatility SharpeRatio'.split()):
    print(f'{j} is : {get_ret_vol_sr(optimal_weights_fs)[i]}\n')
print("\n")
print('For a given portfolio we have: (Using Monte Carlo)\n \n')
for i, j in enumerate('Return Volatility SharpeRatio'.split()):
    print(f'{j} is : {get_ret_vol_sr(monte_c_weights_fs)[i]}\n')
```

    For a given portfolio we have: (Using Markowitz - SciPy)
     
    
    Return is : 0.1892723056031749
    
    Volatility is : 0.2118498940779727
    
    SharpeRatio is : 0.685496229002793
    
    
    
    For a given portfolio we have: (Using Monte Carlo)
     
    
    Return is : 0.17841510723513962
    
    Volatility is : 0.23507638689473154
    
    SharpeRatio is : 0.5715806122085139


### Efficient Frontier


```python
frontier_y_fs = np.linspace(port_return_fs.min(), port_return_fs.max() * 1.1, 100)
frontier_vol_fs = []

def minimize_vol(weights_fs):
    return get_ret_vol_sr(weights_fs)[1]

for possible_ret_fs in frontier_y_fs:
    cons_fs = ({'type':'eq','fun':check_sum},
            {'type':'eq','fun':lambda w:get_ret_vol_sr(w)[0] - possible_ret_fs})
    result_fs = optimize.minimize(minimize_vol, init_guess_fs, method='SLSQP', constraints=cons_fs, bounds=bounds_fs)
    frontier_vol_fs.append(result_fs['fun'])
```

<img src="/assets/proj/Portfolio_construction_files/fs_2.png" alt="fs2" style="max-width: 100%; height: auto;">       

# 7. Energy Sector by **Alberto Pio Stigliani**

> NOTE:
>
> The approach is the same as the Financial Sector

## Data Processing


```python
#Tickers list
tickers_es = ["XOM", "CVX", "COP", "EOG","WMB", "EPD", "OKE", "KMI", "ET", "SLB"]

#Dict
ticker_name_es = {
    "XOM": "Exxon Mobil Corporation",
    "CVX": "Chevron Corporation",
    "COP": "ConocoPhillips",
    "EOG": "EOG Resources,",
    "WMB": "Williams Companies Inc.",
    "EPD": "Enterprise Products Partners",
    "OKE": "ONEOK",
    "KMI": "Kinder Morgan",
    "ET": "Energy Transfer",
    "SLB": "Schlumberger"
}
```

### Optimal Weights


    Stock Chevron Corporation has weight 0.0 %
    Stock ConocoPhillips has weight 0.0 %
    Stock EOG Resources, has weight 0.0 %
    Stock Energy Transfer has weight 0.0 %
    Stock Enterprise Products Partners has weight 0.0 %
    Stock Exxon Mobil Corporation has weight 0.0 %
    Stock Kinder Morgan has weight 0.0 %
    Stock ONEOK has weight 0.0 %
    Stock Schlumberger has weight 0.0 %
    Stock Williams Companies Inc. has weight 100.0 %


### Monte Carlo weights

    Stock Chevron Corporation has weight 8.4 %
    Stock ConocoPhillips has weight 0.91 %
    Stock EOG Resources, has weight 10.74 %
    Stock Energy Transfer has weight 11.04 %
    Stock Enterprise Products Partners has weight 0.2 %
    Stock Exxon Mobil Corporation has weight 6.88 %
    Stock Kinder Morgan has weight 0.12 %
    Stock ONEOK has weight 3.85 %
    Stock Schlumberger has weight 6.66 %
    Stock Williams Companies Inc. has weight 51.2 %


### Comparison and metrics


    For a given portfolio we have: (Using Markowitz - SciPy)
     
    
    Return is : 0.24390270933462732
    
    Volatility is : 0.35756417137002044
    
    SharpeRatio is : 0.5589282239067767
    
    
    
    For a given portfolio we have: (Using Monte Carlo)
     
    
    Return is : 0.1876335670238803
    
    Volatility is : 0.3455768547486279
    
    SharpeRatio is : 0.41548952990569715

### Efficient Frontier
    
<img src="/assets/proj/Portfolio_construction_files/es_2.png" alt="es2" style="max-width: 100%; height: auto;">       


# 8. Technology Sector by [**Robert Roman**](https://www.linkedin.com/in/robert-roman-998b99213/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)

> NOTE:
>
> The approach is the same as the Financial Sector

## Data Processing


```python
# Tickers list
tickers_ts = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "CRM", "AMD", "BABA"]

# Dictionary to store stocks' name
ticker_name_ts = {
    "NVDA": "NVIDIA",
    "AAPL": "APPLE",
    "MSFT": "MICROSOFT",
    "AMZN": "AMAZON",
    "GOOG": "GOOGLE",
    "META": "META",
    "TSLA": "TESLA",
    "CRM": "SALESFORCE",
    "AMD": "AMD",
    "BABA": "ALIBABA"
}
```

### Optimal Weights


    Stock NVIDIA has weight 86.26 %
    Stock TESLA has weight 13.74 %
    Stock APPLE has weight 0.0 %
    Stock AMD has weight 0.0 %
    Stock AMAZON has weight 0.0 %
    Stock ALIBABA has weight 0.0 %
    Stock SALESFORCE has weight 0.0 %
    Stock GOOGLE has weight 0.0 %
    Stock META has weight 0.0 %
    Stock MICROSOFT has weight 0.0 %


### Monte Carlo Weights


    Stock NVIDIA has weight 51.2 %
    Stock SALESFORCE has weight 11.04 %
    Stock AMAZON has weight 10.74 %
    Stock AMD has weight 8.4 %
    Stock TESLA has weight 6.88 %
    Stock MICROSOFT has weight 6.66 %
    Stock META has weight 3.85 %
    Stock APPLE has weight 0.91 %
    Stock ALIBABA has weight 0.2 %
    Stock GOOGLE has weight 0.12 %


### Comparison and metrics

    For a given portfolio we have: (Using Markowitz - SciPy)
     
    
    Return is : 0.620340335724262
    
    Volatility is : 0.5153004255467142
    
    SharpeRatio is : 1.1183579617943382
    
    
    
    For a given portfolio we have: (Using Monte Carlo)
     
    
    Return is : 0.4359913020704139
    
    Volatility is : 0.41792222819928737
    
    SharpeRatio is : 0.9378331027308557


### Efficient Frontier

<img src="/assets/proj/Portfolio_construction_files/ts_2.png" alt="ts2" style="max-width: 100%; height: auto;">      


# 9. Health Sector by [**Sahar Shirazi**](https://www.linkedin.com/in/sahar-shirazi-906b3a167/)

> NOTE:
>
> The approach is the same as the Financial Sector

## Data Processing

```python
# Tickers list
tickers_hs = ["JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "LLY", "MDT", "AMGN", "GILD"]
# Dictionary to store stocks' name

ticker_name_hs = {
    "JNJ": "Johnson & Johnson",
    "PFE": "BPfizer Inc. ",
    "UNH": "UnitedHealth Group Incorporated",
    "ABBV": "MAbbVie Inc.",
    "MRK": "Merck & Co., Inc.",
    "TMO": "Thermo Fisher Scientific Inc.",
    "LLY": "REli Lilly and Company",
    "MDT": "Medtronic plc",
    "AMGN": "GAmgen Inc.",
    "GILD": "Gilead Sciences, Inc."
}
```

### Optimal Weights


    Stock MAbbVie Inc. has weight 19.98 %
    Stock GAmgen Inc. has weight 0.0 %
    Stock Gilead Sciences, Inc. has weight 0.0 %
    Stock Johnson & Johnson has weight 0.0 %
    Stock REli Lilly and Company has weight 80.02 %
    Stock Medtronic plc has weight 0.0 %
    Stock Merck & Co., Inc. has weight 0.0 %
    Stock BPfizer Inc.  has weight 0.0 %
    Stock Thermo Fisher Scientific Inc. has weight 0.0 %
    Stock UnitedHealth Group Incorporated has weight 0.0 %


### Monte Carlo weights


    Stock MAbbVie Inc. has weight 15.94 %
    Stock GAmgen Inc. has weight 4.75 %
    Stock Gilead Sciences, Inc. has weight 22.06 %
    Stock Johnson & Johnson has weight 1.09 %
    Stock REli Lilly and Company has weight 30.06 %
    Stock Medtronic plc has weight 1.31 %
    Stock Merck & Co., Inc. has weight 4.0 %
    Stock BPfizer Inc.  has weight 2.17 %
    Stock Thermo Fisher Scientific Inc. has weight 4.21 %
    Stock UnitedHealth Group Incorporated has weight 14.42 %


### Comparison and metrics


    For a given portfolio we have: (Using Markowitz - SciPy)
     
    
    Return is : 0.3321472564404809
    
    Volatility is : 0.28174588380820687
    
    SharpeRatio is : 1.0225429044369432
    
    
    
    For a given portfolio we have: (Using Monte Carlo)
     
    
    Return is : 0.19652739686583295
    
    Volatility is : 0.19970382890433205
    
    SharpeRatio is : 0.7635176331085451


### Efficient Frontier

    
<img src="/assets/proj/Portfolio_construction_files/hs_2.png" alt="hs2" style="max-width: 100%; height: auto;">      


# 10. General Portfolio

We ended up with the following stocks:

1. GOLDMAN SACHS GROUP

2. COMMONWEALTH BANK OF AUS

3. BERKSHIRE HATHAWAY B

4. Williams Companies Inc.

5. NVIDIA

6. TESLA

7. REli Lilly and Company

8. AbbVie Inc.



## Data Processing


```python
tickers_gp = ["GS", "CBA.AX", "BRK-B", "WMB", "NVDA", "TSLA", "LLY", "ABBV"]
ticker_name_gp = {
    "GS": "GOLDMAN SACHS GROUP",
    "CBA.AX": "COMMONWEALTH BANK OF AUS",
    "BRK-B": "BERKSHIRE HATHAWAY B",
    "WMB": "Williams Companies Inc.",
    "NVDA": "NVIDIA",
    "TSLA": "TESLA",
    "LLY": "REli Lilly and Company",
    "ABBV": "MAbbVie Inc."
}
```

### Optimal Weights

    Stock REli Lilly and Company has weight 42.04 %
    Stock NVIDIA has weight 25.42 %
    Stock COMMONWEALTH BANK OF AUS has weight 21.3 %
    Stock TESLA has weight 5.68 %
    Stock Williams Companies Inc. has weight 4.72 %
    Stock MAbbVie Inc. has weight 0.85 %
    Stock BERKSHIRE HATHAWAY B has weight 0.0 %
    Stock GOLDMAN SACHS GROUP has weight 0.0 %


### Monte Carlo Weights


    Stock REli Lilly and Company has weight 40.77 %
    Stock NVIDIA has weight 23.62 %
    Stock COMMONWEALTH BANK OF AUS has weight 20.53 %
    Stock MAbbVie Inc. has weight 3.76 %
    Stock GOLDMAN SACHS GROUP has weight 3.38 %
    Stock TESLA has weight 3.12 %
    Stock BERKSHIRE HATHAWAY B has weight 2.76 %
    Stock Williams Companies Inc. has weight 2.06 %


### Comparison and metrics


    For a given portfolio we have: (Using Markowitz - SciPy)
     
    
    Return is : 0.40627639504115
    
    Volatility is : 0.25896239653732983
    
    SharpeRatio is : 1.3987605837237878
    
    
    
    For a given portfolio we have: (Using Monte Carlo)
     
    
    Return is : 0.38575239330114636
    
    Volatility is : 0.247542089402651
    
    SharpeRatio is : 1.3803809769386373


### Efficient Frontier

    
<img src="/assets/proj/Portfolio_construction_files/gp_2.png" alt="gp" style="max-width: 100%; height: auto;">      


### Piechart


```python

# Data for Optimization weights
optimization_weights = {
    "REli Lilly and Company": 42.04,
    "NVIDIA": 25.42,
    "COMMONWEALTH BANK OF AUS": 21.3,
    "TESLA": 5.68,
    "Williams Companies Inc.": 4.72,
    "MAbbVie Inc.": 0.85
}

# Data for Monte Carlo weights
monte_carlo_weights = {
    "REli Lilly and Company": 40.77,
    "NVIDIA": 23.62,
    "COMMONWEALTH BANK OF AUS": 20.53,
    "MAbbVie Inc.": 3.76,
    "GOLDMAN SACHS GROUP": 3.38,
    "TESLA": 3.12,
    "BERKSHIRE HATHAWAY B": 2.76,
    "Williams Companies Inc.": 2.06
}

# Extract labels and values for both strategies
labels_opt = list(optimization_weights.keys())
values_opt = list(optimization_weights.values())

labels_mc = list(monte_carlo_weights.keys())
values_mc = list(monte_carlo_weights.values())

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# Plot Optimization weights pie chart
axes[0].pie(
    values_opt,
    labels=None,  # No labels directly on the chart
    startangle=90
)
axes[0].set_title('Markowitz Weights', fontsize=14)
# Add external legend for Optimization weights
opt_labels = [f"{label}: {value:.2f}%" for label, value in zip(labels_opt, values_opt)]
axes[0].legend(opt_labels, loc="center left", bbox_to_anchor=(1, 0.5))

# Plot Monte Carlo weights pie chart
axes[1].pie(
    values_mc,
    labels=None,  # No labels directly on the chart
    startangle=90
)
axes[1].set_title('Monte Carlo Weights', fontsize=14)
# Add external legend for Monte Carlo weights
mc_labels = [f"{label}: {value:.2f}%" for label, value in zip(labels_mc, values_mc)]
axes[1].legend(mc_labels, loc="center left", bbox_to_anchor=(1, 0.5))

# Adjust layout for better display
plt.tight_layout()
plt.show()

```


    
<img src="/assets/proj/Portfolio_construction_files/weights.png" alt="weights" style="max-width: 100%; height: auto;">      


# References

- [Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77-91. Wiley for the American Finance Association](https://www.jstor.org/stable/2975974)

- [https://www.kaggle.com/code/bhavinmoriya/markowitz-portfolio-optimization](https://www.kaggle.com/code/bhavinmoriya/markowitz-portfolio-optimization)

