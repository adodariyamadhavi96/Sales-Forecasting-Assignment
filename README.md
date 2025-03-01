## Sales Forecasting Project

This repository contains code for processing, feature engineering, and forecasting sales data using machine learning and statistical models.

### Notebooks Overview

1. **Data Processing and Feature Engineering**  
   - Implements rolling statistics and lag features for time series forecasting.  
   - Analyzes the impact of holidays and promotions on sales.  
   - Example features:
     ```python
     train['sales_lag_7'] = train.groupby(['store_nbr', 'family'])['sales'].shift(7)
     train['rolling_mean_7'] = train.groupby(['store_nbr', 'family'])['sales'].rolling(7).mean().reset_index(level=[0,1], drop=True)
     ```

2. **Model Selection, Forecasting, and Evaluation**  
   - Uses machine learning models like `RandomForestRegressor` and `XGBRegressor`.  
   - Implements statistical models such as ARIMA.  
   - Evaluates model performance using RMSE, MAPE, and R² scores.  
   - Example imports:
     ```python
     from sklearn.ensemble import RandomForestRegressor
     from xgboost import XGBRegressor
     from statsmodels.tsa.arima.model import ARIMA
     ```

### Requirements

To run the notebooks, install the required dependencies using:

```bash
pip install pandas numpy scikit-learn xgboost statsmodels matplotlib seaborn
```

### How to Use

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. Run Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open the notebooks and execute the cells sequentially.
