## Sales Forecasting Project

### Introduction
Welcome to the Sales Forecasting Project. This repository contains code for processing, feature engineering, and forecasting sales data using machine learning and statistical models. The goal is to build an accurate forecasting model that predicts future sales based on historical data, holidays, promotions, and other external factors.

This project is structured into two main parts:
1. **Data Processing and Feature Engineering** - Cleaning, transforming, and exploring the dataset.
2. **Model Selection, Forecasting, and Evaluation** - Training different forecasting models, comparing their performance, and presenting insights.

### Dataset Overview
The dataset includes:
- **Historical sales data**
- **Store metadata**
- **Economic indicators (e.g., oil prices)**
- **Holidays and promotional events**

The objective is to forecast daily sales for each product category across different stores.

---
## Part 1: Data Processing and Feature Engineering
### 1. Data Cleaning
- Load the dataset using Pandas.
  ```python
  train = pd.read_csv('train.csv', parse_dates=['date'])
  stores = pd.read_csv('stores.csv')
  oil = pd.read_csv('oil.csv', parse_dates=['date'])
  holidays = pd.read_csv('holidays_events.csv', parse_dates=['date'])
  ```
- Handle missing values in oil prices.
  ```python
  oil['dcoilwtico'] = oil['dcoilwtico'].interpolate(method='linear')
  ```
- Merge relevant data sources.
  ```python
  train = train.merge(stores, on='store_nbr', how='left')
  train = train.merge(oil, on='date', how='left')
  train = train.merge(holidays, on='date', how='left')
  ```

### 2. Feature Engineering
- Extract time-based features (day, week, month, year, weekday, etc.).
  ```python
  train['day'] = train['date'].dt.day
  train['month'] = train['date'].dt.month
  train['year'] = train['date'].dt.year
  train['day_of_week'] = train['date'].dt.dayofweek
  ```
- Create event-based features.
  ```python
  train['is_holiday'] = train['type'].notna().astype(int)
  train['is_weekend'] = (train['day_of_week'] >= 5).astype(int)
  ```
- Implement rolling statistics.
  ```python
  train['sales_lag_7'] = train.groupby(['store_nbr', 'family'])['sales'].shift(7)
  train['rolling_mean_7'] = train.groupby(['store_nbr', 'family'])['sales'].rolling(7).mean().reset_index(level=[0,1], drop=True)
  ```

### 3. Exploratory Data Analysis (EDA)
- Visualize sales trends over time.
  ```python
  plt.figure(figsize=(12, 6))
  sns.lineplot(x='date', y='sales', data=train, label='Sales Trend')
  plt.title('Sales Trends Over Time')
  plt.show()
  ```

---
## Part 2: Model Selection, Forecasting, and Evaluation
### 1. Model Training
Train multiple forecasting models:
- **ARIMA (AutoRegressive Integrated Moving Average)**
  ```python
  arima_model = ARIMA(y_train, order=(2,1,0))
  arima_model_fit = arima_model.fit()
  ```
- **Random Forest Regressor**
  ```python
  rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
  rf_model.fit(X_train, y_train)
  ```
- **XGBoost**
  ```python
  xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
  xgb_model.fit(X_train, y_train)
  ```

### 2. Model Evaluation
Evaluate models using:
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R² Score
  ```python
  def evaluate_model(y_true, y_pred, model_name):
      rmse = np.sqrt(mean_squared_error(y_true, y_pred))
      mape = mean_absolute_percentage_error(y_true, y_pred)
      r2 = r2_score(y_true, y_pred)
      print(f"{model_name} - RMSE: {rmse:.2f}, MAPE: {mape:.2%}, R²: {r2:.2f}")
  ```

### 3. Visualization
- Plot actual vs. predicted sales.
  ```python
  plt.figure(figsize=(12, 6))
  sns.lineplot(x=y_val.index, y=y_val, label='Actual Sales')
  sns.lineplot(x=y_val.index, y=rf_preds, label='Random Forest Predictions')
  sns.lineplot(x=y_val.index, y=xgb_preds, label='XGBoost Predictions')
  plt.legend()
  plt.title('Actual vs Predicted Sales')
  plt.show()
  ```

### 4. Interpretation and Business Insights
#### Model Performance
| Model | RMSE ↓ | MAPE ↓ | R² ↑ |
| ------------ | ---------- | --------- | --------- |
| ARIMA | 1411.32 | High | -0.06 |
| Random Forest | 431.90 | Moderate | 0.90 |
| XGBoost | **307.43** | **Best** | **0.95** |

#### Business Strategies
- **Inventory Planning**: Optimize stock levels for peak demand periods.
- **Targeted Promotions**: Schedule promotions during forecasted low-sales periods.
- **Price Adjustments**: Monitor external economic factors and adjust pricing strategies accordingly.

By leveraging these insights, businesses can improve demand forecasting, reduce stockouts, and enhance customer satisfaction.

---
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
