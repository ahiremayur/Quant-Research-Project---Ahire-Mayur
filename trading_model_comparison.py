import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_parquet('data.parquet')

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Calculate spread
data['Spread'] = data['banknifty'] - data['nifty']

# Compute z-score of the spread
data['Z-Score'] = (data['Spread'] - data['Spread'].rolling(window=252).mean()) / data['Spread'].rolling(window=252).std()

# Define trading logic
def zscore_trading_strategy(data, entry_threshold=-2, exit_threshold=0):
    positions = []
    for z_score in data['Z-Score']:
        if z_score < entry_threshold:
            positions.append(1)  # Buy signal
        elif z_score > exit_threshold:
            positions.append(-1)  # Sell signal
        else:
            positions.append(0)  # Hold signal
    return positions

# Apply trading strategy
data['Position'] = zscore_trading_strategy(data)

# Calculate P/L
data['P/L'] = data['Position'] * (data['Spread'] * data['tte'] ** 0.7)

# Evaluate base model
base_model_pl = data['P/L'].sum()
base_model_sharpe_ratio = data['P/L'].mean() / data['P/L'].std()
base_model_drawdown = (data['P/L'].cumsum() - data['P/L'].cumsum().cummax()).min()


# Feature Engineering
X = data[['banknifty', 'nifty', 'tte']]
y = data['Spread']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict spread
data['Predicted Spread'] = rf_model.predict(X)

# Calculate P/L using predicted spread
data['P/L_RF'] = zscore_trading_strategy(data) * (data['Predicted Spread'] * data['tte'] ** 0.7)

# Evaluate Random Forest model
rf_model_pl = data['P/L_RF'].sum()
rf_model_sharpe_ratio = data['P/L_RF'].mean() / data['P/L_RF'].std()
rf_model_drawdown = (data['P/L_RF'].cumsum() - data['P/L_RF'].cumsum().cummax()).min()

# Print results
print("Base Model:")
print("Total P/L:", base_model_pl)
print("Sharpe Ratio:", base_model_sharpe_ratio)
print("Drawdown:", base_model_drawdown)

print("\nProposed Model:")
print("Total P/L:", rf_model_pl)
print("Sharpe Ratio:", rf_model_sharpe_ratio)
print("Drawdown:", rf_model_drawdown)

