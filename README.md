Majorly contains two files namely trading_model_comparison.py and data_test.py. :

1)trading_model_comparison.py :
Data Loading and Preprocessing:
The script loads the dataset from a Parquet file ('data.parquet') containing minute-level implied volatilities and other relevant data.
Missing values in the dataset are handled by forward filling.

Base Model (Z-Score Trading System):
The z-score of the spread between Bank Nifty and Nifty is computed to identify trading signals.
A z-score trading strategy is implemented, where buy signals occur when the z-score falls below a specified threshold, and sell signals occur when it exceeds another threshold.
Profit/Loss (P/L) is calculated based on the trading positions taken.

Proposed Model (Random Forest Regression):
Feature engineering is performed by selecting relevant features ('banknifty', 'nifty', 'tte') and target variable ('Spread').
The dataset is split into training and testing sets using train_test_split.
A Random Forest Regression model is trained on the training data to predict the spread between Bank Nifty and Nifty.
Trading positions are determined based on the predicted spread, and P/L is calculated accordingly.

Evaluation and Comparison:
Performance metrics such as total P/L, Sharpe Ratio, and Drawdown are computed for both the base and proposed models.
Results are printed to compare the performance of the two models.

Usage:
Ensure that the 'data.parquet' file containing the dataset is available in the same directory as the script.
Run the script to compare the performance of the base and proposed trading models.

Dependencies:
pandas
scikit-learn

2) data_test.py :
   
The code utilizes the pandas library to read data from a Parquet file named 'data.parquet' into a DataFrame named 'data'.
Parquet is a columnar storage file format commonly used for storing large datasets efficiently.
