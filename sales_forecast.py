import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Read and prepare the data
df = pd.read_csv('Walmart-Project/Walmart DataSet/Walmart DataSet.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Add time-based features
def add_time_features(df):
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    # Create seasonal features using sine and cosine transforms
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month']/12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month']/12)
    df['Week_Sin'] = np.sin(2 * np.pi * df['Week']/52)
    df['Week_Cos'] = np.cos(2 * np.pi * df['Week']/52)
    return df

# Create lag features
def add_lag_features(df, store_id, lags=[1, 2, 3, 4]):
    store_data = df[df['Store'] == store_id].copy()
    for lag in lags:
        store_data[f'Sales_Lag_{lag}'] = store_data['Weekly_Sales'].shift(lag)
    return store_data.dropna()

# Create rolling mean features
def add_rolling_features(df):
    df['Sales_Rolling_Mean_4'] = df['Weekly_Sales'].rolling(window=4).mean()
    df['Sales_Rolling_Mean_8'] = df['Weekly_Sales'].rolling(window=8).mean()
    df['Sales_Rolling_Mean_12'] = df['Weekly_Sales'].rolling(window=12).mean()
    return df

# Prepare features for modeling
def prepare_features(df):
    feature_columns = [
        'Month_Sin', 'Month_Cos', 'Week_Sin', 'Week_Cos',
        'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
        'Holiday_Flag', 'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_3', 'Sales_Lag_4',
        'Sales_Rolling_Mean_4', 'Sales_Rolling_Mean_8', 'Sales_Rolling_Mean_12'
    ]
    return feature_columns

# Train model for a specific store
def train_store_model(store_data, feature_columns):
    X = store_data[feature_columns]
    y = store_data['Weekly_Sales']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, mae, rmse, r2

# Generate future dates and features
def generate_future_features(last_date, periods=12):
    future_dates = pd.date_range(start=last_date + timedelta(days=7), periods=periods, freq='W')
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['Year'] = future_df['Date'].dt.year
    future_df['Month'] = future_df['Date'].dt.month
    future_df['Week'] = future_df['Date'].dt.isocalendar().week
    future_df['Month_Sin'] = np.sin(2 * np.pi * future_df['Month']/12)
    future_df['Month_Cos'] = np.cos(2 * np.pi * future_df['Month']/12)
    future_df['Week_Sin'] = np.sin(2 * np.pi * future_df['Week']/52)
    future_df['Week_Cos'] = np.cos(2 * np.pi * future_df['Week']/52)
    return future_df

# Forecast future sales for each store
def forecast_store_sales():
    print("Starting sales forecast for each store...")
    
    # Prepare the data
    df_processed = add_time_features(df.copy())
    feature_columns = prepare_features(df_processed)
    
    # Store forecasts and metrics
    store_forecasts = {}
    store_metrics = {}
    
    for store_id in df['Store'].unique():
        print(f"\nProcessing Store {store_id}")
        
        # Prepare store-specific data
        store_data = df_processed[df_processed['Store'] == store_id].copy()
        store_data = add_lag_features(store_data, store_id)
        store_data = add_rolling_features(store_data)
        
        # Train model
        model, scaler, mae, rmse, r2 = train_store_model(store_data, feature_columns)
        store_metrics[store_id] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        
        # Generate future features
        last_date = store_data['Date'].max()
        future_df = generate_future_features(last_date)
        
        # Use last known values for external features
        last_known = store_data.iloc[-1]
        future_df['Temperature'] = last_known['Temperature']
        future_df['Fuel_Price'] = last_known['Fuel_Price']
        future_df['CPI'] = last_known['CPI']
        future_df['Unemployment'] = last_known['Unemployment']
        future_df['Holiday_Flag'] = 0  # Default to non-holiday
        
        # Initialize lag features with last known values
        future_df['Sales_Lag_1'] = last_known['Weekly_Sales']
        future_df['Sales_Lag_2'] = store_data['Weekly_Sales'].iloc[-2]
        future_df['Sales_Lag_3'] = store_data['Weekly_Sales'].iloc[-3]
        future_df['Sales_Lag_4'] = store_data['Weekly_Sales'].iloc[-4]
        
        # Initialize rolling means with last known values
        future_df['Sales_Rolling_Mean_4'] = store_data['Weekly_Sales'].tail(4).mean()
        future_df['Sales_Rolling_Mean_8'] = store_data['Weekly_Sales'].tail(8).mean()
        future_df['Sales_Rolling_Mean_12'] = store_data['Weekly_Sales'].tail(12).mean()
        
        # Make predictions
        X_future = future_df[feature_columns]
        X_future_scaled = scaler.transform(X_future)
        predictions = model.predict(X_future_scaled)
        
        # Store forecasts
        store_forecasts[store_id] = pd.DataFrame({
            'Date': future_df['Date'],
            'Forecasted_Sales': predictions
        })
        
        # Plot actual vs forecasted
        plt.figure(figsize=(12, 6))
        plt.plot(store_data['Date'], store_data['Weekly_Sales'], label='Actual Sales')
        plt.plot(future_df['Date'], predictions, label='Forecasted Sales', linestyle='--')
        plt.title(f'Store {store_id} - Sales Forecast')
        plt.xlabel('Date')
        plt.ylabel('Weekly Sales ($)')
        plt.legend()
        plt.savefig(f'store_{store_id}_forecast.png')
        plt.close()
        
        print(f"Store {store_id} Metrics:")
        print(f"MAE: ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"R2 Score: {r2:.3f}")
        print("\nNext 12 weeks forecast:")
        print(store_forecasts[store_id])
    
    return store_forecasts, store_metrics

if __name__ == "__main__":
    forecasts, metrics = forecast_store_sales() 