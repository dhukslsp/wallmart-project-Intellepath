import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime

# Read the data
df = pd.read_csv('Walmart-Project/Walmart DataSet/Walmart DataSet.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Handle missing values
def handle_missing_values():
    print("\nMissing Values Analysis:")
    print(df.isnull().sum())
    
    # Fill missing values with appropriate methods
    df['Temperature'].fillna(df['Temperature'].mean(), inplace=True)
    df['Fuel_Price'].fillna(df['Fuel_Price'].mean(), inplace=True)
    df['CPI'].fillna(df['CPI'].mean(), inplace=True)
    df['Unemployment'].fillna(df['Unemployment'].mean(), inplace=True)
    
    print("\nMissing values after handling:")
    print(df.isnull().sum())

# Outlier Analysis
def analyze_outliers():
    print("\nOutlier Analysis:")
    for column in ['Weekly_Sales', 'Temperature', 'CPI', 'Unemployment']:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
        print(f"\n{column} outliers: {len(outliers)}")
        print(f"Percentage of outliers: {(len(outliers)/len(df))*100:.2f}%")

# 1. Impact of Unemployment Rate on Sales
def analyze_unemployment_impact():
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='Unemployment', y='Weekly_Sales')
    plt.title('Weekly Sales vs Unemployment Rate')
    plt.xlabel('Unemployment Rate (%)')
    plt.ylabel('Weekly Sales ($)')
    plt.savefig('unemployment_impact.png')
    plt.close()
    
    # Calculate correlation
    correlation = df['Weekly_Sales'].corr(df['Unemployment'])
    print(f"\nCorrelation between Unemployment and Sales: {correlation:.3f}")
    
    # Analyze store-wise impact
    store_unemployment_impact = df.groupby('Store').apply(
        lambda x: x['Weekly_Sales'].corr(x['Unemployment'])
    ).sort_values()
    
    print("\nStores most affected by unemployment (negative correlation):")
    print(store_unemployment_impact.head())
    
    print("\nStores least affected by unemployment (positive correlation):")
    print(store_unemployment_impact.tail())

# 2. Seasonal Trends Analysis
def analyze_seasonal_trends():
    # Add month and year columns
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    
    # Monthly average sales
    monthly_sales = df.groupby('Month')['Weekly_Sales'].mean()
    
    plt.figure(figsize=(12, 6))
    monthly_sales.plot(kind='bar')
    plt.title('Average Monthly Sales')
    plt.xlabel('Month')
    plt.ylabel('Average Weekly Sales ($)')
    plt.savefig('seasonal_trends.png')
    plt.close()
    
    # Analyze peak and low seasons
    print("\nPeak and Low Seasons Analysis:")
    print("Peak months (highest sales):")
    print(monthly_sales.nlargest(3))
    print("\nLow months (lowest sales):")
    print(monthly_sales.nsmallest(3))

# 3. Temperature Impact Analysis
def analyze_temperature_impact():
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='Temperature', y='Weekly_Sales')
    plt.title('Weekly Sales vs Temperature')
    plt.xlabel('Temperature (°F)')
    plt.ylabel('Weekly Sales ($)')
    plt.savefig('temperature_impact.png')
    plt.close()
    
    # Calculate correlation
    correlation = df['Weekly_Sales'].corr(df['Temperature'])
    print(f"\nCorrelation between Temperature and Sales: {correlation:.3f}")
    
    # Analyze temperature ranges impact
    df['Temp_Range'] = pd.qcut(df['Temperature'], q=4)
    temp_range_sales = df.groupby('Temp_Range')['Weekly_Sales'].mean()
    
    print("\nSales by Temperature Range:")
    print(temp_range_sales)

# 4. CPI Impact Analysis
def analyze_cpi_impact():
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='CPI', y='Weekly_Sales')
    plt.title('Weekly Sales vs Consumer Price Index')
    plt.xlabel('CPI')
    plt.ylabel('Weekly Sales ($)')
    plt.savefig('cpi_impact.png')
    plt.close()
    
    # Calculate correlation
    correlation = df['Weekly_Sales'].corr(df['CPI'])
    print(f"\nCorrelation between CPI and Sales: {correlation:.3f}")
    
    # Analyze store-wise CPI impact
    store_cpi_impact = df.groupby('Store').apply(
        lambda x: x['Weekly_Sales'].corr(x['CPI'])
    ).sort_values()
    
    print("\nStores most affected by CPI changes:")
    print(store_cpi_impact.head())
    print("\nStores least affected by CPI changes:")
    print(store_cpi_impact.tail())

# 5. Store Performance Analysis
def analyze_store_performance():
    # Calculate performance metrics
    store_performance = df.groupby('Store')['Weekly_Sales'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).round(2)
    
    # Create box plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Store', y='Weekly_Sales')
    plt.title('Sales Distribution by Store')
    plt.xlabel('Store ID')
    plt.ylabel('Weekly Sales ($)')
    plt.xticks(rotation=45)
    plt.savefig('store_performance.png')
    plt.close()
    
    # Print top and bottom performing stores
    print("\nTop 5 Performing Stores (by average sales):")
    print(store_performance.sort_values('mean', ascending=False).head())
    
    print("\nBottom 5 Performing Stores (by average sales):")
    print(store_performance.sort_values('mean', ascending=True).head())
    
    # Calculate performance gap
    best_store = store_performance['mean'].max()
    worst_store = store_performance['mean'].min()
    performance_gap = best_store - worst_store
    gap_percentage = (performance_gap / worst_store) * 100
    
    print(f"\nPerformance Gap Analysis:")
    print(f"Difference between best and worst performing stores: ${performance_gap:,.2f}")
    print(f"Percentage difference: {gap_percentage:.2f}%")

# Run all analyses
def run_comprehensive_analysis():
    print("Starting comprehensive analysis...")
    
    handle_missing_values()
    analyze_outliers()
    
    analyze_unemployment_impact()
    print("\nUnemployment impact analysis completed.")
    
    analyze_seasonal_trends()
    print("Seasonal trends analysis completed.")
    
    analyze_temperature_impact()
    print("Temperature impact analysis completed.")
    
    analyze_cpi_impact()
    print("CPI impact analysis completed.")
    
    analyze_store_performance()
    print("Store performance analysis completed.")
    
    print("\nAll analyses completed. Check the generated plots for visual insights.")

if __name__ == "__main__":
    run_comprehensive_analysis() 