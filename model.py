import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Apply standard style for seaborn plots
sns.set_theme(style="whitegrid")

def load_data(path):
    print("1. Loading dataset...")
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    print("2. Cleaning, preprocessing and engineering features...")
    # Filter out entries where price is 0 (anomalies)
    df = df[df['price'] > 0]
    
    # Drop columns that have minimal individual predictive value for a simplified regression
    # Or could be overly complex without feature engineering (e.g. date)
    df = df.drop(['date', 'street', 'country'], axis=1)
    
    # Extract 'house_age' and 'is_renovated' to handle years effectively
    df['house_age'] = 2014 - df['yr_built']
    df['is_renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
    
    # Drop original year features
    df = df.drop(['yr_built', 'yr_renovated'], axis=1)

    # Encode categorical variables: city, statezip
    le_city = LabelEncoder()
    df['city'] = le_city.fit_transform(df['city'])

    le_statezip = LabelEncoder()
    df['statezip'] = le_statezip.fit_transform(df['statezip'])
    
    return df

def perform_eda(df):
    print("3. Performing Exploratory Data Analysis (EDA) and saving plots...")
    
    # Feature Correlation Heatmap
    plt.figure(figsize=(14, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    # Distribution of House Prices
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'] / 1e6, bins=50, kde=True, color='blue')
    plt.xlabel('Price (in millions)')
    plt.title('Distribution of House Prices')
    plt.tight_layout()
    plt.savefig('price_distribution.png')
    plt.close()

def train_and_evaluate(df):
    print("4. Training regression models and evaluating accuracy...")
    
    X = df.drop('price', axis=1)
    y = df['price']

    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling Numerical Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------ Linear Regression ------------------
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    lr_r2 = r2_score(y_test, lr_pred)

    print("\n--- Linear Regression Performance ---")
    print(f"RMSE: ${lr_rmse:,.2f}")
    print(f"R² Score: {lr_r2:.4f}")

    # ------------------ Random Forest Regressor ------------------
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)

    print("\n--- Random Forest Regression Performance ---")
    print(f"RMSE: ${rf_rmse:,.2f}")
    print(f"R² Score: {rf_r2:.4f}")

    # Save Feature Importance Plot
    feature_importance = rf_model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.figure(figsize=(10, 8))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, X.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    # Create Actual vs Predicted plot for Random Forest
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, rf_pred, alpha=0.3, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices (Random Forest)')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    plt.close()

if __name__ == "__main__":
    data_file = 'data.csv'
    
    df = load_data(data_file)
    df_processed = preprocess_data(df)
    perform_eda(df_processed)
    train_and_evaluate(df_processed)
    
    print("\nRun complete! Generated visualization files: correlation_heatmap.png, price_distribution.png, feature_importance.png, and actual_vs_predicted.png")
