"""
Traffic Data Preprocessing - Python Script Version
Run this instead of .ipynb if you prefer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import joblib
warnings.filterwarnings('ignore')

def main():
    print("🚦 Addis Ababa Traffic Data Preprocessing")
    print("="*60)
    
    # Load data (you need to have the CSV file)
    try:
        df = pd.read_csv('addis_ababa_traffic_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"✅ Data loaded: {len(df)} records")
    except:
        print("❌ Error: Could not load data. Run collect_traffic_data.py first")
        return
    
    # Create features
    print("\n⚙️ Creating features...")
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 16) & (df['hour'] <= 19))
    df['is_rush_hour'] = df['is_rush_hour'].astype(int)
    
    # One-hot encoding for locations
    location_dummies = pd.get_dummies(df['location'], prefix='loc')
    df = pd.concat([df, location_dummies], axis=1)
    
    # Select features
    feature_columns = [
        'hour', 'day_of_week', 'month', 'is_weekend', 
        'is_holiday', 'is_rush_hour', 'rainfall_mm', 'temperature_c',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
    ] + list(location_dummies.columns)
    
    X = df[feature_columns]
    y = df['congestion_level']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save files
    joblib.dump(scaler, 'traffic_scaler.pkl')
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('X_train_processed.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('X_test_processed.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    
    print(f"✅ Data processed:")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Testing samples: {X_test.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    print("\n💾 Files saved:")
    print("   - traffic_scaler.pkl")
    print("   - X_train_processed.csv")
    print("   - X_test_processed.csv")
    print("   - y_train.csv")
    print("   - y_test.csv")

if __name__ == "__main__":
    main()