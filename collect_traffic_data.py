"""
Addis Ababa Traffic Data Collection Script
Group 5 - Ethiopian Traffic Congestion Predictor
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_traffic_data():
    """Create realistic sample traffic data for Addis Ababa"""
    
    # Major traffic locations in Addis Ababa
    locations = ['Megenagna', 'Bole', 'Mexico', 'Piassa', 'Sar Bet', 'Gotera', 'Kality', 'Bole Airport']
    
    # Generate 60 days of data
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(60)]
    
    traffic_data = []
    
    for date in dates:
        for location in locations:
            for hour in range(5, 23):  # 5 AM to 10 PM
                # Base congestion level
                base_congestion = 35
                
                # RUSH HOUR EFFECT (7-9 AM, 4-7 PM)
                if hour in [7, 8, 9, 16, 17, 18, 19]:
                    base_congestion += 35
                
                # LOCATION EFFECT
                if location == 'Megenagna':
                    base_congestion += 20  # Always congested
                elif location == 'Mexico':
                    base_congestion += 18
                elif location == 'Bole':
                    base_congestion += 15
                
                # DAY OF WEEK (Monday=0)
                day_of_week = date.weekday()
                if day_of_week < 5:  # Weekday
                    base_congestion += 25
                else:  # Weekend
                    base_congestion -= 10
                
                # MONTH/SEASON EFFECT
                month = date.month
                if month in [6, 7, 8, 9]:  # Rainy season
                    base_congestion += 10
                elif month in [12, 1]:  # Holiday season
                    base_congestion += 5
                
                # Add randomness
                congestion = base_congestion + np.random.randint(-8, 8)
                congestion = max(10, min(98, congestion))
                
                # Weather simulation
                if month in [6, 7, 8, 9]:  # Rainy season
                    rainfall = np.random.uniform(0, 15)
                    if rainfall > 5:
                        congestion += min(15, rainfall)  # More rain = more congestion
                else:
                    rainfall = np.random.uniform(0, 3)
                
                # Temperature
                if month in [10, 11, 12, 1, 2]:  # Cooler months
                    temperature = np.random.uniform(15, 22)
                else:
                    temperature = np.random.uniform(18, 28)
                
                # Special days
                is_holiday = 0
                if (date.month == 1 and date.day == 7) or (date.month == 9 and date.day == 11):
                    is_holiday = 1
                    congestion -= 20  # Less traffic on holidays
                
                # Create data point
                traffic_data.append({
                    'timestamp': date.replace(hour=hour),
                    'location': location,
                    'hour': hour,
                    'day_of_week': day_of_week,
                    'month': month,
                    'is_weekend': 1 if day_of_week >= 5 else 0,
                    'is_holiday': is_holiday,
                    'rainfall_mm': round(rainfall, 1),
                    'temperature_c': round(temperature, 1),
                    'congestion_level': int(congestion)
                })
    
    return pd.DataFrame(traffic_data)

def main():
    """Main function to generate and save traffic data"""
    print("🚦 Generating Addis Ababa Traffic Data...")
    print("=" * 60)
    
    # Create sample data
    df = create_sample_traffic_data()
    
    # Display information
    print(f"✅ Data generated successfully!")
    print(f"📊 Total records: {len(df):,}")
    print(f"📍 Locations: {', '.join(df['location'].unique())}")
    print(f"📅 Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    
    # Summary statistics
    print("\n📈 Congestion Statistics:")
    print(f"   Average congestion: {df['congestion_level'].mean():.1f}%")
    print(f"   Maximum congestion: {df['congestion_level'].max()}%")
    print(f"   Minimum congestion: {df['congestion_level'].min()}%")
    
    # Save to CSV
    filename = 'addis_ababa_traffic_data.csv'
    df.to_csv(filename, index=False)
    print(f"\n💾 Data saved to: {filename}")
    
    # Show sample data
    print("\n👀 Sample data (first 5 rows):")
    print(df.head().to_string())
    
    print("\n" + "=" * 60)
    print("🎯 NEXT STEP: Run 'traffic_preprocessing.ipynb' in Google Colab")
    print("=" * 60)

if __name__ == "__main__":
    main()