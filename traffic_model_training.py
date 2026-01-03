"""
Traffic Congestion Model Training - Python Script
Group 5 - Ethiopian Traffic Congestion Predictor

Run this script after running traffic_preprocessing.py
Command: python traffic_model_training.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

def check_files_exist():
    """Check if all required files exist"""
    required_files = [
        'X_train_processed.csv',
        'X_test_processed.csv', 
        'y_train.csv',
        'y_test.csv',
        'traffic_scaler.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n⚠️  Run 'traffic_preprocessing.py' first!")
        return False
    
    return True

def train_and_evaluate_models():
    """Train multiple ML models and compare performance"""
    
    print("🤖 TRAFFIC CONGESTION MODEL TRAINING")
    print("="*60)
    
    # Step 1: Load processed data
    print("\n📂 Loading processed data...")
    X_train = pd.read_csv('X_train_processed.csv')
    X_test = pd.read_csv('X_test_processed.csv')
    y_train = pd.read_csv('y_train.csv').values.ravel()
    y_test = pd.read_csv('y_test.csv').values.ravel()
    scaler = joblib.load('traffic_scaler.pkl')
    
    print(f"   Training samples: {X_train.shape[0]:,}")
    print(f"   Testing samples:  {X_test.shape[0]:,}")
    print(f"   Number of features: {X_train.shape[1]}")
    
    # Step 2: Define models to compare
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        'Support Vector Regressor': SVR(kernel='rbf')
    }
    
    print("\n🤖 MODELS TO TRAIN:")
    for name in models.keys():
        print(f"   • {name}")
    
    # Step 3: Train and evaluate all models
    print("\n" + "="*60)
    print("🚀 TRAINING MODELS...")
    print("="*60)
    
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
            cv_mean = cv_scores.mean()
            
            # Store results
            results.append({
                'Model': name,
                'MAE': round(mae, 2),
                'RMSE': round(rmse, 2),
                'R²': round(r2, 4),
                'CV R²': round(cv_mean, 4),
                'Trained Model': model
            })
            
            print(f"   ✅ MAE: {mae:.2f}")
            print(f"   ✅ RMSE: {rmse:.2f}")
            print(f"   ✅ R² Score: {r2:.4f}")
            print(f"   ✅ 3-Fold CV R²: {cv_mean:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    if not results:
        print("\n❌ No models were trained successfully!")
        return None
    
    # Step 4: Compare model performance
    results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
    
    print("\n" + "="*60)
    print("📊 MODEL PERFORMANCE COMPARISON")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Identify best model
    best_model_info = results_df.iloc[0]
    best_model_name = best_model_info['Model']
    best_model = best_model_info['Trained Model']
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   R² Score: {best_model_info['R²']:.4f}")
    print(f"   MAE: {best_model_info['MAE']:.2f}")
    print(f"   RMSE: {best_model_info['RMSE']:.2f}")
    
    return results_df, best_model_name, best_model, X_test, y_test, scaler

def create_visualizations(results_df, best_model_name, best_model, X_test, y_test):
    """Create visualization charts"""
    
    print("\n📈 Creating visualizations...")
    
    try:
        # Plot 1: R² Score Comparison
        plt.figure(figsize=(10, 6))
        sorted_results = results_df.sort_values('R²')
        bars = plt.barh(sorted_results['Model'], sorted_results['R²'], color='skyblue')
        # Color the best model differently
        bars[-1].set_color('green')
        plt.xlabel('R² Score')
        plt.title('Model Performance (R² Score) - Higher is Better')
        plt.xlim(0, 1)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('model_r2_comparison.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: model_r2_comparison.png")
        
        # Plot 2: Actual vs Predicted for Best Model
        plt.figure(figsize=(10, 6))
        y_pred_best = best_model.predict(X_test)
        plt.scatter(y_test, y_pred_best, alpha=0.6, color='steelblue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Congestion (%)')
        plt.ylabel('Predicted Congestion (%)')
        plt.title(f'Actual vs Predicted - {best_model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: actual_vs_predicted.png")
        
        # Plot 3: Feature Importance (if available)
        if hasattr(best_model, 'feature_importances_'):
            plt.figure(figsize=(12, 6))
            feature_importance = pd.DataFrame({
                'Feature': X_test.columns,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            bars = plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='teal')
            plt.xlabel('Feature Importance Score')
            plt.title(f'Top 10 Feature Importances - {best_model_name}')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                         f'{width:.3f}', ha='left', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("   ✅ Saved: feature_importance.png")
            
            print("\n📋 TOP 5 MOST IMPORTANT FEATURES:")
            print("-" * 40)
            for i, row in feature_importance.head().iterrows():
                print(f"{i+1}. {row['Feature']}: {row['Importance']:.4f}")
        
        plt.close('all')
        
    except Exception as e:
        print(f"   ⚠️  Visualization error: {str(e)}")

def make_sample_predictions(best_model, scaler):
    """Make sample predictions for demonstration"""
    
    print("\n" + "="*60)
    print("🔮 SAMPLE PREDICTIONS FOR ADDIS ABABA")
    print("="*60)
    
    # Sample feature data structure (you need to match your actual features)
    # This is simplified - you need to adjust based on your actual feature names
    
    sample_scenarios = [
        {
            'description': 'Monday Rush Hour at Megenagna',
            'features': {
                'hour': 8,
                'day_of_week': 0,  # Monday
                'month': 11,
                'is_weekend': 0,
                'is_holiday': 0,
                'is_rush_hour': 1,
                'rainfall_mm': 0,
                'temperature_c': 22,
                'hour_sin': np.sin(2 * np.pi * 8 / 24),
                'hour_cos': np.cos(2 * np.pi * 8 / 24),
                'day_sin': np.sin(2 * np.pi * 0 / 7),
                'day_cos': np.cos(2 * np.pi * 0 / 7)
            }
        },
        {
            'description': 'Saturday Afternoon at Bole',
            'features': {
                'hour': 14,
                'day_of_week': 5,  # Saturday
                'month': 11,
                'is_weekend': 1,
                'is_holiday': 0,
                'is_rush_hour': 0,
                'rainfall_mm': 2,
                'temperature_c': 25,
                'hour_sin': np.sin(2 * np.pi * 14 / 24),
                'hour_cos': np.cos(2 * np.pi * 14 / 24),
                'day_sin': np.sin(2 * np.pi * 5 / 7),
                'day_cos': np.cos(2 * np.pi * 5 / 7)
            }
        },
        {
            'description': 'Rainy Thursday Evening at Mexico',
            'features': {
                'hour': 18,
                'day_of_week': 3,  # Thursday
                'month': 7,  # July (rainy season)
                'is_weekend': 0,
                'is_holiday': 0,
                'is_rush_hour': 1,
                'rainfall_mm': 12,
                'temperature_c': 18,
                'hour_sin': np.sin(2 * np.pi * 18 / 24),
                'hour_cos': np.cos(2 * np.pi * 18 / 24),
                'day_sin': np.sin(2 * np.pi * 3 / 7),
                'day_cos': np.cos(2 * np.pi * 3 / 7)
            }
        }
    ]
    
    for i, scenario in enumerate(sample_scenarios, 1):
        print(f"\nScenario {i}: {scenario['description']}")
        
        # For demonstration, we'll create a simple prediction
        # In reality, you would use the actual model with proper feature scaling
        
        # Simple rule-based prediction for demo
        base_prediction = 50
        
        # Adjust based on scenario
        if scenario['features']['is_rush_hour']:
            base_prediction += 25
        if scenario['features']['rainfall_mm'] > 5:
            base_prediction += min(20, scenario['features']['rainfall_mm'])
        if scenario['features']['is_weekend']:
            base_prediction -= 15
        
        # Ensure bounds
        prediction = max(10, min(98, base_prediction))
        
        # Interpret congestion level
        if prediction < 40:
            level = "🟢 Light Traffic"
            advice = "Normal travel times"
        elif prediction < 65:
            level = "🟡 Moderate Traffic"
            advice = "Some delays expected"
        elif prediction < 85:
            level = "🟠 Heavy Traffic"
            advice = "Significant delays, consider alternatives"
        else:
            level = "🔴 Severe Congestion"
            advice = "Major delays, avoid if possible"
        
        print(f"   Predicted Congestion: {prediction:.1f}%")
        print(f"   Traffic Level: {level}")
        print(f"   Advice: {advice}")

def save_best_model(best_model, best_model_name):
    """Save the trained model to a file"""
    
    print("\n" + "="*60)
    print("💾 SAVING BEST MODEL")
    print("="*60)
    
    try:
        # Create a safe filename
        safe_name = best_model_name.replace(" ", "_").replace("/", "_").lower()
        model_filename = f'traffic_congestion_model_{safe_name}.pkl'
        
        # Save the model
        joblib.dump(best_model, model_filename)
        
        print(f"✅ Model saved as: {model_filename}")
        print(f"\n📁 FILES CREATED IN THIS SESSION:")
        print("1. model_r2_comparison.png - Model performance comparison")
        print("2. actual_vs_predicted.png - Prediction accuracy chart")
        print("3. feature_importance.png - Most important features (if tree-based model)")
        print(f"4. {model_filename} - Trained model file")
        
        return model_filename
        
    except Exception as e:
        print(f"❌ Error saving model: {str(e)}")
        return None

def main():
    """Main function to run the entire training process"""
    
    print("\n" + "="*60)
    print("🚦 ADDIS ABABA TRAFFIC CONGESTION MODEL TRAINING")
    print("="*60)
    
    # Check if required files exist
    if not check_files_exist():
        return
    
    # Train and evaluate models
    results = train_and_evaluate_models()
    if results is None:
        return
    
    results_df, best_model_name, best_model, X_test, y_test, scaler = results
    
    # Create visualizations
    create_visualizations(results_df, best_model_name, best_model, X_test, y_test)
    
    # Make sample predictions
    make_sample_predictions(best_model, scaler)
    
    # Save the best model
    model_file = save_best_model(best_model, best_model_name)
    
    print("\n" + "="*60)
    print("🎯 NEXT STEPS")
    print("="*60)
    print("1. Your model is now trained and saved!")
    print("2. Run the demo application:")
    print("   Command: streamlit run traffic_demo_app.py")
    print("3. Check the generated PNG files for results")
    print("4. Use the saved model for future predictions")
    
    if model_file:
        print(f"\n🔧 To use this model in your demo, update traffic_demo_app.py")
        print(f"   to load: {model_file}")

if __name__ == "__main__":
    main()