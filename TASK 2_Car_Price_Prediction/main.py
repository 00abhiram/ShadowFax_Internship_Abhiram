"""
Car Price Prediction - Week 2: ShadowFox AIML Internship
Intermediate Level Project

This script performs:
1. Data Preprocessing (handling missing values, feature engineering)
2. Encoding categorical variables (One-Hot/Label Encoding)
3. Random Forest Regression with Hyperparameter Tuning
4. Model Evaluation (MSE, R² Score)
5. Visualization (Actual vs Predicted Prices)

Author: ShadowFox AIML Intern
Date: February 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import warnings

# Scikit-learn imports
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import warnings

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

CURRENT_YEAR = 2026

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'car_data.csv')
OUTPUT_DIR = BASE_DIR
MODEL_PATH = os.path.join(OUTPUT_DIR, 'car_price_model.pkl')
PLOT_PATH = os.path.join(OUTPUT_DIR, 'prediction_results.png')

TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(filepath):
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    df = pd.read_csv(filepath)
    print(f"\n[INFO] Dataset loaded successfully!")
    print(f"[INFO] Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n[INFO] Column Names:")
    for col in df.columns:
        print(f"       - {col}")
    print(f"\n[INFO] First 5 rows:")
    print(df.head())
    print(f"\n[INFO] Data Types:")
    print(df.dtypes)
    return df


def explore_data(df):
    print("\n" + "=" * 70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    print(f"\n[INFO] Statistical Summary:")
    print(df.describe())
    print(f"\n[INFO] Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("       No missing values found!")
    else:
        for col, count in missing[missing > 0].items():
            print(f"       - {col}: {count} missing values")
    print(f"\n[INFO] Unique Values in Categorical Columns:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"       - {col}: {df[col].nunique()} unique values")


def preprocess_data(df):
    print("\n" + "=" * 70)
    print("STEP 3: DATA PREPROCESSING")
    print("=" * 70)
    df_processed = df.copy()
    print("\n[INFO] Handling missing values...")
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_processed[col].isnull().sum() > 0:
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)
            print(f"       - {col}: Filled with median ({median_val})")
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_processed[col].isnull().sum() > 0:
            mode_val = df_processed[col].mode()[0]
            df_processed[col].fillna(mode_val, inplace=True)
            print(f"       - {col}: Filled with mode ({mode_val})")
    if df_processed.isnull().sum().sum() == 0:
        print("       All missing values handled!")
    print("\n[INFO] Feature Engineering...")
    if 'Car_Name' in df_processed.columns:
        df_processed['Brand'] = df_processed['Car_Name'].apply(lambda x: str(x).split()[0].lower())
        print(f"       - Created 'Brand' column from Car_Name")
        print(f"       - Unique brands: {df_processed['Brand'].nunique()}")
        df_processed.drop('Car_Name', axis=1, inplace=True)
        print(f"       - Dropped original 'Car_Name' column")
    year_col = None
    for col in ['Year', 'year', 'Year_Bought', 'year_bought']:
        if col in df_processed.columns:
            year_col = col
            break
    if year_col:
        df_processed['Years_Driven'] = CURRENT_YEAR - df_processed[year_col]
        print(f"       - Created 'Years_Driven' column (Current Year: {CURRENT_YEAR})")
        print(f"       - Years_Driven range: {df_processed['Years_Driven'].min()} to {df_processed['Years_Driven'].max()}")
        df_processed.drop(year_col, axis=1, inplace=True)
        print(f"       - Dropped original '{year_col}' column")
    print("\n[INFO] Encoding categorical variables...")
    target_col = None
    for col in ['Selling_Price', 'selling_price', 'Price', 'price']:
        if col in df_processed.columns:
            target_col = col
            break
    if target_col is None:
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        target_col = numerical_cols[-1]
    print(f"       - Target variable: {target_col}")
    label_encoders = {}
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
        print(f"       - Label Encoded: {col} ({len(le.classes_)} categories)")
    print(f"\n[INFO] Final preprocessed data shape: {df_processed.shape}")
    print(f"\n[INFO] Final columns:")
    for col in df_processed.columns:
        dtype = df_processed[col].dtype
        print(f"       - {col} ({dtype})")
    return df_processed, target_col, label_encoders


def train_model(X_train, y_train):
    print("\n" + "=" * 70)
    print("STEP 4: MODEL TRAINING WITH HYPERPARAMETER TUNING")
    print("=" * 70)
    print("\n[INFO] Using RandomizedSearchCV for Hyperparameter Tuning...")
    param_distributions = {
        'n_estimators': [50, 100, 150, 200, 250, 300],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    print("\n[INFO] Hyperparameter Search Space:")
    for param, values in param_distributions.items():
        print(f"       - {param}: {values}")
    rf = RandomForestRegressor(random_state=RANDOM_STATE)
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=50,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE
    )
    print("\n[INFO] Starting hyperparameter search (this may take a few minutes)...")
    random_search.fit(X_train, y_train)
    print("\n[INFO] Best Hyperparameters Found:")
    for param, value in random_search.best_params_.items():
        print(f"       - {param}: {value}")
    print(f"\n[INFO] Best Cross-Validation R² Score: {random_search.best_score_:.4f}")
    return random_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    print("\n" + "=" * 70)
    print("STEP 5: MODEL EVALUATION")
    print("=" * 70)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"\n[RESULTS] Evaluation Metrics:")
    print(f"          - Mean Squared Error (MSE): {mse:.4f}")
    print(f"          - Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"          - R² Score: {r2:.4f} ({r2 * 100:.2f}%)")
    print(f"\n[INFO] Feature Importance (Top 10):")
    feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    for idx, row in feature_importance.head(10).iterrows():
        print(f"       - {row['Feature']}: {row['Importance']:.4f}")
    return y_pred, mse, r2


def plot_predictions(y_test, y_pred, save_path):
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING VISUALIZATION")
    print("=" * 70)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5, c='steelblue')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Price (Lakhs)', fontsize=12)
    ax1.set_ylabel('Predicted Price (Lakhs)', fontsize=12)
    ax1.set_title('Actual vs Predicted Car Prices', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    r2 = r2_score(y_test, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2 = axes[1]
    residuals = y_test - y_pred
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax2.set_xlabel('Prediction Error (Residuals)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[INFO] Plot saved to: {save_path}")
    plt.show()
    return fig


def save_model(model, filepath):
    print("\n" + "=" * 70)
    print("STEP 7: SAVING MODEL")
    print("=" * 70)
    joblib.dump(model, filepath)
    print(f"\n[INFO] Model saved to: {filepath}")
    file_size = os.path.getsize(filepath) / (1024 * 1024)
    print(f"[INFO] Model file size: {file_size:.2f} MB")


def main():
    print("\n" + "=" * 70)
    print("CAR PRICE PREDICTION - WEEK 2 SHADOWFOX INTERNSHIP")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    df = load_data(DATA_PATH)
    explore_data(df)
    df_processed, target_col, label_encoders = preprocess_data(df)
    X = df_processed.drop(target_col, axis=1)
    y = df_processed[target_col]
    print(f"\n[INFO] Features shape: {X.shape}")
    print(f"[INFO] Target shape: {y.shape}")
    print("\n" + "=" * 70)
    print("DATA SPLITTING")
    print("=" * 70)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"\n[INFO] Training set: {X_train.shape[0]} samples ({(1-TEST_SIZE)*100:.0f}%)")
    print(f"[INFO] Test set: {X_test.shape[0]} samples ({TEST_SIZE*100:.0f}%)")
    best_model = train_model(X_train, y_train)
    y_pred, mse, r2 = evaluate_model(best_model, X_test, y_test)
    plot_predictions(y_test, y_pred, PLOT_PATH)
    save_model(best_model, MODEL_PATH)
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - FINAL SUMMARY")
    print("=" * 70)
    print(f"\n{'='*40}")
    print(f"   FINAL R² SCORE: {r2:.4f} ({r2 * 100:.2f}%)")
    print(f"{'='*40}")
    print(f"\n[OUTPUT FILES]")
    print(f"   - Prediction Plot: {PLOT_PATH}")
    print(f"   - Trained Model: {MODEL_PATH}")
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    return best_model, r2


if __name__ == "__main__":
    model, score = main()
