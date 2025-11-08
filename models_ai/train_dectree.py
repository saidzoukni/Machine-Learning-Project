"""
Train a Decision Tree model for restaurant online_order prediction
Based on Zomato restaurant dataset

Features:
 - name, rest_type, dish_liked, cuisines, type, city (categorical - LabelEncoded)
 - rate, cost (numerical - StandardScaled)
 - book_table (binary - LabelEncoded)

Target:
 - online_order (0: No, 1: Yes)

Saves:
 - dectree_model.pkl: The trained Decision Tree model
 - label_encoders.pkl: Dictionary of LabelEncoders for categorical features
 - scaler.pkl: StandardScaler for numerical features
"""

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def clean_rate(value):
    """Clean rate column: extract numeric value from '4.1/5' format"""
    try:
        if pd.isna(value):
            return np.nan
        if isinstance(value, str):
            return float(value.split('/')[0])
        return float(value)
    except:
        return np.nan


def load_and_preprocess_data(csv_path):
    """Load and preprocess the Zomato dataset"""
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # Drop unnecessary columns
    columns_to_drop = ['url', 'address', 'phone', 'votes', 'menu_item', 'location', 'reviews_list']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)
    
    # Clean rate column
    df['rate'] = df['rate'].apply(clean_rate)
    
    # Rename columns
    new_names = {
        'listed_in(type)': 'type',
        'listed_in(city)': 'city',
        'approx_cost(for two people)': 'cost'
    }
    df = df.rename(columns=new_names)
    
    # Drop rows with missing values
    print(f"Dataset shape before dropping NaN: {df.shape}")
    df = df.dropna()
    print(f"Dataset shape after dropping NaN: {df.shape}")
    
    return df


def encode_and_scale_features(df):
    """Encode categorical features and scale numerical features"""
    print("\nEncoding categorical features...")
    
    # Initialize LabelEncoders for each categorical column
    label_encoders = {}
    categorical_cols = ['name', 'rest_type', 'dish_liked', 'cuisines', 'type', 'city', 'book_table']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"  Encoded {col}: {len(le.classes_)} unique values")
    
    # Encode online_order (target)
    le_target = LabelEncoder()
    df['online_order'] = le_target.fit_transform(df['online_order'])
    label_encoders['online_order'] = le_target
    
    print("\nScaling numerical features...")
    # Clean and convert cost column (remove commas)
    if 'cost' in df.columns:
        df['cost'] = df['cost'].astype(str).str.replace(',', '').astype(float)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['rate', 'cost']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    print(f"  Scaled {numerical_cols}")
    
    return df, label_encoders, scaler


def prepare_features_and_target(df):
    """Prepare feature matrix X and target vector y"""
    # Feature order: name, rest_type, dish_liked, cuisines, type, city, rate, cost, book_table
    feature_cols = ['name', 'rest_type', 'dish_liked', 'cuisines', 'type', 'city', 'rate', 'cost', 'book_table']
    
    # Check which columns exist
    available_cols = [col for col in feature_cols if col in df.columns]
    X = df[available_cols].values
    
    y = df['online_order'].values
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y


def train_model(X_train, y_train, X_test, y_test):
    """Train Decision Tree model"""
    print("\nTraining Decision Tree model...")
    
    # Create Decision Tree classifier
    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Online Order', 'Online Order']))
    
    return model


def save_models(model, label_encoders, scaler, base_dir):
    """Save all models and preprocessors"""
    print("\nSaving models...")
    
    # Save Decision Tree model
    model_path = base_dir / "dectree_model.pkl"
    joblib.dump(model, model_path)
    print(f"  ✓ Saved model to {model_path}")
    
    # Save LabelEncoders (remove target encoder from the dict for prediction use)
    encoders_for_prediction = {k: v for k, v in label_encoders.items() if k != 'online_order'}
    encoders_path = base_dir / "label_encoders.pkl"
    joblib.dump(encoders_for_prediction, encoders_path)
    print(f"  ✓ Saved label encoders to {encoders_path}")
    
    # Save StandardScaler
    scaler_path = base_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Saved scaler to {scaler_path}")
    
    print("\n✓ All files saved successfully!")


def main():
    """Main training function"""
    # Get the script directory
    script_dir = Path(__file__).resolve().parent
    
    # Look for zomato.csv in common locations
    possible_paths = [
        script_dir / "zomato.csv",
        script_dir.parent / "zomato.csv",
        Path.cwd() / "zomato.csv",
    ]
    
    csv_path = None
    for path in possible_paths:
        if path.exists():
            csv_path = path
            break
    
    if csv_path is None:
        print("ERROR: zomato.csv not found!")
        print("\nPlease place zomato.csv in one of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nOr modify the script to point to your dataset location.")
        return
    
    print(f"Found dataset at: {csv_path}")
    
    # Load and preprocess data
    df = load_and_preprocess_data(csv_path)
    
    # Encode and scale
    df, label_encoders, scaler = encode_and_scale_features(df)
    
    # Prepare features and target
    X, y = prepare_features_and_target(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = train_model(X_train, y_train, X_test, y_test)
    
    # Save everything
    save_models(model, label_encoders, scaler, script_dir)
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()

