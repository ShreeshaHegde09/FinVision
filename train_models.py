#!/usr/bin/env python3
"""
Training script for Financial Advisor ML Models
Run this script to train models with your actual dataset
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_dataset(file_path):
    """Load and validate the dataset"""
    print(f" Loading dataset from: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f" Dataset loaded successfully!")
        print(f" Shape: {df.shape}")
        print(f" Columns: {list(df.columns)}")
        
        # Check for required columns
        required_columns = ['Income', 'Age', 'Dependents', 'Desired_Savings']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return None
            
        return df
        
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {file_path}")
        print("Please ensure your CSV file is in the same directory")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return None

def preprocess_data(df):
    """Preprocess the dataset for training"""
    print("\n Preprocessing dataset...")
    
    # Handle missing values
    print(f"Missing values before cleaning:\n{df.isnull().sum()}")
    
    # Fill numeric columns with median
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical columns with mode
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
    
    # Create financial features
    df['Savings_Rate'] = df['Desired_Savings'] / df['Income']
    df['Expense_Ratio'] = (df.get('Groceries', 0) + df.get('Transport', 0) + 
                          df.get('Eating_Out', 0) + df.get('Entertainment', 0) + 
                          df.get('Utilities', 0) + df.get('Healthcare', 0) + 
                          df.get('Education', 0) + df.get('Miscellaneous', 0)) / df['Income']
    
    df['Discretionary_Spending'] = (df.get('Eating_Out', 0) + df.get('Entertainment', 0)) / df['Income']
    df['Essential_Spending'] = (df.get('Groceries', 0) + df.get('Utilities', 0) + df.get('Rent', 0)) / df['Income']
    df['Investment_Capacity'] = df.get('Disposable_Income', df['Desired_Savings']) / df['Income']
    df['Financial_Burden'] = (df.get('Rent', 0) + df.get('Loan_Repayment', 0)) / df['Income']
    
    # Create age groups
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], 
                            labels=['Young', 'Early_Career', 'Mid_Career', 'Senior', 'Pre_Retirement'])
    
    # Create financial personality
    df = classify_financial_personality(df)
    
    print(" Preprocessing completed!")
    print(f" Final shape: {df.shape}")
    
    return df

def classify_financial_personality(df):
    """Classify users into financial personality types"""
    print(" Classifying financial personalities...")
    
    df = df.copy()
    
    # Create scoring system
    df['Risk_Score'] = 0
    df['Savings_Score'] = 0
    df['Spending_Score'] = 0
    
    # Risk assessment
    df.loc[df['Age'] < 30, 'Risk_Score'] += 2
    df.loc[(df['Age'] >= 30) & (df['Age'] < 40), 'Risk_Score'] += 1
    df.loc[df['Age'] >= 50, 'Risk_Score'] -= 1
    df.loc[df['Dependents'] > 2, 'Risk_Score'] -= 1
    
    # Savings behavior
    df.loc[df['Savings_Rate'] >= 0.3, 'Savings_Score'] += 3
    df.loc[(df['Savings_Rate'] >= 0.2) & (df['Savings_Rate'] < 0.3), 'Savings_Score'] += 2
    df.loc[(df['Savings_Rate'] >= 0.1) & (df['Savings_Rate'] < 0.2), 'Savings_Score'] += 1
    df.loc[df['Savings_Rate'] < 0.1, 'Savings_Score'] -= 1
    
    # Spending behavior
    df.loc[df['Discretionary_Spending'] >= 0.2, 'Spending_Score'] += 2
    df.loc[(df['Discretionary_Spending'] >= 0.15) & (df['Discretionary_Spending'] < 0.2), 'Spending_Score'] += 1
    
    # Classify personality types
    conditions = [
        (df['Savings_Score'] >= 2) & (df['Risk_Score'] >= 1) & (df['Investment_Capacity'] >= 0.2),
        (df['Spending_Score'] >= 2) & (df['Savings_Score'] <= 0),
        (df['Savings_Score'] >= 1) & (df['Risk_Score'] <= 0),
    ]
    
    choices = ['Aggressive_Investor', 'Heavy_Spender', 'Low_Risk_Investor']
    df['Financial_Personality'] = np.select(conditions, choices, default='Moderate_Balanced')
    
    # Show distribution
    print(" Personality Distribution:")
    personality_dist = df['Financial_Personality'].value_counts()
    for personality, count in personality_dist.items():
        print(f"   {personality}: {count} ({count/len(df)*100:.1f}%)")
    
    return df

def train_savings_predictor(df):
    """Train the savings prediction model"""
    print("\n Training Savings Prediction Model...")
    
    # Prepare features
    feature_columns = ['Income', 'Age', 'Dependents', 'Expense_Ratio', 'Financial_Burden', 'Investment_Capacity']
    
    # Encode categorical variables
    label_encoders = {}
    if 'Occupation' in df.columns:
        le_occupation = LabelEncoder()
        df['Occupation_encoded'] = le_occupation.fit_transform(df['Occupation'].astype(str))
        feature_columns.append('Occupation_encoded')
        label_encoders['Occupation'] = le_occupation
    
    if 'City_Tier' in df.columns:
        le_city = LabelEncoder()
        df['City_Tier_encoded'] = le_city.fit_transform(df['City_Tier'].astype(str))
        feature_columns.append('City_Tier_encoded')
        label_encoders['City_Tier'] = le_city
    
    X = df[feature_columns]
    y = df['Desired_Savings']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f" Savings Predictor - R² Score: {r2:.3f}, RMSE: Rs. {rmse:,.0f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(" Feature Importance:")
    for _, row in feature_importance.head().iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    return model, scaler, label_encoders, feature_columns

def train_personality_classifier(df):
    """Train the personality classification model"""
    print("\n Training Personality Classification Model...")
    
    feature_columns = ['Income', 'Age', 'Dependents', 'Savings_Rate', 
                      'Discretionary_Spending', 'Essential_Spending', 
                      'Investment_Capacity', 'Financial_Burden']
    
    X = df[feature_columns]
    y = df['Financial_Personality']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    
    print(f" Personality Classifier - Accuracy: {accuracy:.3f}")
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

def save_models(savings_model, personality_model, scaler, label_encoders, feature_columns):
    """Save all trained models"""
    print("\n Saving trained models...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save models
    joblib.dump(savings_model, 'models/savings_predictor.joblib')
    joblib.dump(personality_model, 'models/personality_classifier.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(label_encoders, 'models/label_encoders.joblib')
    joblib.dump(feature_columns, 'models/feature_columns.joblib')
    
    print(" All models saved successfully!")
    print(" Models saved in 'models/' directory:")
    for file in os.listdir('models'):
        print(f"   - {file}")

def main():
    """Main training function"""
    print("Starting Financial Advisor ML Training")
    print("=" * 50)
    
    # Find dataset file
    dataset_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not dataset_files:
        print("❌ No CSV files found in current directory!")
        print("Please ensure your dataset file is in the same folder as this script")
        return
    
    print(f" Found CSV files: {dataset_files}")
    
    # Use the first CSV file or ask user to specify
    if len(dataset_files) == 1:
        dataset_file = dataset_files[0]
        print(f" Using dataset: {dataset_file}")
    else:
        print("Multiple CSV files found. Please specify which one to use:")
        for i, file in enumerate(dataset_files):
            print(f"   {i+1}. {file}")
        
        try:
            choice = int(input("Enter choice (1-{}): ".format(len(dataset_files)))) - 1
            dataset_file = dataset_files[choice]
        except (ValueError, IndexError):
            print("❌ Invalid choice. Using the first file.")
            dataset_file = dataset_files[0]
    
    # Load and preprocess data
    df = load_dataset(dataset_file)
    if df is None:
        return
    
    df = preprocess_data(df)
    
    # Train models
    savings_model, scaler, label_encoders, feature_columns = train_savings_predictor(df)
    personality_model = train_personality_classifier(df)
    
    # Save models
    save_models(savings_model, personality_model, scaler, label_encoders, feature_columns)
    
    print("\n Training completed successfully!")
    print(" You can now run the web application with: python app.py")

if __name__ == "__main__":
    main()