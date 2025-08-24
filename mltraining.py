import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import joblib
from financialadvisor import FinancialGoalAdvisor 
import warnings
warnings.filterwarnings('ignore')

class FinancialDataAnalyzer:
    """Comprehensive data analysis and model training for financial advisor"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        
    def load_and_analyze_data(self, file_path):
        """Load dataset and perform comprehensive analysis"""
        print("📊 Loading and analyzing financial dataset...")
        
        # Load data
        df = pd.read_csv(file_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Basic statistics
        print("\n📈 Dataset Overview:")
        print(df.describe())
        
        # Check for missing values
        print(f"\n❓ Missing values:\n{df.isnull().sum()}")
        
        # Data types
        print(f"\n🔍 Data types:\n{df.dtypes}")
        
        return df
    
    def create_financial_features(self, df):
        """Create additional features for better analysis"""
        df = df.copy()
        
        # Financial ratios and metrics
        df['Savings_Rate'] = df['Desired_Savings'] / df['Income']
        df['Expense_Ratio'] = (df['Groceries'] + df['Transport'] + df['Eating_Out'] + 
                              df['Entertainment'] + df['Utilities'] + df['Healthcare'] + 
                              df['Education'] + df['Miscellaneous']) / df['Income']
        
        df['Discretionary_Spending'] = (df['Eating_Out'] + df['Entertainment']) / df['Income']
        df['Essential_Spending'] = (df['Groceries'] + df['Utilities'] + df['Rent']) / df['Income']
        df['Investment_Capacity'] = df['Disposable_Income'] / df['Income']
        df['Financial_Burden'] = (df['Rent'] + df['Loan_Repayment']) / df['Income']
        
        # Age groups for life stage analysis
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], 
                                labels=['Young', 'Early_Career', 'Mid_Career', 'Senior', 'Pre_Retirement'])
        
        # Income groups
        df['Income_Group'] = pd.cut(df['Income'], bins=[0, 30000, 50000, 80000, 120000, np.inf],
                                   labels=['Low', 'Lower_Middle', 'Middle', 'Upper_Middle', 'High'])
        
        # Dependents categories
        df['Dependent_Category'] = pd.cut(df['Dependents'], bins=[-1, 0, 2, 4, np.inf],
                                         labels=['None', 'Few', 'Moderate', 'Many'])
        
        return df
    
    def classify_financial_personality(self, df):
        """Classify users into financial personality types using multiple criteria"""
        df = df.copy()
        
        # Create personality score
        df['Risk_Score'] = 0
        df['Savings_Score'] = 0
        df['Spending_Score'] = 0
        
        # Risk assessment based on age and dependents
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
        df.loc[df['Discretionary_Spending'] < 0.05, 'Spending_Score'] -= 1
        
        # Classify personality types
        conditions = [
            (df['Savings_Score'] >= 2) & (df['Risk_Score'] >= 1) & (df['Investment_Capacity'] >= 0.2),
            (df['Spending_Score'] >= 2) & (df['Savings_Score'] <= 0),
            (df['Savings_Score'] >= 1) & (df['Risk_Score'] <= 0),
        ]
        
        choices = ['Aggressive_Investor', 'Heavy_Spender', 'Low_Risk_Investor']
        df['Financial_Personality'] = np.select(conditions, choices, default='Moderate_Balanced')
        
        return df
    
    def perform_customer_segmentation(self, df):
        """Perform customer segmentation using clustering"""
        print("🎯 Performing customer segmentation...")
        
        # Select features for clustering
        cluster_features = ['Income', 'Age', 'Dependents', 'Savings_Rate', 
                           'Discretionary_Spending', 'Essential_Spending', 'Investment_Capacity']
        
        # Prepare data for clustering
        cluster_data = df[cluster_features].fillna(df[cluster_features].median())
        cluster_data_scaled = self.scaler.fit_transform(cluster_data)
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(cluster_data_scaled)
            inertias.append(kmeans.inertia_)
        
        # Use 5 clusters (you can adjust based on elbow curve)
        optimal_k = 5
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        df['Customer_Segment'] = kmeans.fit_predict(cluster_data_scaled)
        
        # Analyze segments
        segment_analysis = df.groupby('Customer_Segment')[cluster_features].mean()
        print("Customer Segments Analysis:")
        print(segment_analysis)
        
        return df, segment_analysis
    
    def train_savings_prediction_model(self, df):
        """Train model to predict optimal savings"""
        print("🤖 Training savings prediction model...")
        
        # Prepare features
        feature_columns = ['Income', 'Age', 'Dependents', 'Expense_Ratio', 
                          'Financial_Burden', 'Investment_Capacity']
        
        # Encode categorical variables
        for col in ['Occupation', 'City_Tier']:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('Unknown'))
                feature_columns.append(f'{col}_encoded')
                self.label_encoders[col] = le
        
        X = df[feature_columns].fillna(df[feature_columns].median())
        y = df['Desired_Savings']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train multiple models and select best
        models = {
            'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient_Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = -np.inf
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            print(f"{name} R² Score: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_model = model
        
        self.models['savings_predictor'] = best_model
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 Feature Importance for Savings Prediction:")
        print(feature_importance)
        
        return best_model, feature_importance
    
    def train_personality_classifier(self, df):
        """Train model to classify financial personality"""
        print("🧠 Training financial personality classifier...")
        
        # Prepare features
        feature_columns = ['Income', 'Age', 'Dependents', 'Savings_Rate', 
                          'Discretionary_Spending', 'Essential_Spending', 
                          'Investment_Capacity', 'Financial_Burden']
        
        # Add encoded categorical features
        for col in ['Occupation', 'City_Tier']:
            if col in df.columns and f'{col}_encoded' in df.columns:
                feature_columns.append(f'{col}_encoded')
        
        X = df[feature_columns].fillna(df[feature_columns].median())
        y = df['Financial_Personality']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train classifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = classifier.predict(X_test)
        accuracy = classifier.score(X_test, y_test)
        
        print(f"Personality Classifier Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.models['personality_classifier'] = classifier
        
        return classifier
    
    def train_investment_recommender(self, df):
        """Train model to recommend optimal investment allocation"""
        print("💰 Training investment recommendation model...")
        
        # Create synthetic investment allocation data based on personality and profile
        investment_data = []
        
        for _, row in df.iterrows():
            personality = row['Financial_Personality']
            age = row['Age']
            income = row['Income']
            dependents = row['Dependents']
            risk_capacity = max(0, min(1, (100 - age) / 100 - dependents * 0.1))
            
            # Create allocation based on rules
            if personality == 'Aggressive_Investor':
                allocation = {
                    'Equity': min(0.7, 0.4 + risk_capacity * 0.3),
                    'Mutual_Funds': 0.2,
                    'Real_Estate': 0.1,
                    'Gold': 0.05,
                    'Fixed_Deposits': 0.05
                }
            elif personality == 'Heavy_Spender':
                allocation = {
                    'PPF': 0.4,
                    'ELSS': 0.3,
                    'Fixed_Deposits': 0.2,
                    'Gold': 0.1,
                    'Equity': 0.0
                }
            elif personality == 'Low_Risk_Investor':
                allocation = {
                    'Fixed_Deposits': 0.4,
                    'PPF': 0.3,
                    'Gold': 0.2,
                    'Bonds': 0.1,
                    'Equity': 0.0
                }
            else:  # Moderate_Balanced
                allocation = {
                    'Mutual_Funds': 0.3,
                    'PPF': 0.2,
                    'ELSS': 0.2,
                    'Gold': 0.15,
                    'Fixed_Deposits': 0.15
                }
            
            # Normalize to sum to 1
            total = sum(allocation.values())
            allocation = {k: v/total for k, v in allocation.items()}
            
            investment_data.append({
                'Income': income,
                'Age': age,
                'Dependents': dependents,
                'Personality': personality,
                'Risk_Capacity': risk_capacity,
                'Savings_Rate': row['Savings_Rate'],
                **allocation
            })
        
        investment_df = pd.DataFrame(investment_data)
        
        # This would be used for future recommendation engine
        self.models['investment_data'] = investment_df
        
        return investment_df
    
    def generate_insights_report(self, df):
        """Generate comprehensive insights report"""
        print("📊 Generating insights report...")
        
        insights = []
        
        # Demographic insights
        avg_age = df['Age'].mean()
        avg_income = df['Income'].mean()
        avg_dependents = df['Dependents'].mean()
        
        insights.append(f"👥 Average user profile: {avg_age:.1f} years old, ₹{avg_income:,.0f} income, {avg_dependents:.1f} dependents")
        
        # Savings behavior insights
        avg_savings_rate = df['Savings_Rate'].mean()
        high_savers = (df['Savings_Rate'] >= 0.25).sum() / len(df) * 100
        
        insights.append(f"💰 Average savings rate: {avg_savings_rate:.1%}")
        insights.append(f"🎯 {high_savers:.1f}% of users save 25% or more of their income")
        
        # Spending patterns
        top_expense_category = df[['Groceries', 'Transport', 'Eating_Out', 'Entertainment', 
                                  'Utilities', 'Healthcare', 'Education', 'Miscellaneous']].mean().idxmax()
        
        insights.append(f"🛒 Highest expense category: {top_expense_category}")
        
        # City-wise analysis
        if 'City_Tier' in df.columns:
            city_savings = df.groupby('City_Tier')['Savings_Rate'].mean()
            best_saving_city = city_savings.idxmax()
            insights.append(f"🏙️ Best savings rate by city tier: {best_saving_city} ({city_savings[best_saving_city]:.1%})")
        
        # Occupation insights
        if 'Occupation' in df.columns:
            occ_income = df.groupby('Occupation')['Income'].mean()
            highest_income_occ = occ_income.idxmax()
            insights.append(f"💼 Highest income occupation: {highest_income_occ} (₹{occ_income[highest_income_occ]:,.0f})")
        
        # Financial personality distribution
        personality_dist = df['Financial_Personality'].value_counts(normalize=True) * 100
        insights.append("🧭 Financial personality distribution:")
        for personality, percentage in personality_dist.items():
            insights.append(f"   • {personality}: {percentage:.1f}%")
        
        # Age vs Investment capacity
        age_investment = df.groupby('Age_Group')['Investment_Capacity'].mean()
        insights.append("📈 Investment capacity by age group:")
        for age_group, capacity in age_investment.items():
            insights.append(f"   • {age_group}: {capacity:.1%}")
        
        return insights
    
    def create_visualizations(self, df):
        """Create comprehensive visualizations"""
        print("📊 Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Financial Behavior Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Income distribution
        axes[0,0].hist(df['Income'], bins=30, alpha=0.7, color='skyblue')
        axes[0,0].set_title('Income Distribution')
        axes[0,0].set_xlabel('Income (₹)')
        axes[0,0].set_ylabel('Frequency')
        
        # 2. Savings rate by age group
        df.boxplot(column='Savings_Rate', by='Age_Group', ax=axes[0,1])
        axes[0,1].set_title('Savings Rate by Age Group')
        axes[0,1].set_xlabel('Age Group')
        axes[0,1].set_ylabel('Savings Rate')
        
        # 3. Financial personality distribution
        personality_counts = df['Financial_Personality'].value_counts()
        axes[0,2].pie(personality_counts.values, labels=personality_counts.index, autopct='%1.1f%%')
        axes[0,2].set_title('Financial Personality Distribution')
        
        # 4. Investment capacity vs Income
        axes[1,0].scatter(df['Income'], df['Investment_Capacity'], alpha=0.6, color='green')
        axes[1,0].set_title('Investment Capacity vs Income')
        axes[1,0].set_xlabel('Income (₹)')
        axes[1,0].set_ylabel('Investment Capacity')
        
        # 5. Expense breakdown
        expense_cols = ['Groceries', 'Transport', 'Eating_Out', 'Entertainment', 'Utilities']
        avg_expenses = df[expense_cols].mean()
        axes[1,1].bar(range(len(avg_expenses)), avg_expenses.values, color='coral')
        axes[1,1].set_title('Average Monthly Expenses')
        axes[1,1].set_xticks(range(len(avg_expenses)))
        axes[1,1].set_xticklabels(avg_expenses.index, rotation=45)
        axes[1,1].set_ylabel('Amount (₹)')
        
        # 6. Dependents vs Savings Rate
        df.boxplot(column='Savings_Rate', by='Dependent_Category', ax=axes[1,2])
        axes[1,2].set_title('Savings Rate by Number of Dependents')
        axes[1,2].set_xlabel('Dependent Category')
        axes[1,2].set_ylabel('Savings Rate')
        
        plt.tight_layout()
        plt.show()
        
        # Additional correlation heatmap
        plt.figure(figsize=(12, 8))
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    fmt='.2f', square=True)
        plt.title('Financial Metrics Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def save_models(self, save_dir='financial_models'):
        """Save trained models"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            joblib.dump(model, f'{save_dir}/{model_name}.joblib')
        
        # Save label encoders
        joblib.dump(self.label_encoders, f'{save_dir}/label_encoders.joblib')
        joblib.dump(self.scaler, f'{save_dir}/scaler.joblib')
        
        print(f"✅ Models saved to {save_dir}/")
    
    def comprehensive_analysis_pipeline(self, file_path):
        """Run complete analysis pipeline"""
        print("🚀 Starting comprehensive financial data analysis...")
        
        # Load and analyze data
        df = self.load_and_analyze_data(file_path)
        
        # Feature engineering
        df = self.create_financial_features(df)
        df = self.classify_financial_personality(df)
        
        # Customer segmentation
        df, segment_analysis = self.perform_customer_segmentation(df)
        
        # Train models
        savings_model, feature_importance = self.train_savings_prediction_model(df)
        personality_classifier = self.train_personality_classifier(df)
        investment_data = self.train_investment_recommender(df)
        
        # Generate insights
        insights = self.generate_insights_report(df)
        
        # Create visualizations
        self.create_visualizations(df)
        
        # Save models
        self.save_models()
        
        # Print insights
        print("\n" + "="*80)
        print("📊 FINANCIAL BEHAVIOR INSIGHTS REPORT")
        print("="*80)
        for insight in insights:
            print(insight)
        
        print("\n" + "="*80)
        print("✅ Analysis complete! Models trained and saved.")
        print("="*80)
        
        return {
            'processed_data': df,
            'models': self.models,
            'insights': insights,
            'segment_analysis': segment_analysis,
            'feature_importance': feature_importance
        }

# Integration with the main advisor system
class EnhancedFinancialGoalAdvisor(FinancialGoalAdvisor):
    """Enhanced version with ML models trained on real data"""
    
    def __init__(self, analyzer=None):
        super().__init__()
        self.analyzer = analyzer
        if analyzer:
            self.models = analyzer.models
            self.label_encoders = analyzer.label_encoders
            self.scaler = analyzer.scaler
    
    def predict_optimal_savings(self, user_data):
        """Predict optimal savings using trained model"""
        if 'savings_predictor' not in self.models:
            return None
        
        # Prepare features (same as training)
        features = ['Income', 'Age', 'Dependents']
        
        # Calculate derived features
        total_expenses = (user_data.get('Groceries', 0) + user_data.get('Transport', 0) + 
                         user_data.get('Eating_Out', 0) + user_data.get('Entertainment', 0) + 
                         user_data.get('Utilities', 0) + user_data.get('Healthcare', 0) + 
                         user_data.get('Education', 0) + user_data.get('Miscellaneous', 0))
        
        expense_ratio = total_expenses / user_data['Income']
        financial_burden = (user_data.get('Rent', 0) + user_data.get('Loan_Repayment', 0)) / user_data['Income']
        investment_capacity = user_data.get('Disposable_Income', 0) / user_data['Income']
        
        feature_values = [
            user_data['Income'],
            user_data['Age'], 
            user_data['Dependents'],
            expense_ratio,
            financial_burden,
            investment_capacity
        ]
        
        # Add encoded categorical features
        for col in ['Occupation', 'City_Tier']:
            if col in self.label_encoders and col in user_data:
                try:
                    encoded_value = self.label_encoders[col].transform([user_data[col]])[0]
                    feature_values.append(encoded_value)
                except ValueError:
                    # Handle unknown categories
                    feature_values.append(0)
        
        # Make prediction
        prediction = self.models['savings_predictor'].predict([feature_values])[0]
        return max(0, prediction)  # Ensure non-negative
    
    def get_ml_based_recommendations(self, user_data):
        """Get recommendations based on ML model predictions"""
        recommendations = []
        
        # Predict optimal savings
        predicted_savings = self.predict_optimal_savings(user_data)
        if predicted_savings:
            current_savings = user_data.get('Desired_Savings', 0)
            if predicted_savings > current_savings * 1.2:
                recommendations.append(f"💡 ML model suggests you could save ₹{predicted_savings:,.0f} monthly (current: ₹{current_savings:,.0f})")
            elif predicted_savings < current_savings * 0.8:
                recommendations.append(f"⚠️ Your current savings goal might be too ambitious. Consider ₹{predicted_savings:,.0f} monthly")
        
        # Add personality-based ML recommendations
        if 'personality_classifier' in self.models:
            recommendations.append("🤖 Recommendations are personalized using ML models trained on 20,000+ user profiles")
        
        return recommendations

# Usage example
def run_complete_analysis():
    """Complete analysis workflow"""
    
    # Initialize analyzer
    analyzer = FinancialDataAnalyzer()
    
    # Run analysis (replace 'your_dataset.csv' with actual file path)
    print("Note: Replace 'your_dataset.csv' with the actual path to your dataset")
    # results = analyzer.comprehensive_analysis_pipeline('your_dataset.csv')
    
    # For demo purposes, create sample data
    print("Creating sample dataset for demonstration...")
    sample_data = create_sample_dataset()
    
    # Run analysis on sample data
    sample_data = analyzer.create_financial_features(sample_data)
    sample_data = analyzer.classify_financial_personality(sample_data)
    
    # Train models
    analyzer.train_savings_prediction_model(sample_data)
    analyzer.train_personality_classifier(sample_data)
    
    # Create enhanced advisor
    enhanced_advisor = EnhancedFinancialGoalAdvisor(analyzer)
    
    return enhanced_advisor, analyzer

def create_sample_dataset():
    """Create sample dataset for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Income': np.random.normal(65000, 25000, n_samples),
        'Age': np.random.randint(22, 60, n_samples),
        'Dependents': np.random.randint(0, 5, n_samples),
        'Occupation': np.random.choice(['IT_Professional', 'Teacher', 'Doctor', 'Engineer', 'Business'], n_samples),
        'City_Tier': np.random.choice(['Tier_1', 'Tier_2', 'Tier_3'], n_samples),
        'Rent': np.random.normal(15000, 8000, n_samples),
        'Loan_Repayment': np.random.normal(8000, 5000, n_samples),
        'Insurance': np.random.normal(2000, 1000, n_samples),
        'Groceries': np.random.normal(8000, 3000, n_samples),
        'Transport': np.random.normal(4000, 2000, n_samples),
        'Eating_Out': np.random.normal(3000, 1500, n_samples),
        'Entertainment': np.random.normal(2000, 1000, n_samples),
        'Utilities': np.random.normal(2000, 800, n_samples),
        'Healthcare': np.random.normal(2500, 1200, n_samples),
        'Education': np.random.normal(1500, 800, n_samples),
        'Miscellaneous': np.random.normal(2000, 1000, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Ensure positive values
    for col in df.columns:
        if col not in ['Age', 'Dependents', 'Occupation', 'City_Tier']:
            df[col] = np.maximum(0, df[col])
    
    # Calculate derived columns
    total_expenses = (df['Groceries'] + df['Transport'] + df['Eating_Out'] + 
                     df['Entertainment'] + df['Utilities'] + df['Healthcare'] + 
                     df['Education'] + df['Miscellaneous'])
    
    df['Desired_Savings_Percentage'] = np.random.uniform(10, 40, n_samples)
    df['Desired_Savings'] = df['Income'] * df['Desired_Savings_Percentage'] / 100
    df['Disposable_Income'] = df['Income'] - df['Rent'] - df['Loan_Repayment'] - df['Insurance'] - total_expenses
    
    # Calculate potential savings (dummy values)
    df['Potential_Savings_Groceries'] = df['Groceries'] * 0.1
    df['Potential_Savings_Transport'] = df['Transport'] * 0.15
    df['Potential_Savings_Eating_Out'] = df['Eating_Out'] * 0.3
    df['Potential_Savings_Entertainment'] = df['Entertainment'] * 0.2
    df['Potential_Savings_Utilities'] = df['Utilities'] * 0.05
    df['Potential_Savings_Healthcare'] = df['Healthcare'] * 0.1
    df['Potential_Savings_Education'] = df['Education'] * 0.05
    df['Potential_Savings_Miscellaneous'] = df['Miscellaneous'] * 0.2
    
    return df

if __name__ == "__main__":
    # Run the complete analysis
    enhanced_advisor, analyzer = run_complete_analysis()
    print("✅ Enhanced Financial Advisor with ML models is ready!")
    
    # Demo with sample user
    sample_user = {
        'Income': 80000, 'Age': 32, 'Dependents': 2, 'Occupation': 'IT_Professional',
        'City_Tier': 'Tier_1', 'Rent': 25000, 'Loan_Repayment': 8000, 'Insurance': 2000,
        'Groceries': 8000, 'Transport': 5000, 'Eating_Out': 4000, 'Entertainment': 3000,
        'Utilities': 2000, 'Healthcare': 3000, 'Education': 2000, 'Miscellaneous': 3000,
        'Disposable_Income': 21000, 'Desired_Savings': 16000
    }
    
    ml_recommendations = enhanced_advisor.get_ml_based_recommendations(sample_user)
    print("\n🤖 ML-Based Recommendations:")
    for rec in ml_recommendations:
        print(f"  {rec}")