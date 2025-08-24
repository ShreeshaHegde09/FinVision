import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FinancialPersonalityAnalyzer:
    """Analyzes user's financial personality based on spending and investment patterns"""
    
    def __init__(self):
        self.personality_model = None
        self.scaler = StandardScaler()
        self.le_occupation = LabelEncoder()
        self.le_city = LabelEncoder()
        
    def create_personality_features(self, df):
        """Create features for personality analysis"""
        df = df.copy()
        
        # Calculate spending ratios
        df['Savings_Rate'] = df['Desired_Savings'] / df['Income']
        df['Discretionary_Spending'] = (df['Eating_Out'] + df['Entertainment']) / df['Income']
        df['Essential_Spending'] = (df['Groceries'] + df['Utilities'] + df['Rent']) / df['Income']
        df['Investment_Capacity'] = df['Disposable_Income'] / df['Income']
        df['Risk_Buffer'] = df['Insurance'] / df['Income']
        
        # Age groups for life stage analysis
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], 
                                labels=['Young', 'Early_Career', 'Mid_Career', 'Pre_Retirement'])
        
        return df
    
    def classify_financial_personality(self, df):
        """Classify users into financial personality types"""
        df = self.create_personality_features(df)
        
        # Define personality based on spending and saving patterns
        conditions = [
            (df['Savings_Rate'] >= 0.3) & (df['Investment_Capacity'] >= 0.2),  # Aggressive Investor
            (df['Discretionary_Spending'] >= 0.15) & (df['Savings_Rate'] < 0.15),  # Heavy Spender
            (df['Savings_Rate'] >= 0.2) & (df['Risk_Buffer'] >= 0.05),  # Low Risk Investor
        ]
        
        choices = ['Aggressive_Investor', 'Heavy_Spender', 'Low_Risk_Investor']
        df['Financial_Personality'] = np.select(conditions, choices, default='Moderate_Balanced')
        
        return df

class InvestmentPlanGenerator:
    """Generates customized investment plans based on goals and user profile"""
    
    def __init__(self):
        # Indian investment options with typical returns (%)
        self.investment_options = {
            'PPF': {'return': 7.1, 'risk': 'Very Low', 'liquidity': 'Low', 'tax_benefit': True},
            'ELSS': {'return': 12.0, 'risk': 'High', 'liquidity': 'Medium', 'tax_benefit': True},
            'Fixed_Deposit': {'return': 6.5, 'risk': 'Very Low', 'liquidity': 'Medium', 'tax_benefit': False},
            'Gold': {'return': 8.5, 'risk': 'Medium', 'liquidity': 'High', 'tax_benefit': False},
            'Real_Estate': {'return': 9.0, 'risk': 'Medium', 'liquidity': 'Very Low', 'tax_benefit': True},
            'Equity_Mutual_Funds': {'return': 14.0, 'risk': 'High', 'liquidity': 'High', 'tax_benefit': False},
            'Direct_Equity': {'return': 15.0, 'risk': 'Very High', 'liquidity': 'High', 'tax_benefit': False},
            'NPS': {'return': 10.0, 'risk': 'Medium', 'liquidity': 'Very Low', 'tax_benefit': True},
            'Bonds': {'return': 7.5, 'risk': 'Low', 'liquidity': 'Medium', 'tax_benefit': False},
            'Insurance_ULIP': {'return': 8.0, 'risk': 'Medium', 'liquidity': 'Low', 'tax_benefit': True}
        }
        
        # Risk profiles for different personalities
        self.risk_profiles = {
            'Aggressive_Investor': ['Direct_Equity', 'Equity_Mutual_Funds', 'ELSS', 'Real_Estate'],
            'Heavy_Spender': ['PPF', 'ELSS', 'Fixed_Deposit'],  # Forced savings
            'Low_Risk_Investor': ['PPF', 'Fixed_Deposit', 'Gold', 'Bonds'],
            'Moderate_Balanced': ['ELSS', 'Equity_Mutual_Funds', 'PPF', 'Gold', 'NPS']
        }
    
    def calculate_monthly_investment_needed(self, goal_amount, years, expected_return=12):
        """Calculate monthly SIP needed using compound interest"""
        monthly_rate = expected_return / (12 * 100)
        months = years * 12
        
        if monthly_rate == 0:
            return goal_amount / months
        
        # Future Value of Annuity formula
        monthly_investment = goal_amount * monthly_rate / ((1 + monthly_rate) ** months - 1)
        return monthly_investment
    
    def generate_investment_allocation(self, user_profile, goal_amount, years):
        """Generate investment allocation based on user profile"""
        personality = user_profile['Financial_Personality']
        age = user_profile['Age']
        income = user_profile['Income']
        dependents = user_profile['Dependents']
        
        # Age-based risk adjustment
        equity_percentage = max(0.2, min(0.8, (100 - age) / 100))
        
        # Dependent-based adjustment (more dependents = lower risk)
        if dependents > 2:
            equity_percentage *= 0.8
        
        # Get suitable investment options
        suitable_investments = self.risk_profiles[personality]
        
        # Calculate required monthly investment
        avg_return = np.mean([self.investment_options[inv]['return'] for inv in suitable_investments])
        monthly_needed = self.calculate_monthly_investment_needed(goal_amount, years, avg_return)
        
        # Create allocation based on personality and constraints
        allocation = {}
        
        if personality == 'Aggressive_Investor':
            allocation = {
                'Direct_Equity': 0.4,
                'Equity_Mutual_Funds': 0.3,
                'ELSS': 0.2,
                'Real_Estate': 0.1
            }
        elif personality == 'Heavy_Spender':
            allocation = {
                'PPF': 0.4,
                'ELSS': 0.4,
                'Fixed_Deposit': 0.2
            }
        elif personality == 'Low_Risk_Investor':
            allocation = {
                'PPF': 0.3,
                'Fixed_Deposit': 0.3,
                'Gold': 0.2,
                'Bonds': 0.2
            }
        else:  # Moderate_Balanced
            allocation = {
                'Equity_Mutual_Funds': 0.3,
                'ELSS': 0.2,
                'PPF': 0.2,
                'Gold': 0.15,
                'NPS': 0.15
            }
        
        # Convert percentages to actual amounts
        investment_plan = {}
        for investment, percentage in allocation.items():
            investment_plan[investment] = {
                'monthly_amount': monthly_needed * percentage,
                'percentage': percentage * 100,
                'expected_return': self.investment_options[investment]['return'],
                'risk_level': self.investment_options[investment]['risk']
            }
        
        return {
            'total_monthly_investment': monthly_needed,
            'investment_breakdown': investment_plan,
            'expected_portfolio_return': avg_return,
            'goal_achievability': 'High' if monthly_needed < (income * 0.3) else 'Medium' if monthly_needed < (income * 0.5) else 'Low'
        }

class FinancialGoalAdvisor:
    """Main class that orchestrates the entire financial advisory system"""
    
    def __init__(self):
        self.personality_analyzer = FinancialPersonalityAnalyzer()
        self.investment_planner = InvestmentPlanGenerator()
        self.savings_predictor = None
        self.expense_optimizer = None
        
    def load_and_preprocess_data(self, df):
        """Load and preprocess the dataset"""
        df = df.copy()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Create additional features
        df = self.personality_analyzer.create_personality_features(df)
        df = self.personality_analyzer.classify_financial_personality(df)
        
        return df
    
    def train_savings_predictor(self, df):
        """Train model to predict potential savings"""
        features = ['Income', 'Age', 'Dependents', 'Savings_Rate', 'Essential_Spending', 
                   'Discretionary_Spending', 'Investment_Capacity']
        
        # Encode categorical variables
        df_encoded = df.copy()
        df_encoded['Occupation_encoded'] = self.personality_analyzer.le_occupation.fit_transform(df['Occupation'])
        df_encoded['City_Tier_encoded'] = self.personality_analyzer.le_city.fit_transform(df['City_Tier'])
        
        features.extend(['Occupation_encoded', 'City_Tier_encoded'])
        
        X = df_encoded[features]
        y = df_encoded['Desired_Savings']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.personality_analyzer.scaler.fit_transform(X_train)
        X_test_scaled = self.personality_analyzer.scaler.transform(X_test)
        
        # Train Random Forest
        self.savings_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.savings_predictor.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.savings_predictor.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Savings Predictor - R² Score: {r2:.3f}, RMSE: ₹{rmse:,.0f}")
        
        return self.savings_predictor
    
    def analyze_user_profile(self, user_data):
        """Analyze individual user profile"""
        # Create DataFrame from user input
        df_user = pd.DataFrame([user_data])
        df_processed = self.load_and_preprocess_data(df_user)
        
        profile = {
            'Financial_Personality': df_processed['Financial_Personality'].iloc[0],
            'Savings_Rate': df_processed['Savings_Rate'].iloc[0],
            'Investment_Capacity': df_processed['Investment_Capacity'].iloc[0],
            'Discretionary_Spending': df_processed['Discretionary_Spending'].iloc[0],
            'Age_Group': df_processed['Age_Group'].iloc[0],
            'Risk_Profile': self.get_risk_profile(df_processed.iloc[0])
        }
        
        return profile
    
    def get_risk_profile(self, user_data):
        """Determine user's risk profile"""
        age = user_data['Age']
        dependents = user_data['Dependents']
        income = user_data['Income']
        personality = user_data['Financial_Personality']
        
        risk_score = 0
        
        # Age factor (younger = higher risk tolerance)
        if age < 30:
            risk_score += 3
        elif age < 40:
            risk_score += 2
        elif age < 50:
            risk_score += 1
        
        # Dependents factor
        risk_score -= dependents
        
        # Income factor
        if income > 100000:
            risk_score += 2
        elif income > 50000:
            risk_score += 1
        
        # Personality factor
        personality_risk = {
            'Aggressive_Investor': 3,
            'Moderate_Balanced': 1,
            'Low_Risk_Investor': -2,
            'Heavy_Spender': 0
        }
        risk_score += personality_risk.get(personality, 0)
        
        if risk_score >= 4:
            return 'High'
        elif risk_score >= 1:
            return 'Medium'
        else:
            return 'Low'
    
    def create_financial_plan(self, user_data, financial_goals):
        """Create comprehensive financial plan"""
        profile = self.analyze_user_profile(user_data)
        
        plans = {}
        for goal in financial_goals:
            goal_name = goal['name']
            goal_amount = goal['amount']
            years = goal['years']
            
            plan = self.investment_planner.generate_investment_allocation(
                user_data, goal_amount, years
            )
            
            plans[goal_name] = {
                'goal_amount': goal_amount,
                'time_horizon': years,
                'monthly_investment_needed': plan['total_monthly_investment'],
                'investment_strategy': plan['investment_breakdown'],
                'expected_return': plan['expected_portfolio_return'],
                'achievability': plan['goal_achievability'],
                'recommendations': self.get_recommendations(profile, plan, user_data)
            }
        
        return {
            'user_profile': profile,
            'financial_plans': plans,
            'overall_recommendations': self.get_overall_recommendations(profile, plans, user_data)
        }
    
    def get_recommendations(self, profile, plan, user_data):
        """Generate specific recommendations"""
        recommendations = []
        
        personality = profile['Financial_Personality']
        monthly_needed = plan['total_monthly_investment']
        income = user_data['Income']
        
        # Investment percentage check
        investment_ratio = monthly_needed / income
        if investment_ratio > 0.5:
            recommendations.append("⚠️  Required investment exceeds 50% of income. Consider extending timeline or reducing goal amount.")
        elif investment_ratio > 0.3:
            recommendations.append("⚠️  High investment requirement. Look for ways to increase income or reduce expenses.")
        
        # Personality-specific advice
        if personality == 'Heavy_Spender':
            recommendations.extend([
                "🔒 Consider automatic SIP investments to enforce discipline",
                "📱 Use expense tracking apps to monitor spending",
                "🎯 Set up separate savings account for goals"
            ])
        elif personality == 'Low_Risk_Investor':
            recommendations.extend([
                "📈 Consider gradually increasing equity exposure as you get comfortable",
                "🛡️ Your conservative approach is good for capital protection",
                "💡 Learn about SIP in mutual funds for rupee cost averaging"
            ])
        elif personality == 'Aggressive_Investor':
            recommendations.extend([
                "⚖️ Maintain some stable investments for emergencies",
                "📊 Regular portfolio rebalancing is crucial",
                "🎓 Stay updated with market trends and company fundamentals"
            ])
        
        # Age-specific advice
        age = user_data['Age']
        if age < 30:
            recommendations.append("🚀 Take advantage of your long time horizon for wealth creation")
        elif age > 50:
            recommendations.append("🛡️ Focus on capital preservation and regular income generation")
        
        return recommendations
    
    def get_overall_recommendations(self, profile, plans, user_data):
        """Generate overall financial recommendations"""
        recommendations = []
        
        # Emergency fund check
        monthly_expenses = user_data['Rent'] + user_data['Groceries'] + user_data['Utilities']
        emergency_fund = monthly_expenses * 6
        recommendations.append(f"💰 Maintain emergency fund of ₹{emergency_fund:,.0f} (6 months expenses)")
        
        # Tax saving recommendations
        recommendations.extend([
            "💸 Maximize Section 80C deductions (₹1.5 lakh annually)",
            "🏥 Consider health insurance for tax benefits under 80D",
            "🏠 Home loan EMI provides tax benefits under 80C and 24(b)"
        ])
        
        # Insurance recommendations
        income = user_data['Income']
        life_cover = income * 120  # 10 times annual income
        recommendations.append(f"🛡️ Ensure adequate life insurance cover: ₹{life_cover:,.0f}")
        
        # Investment discipline
        recommendations.extend([
            "📅 Set up automatic investments on salary day",
            "📊 Review and rebalance portfolio annually",
            "📚 Increase financial literacy through books and courses",
            "🎯 Track progress monthly and adjust if needed"
        ])
        
        return recommendations
    
    def monitor_progress(self, initial_plan, current_investments, months_elapsed):
        """Monitor investment progress and provide alerts"""
        alerts = []
        
        for goal_name, plan in initial_plan['financial_plans'].items():
            expected_amount = plan['monthly_investment_needed'] * months_elapsed
            actual_amount = current_investments.get(goal_name, 0)
            
            variance = (actual_amount - expected_amount) / expected_amount if expected_amount > 0 else 0
            
            if variance < -0.2:  # 20% behind
                alerts.append(f"🚨 {goal_name}: You're {abs(variance)*100:.1f}% behind schedule!")
            elif variance < -0.1:  # 10% behind
                alerts.append(f"⚠️  {goal_name}: Slightly behind schedule ({abs(variance)*100:.1f}%)")
            elif variance > 0.1:  # 10% ahead
                alerts.append(f"🎉 {goal_name}: Great job! You're {variance*100:.1f}% ahead of schedule")
        
        return alerts

# Example usage and testing
def demo_financial_advisor():
    """Demonstrate the financial advisor system"""
    
    # Sample user data
    user_data = {
        'Income': 80000,
        'Age': 32,
        'Dependents': 2,
        'Occupation': 'IT_Professional',
        'City_Tier': 'Tier_1',
        'Rent': 25000,
        'Loan_Repayment': 8000,
        'Insurance': 2000,
        'Groceries': 8000,
        'Transport': 5000,
        'Eating_Out': 4000,
        'Entertainment': 3000,
        'Utilities': 2000,
        'Healthcare': 3000,
        'Education': 2000,
        'Miscellaneous': 3000
    }
    
    # Financial goals
    goals = [
        {'name': 'Child Education', 'amount': 5000000, 'years': 15},
        {'name': 'Retirement', 'amount': 10000000, 'years': 25},
        {'name': 'Home Purchase', 'amount': 3000000, 'years': 8}
    ]
    
    # Initialize advisor
    advisor = FinancialGoalAdvisor()
    
    # Create financial plan
    financial_plan = advisor.create_financial_plan(user_data, goals)
    
    # Display results
    print("=" * 80)
    print("🏦 PERSONALIZED FINANCIAL ADVISORY REPORT")
    print("=" * 80)
    
    print(f"\n👤 USER PROFILE:")
    print(f"Financial Personality: {financial_plan['user_profile']['Financial_Personality']}")
    print(f"Current Savings Rate: {financial_plan['user_profile']['Savings_Rate']:.1%}")
    print(f"Investment Capacity: {financial_plan['user_profile']['Investment_Capacity']:.1%}")
    
    print(f"\n🎯 GOAL-WISE INVESTMENT PLANS:")
    for goal_name, plan in financial_plan['financial_plans'].items():
        print(f"\n📋 {goal_name.upper()}:")
        print(f"  Target Amount: ₹{plan['goal_amount']:,}")
        print(f"  Time Horizon: {plan['time_horizon']} years")
        print(f"  Monthly Investment: ₹{plan['monthly_investment_needed']:,.0f}")
        print(f"  Achievability: {plan['achievability']}")
        
        print(f"\n  💼 Investment Allocation:")
        for investment, details in plan['investment_strategy'].items():
            print(f"    {investment}: ₹{details['monthly_amount']:,.0f} ({details['percentage']:.1f}%)")
        
        print(f"\n  💡 Recommendations:")
        for rec in plan['recommendations']:
            print(f"    {rec}")
    
    print(f"\n🌟 OVERALL RECOMMENDATIONS:")
    for rec in financial_plan['overall_recommendations']:
        print(f"  {rec}")
    
    return financial_plan

# Run the demo
if __name__ == "__main__":
    demo_plan = demo_financial_advisor()