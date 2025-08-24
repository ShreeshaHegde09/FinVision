#!/usr/bin/env python3
"""
Enhanced Flask Web Application for Financial Advisor
Now includes advanced age-group based allocation engine
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import json
from datetime import datetime

# Import the new allocation engine
from investment_allocation_engine import InvestmentAllocationEngine, AllocationConfig

app = Flask(__name__)
CORS(app)

class EnhancedFinancialAdvisorAPI:
    def __init__(self):
        self.models_loaded = False
        self.load_models()
        
        # Initialize the advanced allocation engine
        self.allocation_engine = InvestmentAllocationEngine()
        
        # Investment options with returns (keeping existing for compatibility)
        self.investment_options = {
            'PPF': {'return': 7.1, 'risk': 'Very Low', 'tax_benefit': True},
            'ELSS': {'return': 12.0, 'risk': 'High', 'tax_benefit': True},
            'Fixed_Deposit': {'return': 6.5, 'risk': 'Very Low', 'tax_benefit': False},
            'Gold': {'return': 8.5, 'risk': 'Medium', 'tax_benefit': False},
            'Real_Estate': {'return': 9.0, 'risk': 'Medium', 'tax_benefit': True},
            'Equity_Mutual_Funds': {'return': 14.0, 'risk': 'High', 'tax_benefit': False},
            'Direct_Equity': {'return': 15.0, 'risk': 'Very High', 'tax_benefit': False},
            'NPS': {'return': 10.0, 'risk': 'Medium', 'tax_benefit': True},
            'Bonds': {'return': 7.5, 'risk': 'Low', 'tax_benefit': False},
            'Insurance_ULIP': {'return': 8.0, 'risk': 'Medium', 'tax_benefit': True}
        }
    
    def load_models(self):
        """Load trained ML models"""
        try:
            models_dir = 'models'
            if not os.path.exists(models_dir):
                print("❌ Models directory not found. Please run training first.")
                return
            
            self.savings_model = joblib.load(f'{models_dir}/savings_predictor.joblib')
            self.personality_model = joblib.load(f'{models_dir}/personality_classifier.joblib')
            self.scaler = joblib.load(f'{models_dir}/scaler.joblib')
            self.label_encoders = joblib.load(f'{models_dir}/label_encoders.joblib')
            self.feature_columns = joblib.load(f'{models_dir}/feature_columns.joblib')
            
            self.models_loaded = True
            print("✅ ML models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("Please run train_models.py first to train the models")
            self.models_loaded = False
    
    def predict_personality(self, user_data):
        """Predict financial personality using ML model"""
        if not self.models_loaded:
            return self._rule_based_personality(user_data)
        
        try:
            # Calculate features
            income = user_data.get('income', 0)
            total_expenses = (user_data.get('rent', 0) + user_data.get('groceries', 0) + 
                            user_data.get('transport', 0) + user_data.get('eatingOut', 0) + 
                            user_data.get('entertainment', 0) + user_data.get('healthcare', 0))
            
            savings_rate = max(0, (income - total_expenses)) / income if income > 0 else 0
            discretionary_spending = (user_data.get('eatingOut', 0) + user_data.get('entertainment', 0)) / income if income > 0 else 0
            essential_spending = (user_data.get('groceries', 0) + user_data.get('rent', 0)) / income if income > 0 else 0
            investment_capacity = savings_rate
            financial_burden = user_data.get('rent', 0) / income if income > 0 else 0
            
            features = np.array([[
                income,
                user_data.get('age', 30),
                user_data.get('dependents', 0),
                savings_rate,
                discretionary_spending,
                essential_spending,
                investment_capacity,
                financial_burden
            ]])
            
            personality = self.personality_model.predict(features)[0]
            return personality
            
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            return self._rule_based_personality(user_data)
    
    def _rule_based_personality(self, user_data):
        """Fallback rule-based personality classification"""
        income = user_data.get('income', 0)
        age = user_data.get('age', 30)
        dependents = user_data.get('dependents', 0)
        
        total_expenses = (user_data.get('rent', 0) + user_data.get('groceries', 0) + 
                         user_data.get('transport', 0) + user_data.get('eatingOut', 0) + 
                         user_data.get('entertainment', 0) + user_data.get('healthcare', 0))
        
        savings_capacity = (income - total_expenses) / income if income > 0 else 0
        discretionary_spending = (user_data.get('eatingOut', 0) + user_data.get('entertainment', 0)) / income if income > 0 else 0
        
        # Rule-based classification
        if savings_capacity >= 0.25 and age < 40 and dependents <= 2:
            return 'Aggressive_Investor'
        elif discretionary_spending >= 0.15 and savings_capacity < 0.15:
            return 'Heavy_Spender'
        elif savings_capacity >= 0.15 and (age > 45 or dependents > 2):
            return 'Low_Risk_Investor'
        else:
            return 'Moderate_Balanced'
    
    def calculate_monthly_investment(self, goal_amount, years, expected_return=12):
        """Calculate required monthly investment"""
        monthly_rate = expected_return / (12 * 100)
        months = years * 12
        
        if monthly_rate == 0:
            return goal_amount / months
        
        return goal_amount * monthly_rate / ((1 + monthly_rate) ** months - 1)
    
    def transform_user_data_for_allocation(self, user_data, goals):
        """Transform user data to format expected by allocation engine"""
        # Calculate monthly expenses
        monthly_expenses = (
            user_data.get('rent', 0) +
            user_data.get('groceries', 0) +
            user_data.get('transport', 0) +
            user_data.get('eatingOut', 0) +
            user_data.get('entertainment', 0) +
            user_data.get('healthcare', 0)
        )
        
        # Estimate total investment capacity
        income = user_data.get('income', 0)
        savings_capacity = max(0, income - monthly_expenses) * 12  # Annual savings
        total_investment_amount = savings_capacity * 5  # Assume 5 years of savings available
        
        # Determine job stability based on profession
        occupation = user_data.get('occupation', '')
        stable_jobs = ['Government_Employee', 'Teacher', 'Doctor', 'Banker']
        job_stability = 'stable' if occupation in stable_jobs else 'stable'  # Default to stable
        
        # Calculate risk appetite based on personality and age
        personality = self.predict_personality(user_data)
        age = user_data.get('age', 30)
        
        risk_appetite = 50  # Base risk
        if personality == 'Aggressive_Investor':
            risk_appetite = 80
        elif personality == 'Heavy_Spender':
            risk_appetite = 30
        elif personality == 'Low_Risk_Investor':
            risk_appetite = 25
        else:  # Moderate_Balanced
            risk_appetite = 55
        
        return {
            'age': age,
            'risk_appetite': risk_appetite,
            'income': income,
            'monthly_expenses': monthly_expenses,
            'dependents': user_data.get('dependents', 0),
            'job_stability': job_stability,
            'total_investment_amount': max(total_investment_amount, 100000),  # Minimum 1L
            'goals': goals,
            'preferences': user_data.get('preferences', {}),
            'occupation': occupation,
            'city_tier': user_data.get('cityTier', 'Tier_2')
        }
    
    def generate_enhanced_plan(self, user_data, goals):
        """Generate enhanced financial plan using new allocation engine"""
        try:
            # Transform data for allocation engine
            allocation_input = self.transform_user_data_for_allocation(user_data, goals)
            
            # Get advanced allocation recommendation
            allocation_result = self.allocation_engine.calculate_allocation(allocation_input)
            
            # Generate plan for each goal using the new allocation
            enhanced_plans = {}
            for goal in goals:
                # Calculate monthly investment needed
                goal_specific_allocation = allocation_result['allocation'].copy()
                
                # Calculate average expected return from the allocation
                avg_return = allocation_result['expected_annual_return']
                
                monthly_needed = self.calculate_monthly_investment(
                    goal['amount'], goal['years'], avg_return
                )
                
                # Create detailed investment breakdown using specific recommendations
                investment_breakdown = {}
                specific_recs = allocation_result.get('specific_recommendations', {})
                
                for asset_class, details in specific_recs.items():
                    if details['percentage'] > 1:  # Only include if >1%
                        monthly_amount = monthly_needed * (details['percentage'] / 100)
                        
                        investment_breakdown[asset_class] = {
                            'monthly_amount': monthly_amount,
                            'percentage': details['percentage'],
                            'expected_return': avg_return,  # Use portfolio return
                            'risk_level': self._map_risk_level(asset_class),
                            'tax_benefit': self._has_tax_benefit(asset_class),
                            'recommended_vehicles': details.get('recommended_vehicles', []),
                            'vehicle_split': details.get('suggested_split', {})
                        }
                
                # Determine achievability
                income = user_data.get('income', 1)
                achievability = 'High' if monthly_needed < (income * 0.3) else 'Medium' if monthly_needed < (income * 0.5) else 'Low'
                
                enhanced_plans[goal['name']] = {
                    'goal_amount': goal['amount'],
                    'time_horizon': goal['years'],
                    'monthly_investment': monthly_needed,
                    'investment_breakdown': investment_breakdown,
                    'expected_return': avg_return,
                    'achievability': achievability,
                    'age_group_strategy': allocation_result['age_group_description'],
                    'risk_profile': allocation_result['portfolio_risk_level'],
                    'goal_alignment': allocation_result.get('goal_alignment', {}).get(goal['name'], 'Good')
                }
            
            # Generate enhanced recommendations
            recommendations = self._generate_enhanced_recommendations(
                allocation_result, user_data, enhanced_plans
            )
            
            return {
                'personality': self.predict_personality(user_data),
                'age_group': allocation_result['age_group'],
                'risk_score': allocation_result['risk_score'],
                'plans': enhanced_plans,
                'recommendations': recommendations,
                'allocation_summary': allocation_result['allocation_percentages'],
                'tax_strategy': allocation_result['tax_focus'],
                'rebalancing_frequency': allocation_result['rebalancing_frequency'],
                'next_steps': allocation_result['next_steps'],
                'portfolio_metrics': {
                    'expected_return': allocation_result['expected_annual_return'],
                    'risk_level': allocation_result['portfolio_risk_level'],
                    'liquidity_requirement': allocation_result['liquidity_requirement']
                }
            }
            
        except Exception as e:
            print(f"Error in enhanced plan generation: {e}")
            # Fallback to original plan generation
            return self.generate_plan(user_data, goals)
    
    def _map_risk_level(self, asset_class):
        """Map asset class to risk level"""
        risk_mapping = {
            'equity': 'High',
            'debt': 'Low',
            'real_estate': 'Medium',
            'alternatives': 'Very High',
            'gold': 'Medium',
            'cash': 'Very Low',
            'mutual_funds': 'High',
            'child_plans': 'Low',
            'retirement_plans': 'Medium',
            'healthcare_funds': 'Low'
        }
        return risk_mapping.get(asset_class, 'Medium')
    
    def _has_tax_benefit(self, asset_class):
        """Check if asset class has tax benefits"""
        tax_benefit_assets = ['debt', 'retirement_plans', 'child_plans']
        return asset_class in tax_benefit_assets
    
    def _generate_enhanced_recommendations(self, allocation_result, user_data, plans):
        """Generate enhanced recommendations using allocation engine insights"""
        recommendations = allocation_result.get('next_steps', [])
        
        # Add specific recommendations based on age group and allocation
        age_group = allocation_result['age_group']
        total_monthly = sum(plan['monthly_investment'] for plan in plans.values())
        income = user_data.get('income', 1)
        investment_ratio = total_monthly / income
        
        # Investment feasibility
        if investment_ratio > 0.5:
            recommendations.insert(0, "⚠️ Required investment exceeds 50% of income. Consider extending timeline or reducing goal amounts.")
        elif investment_ratio > 0.3:
            recommendations.insert(0, "⚠️ High investment requirement. Look for additional income sources.")
        else:
            recommendations.insert(0, "✅ Your investment requirements are manageable within your current income.")
        
        # Age-specific enhanced recommendations
        if age_group in ['young_adult', 'early_career']:
            recommendations.extend([
                "🚀 Take advantage of your long investment horizon for wealth creation",
                "📱 Consider using robo-advisors for automated rebalancing",
                "💡 Start with small amounts and increase SIP annually"
            ])
        elif age_group in ['family_building', 'wealth_accumulation']:
            recommendations.extend([
                "👨‍👩‍👧‍👦 Balance family needs with long-term wealth creation",
                "🏠 Consider real estate investment for diversification",
                "📚 Plan for children's education with dedicated funds"
            ])
        elif age_group in ['pre_retirement', 'retired']:
            recommendations.extend([
                "🛡️ Focus on capital preservation and regular income",
                "🏥 Increase healthcare and emergency fund allocation",
                "📊 Consider annuity products for guaranteed income"
            ])
        
        # Tax optimization recommendations
        recommendations.extend([
            f"💰 Follow the tax strategy: {allocation_result['tax_focus']}",
            f"⚖️ Rebalance your portfolio: {allocation_result['rebalancing_frequency']}",
            "📋 Maintain detailed investment records for tax benefits"
        ])
        
        return recommendations
    
    def generate_plan(self, user_data, goals):
        """Original plan generation method (fallback)"""
        personality = self.predict_personality(user_data)
        
        plans = {}
        for goal in goals:
            allocation = self.generate_investment_allocation(
                personality, 
                user_data.get('age', 30), 
                user_data.get('dependents', 0)
            )
            
            # Calculate average expected return
            avg_return = sum(self.investment_options[inv]['return'] * percent 
                           for inv, percent in allocation.items() 
                           if inv in self.investment_options)
            
            monthly_needed = self.calculate_monthly_investment(
                goal['amount'], goal['years'], avg_return
            )
            
            # Create investment breakdown
            investment_breakdown = {}
            for investment, percentage in allocation.items():
                if investment in self.investment_options:
                    investment_breakdown[investment] = {
                        'monthly_amount': monthly_needed * percentage,
                        'percentage': percentage * 100,
                        'expected_return': self.investment_options[investment]['return'],
                        'risk_level': self.investment_options[investment]['risk'],
                        'tax_benefit': self.investment_options[investment]['tax_benefit']
                    }
            
            # Determine achievability
            income = user_data.get('income', 1)
            achievability = 'High' if monthly_needed < (income * 0.3) else 'Medium' if monthly_needed < (income * 0.5) else 'Low'
            
            plans[goal['name']] = {
                'goal_amount': goal['amount'],
                'time_horizon': goal['years'],
                'monthly_investment': monthly_needed,
                'investment_breakdown': investment_breakdown,
                'expected_return': avg_return,
                'achievability': achievability
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(personality, user_data, plans)
        
        return {
            'personality': personality,
            'plans': plans,
            'recommendations': recommendations
        }
    
    def generate_investment_allocation(self, personality, age, dependents):
        """Generate investment allocation based on personality (original method)"""
        allocations = {
            'Aggressive_Investor': {
                'Direct_Equity': 0.35,
                'Equity_Mutual_Funds': 0.30,
                'ELSS': 0.20,
                'Real_Estate': 0.10,
                'Gold': 0.05
            },
            'Heavy_Spender': {
                'PPF': 0.40,
                'ELSS': 0.30,
                'Fixed_Deposit': 0.20,
                'Gold': 0.10
            },
            'Low_Risk_Investor': {
                'PPF': 0.30,
                'Fixed_Deposit': 0.30,
                'Gold': 0.20,
                'Bonds': 0.15,
                'NPS': 0.05
            },
            'Moderate_Balanced': {
                'Equity_Mutual_Funds': 0.25,
                'ELSS': 0.20,
                'PPF': 0.20,
                'Gold': 0.15,
                'NPS': 0.10,
                'Fixed_Deposit': 0.10
            }
        }
        
        allocation = allocations.get(personality, allocations['Moderate_Balanced'])
        
        # Age-based adjustment
        if age > 45:
            # Reduce equity exposure for older users
            for key in list(allocation.keys()):
                if 'Equity' in key:
                    allocation[key] *= 0.7
                elif key in ['PPF', 'Fixed_Deposit']:
                    allocation[key] *= 1.3
        
        # Normalize allocation
        total = sum(allocation.values())
        allocation = {k: v/total for k, v in allocation.items()}
        
        return allocation
    
    def _generate_recommendations(self, personality, user_data, plans):
        """Generate personalized recommendations (original method)"""
        recommendations = []
        
        total_monthly = sum(plan['monthly_investment'] for plan in plans.values())
        income = user_data.get('income', 1)
        investment_ratio = total_monthly / income
        
        # Investment feasibility
        if investment_ratio > 0.5:
            recommendations.append("⚠️ Required investment exceeds 50% of income. Consider extending timeline or reducing goal amounts.")
        elif investment_ratio > 0.3:
            recommendations.append("⚠️ High investment requirement. Look for additional income sources or expense reduction.")
        else:
            recommendations.append("✅ Your investment requirements are manageable within your current income.")
        
        # Personality-specific advice
        if personality == 'Heavy_Spender':
            recommendations.extend([
                "🔒 Set up automatic SIP investments to build financial discipline",
                "📱 Use expense tracking apps to monitor and control spending",
                "🎯 Create separate savings accounts for each financial goal"
            ])
        elif personality == 'Low_Risk_Investor':
            recommendations.extend([
                "📈 Consider gradually increasing equity exposure for better long-term returns",
                "🛡️ Your conservative approach is excellent for capital protection",
                "💡 SIP in mutual funds provides rupee cost averaging benefits"
            ])
        elif personality == 'Aggressive_Investor':
            recommendations.extend([
                "⚖️ Maintain adequate emergency fund despite high-risk investments",
                "📊 Regular portfolio rebalancing is crucial for optimal returns",
                "🎓 Stay updated with market trends and company fundamentals"
            ])
        
        # General recommendations
        recommendations.extend([
            "💸 Maximize Section 80C deductions (₹1.5 lakh annually)",
            "🏥 Consider health insurance for Section 80D tax benefits",
            f"💰 Maintain emergency fund of ₹{user_data.get('rent', 0) * 6:,.0f} (6 months expenses)"
        ])
        
        return recommendations

# Initialize the enhanced advisor
advisor = EnhancedFinancialAdvisorAPI()

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/generate_plan', methods=['POST'])
def generate_plan_api():
    """Enhanced API endpoint to generate financial plan"""
    try:
        data = request.json
        user_data = data.get('user_data', {})
        goals = data.get('goals', [])
        use_enhanced = data.get('use_enhanced', True)  # Option to use enhanced engine
        
        # Validate input
        if not user_data.get('income') or not goals:
            return jsonify({'error': 'Missing required data'}), 400
        
        # Generate plan using enhanced engine by default
        if use_enhanced:
            result = advisor.generate_enhanced_plan(user_data, goals)
        else:
            result = advisor.generate_plan(user_data, goals)
        
        return jsonify({
            'success': True,
            'data': result,
            'enhanced': use_enhanced,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in generate_plan_api: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/get-allocation', methods=['POST'])
def get_allocation_api():
    """New endpoint specifically for allocation calculation"""
    try:
        user_data = request.json
        
        # Validate required fields
        required_fields = ['age', 'income', 'monthly_expenses']
        for field in required_fields:
            if field not in user_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Add default goals if not provided
        if 'goals' not in user_data:
            user_data['goals'] = [
                {'name': 'Wealth Creation', 'amount': 5000000, 'years': 15}
            ]
        
        # Calculate allocation using the engine directly
        allocation_result = advisor.allocation_engine.calculate_allocation(user_data)
        
        return jsonify({
            'success': True,
            'allocation': allocation_result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in get_allocation_api: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': advisor.models_loaded,
        'enhanced_engine': True,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict_personality', methods=['POST'])
def predict_personality_api():
    """API endpoint to predict personality only"""
    try:
        user_data = request.json
        personality = advisor.predict_personality(user_data)
        
        return jsonify({
            'success': True,
            'personality': personality
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/age-group-info/<age>')
def get_age_group_info(age):
    """Get age group information and preferences"""
    try:
        age_int = int(age)
        age_group = advisor.allocation_engine.get_age_group(age_int)
        age_group_data = advisor.allocation_engine.config.AGE_GROUP_PREFERENCES[age_group]
        
        return jsonify({
            'success': True,
            'age_group': age_group,
            'description': age_group_data['description'],
            'preferred_investments': age_group_data['preferred_investments'],
            'investment_vehicles': age_group_data['investment_vehicles'],
            'tax_focus': age_group_data['tax_focus'],
            'liquidity_needs': age_group_data['liquidity_needs']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced Financial Advisor Web Application")
    print("📊 Models loaded:", advisor.models_loaded)
    print("🎯 Enhanced allocation engine: Active")
    print("🇮🇳 Age-group based preferences: Enabled")
    
    if not advisor.models_loaded:
        print("⚠️  Running without ML models. Please run 'python train_models.py' first for better predictions.")
    
    print("🌐 Server starting at http://localhost:5000")
    
    # Use Waitress for production-ready serving
    try:
        from waitress import serve
        serve(app, host='0.0.0.0', port=5000)
    except ImportError:
        # Fallback to Flask dev server
        app.run(host='0.0.0.0', port=5000, debug=False)