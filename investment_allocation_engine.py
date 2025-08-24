#!/usr/bin/env python3
"""
Advanced Investment Allocation Engine for Indian Financial Advisor
Integrates age-group preferences with personalized risk profiling
"""

import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AllocationConfig:
    """Configuration class for allocation weights and preferences"""
    
    # Risk appetite weights (0-100 scale)
    RISK_WEIGHTS = {
        'conservative': {'equity': 0.20, 'debt': 0.50, 'real_estate': 0.15, 'gold': 0.10, 'cash': 0.05},
        'moderate': {'equity': 0.40, 'debt': 0.30, 'real_estate': 0.15, 'gold': 0.10, 'cash': 0.05},
        'aggressive': {'equity': 0.60, 'debt': 0.15, 'real_estate': 0.15, 'gold': 0.05, 'cash': 0.05}
    }
    
    # Age-group investment preferences (Indian context)
    AGE_GROUP_PREFERENCES = {
        'young_adult': {  # 18-28 years
            'age_range': (18, 28),
            'description': 'High growth focus, tech-savvy, experimental',
            'preferred_investments': {
                'equity': 0.45,
                'mutual_funds': 0.25,
                'alternatives': 0.15,  # Crypto, startups, new-age investments
                'debt': 0.10,
                'cash': 0.05
            },
            'investment_vehicles': {
                'equity': ['Direct_Equity', 'Growth_Mutual_Funds', 'Sectoral_Funds'],
                'debt': ['Liquid_Funds', 'Ultra_Short_Funds'],
                'alternatives': ['Crypto', 'P2P_Lending', 'REITs'],
                'cash': ['Savings_Account', 'FD_Short_Term']
            },
            'tax_focus': 'ELSS for 80C',
            'liquidity_needs': 3  # months
        },
        
        'early_career': {  # 28-35 years
            'age_range': (28, 35),
            'description': 'Career building, marriage planning, home buying',
            'preferred_investments': {
                'equity': 0.40,
                'debt': 0.25,
                'real_estate': 0.20,  # Home loan preparation
                'gold': 0.10,
                'cash': 0.05
            },
            'investment_vehicles': {
                'equity': ['Large_Cap_Funds', 'Mid_Cap_Funds', 'ELSS'],
                'debt': ['PPF', 'NSC', 'Corporate_Bonds'],
                'real_estate': ['Home_Down_Payment', 'REITs'],
                'gold': ['Gold_ETF', 'Digital_Gold'],
                'cash': ['Emergency_Fund', 'Fixed_Deposits']
            },
            'tax_focus': 'PPF + ELSS combination',
            'liquidity_needs': 6  # months
        },
        
        'family_building': {  # 35-45 years
            'age_range': (35, 45),
            'description': 'Family expenses, child education, stability focus',
            'preferred_investments': {
                'equity': 0.35,
                'debt': 0.30,
                'child_plans': 0.15,  # Education-specific investments
                'real_estate': 0.10,
                'gold': 0.10
            },
            'investment_vehicles': {
                'equity': ['Diversified_Funds', 'Child_Funds', 'Conservative_Equity'],
                'debt': ['PPF', 'Child_Plans', 'NSC', 'Fixed_Deposits'],
                'child_plans': ['Sukanya_Samriddhi', 'Child_Education_Plans'],
                'real_estate': ['Second_Property', 'Real_Estate_Funds'],
                'gold': ['Physical_Gold', 'Gold_Mutual_Funds']
            },
            'tax_focus': 'Maximum 80C utilization + Child benefits',
            'liquidity_needs': 8  # months
        },
        
        'wealth_accumulation': {  # 45-55 years
            'age_range': (45, 55),
            'description': 'Peak earning years, retirement planning',
            'preferred_investments': {
                'equity': 0.30,
                'debt': 0.35,
                'retirement_plans': 0.20,  # NPS, PF
                'real_estate': 0.10,
                'gold': 0.05
            },
            'investment_vehicles': {
                'equity': ['Blue_Chip_Funds', 'Dividend_Funds', 'Conservative_Funds'],
                'debt': ['PPF', 'Corporate_FD', 'Government_Bonds'],
                'retirement_plans': ['NPS', 'PF_VPF', 'Pension_Plans'],
                'real_estate': ['Commercial_Property', 'REITs'],
                'gold': ['Gold_Bonds', 'Gold_ETF']
            },
            'tax_focus': 'NPS 80CCD + Traditional 80C',
            'liquidity_needs': 12  # months
        },
        
        'pre_retirement': {  # 55-65 years
            'age_range': (55, 65),
            'description': 'Capital preservation, income generation',
            'preferred_investments': {
                'debt': 0.50,
                'equity': 0.20,
                'retirement_income': 0.15,
                'gold': 0.10,
                'cash': 0.05
            },
            'investment_vehicles': {
                'debt': ['Senior_Citizen_FD', 'Government_Bonds', 'Monthly_Income_Plans'],
                'equity': ['Dividend_Funds', 'Large_Cap_Conservative'],
                'retirement_income': ['Annuity_Plans', 'Pension_Funds'],
                'gold': ['Physical_Gold', 'Gold_Bonds'],
                'cash': ['Senior_Savings', 'Liquid_Funds']
            },
            'tax_focus': 'Senior citizen benefits + Health insurance',
            'liquidity_needs': 18  # months
        },
        
        'retired': {  # 65+ years
            'age_range': (65, 100),
            'description': 'Income generation, healthcare focus, legacy planning',
            'preferred_investments': {
                'debt': 0.60,
                'healthcare_funds': 0.15,
                'equity': 0.15,
                'gold': 0.10
            },
            'investment_vehicles': {
                'debt': ['Senior_Citizen_Schemes', 'Government_Bonds', 'Bank_FD'],
                'healthcare_funds': ['Health_Insurance', 'Medical_Emergency_Fund'],
                'equity': ['Dividend_Funds_Conservative', 'Blue_Chip_Defensive'],
                'gold': ['Physical_Gold', 'Gold_Savings']
            },
            'tax_focus': 'Senior citizen exemptions + Healthcare',
            'liquidity_needs': 24  # months
        }
    }

class InvestmentAllocationEngine:
    """
    Advanced investment allocation engine considering Indian investor preferences
    """
    
    def __init__(self, config: AllocationConfig = None):
        self.config = config or AllocationConfig()
        
        # Indian-specific investment options with expected returns
        self.investment_universe = {
            'Direct_Equity': {'return': 15.0, 'risk': 'Very High', 'liquidity': 'High', 'tax_benefit': False},
            'Large_Cap_Funds': {'return': 12.0, 'risk': 'High', 'liquidity': 'High', 'tax_benefit': False},
            'Mid_Cap_Funds': {'return': 14.0, 'risk': 'Very High', 'liquidity': 'High', 'tax_benefit': False},
            'ELSS': {'return': 12.0, 'risk': 'High', 'liquidity': 'Low', 'tax_benefit': True},
            'PPF': {'return': 7.1, 'risk': 'Very Low', 'liquidity': 'Very Low', 'tax_benefit': True},
            'NPS': {'return': 10.0, 'risk': 'Medium', 'liquidity': 'Very Low', 'tax_benefit': True},
            'Fixed_Deposits': {'return': 6.5, 'risk': 'Very Low', 'liquidity': 'Medium', 'tax_benefit': False},
            'Corporate_Bonds': {'return': 7.5, 'risk': 'Low', 'liquidity': 'Medium', 'tax_benefit': False},
            'Real_Estate': {'return': 9.0, 'risk': 'Medium', 'liquidity': 'Very Low', 'tax_benefit': True},
            'REITs': {'return': 8.5, 'risk': 'Medium', 'liquidity': 'High', 'tax_benefit': False},
            'Gold_ETF': {'return': 8.0, 'risk': 'Medium', 'liquidity': 'High', 'tax_benefit': False},
            'Digital_Gold': {'return': 8.0, 'risk': 'Medium', 'liquidity': 'High', 'tax_benefit': False},
            'Liquid_Funds': {'return': 4.0, 'risk': 'Very Low', 'liquidity': 'Very High', 'tax_benefit': False}
        }
    
    def get_age_group(self, age: int) -> str:
        """Determine age group based on age"""
        for group_name, group_data in self.config.AGE_GROUP_PREFERENCES.items():
            age_min, age_max = group_data['age_range']
            if age_min <= age <= age_max:
                return group_name
        return 'young_adult'  # default
    
    def calculate_risk_score(self, user_data: dict) -> float:
        """
        Calculate risk score (0-100) based on user profile
        """
        risk_score = user_data.get('risk_appetite', 50)  # Base risk from questionnaire
        age = user_data.get('age', 30)
        income = user_data.get('income', 50000)
        dependents = user_data.get('dependents', 0)
        job_stability = user_data.get('job_stability', 'stable')  # stable/unstable
        
        # Age adjustment: younger people can take more risk
        if age < 30:
            risk_score += 15
        elif age < 40:
            risk_score += 5
        elif age > 55:
            risk_score -= 15
        
        # Income stability adjustment
        if job_stability == 'stable':
            risk_score += 10
        elif job_stability == 'unstable':
            risk_score -= 20
        
        # Dependents adjustment: more dependents = less risk
        risk_score -= dependents * 5
        
        # Income level adjustment
        if income > 150000:
            risk_score += 10
        elif income < 30000:
            risk_score -= 10
        
        return max(0, min(100, risk_score))
    
    def calculate_time_horizon_score(self, goals: List[dict]) -> dict:
        """
        Calculate time horizon scores for different goals
        """
        horizon_scores = {}
        for goal in goals:
            years = goal.get('years', 10)
            if years <= 3:
                horizon_scores[goal['name']] = 'short'
            elif years <= 7:
                horizon_scores[goal['name']] = 'medium'
            else:
                horizon_scores[goal['name']] = 'long'
        return horizon_scores
    
    def calculate_liquidity_needs(self, user_data: dict, age_group: str) -> float:
        """
        Calculate required liquidity as percentage of portfolio
        """
        monthly_expenses = user_data.get('monthly_expenses', 50000)
        age_group_data = self.config.AGE_GROUP_PREFERENCES[age_group]
        emergency_months = age_group_data['liquidity_needs']
        
        # Calculate emergency fund requirement
        emergency_fund = monthly_expenses * emergency_months
        total_investable = user_data.get('total_investment_amount', 1000000)
        
        liquidity_percentage = min(0.3, emergency_fund / total_investable)  # Max 30%
        return liquidity_percentage
    
    def apply_age_group_preferences(self, base_allocation: dict, age_group: str) -> dict:
        """
        Adjust allocation based on age group preferences
        """
        age_preferences = self.config.AGE_GROUP_PREFERENCES[age_group]['preferred_investments']
        
        # Blend base allocation with age preferences (70% age preference, 30% risk-based)
        adjusted_allocation = {}
        for asset_class in base_allocation:
            base_weight = base_allocation.get(asset_class, 0)
            age_weight = age_preferences.get(asset_class, 0)
            adjusted_allocation[asset_class] = (age_weight * 0.7) + (base_weight * 0.3)
        
        # Add age-specific asset classes that might not be in base allocation
        for asset_class, weight in age_preferences.items():
            if asset_class not in adjusted_allocation:
                adjusted_allocation[asset_class] = weight * 0.3
        
        # Normalize to ensure sum = 1
        total_weight = sum(adjusted_allocation.values())
        if total_weight > 0:
            adjusted_allocation = {k: v/total_weight for k, v in adjusted_allocation.items()}
        
        return adjusted_allocation
    
    def get_specific_investment_recommendations(self, allocation: dict, age_group: str) -> dict:
        """
        Convert asset class allocation to specific investment vehicle recommendations
        """
        age_group_data = self.config.AGE_GROUP_PREFERENCES[age_group]
        investment_vehicles = age_group_data.get('investment_vehicles', {})
        
        specific_recommendations = {}
        for asset_class, percentage in allocation.items():
            if asset_class in investment_vehicles and percentage > 0:
                specific_recommendations[asset_class] = {
                    'percentage': round(percentage * 100, 2),
                    'recommended_vehicles': investment_vehicles[asset_class],
                    'suggested_split': self._suggest_vehicle_split(
                        investment_vehicles[asset_class], percentage
                    )
                }
        
        return specific_recommendations
    
    def _suggest_vehicle_split(self, vehicles: List[str], total_percentage: float) -> dict:
        """
        Suggest how to split allocation within an asset class
        """
        if not vehicles:
            return {}
        
        # Simple equal split for now - can be made more sophisticated
        split_percentage = total_percentage / len(vehicles)
        return {vehicle: round(split_percentage * 100, 2) for vehicle in vehicles}
    
    def calculate_allocation(self, user_data: dict) -> dict:
        """
        Main function to calculate personalized investment allocation
        
        Args:
            user_data (dict): User profile data containing:
                - age: int
                - risk_appetite: int (0-100)
                - income: float
                - monthly_expenses: float
                - dependents: int
                - goals: list of dict with 'name', 'amount', 'years'
                - preferences: dict (optional overrides)
                - job_stability: str ('stable'/'unstable')
                - total_investment_amount: float
        
        Returns:
            dict: Complete allocation recommendation
        """
        
        # Step 1: Determine age group
        age = user_data.get('age', 30)
        age_group = self.get_age_group(age)
        
        # Step 2: Calculate various scores
        risk_score = self.calculate_risk_score(user_data)
        goals = user_data.get('goals', [])
        time_horizons = self.calculate_time_horizon_score(goals)
        liquidity_percentage = self.calculate_liquidity_needs(user_data, age_group)
        
        # Step 3: Get base allocation based on risk level
        if risk_score <= 35:
            base_allocation = self.config.RISK_WEIGHTS['conservative'].copy()
        elif risk_score <= 70:
            base_allocation = self.config.RISK_WEIGHTS['moderate'].copy()
        else:
            base_allocation = self.config.RISK_WEIGHTS['aggressive'].copy()
        
        # Step 4: Apply age group preferences
        age_adjusted_allocation = self.apply_age_group_preferences(base_allocation, age_group)
        
        # Step 5: Adjust for liquidity needs
        if liquidity_percentage > 0:
            # Increase cash/liquid allocation
            cash_increase = liquidity_percentage - age_adjusted_allocation.get('cash', 0)
            if cash_increase > 0:
                age_adjusted_allocation['cash'] = liquidity_percentage
                # Proportionally reduce other allocations
                remaining_allocation = 1 - liquidity_percentage
                other_total = sum(v for k, v in age_adjusted_allocation.items() if k != 'cash')
                if other_total > 0:
                    for asset_class in age_adjusted_allocation:
                        if asset_class != 'cash':
                            age_adjusted_allocation[asset_class] = (
                                age_adjusted_allocation[asset_class] / other_total * remaining_allocation
                            )
        
        # Step 6: Apply user preferences (if any)
        user_preferences = user_data.get('preferences', {})
        if user_preferences:
            for asset_class, preferred_percentage in user_preferences.items():
                if asset_class in age_adjusted_allocation:
                    # Blend user preference with calculated allocation (50-50)
                    age_adjusted_allocation[asset_class] = (
                        age_adjusted_allocation[asset_class] * 0.5 + 
                        preferred_percentage * 0.5
                    )
        
        # Step 7: Get specific investment recommendations
        specific_recommendations = self.get_specific_investment_recommendations(
            age_adjusted_allocation, age_group
        )
        
        # Step 8: Calculate expected returns and risk metrics
        expected_return = self._calculate_portfolio_return(age_adjusted_allocation)
        risk_level = self._assess_portfolio_risk(age_adjusted_allocation)
        
        return {
            'allocation': {k: round(v, 4) for k, v in age_adjusted_allocation.items() if v > 0.01},
            'allocation_percentages': {k: round(v * 100, 2) for k, v in age_adjusted_allocation.items() if v > 0.01},
            'age_group': age_group,
            'age_group_description': self.config.AGE_GROUP_PREFERENCES[age_group]['description'],
            'risk_score': round(risk_score, 2),
            'liquidity_requirement': round(liquidity_percentage * 100, 2),
            'specific_recommendations': specific_recommendations,
            'expected_annual_return': round(expected_return, 2),
            'portfolio_risk_level': risk_level,
            'tax_focus': self.config.AGE_GROUP_PREFERENCES[age_group]['tax_focus'],
            'rebalancing_frequency': self._suggest_rebalancing_frequency(age_group),
            'goal_alignment': self._assess_goal_alignment(goals, age_adjusted_allocation),
            'next_steps': self._generate_action_items(age_group, specific_recommendations)
        }
    
    def _calculate_portfolio_return(self, allocation: dict) -> float:
        """Calculate expected portfolio return based on allocation"""
        expected_returns = {
            'equity': 13.0,
            'debt': 7.0,
            'real_estate': 9.0,
            'alternatives': 12.0,
            'gold': 8.0,
            'cash': 4.0,
            'mutual_funds': 12.0,
            'child_plans': 8.5,
            'retirement_plans': 9.5,
            'healthcare_funds': 6.0,
            'retirement_income': 7.5
        }
        
        portfolio_return = 0
        for asset_class, percentage in allocation.items():
            expected_return = expected_returns.get(asset_class, 8.0)
            portfolio_return += percentage * expected_return
        
        return portfolio_return
    
    def _assess_portfolio_risk(self, allocation: dict) -> str:
        """Assess overall portfolio risk level"""
        high_risk_assets = ['equity', 'alternatives', 'mutual_funds']
        high_risk_percentage = sum(allocation.get(asset, 0) for asset in high_risk_assets)
        
        if high_risk_percentage > 0.6:
            return 'High'
        elif high_risk_percentage > 0.3:
            return 'Medium'
        else:
            return 'Low'
    
    def _suggest_rebalancing_frequency(self, age_group: str) -> str:
        """Suggest how often to rebalance portfolio"""
        frequency_map = {
            'young_adult': 'Quarterly',
            'early_career': 'Half-yearly',
            'family_building': 'Half-yearly',
            'wealth_accumulation': 'Annually',
            'pre_retirement': 'Annually',
            'retired': 'Half-yearly'
        }
        return frequency_map.get(age_group, 'Annually')
    
    def _assess_goal_alignment(self, goals: List[dict], allocation: dict) -> dict:
        """Assess how well the allocation aligns with user goals"""
        alignment_score = {}
        for goal in goals:
            goal_name = goal['name'].lower()
            years = goal.get('years', 10)
            
            if 'education' in goal_name or 'child' in goal_name:
                if allocation.get('child_plans', 0) > 0.1 or allocation.get('equity', 0) > 0.3:
                    alignment_score[goal['name']] = 'Good'
                else:
                    alignment_score[goal['name']] = 'Needs Improvement'
            elif 'retirement' in goal_name:
                if allocation.get('retirement_plans', 0) > 0.15 or years > 15:
                    alignment_score[goal['name']] = 'Good'
                else:
                    alignment_score[goal['name']] = 'Needs Improvement'
            elif 'home' in goal_name or 'house' in goal_name:
                if allocation.get('real_estate', 0) > 0.1 or allocation.get('debt', 0) > 0.2:
                    alignment_score[goal['name']] = 'Good'
                else:
                    alignment_score[goal['name']] = 'Needs Improvement'
            else:
                alignment_score[goal['name']] = 'Good'  # Default
        
        return alignment_score
    
    def _generate_action_items(self, age_group: str, recommendations: dict) -> List[str]:
        """Generate specific action items for the user"""
        action_items = []
        
        # Common action items
        action_items.append("Set up automatic SIP for equity investments")
        action_items.append("Review and optimize existing investments")
        
        # Age-specific action items
        if age_group in ['young_adult', 'early_career']:
            action_items.extend([
                "Start ELSS investment for tax saving",
                "Consider increasing risk exposure gradually",
                "Build emergency fund equal to 6 months expenses"
            ])
        elif age_group in ['family_building', 'wealth_accumulation']:
            action_items.extend([
                "Maximize PPF contribution for tax benefits",
                "Consider child education funds if applicable",
                "Review life and health insurance coverage"
            ])
        elif age_group in ['pre_retirement', 'retired']:
            action_items.extend([
                "Focus on income-generating investments",
                "Consider senior citizen saving schemes",
                "Plan for healthcare expenses"
            ])
        
        # Add specific recommendations based on allocation
        if recommendations:
            action_items.append("Diversify within recommended asset classes")
            action_items.append("Monitor and rebalance portfolio regularly")
        
        return action_items

# Integration with Flask/FastAPI
def create_allocation_endpoint():
    """
    Example Flask/FastAPI endpoint integration
    """
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    allocation_engine = InvestmentAllocationEngine()
    
    @app.route('/api/get-allocation', methods=['POST'])
    def get_allocation():
        try:
            user_data = request.json
            
            # Validate required fields
            required_fields = ['age', 'income', 'monthly_expenses']
            for field in required_fields:
                if field not in user_data:
                    return jsonify({'error': f'Missing required field: {field}'}), 400
            
            # Calculate allocation
            allocation_result = allocation_engine.calculate_allocation(user_data)
            
            return jsonify({
                'success': True,
                'data': allocation_result,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    return app

# Example usage
def example_usage():
    """Example of how to use the allocation engine"""
    
    # Sample user data
    user_data = {
        'age': 32,
        'risk_appetite': 65,
        'income': 80000,
        'monthly_expenses': 45000,
        'dependents': 2,
        'job_stability': 'stable',
        'total_investment_amount': 500000,
        'goals': [
            {'name': 'Child Education', 'amount': 5000000, 'years': 15},
            {'name': 'Retirement', 'amount': 10000000, 'years': 25},
            {'name': 'Home Purchase', 'amount': 3000000, 'years': 8}
        ],
        'preferences': {
            'real_estate': 0.15,  # User prefers 15% in real estate
            'gold': 0.05         # User wants some gold exposure
        }
    }
    
    # Initialize engine and calculate allocation
    engine = InvestmentAllocationEngine()
    result = engine.calculate_allocation(user_data)
    
    # Print results
    print("=== PERSONALIZED INVESTMENT ALLOCATION ===")
    print(f"Age Group: {result['age_group']}")
    print(f"Risk Score: {result['risk_score']}")
    print(f"Expected Return: {result['expected_annual_return']}%")
    print(f"Risk Level: {result['portfolio_risk_level']}")
    
    print("\n=== ALLOCATION BREAKDOWN ===")
    for asset_class, percentage in result['allocation_percentages'].items():
        print(f"{asset_class.replace('_', ' ').title()}: {percentage}%")
    
    print(f"\n=== TAX STRATEGY ===")
    print(f"Focus: {result['tax_focus']}")
    
    print(f"\n=== NEXT STEPS ===")
    for step in result['next_steps']:
        print(f"• {step}")
    
    return result

if __name__ == "__main__":
    # Run example
    example_result = example_usage()
    
    # Pretty print the full result
    print("\n" + "="*80)
    print("COMPLETE ALLOCATION RESULT")
    print("="*80)
    print(json.dumps(example_result, indent=2))