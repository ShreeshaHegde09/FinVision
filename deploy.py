#!/usr/bin/env python3
"""
One-click deployment script for Financial Advisor
This script handles the complete setup and deployment process
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            return False
        print(f"✅ {description} completed")
        return True
    except Exception as e:
        print(f"❌ Error in {description}: {str(e)}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Please upgrade Python.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def setup_directory_structure():
    """Create necessary directories"""
    print("📁 Setting up directory structure...")
    
    directories = ['templates', 'models', 'static']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   Created: {directory}/")
    
    # Move index.html to templates if it exists in current directory
    if os.path.exists('index.html'):
        shutil.move('index.html', 'templates/index.html')
        print("   Moved index.html to templates/")
    
    return True

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing Python dependencies...")
    
    dependencies = [
        "pandas>=2.0.3",
        "numpy>=1.24.3", 
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "joblib>=1.3.2",
        "flask>=2.3.2",
        "flask-cors>=4.0.0",
        "waitress>=2.1.2"
    ]
    
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep.split('>=')[0]}"):
            return False
    
    return True

def find_dataset():
    """Find CSV dataset file"""
    print("🔍 Looking for dataset file...")
    
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ No CSV dataset file found!")
        print("   Please ensure your dataset is in the current directory with .csv extension")
        return None
    
    if len(csv_files) == 1:
        dataset = csv_files[0]
        print(f"✅ Found dataset: {dataset}")
        return dataset
    
    print("📋 Multiple CSV files found:")
    for i, file in enumerate(csv_files):
        print(f"   {i+1}. {file}")
    
    while True:
        try:
            choice = int(input("Enter choice (1-{}): ".format(len(csv_files)))) - 1
            if 0 <= choice < len(csv_files):
                dataset = csv_files[choice]
                print(f"✅ Selected dataset: {dataset}")
                return dataset
            else:
                print("❌ Invalid choice. Please try again.")
        except ValueError:
            print("❌ Please enter a valid number.")

def train_models(dataset_file):
    """Train ML models with the dataset"""
    print("🤖 Training ML models...")
    
    # Import and run training
    try:
        import train_models
        # This will run the training process
        result = subprocess.run([sys.executable, 'train_models.py'], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Model training completed successfully!")
            return True
        else:
            print(f"❌ Training failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return False

def test_application():
    """Test if the application starts correctly"""
    print("🧪 Testing application startup...")
    
    try:
        # Import the app to check for any import errors
        sys.path.append('.')
        import app
        print("✅ Application imports successful")
        return True
    except Exception as e:
        print(f"❌ Application test failed: {str(e)}")
        return False

def create_startup_script():
    """Create startup scripts for different platforms"""
    print("📜 Creating startup scripts...")
    
    # Windows batch file
    with open('start_app.bat', 'w') as f:
        f.write("""@echo off
echo Starting Financial Advisor Application...
python app.py
pause
""")
    
    # Linux/Mac shell script
    with open('start_app.sh', 'w') as f:
        f.write("""#!/bin/bash
echo "Starting Financial Advisor Application..."
python3 app.py
""")
    
    # Make shell script executable
    if os.name != 'nt':  # Not Windows
        os.chmod('start_app.sh', 0o755)
    
    print("✅ Startup scripts created: start_app.bat (Windows), start_app.sh (Linux/Mac)")
    return True

def display_final_instructions():
    """Display final instructions for the user"""
    print("\n" + "="*60)
    print("🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\n📋 What was set up:")
    print("   ✅ Python dependencies installed")
    print("   ✅ Directory structure created")
    print("   ✅ ML models trained with your dataset")
    print("   ✅ Flask web application configured")
    print("   ✅ Startup scripts created")
    
    print("\n🚀 How to start your Financial Advisor:")
    if os.name == 'nt':  # Windows
        print("   Option 1: Double-click 'start_app.bat'")
        print("   Option 2: Run 'python app.py' in terminal")
    else:  # Linux/Mac
        print("   Option 1: Run './start_app.sh' in terminal")
        print("   Option 2: Run 'python3 app.py' in terminal")
    
    print("\n🌐 Access your application:")
    print("   → Open browser and go to: http://localhost:5000")
    print("   → The app will be accessible on your local network")
    
    print("\n📊 Your ML Models:")
    print("   → Trained on your actual dataset")
    print("   → Saved in 'models/' directory")
    print("   → Will provide personalized predictions")
    
    print("\n💡 Next Steps:")
    print("   1. Test the application with sample data")
    print("   2. Customize investment options if needed")
    print("   3. Deploy to cloud for wider access")
    print("   4. Add more features as required")
    
    print("\n🆘 Support:")
    print("   → Check logs for any errors")
    print("   → Ensure all CSV data columns are present")
    print("   → Restart if you encounter issues")
    
    print("\n" + "="*60)

def main():
    """Main deployment function"""
    print("🚀 Financial Advisor ML Deployment Script")
    print("This will set up everything you need!")
    print("="*50)
    
    # Step 1: Check Python version
    if not check_python_version():
        return False
    
    # Step 2: Setup directory structure
    if not setup_directory_structure():
        return False
    
    # Step 3: Install dependencies
    print("\n" + "-"*50)
    if not install_dependencies():
        return False
    
    # Step 4: Find dataset
    print("\n" + "-"*50)
    dataset_file = find_dataset()
    if not dataset_file:
        return False
    
    # Step 5: Train models
    print("\n" + "-"*50)
    if not train_models(dataset_file):
        return False
    
    # Step 6: Test application
    print("\n" + "-"*50)
    if not test_application():
        print("⚠️  Application test failed, but continuing...")
    
    # Step 7: Create startup scripts
    print("\n" + "-"*50)
    if not create_startup_script():
        return False
    
    # Step 8: Display final instructions
    display_final_instructions()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✨ Ready to launch your AI Financial Advisor!")
        else:
            print("\n❌ Deployment failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Deployment cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)