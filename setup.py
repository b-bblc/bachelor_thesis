#!/usr/bin/env python3
"""
Setup script for the EDU dependency analysis project.
This script installs all required dependencies and downloads language models.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description=""):
    """Run a command and handle errors."""
    print(f"📦 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e.stderr}")
        return False


def install_requirements():
    """Install Python requirements."""
    print("🐍 Installing Python requirements...")
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python packages"
    )


def download_spacy_models():
    """Download required spaCy language models."""
    print("🌍 Downloading spaCy language models...")
    
    models = [
        ("de_core_news_md", "German model"),
        ("en_core_web_sm", "English model"), 
        ("ru_core_news_sm", "Russian model")
    ]
    
    success_count = 0
    for model, description in models:
        if run_command(
            f"{sys.executable} -m spacy download {model}",
            f"Downloading {description} ({model})"
        ):
            success_count += 1
    
    return success_count == len(models)


def create_directories():
    """Create necessary project directories."""
    print("📁 Creating project directories...")
    
    directories = [
        "results",
        "results/visualizations", 
        "results/analysis",
        "results/reports",
        "extracted_txts_russian",
        "extracted_txts_english", 
        "parsed_results_russian",
        "parsed_results_english",
        "docs",
        "reports"
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(exist_ok=True, parents=True)
        print(f"  Created: {directory}")
    
    return True


def check_data_availability():
    """Check if required data files are available."""
    print("📊 Checking data availability...")
    
    data_sources = [
        ("PotsdamCommentaryCorpus", "German RST corpus"),
        ("RuRsTreebank_full", "Russian RST corpus"),
        ("en_example.rs3", "English RST example")
    ]
    
    available = []
    for source, description in data_sources:
        if Path(source).exists():
            print(f"  ✅ {description}: Available")
            available.append(source)
        else:
            print(f"  ⚠️  {description}: Not found")
    
    return available


def verify_installation():
    """Verify that the installation was successful."""
    print("🔍 Verifying installation...")
    
    try:
        # Test imports
        import pandas
        import numpy
        import matplotlib
        import seaborn
        import spacy
        import conllu
        import sklearn
        
        print("  ✅ All Python packages imported successfully")
        
        # Test spaCy models
        models_to_test = ["de_core_news_md", "en_core_web_sm", "ru_core_news_sm"]
        working_models = []
        
        for model in models_to_test:
            try:
                nlp = spacy.load(model)
                doc = nlp("Test sentence.")
                working_models.append(model)
                print(f"  ✅ {model}: Working")
            except OSError:
                print(f"  ❌ {model}: Not available")
        
        return len(working_models) >= 2  # At least 2 models should work
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up EDU Dependency Analysis Project")
    print("=" * 50)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please check your environment.")
        return False
    
    # Step 2: Download spaCy models
    if not download_spacy_models():
        print("⚠️  Some spaCy models failed to download. The project may have limited functionality.")
    
    # Step 3: Create directories
    create_directories()
    
    # Step 4: Check data availability
    available_data = check_data_availability()
    
    # Step 5: Verify installation
    if verify_installation():
        print("\n✅ Setup completed successfully!")
    else:
        print("\n⚠️  Setup completed with some issues.")
    
    # Step 6: Print next steps
    print("\n📋 NEXT STEPS:")
    print("1. Run the main processing script:")
    print("   python main.py")
    print("\n2. Open the comprehensive analysis notebook:")
    print("   jupyter notebook notebooks/comprehensive_multilingual_analysis.ipynb")
    print("\n3. Explore EDU boundary detection:")
    print("   jupyter notebook notebooks/edu_boundary_detection.ipynb")
    
    if len(available_data) < 3:
        print("\n⚠️  DATA NOTICE:")
        print("Some data sources are missing. You may need to:")
        print("- Download the Potsdam Commentary Corpus for German data")
        print("- Download the Russian RST Treebank for Russian data") 
        print("- Ensure the English example file is present")
    
    print("\n🎓 Happy researching!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
