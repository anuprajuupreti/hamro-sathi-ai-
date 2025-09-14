#!/usr/bin/env python3
"""
Setup script for the Global AI Question Answering System
Installs dependencies and downloads required models
"""

import subprocess
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_requirements():
    """Install Python requirements"""
    logger.info("Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("✅ Python requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install requirements: {e}")
        return False
    return True

def download_spacy_models():
    """Download spaCy language models"""
    models = [
        "en_core_web_sm",  # English
        "de_core_news_sm", # German
        "es_core_news_sm", # Spanish
        "fr_core_news_sm", # French
        "it_core_news_sm", # Italian
        "pt_core_news_sm", # Portuguese
        "nl_core_news_sm", # Dutch
        "zh_core_web_sm",  # Chinese
        "ja_core_news_sm"  # Japanese
    ]
    
    logger.info("Downloading spaCy language models...")
    for model in models:
        try:
            logger.info(f"Downloading {model}...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
            logger.info(f"✅ {model} downloaded successfully!")
        except subprocess.CalledProcessError:
            logger.warning(f"⚠️ Failed to download {model} - continuing with other models")

def setup_directories():
    """Create necessary directories"""
    directories = [
        "chroma_db",
        "models",
        "logs",
        "data"
    ]
    
    logger.info("Creating directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")

def test_imports():
    """Test if all required packages can be imported"""
    logger.info("Testing imports...")
    
    test_packages = [
        "flask",
        "transformers",
        "torch",
        "sentence_transformers",
        "spacy",
        "nltk",
        "numpy",
        "pandas",
        "sklearn",
        "requests",
        "beautifulsoup4",
        "wikipedia",
        "langdetect",
        "googletrans",
        "faiss",
        "chromadb"
    ]
    
    failed_imports = []
    
    for package in test_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} imported successfully")
        except ImportError:
            failed_imports.append(package)
            logger.error(f"❌ Failed to import {package}")
    
    if failed_imports:
        logger.error(f"Failed to import: {', '.join(failed_imports)}")
        return False
    
    logger.info("✅ All packages imported successfully!")
    return True

def download_nltk_data():
    """Download required NLTK data"""
    logger.info("Downloading NLTK data...")
    
    import nltk
    
    nltk_data = [
        'punkt',
        'stopwords',
        'wordnet',
        'vader_lexicon',
        'averaged_perceptron_tagger',
        'omw-1.4',
        'brown',
        'names',
        'words',
        'maxent_ne_chunker'
    ]
    
    for data in nltk_data:
        try:
            nltk.download(data, quiet=True)
            logger.info(f"✅ Downloaded NLTK data: {data}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to download {data}: {e}")

def main():
    """Main setup function"""
    logger.info("🚀 Starting Global AI System Setup...")
    
    # Step 1: Install requirements
    if not install_requirements():
        logger.error("❌ Setup failed at requirements installation")
        return False
    
    # Step 2: Test imports
    if not test_imports():
        logger.error("❌ Setup failed at import testing")
        return False
    
    # Step 3: Setup directories
    setup_directories()
    
    # Step 4: Download NLTK data
    download_nltk_data()
    
    # Step 5: Download spaCy models
    download_spacy_models()
    
    logger.info("🎉 Setup completed successfully!")
    logger.info("📝 Next steps:")
    logger.info("   1. Run 'python ai_backend.py' to start the AI backend")
    logger.info("   2. Open 'index.html' in your web browser")
    logger.info("   3. Start asking questions in any language!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
