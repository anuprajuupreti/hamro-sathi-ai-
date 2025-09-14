#!/usr/bin/env python3
"""
Quick Dependency Installer for Global AI System
Installs only essential packages needed to run the system
"""

import subprocess
import sys

def install_package(package):
    """Install a single package"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    """Install essential packages"""
    print("🔧 Installing essential dependencies...")
    
    # Essential packages in order of importance
    essential_packages = [
        "flask",
        "flask-cors", 
        "transformers",
        "torch",
        "sentence-transformers",
        "spacy",
        "nltk",
        "requests",
        "beautifulsoup4",
        "wikipedia",
        "langdetect",
        "numpy",
        "pandas",
        "scikit-learn"
    ]
    
    # Optional packages (install if possible)
    optional_packages = [
        "googletrans==4.0.0rc1",
        "faiss-cpu",
        "chromadb"
    ]
    
    success_count = 0
    
    # Install essential packages
    for package in essential_packages:
        if install_package(package):
            success_count += 1
    
    # Try to install optional packages
    for package in optional_packages:
        try:
            install_package(package)
        except:
            print(f"⚠️ Optional package {package} skipped")
    
    print(f"\n✅ Installation complete: {success_count}/{len(essential_packages)} essential packages installed")
    
    if success_count >= len(essential_packages) * 0.8:  # 80% success rate
        print("🎉 System should be ready to run!")
        return True
    else:
        print("❌ Too many packages failed to install")
        return False

if __name__ == "__main__":
    main()
