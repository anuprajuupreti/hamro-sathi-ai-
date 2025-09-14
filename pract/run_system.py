#!/usr/bin/env python3
"""
System Runner - Starts all components of the Global AI System
"""

import subprocess
import sys
import os
import time
import threading
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import flask
        import transformers
        import torch
        import sentence_transformers
        import spacy
        import nltk
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run 'python setup.py' first")
        return False

def start_backend():
    """Start the AI backend server"""
    print("🚀 Starting Enhanced Web Search AI Backend...")
    try:
        # Start the enhanced web search backend
        backend_process = subprocess.Popen([
            sys.executable, "simple_web_search_backend.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Check if the process is still running
        if backend_process.poll() is None:
            print("✅ AI Backend Server started successfully on http://localhost:5000")
            return backend_process
        else:
            stdout, stderr = backend_process.communicate()
            print(f"❌ Backend failed to start:")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def open_frontend():
    """Open the frontend in the default web browser"""
    print("🌐 Opening Frontend...")
    
    # Get the path to index.html
    frontend_path = Path("index.html").absolute()
    
    if frontend_path.exists():
        # Open in default browser
        webbrowser.open(f"file://{frontend_path}")
        print("✅ Frontend opened in browser")
        return True
    else:
        print("❌ index.html not found")
        return False

def monitor_system():
    """Monitor the system and provide status updates"""
    print("\n" + "="*50)
    print("🎉 Global AI System is now running!")
    print("="*50)
    print("📊 System Status:")
    print("   • Enhanced AI Backend: http://localhost:5000")
    print("   • Frontend: Opened in browser")
    print("   • Web Search: Google Custom Search + SerpAPI + DuckDuckGo")
    print("   • Query Classification: Intelligent routing enabled")
    print("\n💡 Usage Tips:")
    print("   • Ask current events questions for real-time web search")
    print("   • Try factual queries like 'What is the latest news about...'")
    print("   • Configure Google Custom Search ID in Settings for best results")
    print("   • The system automatically determines when to search the web")
    print("\n🛑 To stop the system:")
    print("   • Close this terminal window")
    print("   • Or press Ctrl+C")
    print("="*50)

def main():
    """Main function to start the entire system"""
    print("🌍 Enhanced AI Assistant with Real-time Web Search")
    print("Created by Anup Raj Uprety")
    print("="*50)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        return False
    
    # Wait a moment for backend to fully initialize
    time.sleep(2)
    
    # Open frontend
    if not open_frontend():
        backend_process.terminate()
        return False
    
    # Monitor system
    try:
        monitor_system()
        
        # Keep the system running
        while True:
            time.sleep(1)
            
            # Check if backend is still running
            if backend_process.poll() is not None:
                print("\n❌ Backend process stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down system...")
        backend_process.terminate()
        print("✅ System stopped successfully")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
