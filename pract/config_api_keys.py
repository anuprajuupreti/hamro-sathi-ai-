#!/usr/bin/env python3
"""
API Key Configuration Script
Automatically configures the system with your API keys
"""

import os
import json
from pathlib import Path

def configure_api_keys():
    """Configure API keys in the system"""
    
    # Your API keys
    api_keys = {
        'OPENAI_API_KEY': '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',
        'GOOGLE_API_KEY': 'AIzaSyCz6IW48uLukc9E0fd0MySk8PMeWT59foI',
        'GOOGLE_API_KEY_BACKUP': 'AIzaSyAaDiMSw8a_rCMR4oAc8EA1mGZgHaBFUIg',
        'GEMINI_API_KEY': 'AIzaSyD6ky2B-vsEwcH_lhn-LgCS_cOVc9Mp-kI',
        'ANUP_API_KEY': 'fXEdtWHFxJnVZb4FOWLoqJC4skDhxh0VRk9XH9Hq'
    }
    
    # Set environment variables
    for key, value in api_keys.items():
        os.environ[key] = value
        print(f"✅ Set {key}")
    
    # Create settings file for frontend
    frontend_settings = {
        'openaiApiKey': api_keys['OPENAI_API_KEY'],
        'googleApiKey': api_keys['GOOGLE_API_KEY'],
        'geminiApiKey': api_keys['GEMINI_API_KEY'],
        'cohereApiKey': api_keys['ANUP_API_KEY'],
        'useStreaming': True,
        'maxTokens': 1000,
        'temperature': 0.7
    }
    
    # Save to local storage format for frontend
    with open('frontend_settings.json', 'w') as f:
        json.dump(frontend_settings, f, indent=2)
    
    print("✅ Frontend settings saved to frontend_settings.json")
    print("✅ All API keys configured successfully!")
    
    return True

if __name__ == "__main__":
    configure_api_keys()
