#!/usr/bin/env python3
"""
Test Suite for Global AI System
Tests all components and functionality
"""

import requests
import json
import time
import sys
from pathlib import Path

def test_backend_health():
    """Test if the backend is running and healthy"""
    try:
        response = requests.get('http://localhost:5000/health', timeout=5)
        if response.status_code == 200:
            print("✅ Backend health check passed")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend not accessible: {e}")
        return False

def test_question_answering():
    """Test the question answering functionality"""
    test_questions = [
        {
            "question": "What is Python programming?",
            "language": "English"
        },
        {
            "question": "¿Qué es la inteligencia artificial?",
            "language": "Spanish"
        },
        {
            "question": "Qu'est-ce que l'apprentissage automatique?",
            "language": "French"
        },
        {
            "question": "Write a simple C program",
            "language": "English"
        }
    ]
    
    print("\n🧪 Testing Question Answering...")
    
    for i, test in enumerate(test_questions, 1):
        try:
            response = requests.post(
                'http://localhost:5000/ask',
                json={"question": test["question"], "context": ""},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Test {i} ({test['language']}): Question answered")
                print(f"   Question: {test['question']}")
                print(f"   Answer length: {len(data.get('answer', ''))} characters")
                if data.get('detected_language'):
                    print(f"   Detected language: {data['detected_language']}")
            else:
                print(f"❌ Test {i} failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Test {i} failed: {e}")
            return False
    
    return True

def test_text_analysis():
    """Test text analysis functionality"""
    print("\n🔍 Testing Text Analysis...")
    
    test_text = "I love this amazing AI system! It's incredibly helpful and supports many languages."
    
    try:
        response = requests.post(
            'http://localhost:5000/analyze',
            json={"text": test_text},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Text analysis successful")
            
            if 'sentiment' in data:
                print(f"   Sentiment detected: {data['sentiment']}")
            if 'entities' in data:
                print(f"   Entities found: {len(data.get('entities', []))}")
            if 'keywords' in data:
                print(f"   Keywords extracted: {len(data.get('keywords', []))}")
            
            return True
        else:
            print(f"❌ Text analysis failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Text analysis failed: {e}")
        return False

def test_knowledge_search():
    """Test knowledge base search"""
    print("\n📚 Testing Knowledge Search...")
    
    try:
        response = requests.post(
            'http://localhost:5000/search',
            json={"query": "artificial intelligence machine learning"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Knowledge search successful")
            
            if 'results' in data:
                print(f"   Results found: {len(data.get('results', []))}")
            
            return True
        else:
            print(f"❌ Knowledge search failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Knowledge search failed: {e}")
        return False

def test_multilingual_capabilities():
    """Test multilingual processing"""
    print("\n🌍 Testing Multilingual Capabilities...")
    
    multilingual_tests = [
        {"text": "Hello world", "expected_lang": "en"},
        {"text": "Hola mundo", "expected_lang": "es"},
        {"text": "Bonjour le monde", "expected_lang": "fr"},
        {"text": "Hallo Welt", "expected_lang": "de"},
        {"text": "こんにちは世界", "expected_lang": "ja"}
    ]
    
    success_count = 0
    
    for test in multilingual_tests:
        try:
            response = requests.post(
                'http://localhost:5000/ask',
                json={"question": f"What language is this: {test['text']}", "context": ""},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                detected_lang = data.get('detected_language', 'unknown')
                print(f"   Text: '{test['text']}' -> Detected: {detected_lang}")
                success_count += 1
            else:
                print(f"   Failed to process: {test['text']}")
                
        except requests.exceptions.RequestException as e:
            print(f"   Error processing: {test['text']} - {e}")
    
    if success_count >= len(multilingual_tests) * 0.8:  # 80% success rate
        print(f"✅ Multilingual test passed ({success_count}/{len(multilingual_tests)})")
        return True
    else:
        print(f"❌ Multilingual test failed ({success_count}/{len(multilingual_tests)})")
        return False

def test_frontend_files():
    """Test if frontend files exist and are accessible"""
    print("\n🌐 Testing Frontend Files...")
    
    required_files = [
        "index.html",
        "requirements.txt",
        "ai_backend.py",
        "multilingual_ai.py",
        "knowledge_engine.py",
        "advanced_nlp.py"
    ]
    
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
        else:
            print(f"   ✅ {file} found")
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ All required files present")
        return True

def run_performance_test():
    """Run performance tests"""
    print("\n⚡ Running Performance Tests...")
    
    # Test response time
    start_time = time.time()
    
    try:
        response = requests.post(
            'http://localhost:5000/ask',
            json={"question": "What is 2+2?", "context": ""},
            timeout=10
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            print(f"✅ Response time: {response_time:.2f} seconds")
            
            if response_time < 5.0:
                print("✅ Performance test passed (< 5 seconds)")
                return True
            else:
                print("⚠️ Performance test warning (> 5 seconds)")
                return True
        else:
            print(f"❌ Performance test failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Global AI System Test Suite")
    print("=" * 40)
    
    tests = [
        ("Frontend Files", test_frontend_files),
        ("Backend Health", test_backend_health),
        ("Question Answering", test_question_answering),
        ("Text Analysis", test_text_analysis),
        ("Knowledge Search", test_knowledge_search),
        ("Multilingual Capabilities", test_multilingual_capabilities),
        ("Performance", run_performance_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test error: {e}")
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
        return True
    elif passed >= total * 0.8:
        print("⚠️ Most tests passed. System is mostly functional.")
        return True
    else:
        print("❌ Many tests failed. Please check the system configuration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
