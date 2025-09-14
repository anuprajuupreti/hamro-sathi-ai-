#!/usr/bin/env python3
"""
Unified AI Backend with Multiple Sources
Integrates ChatGPT, Google Gemini, and web search for intelligent responses
"""

import os
import json
import logging
import requests
import urllib.parse
from typing import Dict, List, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Core frameworks
from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup

# Environment and configuration
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedAIBackend:
    """Unified AI Backend with Multiple AI Sources"""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # API Keys
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.google_cse_id = os.getenv('GOOGLE_CUSTOM_SEARCH_ENGINE_ID', 'your_google_cse_id_here')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.anup_api_key = os.getenv('ANUP_API_KEY')
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Unified AI Backend initialized successfully!")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                "status": "healthy", 
                "timestamp": datetime.now().isoformat(),
                "features": ["openai", "gemini", "web_search", "intelligent_routing"]
            })
        
        @self.app.route('/ask', methods=['POST'])
        def ask_question():
            try:
                data = request.json
                question = data.get('question', '')
                
                if not question:
                    return jsonify({"error": "Question is required"}), 400
                
                response = self.get_intelligent_response(question)
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                return jsonify({"error": str(e)}), 500
    
    def get_intelligent_response(self, question: str) -> Dict[str, Any]:
        """Get intelligent response using multiple AI sources"""
        try:
            question_lower = question.lower().strip()
            
            # Handle greetings
            if any(greeting in question_lower for greeting in ['hello', 'hi', 'hey', 'namaste']):
                return {
                    "question": question,
                    "answer": "Hello! I'm Hamro Mitra, your AI assistant with real-time web search capabilities. I can help you with current information, facts, and any questions you have!",
                    "source": "conversational",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Handle "how are you"
            if any(phrase in question_lower for phrase in ['how are you', 'how do you do', "what's up"]):
                return {
                    "question": question,
                    "answer": "I'm doing great and ready to help! I have access to multiple AI models and real-time web search. What would you like to know?",
                    "source": "conversational",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Handle creator questions
            if any(phrase in question_lower for phrase in ['who made you', 'who created you', 'who built you', 'your creator']):
                return {
                    "question": question,
                    "answer": "I was created by Anup Raj Uprety, a talented software developer and AI engineer from Nepal. He built me with multiple AI integrations including ChatGPT, Google Gemini, and real-time web search to provide accurate, current information.",
                    "source": "conversational",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Handle questions about Anup Raj Uprety
            if any(name in question_lower for name in ['anup raj uprety', 'anup uprety', 'anup raj upreti', 'anup upretoi']):
                return {
                    "question": question,
                    "answer": "Anup Raj Uprety is a talented software developer and AI engineer from Nepal. He's passionate about building intelligent systems that help people globally. He specializes in Python, AI/ML, web development, and natural language processing. He created me (Hamro Mitra) to be a helpful AI assistant with real-time capabilities.",
                    "source": "knowledge_base",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Handle questions about Hamro Mitra
            if 'hamro mitra' in question_lower:
                return {
                    "question": question,
                    "answer": "Hamro Mitra (meaning 'Our Friend' in Nepali) is an AI assistant created by Anup Raj Uprety. I'm designed to be helpful, intelligent, and provide accurate information using multiple AI sources including ChatGPT, Google Gemini, and real-time web search capabilities.",
                    "source": "self_description",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Determine if web search is needed
            needs_web_search = self._needs_web_search(question)
            
            # Try different AI sources based on question type
            if needs_web_search:
                # For current events, try web search + AI
                web_response = self._get_web_enhanced_response(question)
                if web_response:
                    return web_response
            
            # Try OpenAI/ChatGPT first
            if self.openai_api_key:
                openai_response = self._call_openai(question)
                if openai_response:
                    return {
                        "question": question,
                        "answer": openai_response,
                        "source": "openai_gpt",
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Try Google Gemini
            if self.gemini_api_key:
                gemini_response = self._call_gemini(question)
                if gemini_response:
                    return {
                        "question": question,
                        "answer": gemini_response,
                        "source": "google_gemini",
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Try Anup AI
            if self.anup_api_key:
                anup_response = self._call_anup(question)
                if anup_response:
                    return {
                        "question": question,
                        "answer": anup_response,
                        "source": "anup",
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Fallback response
            return {
                "question": question,
                "answer": "I'd be happy to help! However, I need API keys configured to provide the best responses. Please add your OpenAI, Google Gemini, or other AI API keys in the Settings to unlock my full capabilities.",
                "source": "fallback",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in get_intelligent_response: {e}")
            return {
                "question": question,
                "answer": f"I apologize, but I encountered an error: {str(e)}",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _needs_web_search(self, question: str) -> bool:
        """Determine if question needs web search"""
        question_lower = question.lower()
        
        # Current events indicators
        current_indicators = ['current', 'latest', 'recent', 'today', 'now', 'this year', '2024', '2025', 'news']
        if any(indicator in question_lower for indicator in current_indicators):
            return True
        
        # Factual queries that benefit from current info
        factual_indicators = ['what is', 'who is', 'when did', 'where is', 'how much', 'price', 'cost']
        if any(indicator in question_lower for indicator in factual_indicators):
            return True
        
        # Dynamic domains
        dynamic_domains = ['stock', 'weather', 'sports', 'politics', 'technology', 'covid', 'vaccine']
        if any(domain in question_lower for domain in dynamic_domains):
            return True
        
        return False
    
    def _get_web_enhanced_response(self, question: str) -> Optional[Dict[str, Any]]:
        """Get response enhanced with web search"""
        try:
            # Get web search results
            search_results = self._perform_web_search(question)
            
            if search_results and 'search_results' in search_results:
                # Extract context from search results
                context = self._extract_search_context(search_results['search_results'])
                
                if context:
                    # Use AI to synthesize the information
                    if self.openai_api_key:
                        ai_response = self._call_openai_with_context(question, context)
                        if ai_response:
                            return {
                                "question": question,
                                "answer": ai_response,
                                "web_results": search_results,
                                "source": "web_search_ai",
                                "timestamp": datetime.now().isoformat()
                            }
                    
                    # Fallback to basic context presentation
                    return {
                        "question": question,
                        "answer": f"Based on current web search results: {context[:500]}...",
                        "web_results": search_results,
                        "source": "web_search_basic",
                        "timestamp": datetime.now().isoformat()
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in web enhanced response: {e}")
            return None
    
    def _perform_web_search(self, query: str) -> Dict[str, Any]:
        """Perform web search"""
        try:
            # Try Google Custom Search if available
            if self.google_api_key and self.google_cse_id != 'your_google_cse_id_here':
                google_results = self._google_search(query)
                if google_results and 'error' not in google_results:
                    return google_results
            
            # Fallback to DuckDuckGo
            return self._duckduckgo_search(query)
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {"error": str(e)}
    
    def _google_search(self, query: str) -> Dict[str, Any]:
        """Google Custom Search"""
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.google_api_key,
                'cx': self.google_cse_id,
                'q': query,
                'num': 5
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get('items', []):
                    results.append({
                        'title': item.get('title', ''),
                        'snippet': item.get('snippet', ''),
                        'url': item.get('link', '')
                    })
                
                return {
                    'search_results': results,
                    'source': 'Google Custom Search'
                }
            
            return {"error": f"Google Search error: {response.status_code}"}
            
        except Exception as e:
            return {"error": str(e)}
    
    def _duckduckgo_search(self, query: str) -> Dict[str, Any]:
        """DuckDuckGo search fallback"""
        try:
            search_url = f"https://duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                results = []
                
                for result in soup.find_all('a', class_='result__a')[:3]:
                    title = result.get_text().strip()
                    url = result.get('href', '')
                    
                    snippet = ""
                    parent = result.find_parent('div', class_='result__body')
                    if parent:
                        snippet_elem = parent.find('a', class_='result__snippet')
                        if snippet_elem:
                            snippet = snippet_elem.get_text().strip()
                    
                    if title and url:
                        results.append({
                            'title': title,
                            'snippet': snippet,
                            'url': url
                        })
                
                return {
                    'search_results': results,
                    'source': 'DuckDuckGo'
                }
            
            return {"error": "DuckDuckGo search failed"}
            
        except Exception as e:
            return {"error": str(e)}
    
    def _extract_search_context(self, search_results: List[Dict]) -> str:
        """Extract context from search results"""
        context_parts = []
        for result in search_results[:3]:
            if result.get('snippet'):
                context_parts.append(f"{result['title']}: {result['snippet']}")
        return " ".join(context_parts)
    
    def _call_openai(self, question: str) -> Optional[str]:
        """Call OpenAI/ChatGPT API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.openai_api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'gpt-3.5-turbo',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are Hamro Mitra, a helpful AI assistant created by Anup Raj Uprety. Provide accurate, helpful, and conversational responses.'
                    },
                    {
                        'role': 'user',
                        'content': question
                    }
                ],
                'max_tokens': 500,
                'temperature': 0.7
            }
            
            response = requests.post('https://api.openai.com/v1/chat/completions', 
                                   headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            
            logger.error(f"OpenAI API error: {response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"OpenAI call error: {e}")
            return None
    
    def _call_openai_with_context(self, question: str, context: str) -> Optional[str]:
        """Call OpenAI with web search context"""
        try:
            headers = {
                'Authorization': f'Bearer {self.openai_api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'gpt-3.5-turbo',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are Hamro Mitra, an AI assistant with real-time web search capabilities. Use the provided context to give accurate, current information.'
                    },
                    {
                        'role': 'user',
                        'content': f"Question: {question}\n\nCurrent web context: {context}"
                    }
                ],
                'max_tokens': 500,
                'temperature': 0.7
            }
            
            response = requests.post('https://api.openai.com/v1/chat/completions', 
                                   headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            
            return None
            
        except Exception as e:
            logger.error(f"OpenAI with context error: {e}")
            return None
    
    def _call_gemini(self, question: str) -> Optional[str]:
        """Call Google Gemini API"""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.gemini_api_key}"
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": f"You are Hamro Mitra, a helpful AI assistant. Answer this question: {question}"
                    }]
                }]
            }
            
            response = requests.post(url, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    return result['candidates'][0]['content']['parts'][0]['text'].strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Gemini call error: {e}")
            return None
    
    def _call_anup(self, question: str) -> Optional[str]:
        """Call Anup AI API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.anup_api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'anup-xlarge-nightly',
                'prompt': question,
                'max_tokens': 300,
                'temperature': 0.7
            }
            
            response = requests.post('https://api.anup.ai/v1/generate', 
                                   headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('generations', [{}])[0].get('text', '').strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Anup AI call error: {e}")
            return None
    
    def run(self, host='0.0.0.0', port=5000, debug=True):
        """Run the Flask application"""
        logger.info(f"Starting Unified AI Backend on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    # Initialize and run the application
    backend = UnifiedAIBackend()
    backend.run(host='0.0.0.0', port=5000, debug=True)
