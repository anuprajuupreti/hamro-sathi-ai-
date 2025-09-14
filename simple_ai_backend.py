#!/usr/bin/env python3
"""
Simplified AI Backend - Compatible version without problematic dependencies
"""

import os
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from datetime import datetime
import random
import re
from urllib.parse import quote_plus
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleAIAssistant:
    """ChatGPT-like AI Assistant with web search and API integration"""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Load API keys from environment or use placeholders
        self.openai_key = os.getenv('OPENAI_API_KEY', '')
        self.google_key = os.getenv('GOOGLE_API_KEY', '')
        self.gemini_key = os.getenv('GEMINI_API_KEY', '')
        self.anup_key = os.getenv('ANUP_API_KEY', '')
        
        # Setup routes
        self._setup_routes()
        
        logger.info("ChatGPT-like AI Assistant initialized successfully!")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def home():
            """Serve the main frontend interface"""
            try:
                with open('index.html', 'r', encoding='utf-8') as f:
                    return f.read()
            except FileNotFoundError:
                return """
                <!DOCTYPE html>
                <html>
                <head><title>AI Assistant</title></head>
                <body>
                    <h1>AI Assistant</h1>
                    <p>Frontend file not found. Please ensure index.html is in the same directory.</p>
                </body>
                </html>
                """
        
        @self.app.route('/health')
        def health_check():
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.route('/ask', methods=['POST'])
        def ask_question():
            try:
                data = request.get_json()
                question = data.get('question', '').strip()
                
                if not question:
                    return jsonify({"error": "No question provided"}), 400
                
                # Try AI-generated answers first
                response = self.get_ai_generated_answer(question)
                
                # If no AI answer, try direct answers
                if not response:
                    response = self.get_direct_answer(question)
                
                # If still no good response, try OpenAI API
                if not response or "Thank you for your question" in response:
                    api_response = self.get_chatgpt_response(question)
                    if api_response:
                        response = api_response
                    else:
                        # Try multi-source search as last resort
                        search_info = self.search_multiple_sources(question)
                        if search_info:
                            response = self.format_search_response(search_info, question)
                        else:
                            # Ensure we always have a response
                            if not response:
                                response = self.get_comprehensive_answer(question)
                
                return jsonify({
                    "answer": response,
                    "detected_language": "en",
                    "language_name": "English",
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                return jsonify({"error": "Failed to process question"}), 500
        
        @self.app.route('/analyze', methods=['POST'])
        def analyze_text():
            try:
                data = request.get_json()
                text = data.get('text', '').strip()
                
                if not text:
                    return jsonify({"error": "No text provided"}), 400
                
                # Basic analysis
                analysis = {
                    "sentiment": "positive" if any(word in text.lower() for word in ['good', 'great', 'excellent', 'amazing', 'love']) else "neutral",
                    "entities": [],
                    "keywords": text.split()[:5],
                    "summary": text[:100] + "..." if len(text) > 100 else text
                }
                
                return jsonify(analysis)
                
            except Exception as e:
                logger.error(f"Error analyzing text: {e}")
                return jsonify({"error": "Failed to analyze text"}), 500
        
        @self.app.route('/search', methods=['POST'])
        def search_knowledge():
            try:
                data = request.get_json()
                query = data.get('query', '').strip()
                
                if not query:
                    return jsonify({"error": "No query provided"}), 400
                
                # Basic search response
                results = {
                    "results": [
                        {"title": f"Information about {query}", "content": f"This is relevant information about {query}."}
                    ],
                    "total": 1
                }
                
                return jsonify(results)
                
            except Exception as e:
                logger.error(f"Error searching: {e}")
                return jsonify({"error": "Failed to search"}), 500
    
    def get_chatgpt_response(self, question: str) -> str:
        """Get ChatGPT-like response using OpenAI API (only if API key is valid)"""
        # Only try API if we have a valid key and it's not a placeholder
        if not self.openai_key or self.openai_key.startswith('sk-proj-') or len(self.openai_key) < 20:
            return None
            
        try:
            headers = {
                'Authorization': f'Bearer {self.openai_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are Hamro Mitra, an intelligent AI assistant created by Anup Raj Uprety, a software developer from Nepal. You are helpful, knowledgeable, and conversational. Answer questions naturally without citing sources unless specifically asked. When asked about your creator, mention Anup Raj Uprety proudly."
                    },
                    {"role": "user", "content": question}
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                logger.error(f"OpenAI API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            return None
    
    def search_multiple_sources(self, query: str) -> dict:
        """Search multiple sources and return aggregated results"""
        try:
            results = {
                'sources': [],
                'aggregated_info': {},
                'search_query': query,
                'timestamp': datetime.now().isoformat()
            }
            
            # Try multiple search methods
            duckduckgo_results = self.search_duckduckgo(query)
            wikipedia_results = self.search_wikipedia(query)
            serpapi_results = self.search_serpapi(query)
            
            # Aggregate results from different sources
            if duckduckgo_results:
                results['sources'].append(duckduckgo_results)
            if wikipedia_results:
                results['sources'].append(wikipedia_results)
            if serpapi_results:
                results['sources'].append(serpapi_results)
            
            # Create aggregated information
            if results['sources']:
                results['aggregated_info'] = self.aggregate_search_results(results['sources'], query)
            
            return results if results['sources'] else None
            
        except Exception as e:
            logger.error(f"Multi-source search error: {e}")
            return None
    
    def search_duckduckgo(self, query: str) -> dict:
        """Search using DuckDuckGo Instant Answer API"""
        try:
            url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('Abstract') or data.get('Answer'):
                    return {
                        'source': 'DuckDuckGo',
                        'title': data.get('Heading', query),
                        'content': data.get('Abstract') or data.get('Answer'),
                        'url': data.get('AbstractURL', ''),
                        'type': 'search_result'
                    }
            return None
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return None
    
    def search_wikipedia(self, query: str) -> dict:
        """Search Wikipedia for information"""
        try:
            # Wikipedia API search
            search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(query)}"
            response = requests.get(search_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('extract'):
                    return {
                        'source': 'Wikipedia',
                        'title': data.get('title', query),
                        'content': data.get('extract'),
                        'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                        'type': 'encyclopedia'
                    }
            return None
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return None
    
    def search_serpapi(self, query: str) -> dict:
        """Search using SerpAPI (Google Search API) - requires API key"""
        try:
            # Only use if API key is available
            serpapi_key = os.getenv('SERPAPI_KEY', '')
            if not serpapi_key:
                return None
                
            url = "https://serpapi.com/search"
            params = {
                'q': query,
                'api_key': serpapi_key,
                'engine': 'google',
                'num': 3
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                organic_results = data.get('organic_results', [])
                
                if organic_results:
                    # Get the first result
                    first_result = organic_results[0]
                    return {
                        'source': 'Google Search',
                        'title': first_result.get('title', query),
                        'content': first_result.get('snippet', ''),
                        'url': first_result.get('link', ''),
                        'type': 'web_search'
                    }
            return None
        except Exception as e:
            logger.error(f"SerpAPI search error: {e}")
            return None
    
    def aggregate_search_results(self, sources: list, query: str) -> dict:
        """Aggregate information from multiple sources"""
        try:
            # Randomly select primary source for variety
            primary_source = random.choice(sources)
            
            # Detect query type for structured response
            query_lower = query.lower()
            
            if any(phrase in query_lower for phrase in ['prime minister', 'president', 'leader']):
                return self.create_political_leader_aggregate(sources, query)
            elif any(phrase in query_lower for phrase in ['biography', 'who is', 'about']):
                return self.create_biography_aggregate(sources, query)
            else:
                return self.create_general_aggregate(sources, query)
                
        except Exception as e:
            logger.error(f"Aggregation error: {e}")
            return {'content': 'Unable to aggregate search results', 'sources': len(sources)}
    
    def create_political_leader_aggregate(self, sources: list, query: str) -> dict:
        """Create structured political leader information from multiple sources"""
        # Extract information from different sources
        all_content = ' '.join([source.get('content', '') for source in sources])
        
        # Use different sources for different information
        primary_source = random.choice(sources)
        
        return {
            'type': 'political_leader',
            'title': f"Information about {query}",
            'primary_content': primary_source.get('content', ''),
            'primary_source': primary_source.get('source', 'Unknown'),
            'additional_sources': [s.get('source') for s in sources if s != primary_source],
            'source_count': len(sources),
            'aggregated_content': all_content[:500] + '...' if len(all_content) > 500 else all_content
        }
    
    def create_biography_aggregate(self, sources: list, query: str) -> dict:
        """Create structured biography from multiple sources"""
        # Randomly prioritize different sources
        shuffled_sources = sources.copy()
        random.shuffle(shuffled_sources)
        
        primary_source = shuffled_sources[0]
        
        return {
            'type': 'biography',
            'title': primary_source.get('title', query),
            'main_content': primary_source.get('content', ''),
            'source': primary_source.get('source', 'Unknown'),
            'additional_info': [s.get('content', '')[:100] + '...' for s in shuffled_sources[1:3]],
            'all_sources': [s.get('source') for s in sources],
            'source_urls': [s.get('url', '') for s in sources if s.get('url')]
        }
    
    def create_general_aggregate(self, sources: list, query: str) -> dict:
        """Create general aggregated response"""
        # Randomize source priority
        selected_source = random.choice(sources)
        
        return {
            'type': 'general',
            'content': selected_source.get('content', ''),
            'source': selected_source.get('source', 'Unknown'),
            'title': selected_source.get('title', query),
            'alternative_sources': [s.get('source') for s in sources if s != selected_source],
            'total_sources': len(sources)
        }
    
    def search_google(self, query: str) -> dict:
        """Legacy method - now calls multi-source search"""
        return self.search_multiple_sources(query)
    
    def format_search_response(self, search_info, question: str) -> str:
        """Format multi-source search results into attractive, structured responses"""
        try:
            if isinstance(search_info, dict):
                if search_info.get('sources'):
                    return self.format_multi_source_response(search_info)
                elif search_info.get('type') == 'political_leader':
                    return self.format_political_leader_response(search_info)
                elif search_info.get('type') == 'biography':
                    return f"{search_info['content']}"
                else:
                    return f"{search_info.get('content', 'Information not available')}"
            elif isinstance(search_info, str):
                cleaned_info = search_info.replace('...', '').strip()
                return cleaned_info
            else:
                return "I'd be happy to help you with that question. Could you provide a bit more context so I can give you the most accurate information?"
        except Exception as e:
            logger.error(f"Error formatting search response: {e}")
            return "I apologize, but I'm having trouble accessing the most current information right now. Please feel free to ask me about other topics, and I'll do my best to help!"
    
    def format_multi_source_response(self, search_info: dict) -> str:
        """Format responses from multiple sources in a structured, conversational way"""
        try:
            aggregated = search_info.get('aggregated_info', {})
            sources = search_info.get('sources', [])
            
            # Add conversational intros without mentioning sources
            intros = [
                "Here's what I found! 🔍",
                "Great question! 📚",
                "Let me share what I discovered! 🌐",
                "Interesting! 📊"
            ]
            
            intro = random.choice(intros)
            
            if aggregated.get('type') == 'political_leader':
                response = f"{intro}\n\n**{aggregated['title']}**\n\n"
                response += f"📊 **Primary Source:** {aggregated['primary_source']}\n"
                
                # Make content more natural and complete
                content = self.make_content_natural(aggregated['primary_content'])
                response += f"🔍 **Information:** {content}\n\n"
                
                if aggregated.get('additional_sources'):
                    response += f"📚 **I also checked:** {', '.join(aggregated['additional_sources'])}\n"
                
                response += f"\n*I consulted {aggregated['source_count']} sources to give you the most accurate information! 🎯*"
                return response
                
            elif aggregated.get('type') == 'biography':
                response = f"{intro}\n\n**{aggregated['title']}**\n\n"
                
                # Make content more natural
                main_content = self.make_content_natural(aggregated['main_content'])
                response += f"📖 **Here's what I found:** {main_content}\n\n"
                response += f"🌐 **Primary Source:** {aggregated['source']}\n"
                
                if aggregated.get('additional_info'):
                    response += f"\n📋 **Additional interesting details:**\n"
                    for i, info in enumerate(aggregated['additional_info'], 1):
                        natural_info = self.make_content_natural(info)
                        response += f"{i}. {natural_info}\n"
                
                response += f"\n*I gathered this from {len(aggregated.get('all_sources', []))} sources: {', '.join(aggregated.get('all_sources', []))} to ensure accuracy! ✨*"
                return response
                
            else:
                # General response
                response = f"{intro}\n\n**{aggregated.get('title', 'Search Results')}**\n\n"
                
                content = self.make_content_natural(aggregated.get('content', ''))
                response += f"📄 **Here's what I discovered:** {content}\n\n"
                response += f"🔗 **Primary Source:** {aggregated.get('source', 'Unknown')}\n"
                
                if aggregated.get('alternative_sources'):
                    response += f"📚 **Also consulted:** {', '.join(aggregated['alternative_sources'])}\n"
                
                response += f"\n*I checked {aggregated.get('total_sources', 0)} sources to bring you comprehensive information! 🚀*"
                return response
                
        except Exception as e:
            logger.error(f"Error formatting multi-source response: {e}")
            return "Oops! I had trouble organizing the search results properly. Let me try a different approach for your question! 😅"
    
    def make_content_natural(self, content: str) -> str:
        """Convert truncated or incomplete content into natural, complete sentences"""
        if not content:
            return "Information not available at the moment."
        
        # Clean up content
        content = content.strip()
        
        # Remove "Based on current web search results:" prefix
        if content.startswith("Based on current web search results:"):
            content = content.replace("Based on current web search results:", "").strip()
        
        # Fix common truncation issues
        if content.endswith('...'):
            content = content[:-3].strip()
        
        # Ensure proper sentence ending
        if content and not content.endswith(('.', '!', '?')):
            content += '.'
        
        # Fix incomplete sentences that end abruptly
        if content.endswith((' is', ' was', ' are', ' were', ' has', ' have', ' and', ' or', ' but')):
            content = content.rsplit(' ', 1)[0] + '.'
        
        return content
    
    def get_ai_generated_answer(self, question: str) -> str:
        """Generate accurate, polite, and concise AI answers like ChatGPT"""
        question_lower = question.lower().strip()
        
        # Biology questions - factual and concise
        if 'mitochondria' in question_lower:
            return "Mitochondria are membrane-bound organelles known as the 'powerhouse of the cell.' They generate ATP through cellular respiration, converting glucose and oxygen into energy. Found in most eukaryotic cells, they contain their own DNA and can self-replicate."
        
        if 'powerhouse of cell' in question_lower or 'powerhouse of the cell' in question_lower:
            return "Mitochondria are called the powerhouse of the cell because they produce ATP (adenosine triphosphate), the primary energy currency used by cells for various biological processes."
        
        if 'chloroplast' in question_lower:
            return "Chloroplasts are organelles in plant cells that conduct photosynthesis. They contain chlorophyll, which captures light energy to convert carbon dioxide and water into glucose and oxygen."
        
        if 'dna' in question_lower and any(word in question_lower for word in ['what is', 'define']):
            return "DNA (deoxyribonucleic acid) is the hereditary material in all living organisms. It carries genetic instructions for development, functioning, and reproduction, structured as a double helix of nucleotides."
        
        # Chemistry questions
        if 'photosynthesis' in question_lower:
            return "Photosynthesis is the process by which plants convert light energy into chemical energy. Using chlorophyll, plants combine carbon dioxide and water to produce glucose and oxygen: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂."
        
        if 'water formula' in question_lower or 'h2o' in question_lower:
            return "Water's chemical formula is H₂O, meaning each molecule contains two hydrogen atoms bonded to one oxygen atom. It's essential for all known forms of life."
        
        # Physics questions
        if 'gravity' in question_lower and any(word in question_lower for word in ['what is', 'define']):
            return "Gravity is a fundamental force that attracts objects with mass toward each other. On Earth, it gives weight to objects and causes them to fall toward the ground at 9.8 m/s²."
        
        if 'speed of light' in question_lower:
            return "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 300,000 km/s). It's a fundamental constant in physics and the maximum speed at which information can travel."
        
        # Mathematics
        if any(phrase in question_lower for phrase in ['what is 2+2', '2+2', '2 + 2']):
            return "2 + 2 = 4"
        
        if 'pi' in question_lower and any(word in question_lower for word in ['what is', 'value']):
            return "Pi (π) is approximately 3.14159. It's the ratio of a circle's circumference to its diameter, and it's an irrational number with infinite decimal places."
        
        # Geography
        if 'capital of' in question_lower:
            if 'nepal' in question_lower:
                return "The capital of Nepal is Kathmandu."
            if 'india' in question_lower:
                return "The capital of India is New Delhi."
            if 'usa' in question_lower or 'america' in question_lower:
                return "The capital of the United States is Washington, D.C."
            if 'china' in question_lower:
                return "The capital of China is Beijing."
        
        if 'mount everest' in question_lower:
            return "Mount Everest is the world's highest mountain at 8,848.86 meters (29,031.7 feet) above sea level. It's located in the Himalayas on the border between Nepal and Tibet."
        
        # Technology
        if 'artificial intelligence' in question_lower or 'what is ai' in question_lower:
            return "Artificial Intelligence (AI) is technology that enables machines to simulate human intelligence, including learning, reasoning, and problem-solving. It includes machine learning, natural language processing, and computer vision."
        
        if 'anup' in question_lower and any(word in question_lower for word in ['what is', 'company', 'who is']):
            return "Anup is the creator and developer of advanced AI systems including me, Hamro Mitra. Anup Raj Uprety from Nepal is a talented software developer, AI engineer, poet, and researcher who built me as an independent AI assistant to help users with various questions and tasks."
        
        # History
        if 'world war 2' in question_lower or 'wwii' in question_lower:
            return "World War II (1939-1945) was a global conflict involving most nations. It ended with the Allied victory over the Axis powers, fundamentally reshaping international relations and leading to the establishment of the United Nations."
        
        return None
    
    def format_political_leader_response(self, info: dict) -> str:
        """Format political leader information in an attractive, structured way"""
        response = f"**{info['title']}**\n\n"
        response += f"👤 **Current Leader:** {info['person']}\n"
        response += f"🏛️ **Position:** {info['position']}\n"
        response += f"🎯 **Political Party:** {info['party']}\n"
        response += f"📅 **Tenure:** {info['tenure']}\n\n"
        response += f"📋 **Key Information:**\n{info['key_info']}\n\n"
        if info.get('context'):
            response += f"🔍 **Context:** {info['context']}\n\n"
        response += f"*Last updated: {info.get('last_updated', 'Recent')}*"
        return response
    
    def get_structured_political_answer(self, topic: str, question: str) -> str:
        """Get structured answers for political questions"""
        if topic == 'nepal_pm':
            return """**Current Prime Minister of Nepal**

👤 **Current Leader:** KP Sharma Oli
🏛️ **Position:** Prime Minister of Nepal
🎯 **Political Party:** Communist Party of Nepal (Unified Marxist-Leninist)
📅 **Tenure:** Current (as of 2024)

📋 **Key Information:**
KP Sharma Oli is currently serving as the Prime Minister of Nepal. He has held this position multiple times throughout his political career and is known for his significant role in Nepalese politics. Oli came to power for the third time in 2024 after forming strategic political alliances.

🔍 **Context:** Nepal operates under a multi-party democratic system where government leadership can change through coalition politics and parliamentary procedures.

*Note: Political situations can change rapidly. For the most current information, please check recent news sources.*"""
        return None
    
    def generate_intelligent_response(self, question: str) -> str:
        """Generate intelligent responses with comprehensive knowledge"""
        question_lower = question.lower().strip()
        
        # Handle greetings - polite and concise
        if any(greeting in question_lower for greeting in ['hello', 'hi', 'hey', 'namaste', 'sathi']):
            return "Hello! I'm Hamro Mitra, your AI assistant created by Anup Raj Uprety. How can I help you today?"
        
        if any(phrase in question_lower for phrase in ['how are you', 'how do you do', 'what\'s up']):
            responses = [
                "I'm doing well, thank you! How can I help you today?",
                "I'm great! What questions do you have?",
                "I'm here and ready to help. What would you like to know?",
                "I'm doing fine! What can I assist you with?"
            ]
            return random.choice(responses)
        
        # Handle creator questions
        if any(phrase in question_lower for phrase in ['who made you', 'who created you', 'who built you', 'your creator', 'your maker']):
            return "I was created by Anup Raj Uprety, a talented software developer, AI engineer, poet, and researcher from Nepal. He specializes in Python, AI/ML, web development, and natural language processing. He's also a researcher working on consciousness and Hindu philosophy, with published books in these areas. I am Anup's independent creation, built with his own AI technology."
        
        # Handle specific knowledge questions
        if 'prachanda' in question_lower:
            return "Prachanda (Pushpa Kamal Dahal) is a prominent Nepalese politician and the chairman of the Communist Party of Nepal (Maoist Centre). He served as Prime Minister of Nepal multiple times and was a key leader during Nepal's civil war (1996-2006). He played a crucial role in Nepal's transition from a monarchy to a federal democratic republic. Prachanda is known for leading the Maoist insurgency and later participating in the peace process that transformed Nepal's political landscape."
        
        if any(word in question_lower for word in ['nepal', 'nepali', 'kathmandu', 'himalaya']):
            return "Nepal is a beautiful landlocked country in South Asia, nestled between China and India. It's famous for the Himalayas, including Mount Everest, the world's highest peak. Nepal has a rich cultural heritage with diverse ethnic groups, languages, and traditions. Kathmandu is the capital city, known for its ancient temples and vibrant culture. Nepal transitioned from a monarchy to a federal democratic republic in 2008. The country is known for its hospitality, stunning landscapes, and as the birthplace of Lord Buddha."
        
        # Handle programming questions
        if any(word in question_lower for word in ['python', 'programming', 'code', 'function', 'algorithm']):
            if 'python' in question_lower:
                return "Python is a powerful, versatile programming language that's great for beginners and experts alike. It's known for its clean, readable syntax and extensive libraries. Python is widely used in web development, data science, AI/ML, automation, and more. Key features include dynamic typing, automatic memory management, and a vast ecosystem of packages. Would you like help with a specific Python concept or problem?"
            else:
                return "Programming is the art and science of creating software solutions to solve problems. It involves writing instructions for computers using programming languages. Modern programming emphasizes clean code, good design patterns, and efficient algorithms. Whether you're interested in web development, mobile apps, data analysis, or AI, programming skills can help you build amazing things. What specific programming topic interests you?"
        
        # Handle AI/ML questions
        if any(word in question_lower for word in ['ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning']):
            return "Artificial Intelligence is a rapidly evolving field that aims to create intelligent machines capable of performing tasks that typically require human intelligence. Machine learning, a subset of AI, enables computers to learn and improve from data without explicit programming. Deep learning uses neural networks with multiple layers to solve complex problems. AI is transforming industries including healthcare, finance, transportation, and entertainment. Current breakthroughs include large language models, computer vision, and autonomous systems."
        
        # Handle science questions
        if any(word in question_lower for word in ['science', 'physics', 'chemistry', 'biology', 'universe', 'space']):
            return "Science is humanity's systematic approach to understanding the natural world through observation, experimentation, and analysis. Physics explores matter, energy, and their interactions. Chemistry studies the composition and behavior of substances. Biology examines living organisms and life processes. Modern science has revealed incredible insights about our universe, from quantum mechanics to cosmology, from DNA to ecosystems. What specific scientific topic would you like to explore?"
        
        # Handle math questions
        if any(word in question_lower for word in ['math', 'mathematics', 'equation', 'solve', 'calculate']):
            return "Mathematics is the foundation of science, technology, and logical reasoning. It includes areas like algebra, calculus, geometry, statistics, and number theory. Math helps us model real-world phenomena, solve complex problems, and make predictions. From basic arithmetic to advanced topics like differential equations and abstract algebra, mathematics provides powerful tools for understanding patterns and relationships. What mathematical concept or problem would you like help with?"
        
        # Handle history and politics
        if any(word in question_lower for word in ['history', 'politics', 'government', 'democracy', 'election']):
            return "History and politics are fascinating subjects that help us understand how societies develop and govern themselves. Political systems vary from democracies to various forms of authoritarianism. Historical events shape current political landscapes and social structures. Understanding these topics helps us become informed citizens and make better decisions about our communities and nations. What specific historical period or political topic interests you?"
        
        # Handle specific 'Anup' questions with context awareness
        if any(name in question_lower for name in ['who is anup', 'anup', 'tell me about anup', 'anup raj uprety', 'anup uprety', 'anup raj upreti', 'anup upreti']):
            return self.handle_anup_query_interactive(question)
        
        # Handle general knowledge questions - concise and helpful
        if question_lower.startswith(('what', 'how', 'why', 'when', 'where', 'who')):
            # Check for specific people or topics first
            if 'narendra modi' in question_lower or ('modi' in question_lower and any(word in question_lower for word in ['prime minister', 'india', 'bjp'])):
                return "Narendra Modi is the current Prime Minister of India, serving since May 2014. He's a member of the Bharatiya Janata Party (BJP) and previously served as Chief Minister of Gujarat from 2001-2014. He's known for economic reforms and digital initiatives like Digital India."
            
            # Try to provide a helpful response based on context
            if 'what is' in question_lower or 'who is' in question_lower:
                topic = question_lower.replace('what is', '').replace('who is', '').strip().rstrip('?')
                return f"I'd be happy to help you learn about {topic}. Could you be more specific about what aspect interests you most?"
            else:
                return "I can help with questions about science, technology, history, geography, and many other topics. What would you like to know?"
        
        # Default response - simple and helpful
        return "I'm here to help answer your questions on various topics. What would you like to know?"
    
    def get_direct_answer(self, question: str) -> str:
        """Provide direct answers to common questions"""
        question_lower = question.lower().strip()
        
        # Handle greetings
        if any(greeting in question_lower for greeting in ['hello', 'hi', 'hey', 'namaste', 'sathi']):
            return "Hello! I'm Hamro Mitra, your AI assistant created by Anup Raj Uprety. How can I help you today?"
        
        # Handle math questions
        if '2+2' in question_lower or '2 + 2' in question_lower:
            return "2 + 2 = 4. This is basic arithmetic addition where we combine two quantities of 2 to get a sum of 4."
        
        # Handle ChatGPT questions
        if 'chatgpt' in question_lower:
            return "ChatGPT is an AI chatbot developed by OpenAI that uses large language models to have conversations and answer questions. It's trained on vast amounts of text data and can help with writing, coding, analysis, and many other tasks. ChatGPT represents a major breakthrough in conversational AI technology."
        
        # Handle velocity questions
        if 'velocity' in question_lower:
            return "Velocity is a vector quantity in physics that describes the rate of change of an object's position with respect to time. It includes both speed (magnitude) and direction. For example, a car traveling at 60 km/h north has a velocity that specifies both how fast it's moving and in which direction."
        
        # Handle space-time curvature
        if 'space' in question_lower and 'time' in question_lower and 'curvature' in question_lower:
            return "Space-time curvature is a concept from Einstein's General Theory of Relativity. According to this theory, massive objects like stars and planets actually bend or curve the fabric of space and time around them. This curvature is what we experience as gravity. The more massive an object, the more it curves space-time, creating stronger gravitational effects."
        
        # Handle biology questions
        if 'biology' in question_lower and not any(word in question_lower for word in ['space', 'physics', 'chemistry']):
            return "Biology is the scientific study of life and living organisms. It examines the structure, function, growth, evolution, distribution, and taxonomy of all living things. Biology includes many branches like molecular biology, genetics, ecology, botany (plants), zoology (animals), and microbiology (microorganisms). It helps us understand how life works at all levels, from cells to ecosystems."
        
        # Handle Sita questions
        if 'sita' in question_lower:
            return "Sita is a central character in the Hindu epic Ramayana. She is the wife of Lord Rama and is revered as an ideal woman in Hindu culture. Sita is known for her devotion, purity, and strength. In the epic, she is kidnapped by the demon king Ravana, leading to the great war that forms the main plot of the Ramayana. Sita is considered an incarnation of the goddess Lakshmi."
        
        # Handle Prachanda questions
        if 'prachanda' in question_lower:
            return "Prachanda (Pushpa Kamal Dahal) is a prominent Nepalese politician and the chairman of the Communist Party of Nepal (Maoist Centre). He served as Prime Minister of Nepal multiple times and was a key leader during Nepal's civil war (1996-2006). He played a crucial role in Nepal's transition from a monarchy to a federal democratic republic. Prachanda is known for leading the Maoist insurgency and later participating in the peace process that transformed Nepal's political landscape."
        
        # Handle Narendra Modi questions
        if 'narendra modi' in question_lower or 'modi' in question_lower:
            return "Narendra Modi is the current Prime Minister of India, serving since May 2014. Born on September 17, 1950, in Vadnagar, Gujarat, he is a member of the Bharatiya Janata Party (BJP). Before becoming Prime Minister, Modi served as Chief Minister of Gujarat from 2001 to 2014. He is known for his economic policies, digital initiatives like Digital India, and infrastructure development projects. Modi has been a significant figure in Indian politics, leading the BJP to decisive victories in 2014 and 2019 general elections. His leadership style emphasizes development, good governance, and India's growing role on the global stage."
        
        # Handle KP Sharma Oli questions
        if 'kp sharma oli' in question_lower or ('oli' in question_lower and 'nepal' in question_lower):
            return "KP Sharma Oli is a prominent Nepalese politician who has served as Prime Minister of Nepal multiple times. He came to power for the third time in 2024 after forming alliances with various political parties. He is known for his political maneuvering and has been a significant figure in Nepal's contemporary politics. His leadership has been marked by both achievements and controversies, including handling of various national issues and international relations. Oli is the chairman of the Communist Party of Nepal (Unified Marxist-Leninist) and has played a key role in Nepal's political landscape for decades."
        
        # Handle current Nepal PM questions
        if any(phrase in question_lower for phrase in ['current prime minister of nepal', 'nepal prime minister', 'pm of nepal', 'who is prime minister nepal']):
            return self.get_structured_political_answer('nepal_pm', question)
        
        # Handle creator questions with shorter, natural responses
        if any(phrase in question_lower for phrase in ['who made you', 'who created you', 'who built you', 'your creator', 'your maker']):
            return "I was created by Anup Raj Uprety, a software developer from Nepal! 😊"
        
        # Handle Nepal-related questions
        if any(word in question_lower for word in ['nepal', 'nepali', 'kathmandu']):
            return "Nepal is a beautiful landlocked country in South Asia, nestled between China and India. It's famous for the Himalayas, including Mount Everest, the world's highest peak. Nepal has a rich cultural heritage with diverse ethnic groups, languages, and traditions. Kathmandu is the capital city, known for its ancient temples and vibrant culture. Nepal transitioned from a monarchy to a federal democratic republic in 2008."
        
        # Try comprehensive answer before defaulting
        comprehensive_answer = self.get_comprehensive_answer(question)
        if comprehensive_answer:
            return comprehensive_answer
        
        # Default for unrecognized questions
        return "Sorry for the inconvenience, but I am not getting this answer right now. Please try rephrasing your question or ask me something else I can help with."
    
    def get_comprehensive_answer(self, question: str) -> str:
        """Provide comprehensive answers for common biographical and factual questions"""
        question_lower = question.lower().strip()
        
        # Biography questions
        if any(word in question_lower for word in ['biography', 'bio', 'life story', 'about']):
            if 'narendra modi' in question_lower or 'modi' in question_lower:
                return "Narendra Modi is the current Prime Minister of India, serving since May 2014. Born on September 17, 1950, in Vadnagar, Gujarat, he comes from a humble background - his father sold tea at a railway station. Modi joined the Rashtriya Swayamsevak Sangh (RSS) as a young man and later became involved with the Bharatiya Janata Party (BJP). He served as Chief Minister of Gujarat from 2001 to 2014, where he focused on economic development and infrastructure. As Prime Minister, Modi has launched several major initiatives including Digital India, Make in India, Swachh Bharat (Clean India), and the Goods and Services Tax (GST). He is known for his oratory skills, use of social media, and efforts to position India as a major global power. Modi has been re-elected twice, leading the BJP to significant victories in 2014 and 2019."
            
            if 'prachanda' in question_lower:
                return "Prachanda (Pushpa Kamal Dahal) was born on December 11, 1954, in Dhikurpokhari, Nepal. He became involved in communist politics in his youth and eventually led the Communist Party of Nepal (Maoist) in a decade-long armed insurgency (1996-2006) against the monarchy. The conflict ended with the Comprehensive Peace Agreement in 2006. Prachanda played a crucial role in Nepal's transition to a federal democratic republic, including the abolition of the monarchy in 2008. He has served as Prime Minister multiple times and remains chairman of the Communist Party of Nepal (Maoist Centre). His political career has been marked by pragmatic alliances and his ability to navigate Nepal's complex multi-party democracy."
        
        # Handle specific person queries
        if 'narendra modi' in question_lower or ('modi' in question_lower and any(word in question_lower for word in ['prime minister', 'india', 'bjp'])):
            return "Narendra Modi is the current Prime Minister of India, serving since May 2014. Born on September 17, 1950, in Vadnagar, Gujarat, he is a member of the Bharatiya Janata Party (BJP). Before becoming Prime Minister, Modi served as Chief Minister of Gujarat from 2001 to 2014. He is known for his economic policies, digital initiatives like Digital India, and infrastructure development projects. Modi has been a significant figure in Indian politics, leading the BJP to decisive victories in 2014 and 2019 general elections. His leadership style emphasizes development, good governance, and India's growing role on the global stage."
        
        # Handle current affairs and political questions
        if any(phrase in question_lower for phrase in ['current', 'latest', 'recent', 'now', 'today']):
            if 'prime minister' in question_lower and 'nepal' in question_lower:
                return self.get_structured_political_answer('nepal_pm', question)
            elif 'prime minister' in question_lower and 'india' in question_lower:
                return "**Current Prime Minister of India**\n\n👤 **Current Leader:** Narendra Modi\n🏛️ **Position:** Prime Minister of India\n🎯 **Political Party:** Bharatiya Janata Party (BJP)\n📅 **Tenure:** Since May 2014 (3rd term)\n\n📋 **Key Information:**\nNarendra Modi has been serving as India's Prime Minister since 2014, leading the BJP to consecutive electoral victories in 2014, 2019, and 2024. He previously served as Chief Minister of Gujarat from 2001-2014.\n\n🔍 **Context:** India follows a parliamentary system where the Prime Minister is the head of government, leading the party or coalition with majority support in the Lok Sabha."
        
        return None
    
    def handle_anup_query_interactive(self, question: str) -> str:
        """Handle questions about 'Anup' with accurate information"""
        return "Anup Raj Uprety (also known as Anup Upreti) is a multi-talented individual from Nepal. He's a software developer and AI engineer who created me (Hamro Mitra), specializing in Python, AI/ML, web development, and natural language processing. Beyond technology, he's also a poet and researcher working on consciousness and Hindu philosophy, with two published books in these areas. He's passionate about building intelligent systems that help people globally. I am his independent creation, not affiliated with any company like Cohere."
    
    def run(self, host='localhost', port=5000, debug=True):
        """Run the Flask application"""
        logger.info(f"Starting Simple AI Assistant on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    assistant = SimpleAIAssistant()
    assistant.run()
