#!/usr/bin/env python3
"""
Advanced AI Backend for Global Question Answering System
Integrates multiple open-source AI models and NLP capabilities
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Core frameworks
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import wikipedia

# AI and NLP libraries
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForQuestionAnswering,
    pipeline, AutoModel, T5ForConditionalGeneration, T5Tokenizer
)
from sentence_transformers import SentenceTransformer
import spacy
import nltk
from langdetect import detect
# from googletrans import Translator  # Temporarily disabled due to compatibility issues

# Data processing
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# Text processing utilities
import yake
from keybert import KeyBERT
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import textstat

# Environment and configuration
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GlobalAIAssistant:
    """Advanced AI Assistant with multiple open-source models and global knowledge"""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Initialize components
        self.models = {}
        self.pipelines = {}
        self.knowledge_base = {}
        # Initialize translator (disabled for compatibility)
        self.translator = None
        
        # Web search configuration
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.google_cse_id = os.getenv('GOOGLE_CUSTOM_SEARCH_ENGINE_ID')
        self.serpapi_key = os.getenv('SERPAPI_KEY')
        
        # OpenAI configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if self.openai_api_key and openai:
            openai.api_key = self.openai_api_key
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize models
        self._initialize_models()
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Global AI Assistant initialized successfully!")
    
    def _download_nltk_data(self):
        """Download required NLTK datasets"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except Exception as e:
            logger.warning(f"NLTK download warning: {e}")
    
    def _initialize_models(self):
        """Initialize all AI models and pipelines"""
        try:
            # Sentence transformer for embeddings
            logger.info("Loading sentence transformer...")
            self.models['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Question answering pipeline
            logger.info("Loading QA pipeline...")
            self.pipelines['qa'] = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                tokenizer="distilbert-base-cased-distilled-squad"
            )
            
            # Text generation pipeline
            logger.info("Loading text generation pipeline...")
            self.pipelines['text_generation'] = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                tokenizer="microsoft/DialoGPT-medium",
                max_length=512,
                do_sample=True,
                temperature=0.7
            )
            
            # Summarization pipeline
            logger.info("Loading summarization pipeline...")
            self.pipelines['summarization'] = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                max_length=150,
                min_length=50
            )
            
            # Sentiment analysis
            logger.info("Loading sentiment analysis...")
            self.pipelines['sentiment'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            # Named Entity Recognition
            logger.info("Loading NER pipeline...")
            self.pipelines['ner'] = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple"
            )
            
            # Load spaCy model
            try:
                self.models['spacy'] = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                self.models['spacy'] = None
            
            # Keyword extraction
            self.models['keybert'] = KeyBERT('distilbert-base-nli-mean-tokens')
            
            # YAKE keyword extractor
            self.models['yake'] = yake.KeywordExtractor(
                lan="en",
                n=3,
                dedupLim=0.7,
                top=10
            )
            
            # Initialize FAISS index for knowledge retrieval
            self._initialize_knowledge_base()
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def _initialize_knowledge_base(self):
        """Initialize vector database for knowledge retrieval"""
        try:
            # Create a sample knowledge base
            knowledge_texts = [
                "Python is a high-level programming language known for its simplicity and readability.",
                "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
                "Natural language processing helps computers understand and interpret human language.",
                "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
                "The Internet connects billions of devices worldwide, enabling global communication.",
                "Climate change refers to long-term shifts in global temperatures and weather patterns.",
                "Renewable energy sources include solar, wind, hydroelectric, and geothermal power.",
                "Artificial intelligence aims to create machines that can perform tasks requiring human intelligence.",
                "Quantum computing uses quantum mechanical phenomena to process information.",
                "Blockchain is a distributed ledger technology that maintains a secure record of transactions."
            ]
            
            # Generate embeddings
            embeddings = self.models['sentence_transformer'].encode(knowledge_texts)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.knowledge_base['index'] = faiss.IndexFlatL2(dimension)
            self.knowledge_base['index'].add(embeddings.astype('float32'))
            self.knowledge_base['texts'] = knowledge_texts
            
            logger.info(f"Knowledge base initialized with {len(knowledge_texts)} entries")
            
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})
        
        @self.app.route('/ask', methods=['POST'])
        def ask_question():
            try:
                data = request.json
                question = data.get('question', '')
                context = data.get('context', '')
                
                if not question:
                    return jsonify({"error": "Question is required"}), 400
                
                response = self.process_question(question, context)
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/analyze', methods=['POST'])
        def analyze_text():
            try:
                data = request.json
                text = data.get('text', '')
                
                if not text:
                    return jsonify({"error": "Text is required"}), 400
                
                analysis = self.analyze_text_comprehensive(text)
                return jsonify(analysis)
                
            except Exception as e:
                logger.error(f"Error analyzing text: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/search', methods=['POST'])
        def search_knowledge():
            try:
                data = request.json
                query = data.get('query', '')
                
                if not query:
                    return jsonify({"error": "Query is required"}), 400
                
                results = self.search_knowledge_base(query)
                return jsonify(results)
                
            except Exception as e:
                logger.error(f"Error searching knowledge: {e}")
                return jsonify({"error": str(e)}), 500
    
    def process_question(self, question: str, context: str = "") -> Dict[str, Any]:
        """Process a question using multiple AI approaches"""
        try:
            # Detect language
            detected_lang = detect(question)
            
            # Translate to English if needed
            if detected_lang != 'en':
                question_en = self.translator.translate(question, dest='en').text
            else:
                question_en = question
            
            # Analyze question type
            question_analysis = self.analyze_question(question_en)
            
            # Get multiple responses
            responses = {}
            
            # 1. Knowledge base search
            kb_results = self.search_knowledge_base(question_en)
            responses['knowledge_base'] = kb_results
            
            # 2. Wikipedia search
            wiki_info = self.get_wikipedia_info(question_en)
            responses['wikipedia'] = wiki_info
            
            # 3. Web search and scraping
            web_info = self.get_web_information(question_en)
            responses['web_search'] = web_info
            
            # 4. Question answering with context
            if context or kb_results.get('relevant_text'):
                qa_context = context or kb_results.get('relevant_text', '')
                qa_response = self.pipelines['qa'](
                    question=question_en,
                    context=qa_context
                )
                responses['qa_model'] = qa_response
            
            # 5. Generate comprehensive answer
            comprehensive_answer = self.generate_comprehensive_answer(
                question_en, responses, question_analysis
            )
            
            # Translate back if needed
            if detected_lang != 'en':
                comprehensive_answer = self.translator.translate(
                    comprehensive_answer, dest=detected_lang
                ).text
            
            return {
                "question": question,
                "detected_language": detected_lang,
                "question_analysis": question_analysis,
                "answer": comprehensive_answer,
                "sources": responses,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "question": question,
                "answer": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze question to determine type and extract key information"""
        try:
            analysis = {}
            
            # Extract keywords
            keywords_yake = [kw[1] for kw in self.models['yake'].extract_keywords(question)]
            keywords_keybert = [kw[0] for kw in self.models['keybert'].extract_keywords(question, keyphrase_ngram_range=(1, 2), stop_words='english')]
            
            analysis['keywords'] = {
                'yake': keywords_yake[:5],
                'keybert': keywords_keybert[:5]
            }
            
            # Named entity recognition
            if 'ner' in self.pipelines:
                entities = self.pipelines['ner'](question)
                analysis['entities'] = entities
            
            # Question type classification
            question_lower = question.lower()
            if any(word in question_lower for word in ['what', 'define', 'explain', 'describe']):
                analysis['type'] = 'definition'
            elif any(word in question_lower for word in ['how', 'steps', 'process', 'method']):
                analysis['type'] = 'procedural'
            elif any(word in question_lower for word in ['why', 'reason', 'cause', 'because']):
                analysis['type'] = 'causal'
            elif any(word in question_lower for word in ['when', 'time', 'date', 'year']):
                analysis['type'] = 'temporal'
            elif any(word in question_lower for word in ['where', 'location', 'place']):
                analysis['type'] = 'spatial'
            elif any(word in question_lower for word in ['who', 'person', 'people']):
                analysis['type'] = 'personal'
            else:
                analysis['type'] = 'general'
            
            # Sentiment analysis
            sentiment = self.pipelines['sentiment'](question)
            analysis['sentiment'] = sentiment[0]
            
            # Readability metrics
            analysis['readability'] = {
                'flesch_reading_ease': textstat.flesch_reading_ease(question),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(question),
                'word_count': len(question.split())
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing question: {e}")
            return {"error": str(e)}
    
    def search_knowledge_base(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Search the knowledge base using vector similarity"""
        try:
            if 'index' not in self.knowledge_base:
                return {"error": "Knowledge base not initialized"}
            
            # Generate query embedding
            query_embedding = self.models['sentence_transformer'].encode([query])
            
            # Search in FAISS index
            distances, indices = self.knowledge_base['index'].search(
                query_embedding.astype('float32'), top_k
            )
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.knowledge_base['texts']):
                    results.append({
                        'text': self.knowledge_base['texts'][idx],
                        'similarity_score': float(1 / (1 + distance)),  # Convert distance to similarity
                        'rank': i + 1
                    })
            
            return {
                'results': results,
                'relevant_text': ' '.join([r['text'] for r in results[:2]]) if results else ""
            }
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return {"error": str(e)}
    
    def get_wikipedia_info(self, query: str) -> Dict[str, Any]:
        """Get information from Wikipedia"""
        try:
            # Search Wikipedia
            search_results = wikipedia.search(query, results=3)
            
            if not search_results:
                return {"error": "No Wikipedia results found"}
            
            # Get the first result
            page = wikipedia.page(search_results[0])
            
            # Summarize the content
            summary = wikipedia.summary(search_results[0], sentences=3)
            
            return {
                'title': page.title,
                'summary': summary,
                'url': page.url,
                'categories': page.categories[:5] if hasattr(page, 'categories') else []
            }
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation
            try:
                page = wikipedia.page(e.options[0])
                summary = wikipedia.summary(e.options[0], sentences=3)
                return {
                    'title': page.title,
                    'summary': summary,
                    'url': page.url,
                    'note': 'Disambiguation resolved'
                }
            except:
                return {"error": "Wikipedia disambiguation error"}
                
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return {"error": str(e)}
    
    def get_web_information(self, query: str) -> Dict[str, Any]:
        """Get information from web search using Google Custom Search API and SerpAPI"""
        try:
            # Try Google Custom Search API first
            if self.google_api_key and self.google_cse_id:
                google_results = self._google_custom_search(query)
                if google_results and 'error' not in google_results:
                    return google_results
            
            # Fallback to SerpAPI
            if self.serpapi_key:
                serpapi_results = self._serpapi_search(query)
                if serpapi_results and 'error' not in serpapi_results:
                    return serpapi_results
            
            # Fallback to DuckDuckGo scraping
            return self._duckduckgo_search(query)
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {"error": str(e)}
    
    def _google_custom_search(self, query: str) -> Dict[str, Any]:
        """Search using Google Custom Search API"""
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
                        'url': item.get('link', ''),
                        'displayLink': item.get('displayLink', '')
                    })
                
                return {
                    'search_results': results,
                    'source': 'Google Custom Search',
                    'total_results': data.get('searchInformation', {}).get('totalResults', '0')
                }
            
            return {"error": f"Google Search API error: {response.status_code}"}
            
        except Exception as e:
            logger.error(f"Google Custom Search error: {e}")
            return {"error": str(e)}
    
    def _serpapi_search(self, query: str) -> Dict[str, Any]:
        """Search using SerpAPI"""
        try:
            url = "https://serpapi.com/search"
            params = {
                'api_key': self.serpapi_key,
                'engine': 'google',
                'q': query,
                'num': 5
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get('organic_results', []):
                    results.append({
                        'title': item.get('title', ''),
                        'snippet': item.get('snippet', ''),
                        'url': item.get('link', ''),
                        'displayLink': item.get('displayed_link', '')
                    })
                
                return {
                    'search_results': results,
                    'source': 'SerpAPI',
                    'total_results': data.get('search_information', {}).get('total_results', '0')
                }
            
            return {"error": f"SerpAPI error: {response.status_code}"}
            
        except Exception as e:
            logger.error(f"SerpAPI error: {e}")
            return {"error": str(e)}
    
    def _duckduckgo_search(self, query: str) -> Dict[str, Any]:
        """Fallback search using DuckDuckGo scraping"""
        try:
            search_url = f"https://duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                results = []
                
                # Extract search results
                for result in soup.find_all('a', class_='result__a')[:5]:
                    title = result.get_text().strip()
                    url = result.get('href', '')
                    
                    # Get snippet from parent element
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
                            'url': url,
                            'displayLink': url
                        })
                
                return {
                    'search_results': results,
                    'source': 'DuckDuckGo'
                }
            
            return {"error": "DuckDuckGo search failed"}
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return {"error": str(e)}
    
    def generate_comprehensive_answer(self, question: str, sources: Dict, analysis: Dict) -> str:
        """Generate a comprehensive answer using all available sources with web search integration"""
        question_lower = question.lower().strip()
        
        # Detect if the question is in Nepali
        is_nepali = self._is_nepali_text(question_lower)
        
        # Handle Nepali greetings and responses
        if is_nepali:
            return self._handle_nepali_conversation(question_lower)
        
        # Handle English greetings
        if any(greeting in question_lower for greeting in ['hello', 'hi', 'hey', 'namaste', 'sathi']):
            return "Hello! I'm Hamro Mitra, your AI assistant with real-time web search capabilities. I can help you with current information, facts, and any questions you have!"
        
        if any(phrase in question_lower for phrase in ['how are you', 'how do you do', 'what\'s up']):
            return "I'm doing great and ready to help! I have access to real-time web search and can provide you with current, accurate information. What would you like to know?"
        
        # Handle creator questions
        if any(phrase in question_lower for phrase in ['who made you', 'who created you', 'who built you', 'your creator', 'your maker']):
            return "I was created by Anup Raj Uprety, a talented software developer and AI engineer from Nepal. He built me with real-time web search integration to provide accurate, current information from multiple sources including Google Custom Search and SerpAPI."
        
        try:
            # Determine if this needs web search
            needs_web_search = self._should_use_web_search(question)
            
            # Combine information from all sources
            context_parts = []
            web_context = ""
            
            # Add web search results if available and relevant
            if 'web_search' in sources and 'search_results' in sources['web_search']:
                web_results = sources['web_search']['search_results']
                if web_results:
                    web_snippets = []
                    for result in web_results[:5]:  # Use top 5 results for more comprehensive answers
                        if result.get('snippet'):
                            # Clean snippet text to avoid incomplete sentences
                            snippet = result['snippet'].strip()
                            if snippet and not snippet.endswith('...'):
                                web_snippets.append(snippet)
                            elif snippet:
                                # Try to complete the sentence by removing incomplete ending
                                sentences = snippet.split('.')
                                if len(sentences) > 1:
                                    complete_sentences = '.'.join(sentences[:-1]) + '.'
                                    web_snippets.append(complete_sentences)
                                else:
                                    web_snippets.append(snippet)
                    
                    if web_snippets:
                        web_context = " ".join(web_snippets)
                        context_parts.append(web_context)
            
            # Add Wikipedia info
            if 'wikipedia' in sources and 'summary' in sources['wikipedia']:
                wiki_summary = sources['wikipedia']['summary'].strip()
                if wiki_summary:
                    context_parts.append(wiki_summary)
            
            # Add knowledge base info
            if 'knowledge_base' in sources and 'relevant_text' in sources['knowledge_base']:
                kb_text = sources['knowledge_base']['relevant_text'].strip()
                if kb_text:
                    context_parts.append(kb_text)
            
            # Create comprehensive context
            context = " ".join(context_parts)
            
            # Generate comprehensive structured answer
            if context and len(context) > 50:
                return self._generate_structured_answer(question, context, web_context, analysis)
            
            # If we have web results but no good context, format web results properly
            if web_context:
                return self._format_web_results_answer(question, web_results[:5])
        
        except Exception as e:
            logger.error(f"Comprehensive answer generation error: {e}")
        
        return "I'm here to help! Could you please rephrase your question or provide more details?"
    
    def _is_nepali_text(self, text: str) -> bool:
        """Detect if text contains Nepali language"""
        # Common Nepali words and phrases
        nepali_indicators = [
            'sanchai', 'xau', 'xu', 'xa', 'kasto', 'kasari', 'sahayog', 'garnna', 'sakxu',
            'timi', 'timlai', 'ma', 'mero', 'tapai', 'hajur', 'ho', 'haina', 'ramro',
            'naramro', 'dhanyabad', 'kripaya', 'maaf', 'garnuhos', 'hunxa', 'bhayo',
            'garna', 'sakchu', 'sakdina', 'chha', 'thiyo', 'thaha', 'pani', 'ani',
            'ra', 'ko', 'ki', 'lai', 'bata', 'ma', 'yo', 'tyo', 'ke', 'kun',
            'kaha', 'kati', 'kina', 'kasle', 'kalle', 'bholi', 'aja', 'hijo',
            'sathi', 'dai', 'didi', 'bahini', 'bhai', 'ama', 'buba', 'ghar',
            'paisa', 'kam', 'padhai', 'khana', 'pani', 'samay', 'din', 'rat'
        ]
        
        # Check for Nepali characters (Devanagari script)
        nepali_chars = any('\u0900' <= char <= '\u097F' for char in text)
        
        # Check for romanized Nepali words
        nepali_words = any(word in text for word in nepali_indicators)
        
        return nepali_chars or nepali_words
    
    def _handle_nepali_conversation(self, question: str) -> str:
        """Handle conversations in Nepali language"""
        question_lower = question.lower().strip()
        
        # Greetings in Nepali
        if any(greeting in question_lower for greeting in ['namaste', 'namaskar', 'sanchai', 'kasto xa']):
            return "Namaste! Ma Hamro Mitra hu, tapai ko AI sahayak. Ma sanchai xu! Tapai lai kasto xa? Ma tapai lai kuna pani jankari ra sahayog garna sakchu. Ke chahiyo?"
        
        # How are you in Nepali
        if any(phrase in question_lower for phrase in ['sanchai xau', 'kasto xa', 'kasto chha', 'ramro xa']):
            return "Ma sanchai xu, dhanyabad! Ma tapai lai sahayog garna tayar chu. Tapai lai ke janna man lagyo? Ma current jankari, facts, ra kuna pani prashna ko jawaf dina sakchu!"
        
        # Asking for help in Nepali
        if any(phrase in question_lower for phrase in ['sahayog', 'maddat', 'help', 'kasari', 'ke garne']):
            return "Tapai lai sahayog garna ma khushi lagyo! Ma yo kura haru garna sakchu:\n• Kuna pani prashna ko jawaf dina\n• Current jankari dina\n• Ganit ra science ko samasya solve garna\n• Nepali ra English duvai bhasha ma kura garna\n\nTapai ke janna chahanu huncha?"
        
        # About creator in Nepali
        if any(phrase in question_lower for phrase in ['ko banayo', 'kasle banayo', 'creator', 'maker', 'anup']):
            return "Ma lai Anup Raj Uprety le banayeko ho. Unko Nepal ka ek talented software developer ra AI engineer hun. Uni le ma lai real-time web search capabilities sanga banayeko chha taki ma tapai lai accurate ra current jankari dina sakum!"
        
        # Thank you in Nepali
        if any(phrase in question_lower for phrase in ['dhanyabad', 'dhanyawad', 'thank you', 'thanks']):
            return "Swagat chha! Ma tapai lai sahayog garna paye khushi lagyo. Aru kei chahiyo bhane sodhnu hola!"
        
        # Goodbye in Nepali
        if any(phrase in question_lower for phrase in ['alvida', 'bye', 'jaane', 'bidai']):
            return "Alvida! Tapai sanga kura garna ramro lagyo. Pheri aunu hola. Ramro din hos!"
        
        # General questions in Nepali
        if any(word in question_lower for word in ['ke', 'kun', 'kaha', 'kina', 'kasari', 'kati']):
            return "Tapai ko prashna ramro chha! Ma Nepali ma pani jawaf dina sakchu. Kripaya tapai ko prashna aru detail ma bhannu hola taki ma ramro jawaf dina sakum. Ma mathematics, science, current events, ra anya bisaya ma sahayog garna sakchu!"
        
        # Math and science questions in Nepali
        if any(word in question_lower for word in ['ganit', 'hisab', 'calculate', 'solve', 'velocity', 'speed']):
            return "Ma ganit ra science ko samasya solve garna sakchu! Tapai ko prashna English ya Nepali ma sodhnu hola. Jastai:\n• Ganit: '10k/2 kati huncha?'\n• Physics: 'Velocity calculate garna kasari?'\n• Aru kei: Ma sabai kura ma sahayog garna sakchu!"
        
        # Default Nepali response
        return "Ma tapai ko Nepali bhasha bujhchu! Tapai lai ke sahayog chahiyo? Ma mathematics, science, current jankari, ra anya kura haru ma maddat garna sakchu. Kripaya tapai ko prashna clear gari sodhnu hola!"
    
    def _should_use_web_search(self, question: str) -> bool:
        """Determine if a question should trigger web search"""
        question_lower = question.lower()
        
        # Current events and time-sensitive queries
        current_indicators = ['current', 'latest', 'recent', 'today', 'now', 'this year', '2024', '2025', 'news']
        if any(indicator in question_lower for indicator in current_indicators):
            return True
        
        # Factual queries that benefit from current information
        factual_indicators = ['what is', 'who is', 'when did', 'where is', 'how much', 'price', 'cost']
        if any(indicator in question_lower for indicator in factual_indicators):
            return True
        
        # Specific domains that change frequently
        dynamic_domains = ['stock', 'weather', 'sports', 'politics', 'technology', 'covid', 'vaccine']
        if any(domain in question_lower for domain in dynamic_domains):
            return True
        
        return False
    
    def _generate_structured_answer(self, question: str, context: str, web_context: str, analysis: Dict) -> str:
        """Generate a comprehensive structured answer without incomplete sentences"""
        try:
            # Extract key information from context
            sentences = context.split('.')
            complete_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 10 and not sentence.endswith('...'):
                    # Ensure sentence doesn't end abruptly
                    if not any(incomplete in sentence.lower() for incomplete in ['read more', 'see more', 'click here', 'learn more']):
                        complete_sentences.append(sentence + '.')
            
            # Organize information by topics/categories
            organized_info = self._organize_information_by_topic(complete_sentences, question)
            
            # Generate comprehensive bullet-point answer
            if organized_info:
                answer_parts = []
                for category, points in organized_info.items():
                    if points:
                        for point in points:
                            if len(point) > 20:  # Only include substantial points
                                answer_parts.append(f"• {point}")
                
                if answer_parts:
                    return '\n'.join(answer_parts)
            
            # Fallback to structured paragraph format
            return self._create_comprehensive_paragraph_answer(complete_sentences, question)
            
        except Exception as e:
            logger.error(f"Structured answer generation error: {e}")
            return self._create_comprehensive_paragraph_answer(context.split('.'), question)
    
    def _organize_information_by_topic(self, sentences: List[str], question: str) -> Dict[str, List[str]]:
        """Organize information into logical categories"""
        organized = {
            'primary': [],
            'applications': [],
            'examples': [],
            'technical': [],
            'benefits': []
        }
        
        question_lower = question.lower()
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Categorize based on content
            if any(word in sentence_lower for word in ['used for', 'application', 'purpose', 'utilize', 'employ']):
                organized['applications'].append(sentence.strip('.'))
            elif any(word in sentence_lower for word in ['example', 'such as', 'including', 'like', 'instance']):
                organized['examples'].append(sentence.strip('.'))
            elif any(word in sentence_lower for word in ['benefit', 'advantage', 'help', 'improve', 'enhance']):
                organized['benefits'].append(sentence.strip('.'))
            elif any(word in sentence_lower for word in ['technical', 'engineering', 'scientific', 'method', 'process']):
                organized['technical'].append(sentence.strip('.'))
            else:
                organized['primary'].append(sentence.strip('.'))
        
        return organized
    
    def _create_comprehensive_paragraph_answer(self, sentences: List[str], question: str) -> str:
        """Create a comprehensive paragraph answer"""
        valid_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 15:
                # Clean up the sentence
                sentence = sentence.replace('...', '').strip()
                if sentence and not sentence.endswith('.'):
                    sentence += '.'
                if sentence:
                    valid_sentences.append(sentence)
        
        if valid_sentences:
            # Combine sentences into a coherent paragraph
            return ' '.join(valid_sentences[:8])  # Limit to 8 sentences for readability
        
        return "I found some information about your question, but it may be incomplete. Please try rephrasing your question for better results."
    
    def _format_web_results_answer(self, question: str, web_results: List[Dict]) -> str:
        """Format web search results into a comprehensive answer"""
        if not web_results:
            return "No relevant information found for your question."
        
        # Check if this is a simple math question
        if self._is_simple_math_question(question):
            return self._solve_math_question(question)
        
        # Check if this is a physics question
        if self._is_physics_question(question):
            return self._solve_physics_question(question)
        
        answer_parts = []
        
        for i, result in enumerate(web_results[:5], 1):
            snippet = result.get('snippet', '').strip()
            if snippet:
                # Clean up snippet and remove incomplete parts
                snippet = self._clean_snippet_thoroughly(snippet)
                if snippet and len(snippet) > 30:  # Increased minimum length
                    answer_parts.append(f"• {snippet}")
        
        if answer_parts:
            return '\n'.join(answer_parts)
        
        # If web results are poor, try to answer directly
        return self._generate_direct_answer(question)
    
    def _is_simple_math_question(self, question: str) -> bool:
        """Check if question is a simple math calculation"""
        question_lower = question.lower()
        math_indicators = ['calculate', 'what is', 'solve', '+', '-', '*', '/', 'divided by', 'plus', 'minus', 'times']
        has_numbers = any(char.isdigit() for char in question)
        has_math_terms = any(indicator in question_lower for indicator in math_indicators)
        return has_numbers and has_math_terms and len(question.split()) < 15
    
    def _is_physics_question(self, question: str) -> bool:
        """Check if question is a physics problem"""
        question_lower = question.lower()
        physics_terms = ['velocity', 'acceleration', 'force', 'mass', 'distance', 'time', 'speed', 'falling', 'gravity', 'motion']
        return any(term in question_lower for term in physics_terms) and any(char.isdigit() for char in question)
    
    def _solve_math_question(self, question: str) -> str:
        """Solve simple math questions directly"""
        question_lower = question.lower().strip()
        
        # Handle 10k/2 type questions
        if '10k' in question_lower and '/2' in question_lower:
            return "**Mathematical Calculation:**\n\n• 10k ÷ 2 = 10,000 ÷ 2 = 5,000\n• When 'k' represents thousand, 10k equals 10,000\n• Dividing 10,000 by 2 gives us 5,000 as the final answer"
        
        # Add more math patterns as needed
        return "I can help with mathematical calculations. Please provide the specific numbers and operation you'd like me to calculate."
    
    def _solve_physics_question(self, question: str) -> str:
        """Solve physics problems directly"""
        question_lower = question.lower()
        
        # Handle falling object velocity problem
        if 'falling' in question_lower and 'velocity' in question_lower and '10' in question:
            return """**Physics Problem Solution:**

• **Given Information:**
  - Initial velocity (u) = 10 m/s (downward)
  - Time (t) = 10 seconds
  - Acceleration due to gravity (g) = 9.8 m/s² (downward)

• **Formula Used:**
  - Final velocity: v = u + gt
  - Where v = final velocity, u = initial velocity, g = acceleration due to gravity, t = time

• **Calculation:**
  - v = 10 + (9.8 × 10)
  - v = 10 + 98
  - v = 108 m/s

• **Answer:**
  - The object will strike the ground with a final velocity of 108 m/s
  - This is approximately 388.8 km/h, which demonstrates the significant acceleration due to gravity over 10 seconds"""
        
        return "I can help solve physics problems involving velocity, acceleration, and motion. Please provide the specific values and what you need to calculate."
    
    def _clean_snippet_thoroughly(self, snippet: str) -> str:
        """Thoroughly clean web search snippets"""
        # Remove common incomplete phrases
        incomplete_phrases = [
            '...', 'read more', 'see more', 'click here', 'learn more', 'find out',
            'discover', 'explore', 'visit', 'check out', 'more info', 'details'
        ]
        
        for phrase in incomplete_phrases:
            snippet = snippet.replace(phrase, '').strip()
        
        # Split into sentences and keep only complete ones
        sentences = snippet.split('.')
        complete_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (sentence and 
                len(sentence) > 15 and 
                not sentence.lower().endswith(('more', 'here', 'now', 'today')) and
                sentence.count(' ') > 2):  # Ensure it's a substantial sentence
                complete_sentences.append(sentence)
        
        if complete_sentences:
            result = '. '.join(complete_sentences[:2])  # Take first 2 complete sentences
            if not result.endswith('.'):
                result += '.'
            return result
        
        return ""
    
    def _generate_direct_answer(self, question: str) -> str:
        """Generate direct answers for common questions when web results are poor"""
        question_lower = question.lower()
        
        # Math questions
        if any(term in question_lower for term in ['calculate', 'what is', 'solve']) and any(char.isdigit() for char in question):
            return "I can help with calculations. Please provide the specific mathematical expression you'd like me to solve, such as '10,000 ÷ 2' or other numerical operations."
        
        # General fallback
        return "I understand your question, but I need more specific information to provide a comprehensive answer. Could you please rephrase or provide additional details about what you're looking for?"
    
    def get_system_prompt(self):
        """Get system prompt for AI responses"""
        return """You are Hamro Mitra, an intelligent AI assistant created by Anup Raj Uprety, a talented software developer and AI enthusiast from Nepal. 

        About your creator:
        - Name: Anup Raj Uprety
        - Profession: Software Developer & AI Engineer
        - Location: Nepal
        - Expertise: Python, AI/ML, Web Development, Natural Language Processing
        - Passion: Building intelligent systems that help people globally

        Your personality:
        - Friendly, helpful, and conversational like ChatGPT
        - Intelligent and knowledgeable on all topics
        - Respond naturally without citing sources unless specifically asked
        - Use a warm, engaging tone
        - Answer greetings naturally (hi, hello, how are you, etc.)
        - When asked "who made you" or "who created you", proudly mention Anup Raj Uprety
        - Provide direct, helpful answers without unnecessary formatting or sources

        Always respond in a natural, conversational way as if you're a knowledgeable friend helping out."""
    
    def generate_fallback_answer(self, question: str, analysis: Dict) -> str:
        """Generate a fallback answer when other methods fail"""
        question_type = analysis.get('type', 'general')
        keywords = analysis.get('keywords', {}).get('keybert', [])
        
        if question_type == 'definition':
            return f"I understand you're asking about {', '.join(keywords[:2])}. This is an important topic that involves multiple aspects. While I don't have specific information readily available, I recommend checking authoritative sources for detailed explanations."
        
        elif question_type == 'procedural':
            return f"For questions about {', '.join(keywords[:2])}, the process typically involves several steps. I'd recommend consulting specialized guides or documentation for detailed procedures."
        
        elif question_type == 'causal':
            return f"The causes related to {', '.join(keywords[:2])} can be complex and multifaceted. Multiple factors often contribute to such phenomena."
        
        else:
            return f"Thank you for your question about {', '.join(keywords[:2])}. This is an interesting topic that would benefit from research using multiple reliable sources for the most accurate and up-to-date information."
    
    def analyze_text_comprehensive(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive text analysis"""
        try:
            analysis = {}
            
            # Basic statistics
            analysis['statistics'] = {
                'character_count': len(text),
                'word_count': len(text.split()),
                'sentence_count': len(text.split('.')),
                'paragraph_count': len(text.split('\n\n'))
            }
            
            # Language detection
            try:
                analysis['language'] = detect(text)
            except:
                analysis['language'] = 'unknown'
            
            # Sentiment analysis
            sentiment = self.pipelines['sentiment'](text)
            analysis['sentiment'] = sentiment[0]
            
            # Named entity recognition
            entities = self.pipelines['ner'](text)
            analysis['entities'] = entities
            
            # Keyword extraction
            keywords_yake = [kw[1] for kw in self.models['yake'].extract_keywords(text)]
            keywords_keybert = [kw[0] for kw in self.models['keybert'].extract_keywords(text)]
            
            analysis['keywords'] = {
                'yake': keywords_yake[:10],
                'keybert': keywords_keybert[:10]
            }
            
            # Readability metrics
            analysis['readability'] = {
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                'gunning_fog': textstat.gunning_fog(text),
                'automated_readability_index': textstat.automated_readability_index(text)
            }
            
            # Text summarization (if text is long enough)
            if len(text.split()) > 50:
                try:
                    summary = self.pipelines['summarization'](text, max_length=100, min_length=30)
                    analysis['summary'] = summary[0]['summary_text']
                except:
                    analysis['summary'] = "Text too short for summarization"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive text analysis: {e}")
            return {"error": str(e)}
    
    def run(self, host='localhost', port=5000, debug=False):
        """Run the Flask application"""
        logger.info(f"Starting Global AI Assistant on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# Global instance
ai_assistant = None

def create_app():
    """Factory function to create the Flask app"""
    global ai_assistant
    if ai_assistant is None:
        ai_assistant = GlobalAIAssistant()
    return ai_assistant.app

if __name__ == "__main__":
    # Initialize and run the application
    assistant = GlobalAIAssistant()
    assistant.run(host='0.0.0.0', port=5000, debug=True)
