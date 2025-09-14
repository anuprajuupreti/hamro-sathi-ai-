#!/usr/bin/env python3
"""
Multilingual AI Module for Global Question Answering
Supports 100+ languages with advanced NLP capabilities
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Core AI libraries
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    M2M100ForConditionalGeneration, M2M100Tokenizer,
    MBartForConditionalGeneration, MBart50TokenizerFast,
    pipeline
)
from sentence_transformers import SentenceTransformer
import spacy
from langdetect import detect, LangDetectException
from googletrans import Translator

# Specialized multilingual models
import polyglot
from polyglot.detect import Detector
from polyglot.text import Text

# Knowledge and information retrieval
import wikipedia
import requests
from bs4 import BeautifulSoup
import feedparser

# Data processing
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultilingualAI:
    """Advanced multilingual AI system for global question answering"""
    
    def __init__(self):
        self.models = {}
        self.pipelines = {}
        self.translators = {}
        self.language_codes = self._get_supported_languages()
        
        # Initialize multilingual models
        self._initialize_multilingual_models()
        
        # Initialize knowledge sources
        self._initialize_global_knowledge()
        
        logger.info("Multilingual AI system initialized successfully!")
    
    def _get_supported_languages(self) -> Dict[str, str]:
        """Get supported language codes and names"""
        return {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
            'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi',
            'bn': 'Bengali', 'ur': 'Urdu', 'fa': 'Persian', 'tr': 'Turkish',
            'nl': 'Dutch', 'sv': 'Swedish', 'no': 'Norwegian', 'da': 'Danish',
            'fi': 'Finnish', 'pl': 'Polish', 'cs': 'Czech', 'sk': 'Slovak',
            'hu': 'Hungarian', 'ro': 'Romanian', 'bg': 'Bulgarian', 'hr': 'Croatian',
            'sr': 'Serbian', 'sl': 'Slovenian', 'et': 'Estonian', 'lv': 'Latvian',
            'lt': 'Lithuanian', 'el': 'Greek', 'he': 'Hebrew', 'th': 'Thai',
            'vi': 'Vietnamese', 'id': 'Indonesian', 'ms': 'Malay', 'tl': 'Filipino',
            'sw': 'Swahili', 'am': 'Amharic', 'yo': 'Yoruba', 'ig': 'Igbo',
            'ha': 'Hausa', 'zu': 'Zulu', 'af': 'Afrikaans', 'ne': 'Nepali',
            'si': 'Sinhala', 'my': 'Myanmar', 'km': 'Khmer', 'lo': 'Lao',
            'ka': 'Georgian', 'hy': 'Armenian', 'az': 'Azerbaijani', 'kk': 'Kazakh',
            'ky': 'Kyrgyz', 'uz': 'Uzbek', 'tg': 'Tajik', 'mn': 'Mongolian'
        }
    
    def _initialize_multilingual_models(self):
        """Initialize multilingual AI models"""
        try:
            # Multilingual sentence transformer
            logger.info("Loading multilingual sentence transformer...")
            self.models['multilingual_embeddings'] = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            # M2M100 for translation (supports 100 languages)
            logger.info("Loading M2M100 translation model...")
            self.models['m2m100_tokenizer'] = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
            self.models['m2m100_model'] = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
            
            # mBERT for multilingual understanding
            logger.info("Loading multilingual BERT...")
            self.pipelines['multilingual_qa'] = pipeline(
                "question-answering",
                model="deepset/multilingual-bert-base-uncased-squad2",
                tokenizer="deepset/multilingual-bert-base-uncased-squad2"
            )
            
            # Multilingual text classification
            self.pipelines['multilingual_sentiment'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
            )
            
            # Multilingual NER
            self.pipelines['multilingual_ner'] = pipeline(
                "ner",
                model="Babelscape/wikineural-multilingual-ner",
                aggregation_strategy="simple"
            )
            
            # Google Translator as fallback
            self.translators['google'] = Translator()
            
            # Load spaCy multilingual models if available
            self._load_spacy_models()
            
        except Exception as e:
            logger.error(f"Error initializing multilingual models: {e}")
    
    def _load_spacy_models(self):
        """Load available spaCy models for different languages"""
        spacy_models = {
            'en': 'en_core_web_sm',
            'de': 'de_core_news_sm',
            'es': 'es_core_news_sm',
            'fr': 'fr_core_news_sm',
            'it': 'it_core_news_sm',
            'pt': 'pt_core_news_sm',
            'nl': 'nl_core_news_sm',
            'zh': 'zh_core_web_sm',
            'ja': 'ja_core_news_sm'
        }
        
        self.models['spacy'] = {}
        for lang, model_name in spacy_models.items():
            try:
                self.models['spacy'][lang] = spacy.load(model_name)
                logger.info(f"Loaded spaCy model for {lang}")
            except OSError:
                logger.warning(f"spaCy model {model_name} not found for {lang}")
    
    def _initialize_global_knowledge(self):
        """Initialize global knowledge sources"""
        self.knowledge_sources = {
            'wikipedia_languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar', 'hi'],
            'news_sources': {
                'global': [
                    'https://feeds.bbci.co.uk/news/world/rss.xml',
                    'https://rss.cnn.com/rss/edition.rss',
                    'https://feeds.reuters.com/reuters/topNews'
                ],
                'regional': {
                    'asia': 'https://feeds.bbci.co.uk/news/world/asia/rss.xml',
                    'europe': 'https://feeds.bbci.co.uk/news/world/europe/rss.xml',
                    'africa': 'https://feeds.bbci.co.uk/news/world/africa/rss.xml',
                    'americas': 'https://feeds.bbci.co.uk/news/world/latin_america/rss.xml'
                }
            }
        }
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language with confidence score"""
        try:
            # Primary detection with langdetect
            lang = detect(text)
            confidence = 0.9  # langdetect doesn't provide confidence, assume high
            
            # Validate with polyglot if available
            try:
                detector = Detector(text)
                if detector.language.code == lang:
                    confidence = min(detector.language.confidence, 1.0)
            except:
                pass
            
            return lang, confidence
            
        except LangDetectException:
            # Fallback to English
            return 'en', 0.5
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return 'en', 0.3
    
    def translate_text(self, text: str, target_lang: str, source_lang: str = None) -> str:
        """Translate text using multiple translation methods"""
        try:
            if not source_lang:
                source_lang, _ = self.detect_language(text)
            
            if source_lang == target_lang:
                return text
            
            # Try M2M100 first (more accurate for supported languages)
            if source_lang in ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar', 'hi']:
                try:
                    return self._translate_with_m2m100(text, source_lang, target_lang)
                except Exception as e:
                    logger.warning(f"M2M100 translation failed: {e}")
            
            # Fallback to Google Translate
            try:
                result = self.translators['google'].translate(text, src=source_lang, dest=target_lang)
                return result.text
            except Exception as e:
                logger.error(f"Google Translate failed: {e}")
                return text
                
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text
    
    def _translate_with_m2m100(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using M2M100 model"""
        try:
            # Set source language
            self.models['m2m100_tokenizer'].src_lang = source_lang
            
            # Encode text
            encoded = self.models['m2m100_tokenizer'](text, return_tensors="pt")
            
            # Generate translation
            generated_tokens = self.models['m2m100_model'].generate(
                **encoded,
                forced_bos_token_id=self.models['m2m100_tokenizer'].get_lang_id(target_lang),
                max_length=512
            )
            
            # Decode translation
            translation = self.models['m2m100_tokenizer'].batch_decode(
                generated_tokens, skip_special_tokens=True
            )[0]
            
            return translation
            
        except Exception as e:
            logger.error(f"M2M100 translation error: {e}")
            raise
    
    def process_multilingual_question(self, question: str, context: str = "") -> Dict[str, Any]:
        """Process question in any language"""
        try:
            # Detect language
            detected_lang, confidence = self.detect_language(question)
            
            # Translate to English for processing if needed
            question_en = question
            if detected_lang != 'en':
                question_en = self.translate_text(question, 'en', detected_lang)
            
            # Get multilingual context if available
            multilingual_context = self._get_multilingual_context(question_en, detected_lang)
            
            # Combine contexts
            full_context = context
            if multilingual_context:
                full_context = f"{context}\n{multilingual_context}" if context else multilingual_context
            
            # Process with multilingual QA
            if full_context:
                qa_result = self.pipelines['multilingual_qa'](
                    question=question if detected_lang in ['en', 'es', 'fr', 'de', 'it', 'pt'] else question_en,
                    context=full_context
                )
            else:
                qa_result = {"answer": "", "score": 0.0}
            
            # Get additional information
            wiki_info = self._get_multilingual_wikipedia(question_en, detected_lang)
            news_info = self._get_relevant_news(question_en)
            
            # Generate comprehensive answer
            answer = self._generate_multilingual_answer(
                question_en, qa_result, wiki_info, news_info, detected_lang
            )
            
            # Translate answer back if needed
            if detected_lang != 'en' and answer:
                answer = self.translate_text(answer, detected_lang, 'en')
            
            # Analyze question in original language
            analysis = self._analyze_multilingual_text(question, detected_lang)
            
            return {
                "original_question": question,
                "detected_language": detected_lang,
                "language_confidence": confidence,
                "language_name": self.language_codes.get(detected_lang, "Unknown"),
                "question_english": question_en if detected_lang != 'en' else None,
                "answer": answer,
                "analysis": analysis,
                "sources": {
                    "qa_model": qa_result,
                    "wikipedia": wiki_info,
                    "news": news_info
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing multilingual question: {e}")
            return {
                "original_question": question,
                "answer": f"I apologize, but I encountered an error processing your question: {str(e)}",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_multilingual_context(self, question: str, lang: str) -> str:
        """Get context in multiple languages"""
        try:
            contexts = []
            
            # Get Wikipedia context in original language
            if lang in self.knowledge_sources['wikipedia_languages']:
                try:
                    wikipedia.set_lang(lang)
                    search_results = wikipedia.search(question, results=2)
                    if search_results:
                        summary = wikipedia.summary(search_results[0], sentences=2)
                        contexts.append(summary)
                except:
                    pass
            
            # Get English Wikipedia context
            try:
                wikipedia.set_lang('en')
                search_results = wikipedia.search(question, results=2)
                if search_results:
                    summary = wikipedia.summary(search_results[0], sentences=2)
                    contexts.append(summary)
            except:
                pass
            
            return " ".join(contexts)
            
        except Exception as e:
            logger.error(f"Error getting multilingual context: {e}")
            return ""
    
    def _get_multilingual_wikipedia(self, question: str, lang: str) -> Dict[str, Any]:
        """Get Wikipedia information in multiple languages"""
        try:
            results = {}
            
            # Try original language first
            if lang in self.knowledge_sources['wikipedia_languages']:
                try:
                    wikipedia.set_lang(lang)
                    search_results = wikipedia.search(question, results=3)
                    if search_results:
                        page = wikipedia.page(search_results[0])
                        results[lang] = {
                            'title': page.title,
                            'summary': wikipedia.summary(search_results[0], sentences=3),
                            'url': page.url
                        }
                except:
                    pass
            
            # Always try English
            try:
                wikipedia.set_lang('en')
                search_results = wikipedia.search(question, results=3)
                if search_results:
                    page = wikipedia.page(search_results[0])
                    results['en'] = {
                        'title': page.title,
                        'summary': wikipedia.summary(search_results[0], sentences=3),
                        'url': page.url
                    }
            except:
                pass
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting multilingual Wikipedia: {e}")
            return {}
    
    def _get_relevant_news(self, question: str) -> Dict[str, Any]:
        """Get relevant news from global sources"""
        try:
            news_items = []
            
            # Fetch from global news sources
            for source_url in self.knowledge_sources['news_sources']['global']:
                try:
                    feed = feedparser.parse(source_url)
                    for entry in feed.entries[:5]:  # Get top 5 items
                        # Simple relevance check
                        if any(word.lower() in entry.title.lower() or word.lower() in entry.summary.lower() 
                               for word in question.split() if len(word) > 3):
                            news_items.append({
                                'title': entry.title,
                                'summary': entry.summary[:200] + "..." if len(entry.summary) > 200 else entry.summary,
                                'link': entry.link,
                                'published': entry.published if hasattr(entry, 'published') else None
                            })
                except Exception as e:
                    logger.warning(f"Error fetching news from {source_url}: {e}")
            
            return {
                'relevant_news': news_items[:3],  # Return top 3 relevant items
                'total_found': len(news_items)
            }
            
        except Exception as e:
            logger.error(f"Error getting relevant news: {e}")
            return {}
    
    def _analyze_multilingual_text(self, text: str, lang: str) -> Dict[str, Any]:
        """Analyze text in its original language"""
        try:
            analysis = {}
            
            # Basic statistics
            analysis['statistics'] = {
                'character_count': len(text),
                'word_count': len(text.split()),
                'language': lang,
                'language_name': self.language_codes.get(lang, "Unknown")
            }
            
            # Sentiment analysis (works with multiple languages)
            try:
                sentiment = self.pipelines['multilingual_sentiment'](text)
                analysis['sentiment'] = sentiment[0]
            except:
                analysis['sentiment'] = {"label": "UNKNOWN", "score": 0.0}
            
            # Named Entity Recognition
            try:
                entities = self.pipelines['multilingual_ner'](text)
                analysis['entities'] = entities
            except:
                analysis['entities'] = []
            
            # spaCy analysis if model available
            if lang in self.models.get('spacy', {}):
                try:
                    doc = self.models['spacy'][lang](text)
                    analysis['spacy'] = {
                        'tokens': [token.text for token in doc],
                        'pos_tags': [(token.text, token.pos_) for token in doc],
                        'lemmas': [token.lemma_ for token in doc],
                        'entities': [(ent.text, ent.label_) for ent in doc.ents]
                    }
                except:
                    pass
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing multilingual text: {e}")
            return {"error": str(e)}
    
    def _generate_multilingual_answer(self, question: str, qa_result: Dict, 
                                    wiki_info: Dict, news_info: Dict, target_lang: str) -> str:
        """Generate comprehensive answer using all sources"""
        try:
            answer_parts = []
            
            # Use QA model result if confident
            if qa_result.get('score', 0) > 0.3 and qa_result.get('answer'):
                answer_parts.append(qa_result['answer'])
            
            # Add Wikipedia information
            for lang, info in wiki_info.items():
                if info.get('summary'):
                    answer_parts.append(f"According to Wikipedia: {info['summary']}")
                    break  # Use first available summary
            
            # Add relevant news if available
            if news_info.get('relevant_news'):
                news_item = news_info['relevant_news'][0]
                answer_parts.append(f"Recent news: {news_item['title']} - {news_item['summary'][:100]}...")
            
            # Combine all parts
            if answer_parts:
                return " ".join(answer_parts)
            else:
                return "I apologize, but I couldn't find specific information to answer your question. Could you please provide more context or rephrase your question?"
                
        except Exception as e:
            logger.error(f"Error generating multilingual answer: {e}")
            return "I encountered an error while generating the answer. Please try again."
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages"""
        return self.language_codes
    
    def get_language_stats(self) -> Dict[str, Any]:
        """Get statistics about language support"""
        return {
            "total_languages": len(self.language_codes),
            "translation_models": ["M2M100", "Google Translate"],
            "qa_languages": ["en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "ar", "hi", "th", "zh", "ja", "ko"],
            "spacy_languages": list(self.models.get('spacy', {}).keys()),
            "wikipedia_languages": self.knowledge_sources['wikipedia_languages']
        }

# Global instance
multilingual_ai = None

def get_multilingual_ai():
    """Get or create the global multilingual AI instance"""
    global multilingual_ai
    if multilingual_ai is None:
        multilingual_ai = MultilingualAI()
    return multilingual_ai
