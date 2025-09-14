#!/usr/bin/env python3
"""
Advanced NLP Module with Sentiment Analysis, Entity Recognition, and Text Analytics
Provides comprehensive text analysis capabilities for global AI assistant
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import re

# Core NLP libraries
import spacy
import nltk
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, pipeline
)
import torch

# Specialized NLP tools
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yake
from keybert import KeyBERT
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

# Text metrics and analysis
import textstat
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Language detection and translation
from langdetect import detect, LangDetectException
from googletrans import Translator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedNLP:
    """Advanced NLP processing with sentiment analysis, NER, and text analytics"""
    
    def __init__(self):
        self.models = {}
        self.pipelines = {}
        self.analyzers = {}
        
        # Initialize components
        self._initialize_sentiment_models()
        self._initialize_ner_models()
        self._initialize_text_analytics()
        self._download_nltk_resources()
        
        logger.info("Advanced NLP module initialized successfully!")
    
    def _download_nltk_resources(self):
        """Download required NLTK resources"""
        resources = [
            'punkt', 'stopwords', 'wordnet', 'vader_lexicon',
            'averaged_perceptron_tagger', 'omw-1.4', 'brown',
            'names', 'words', 'maxent_ne_chunker'
        ]
        
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download NLTK resource {resource}: {e}")
    
    def _initialize_sentiment_models(self):
        """Initialize sentiment analysis models"""
        try:
            # VADER sentiment analyzer
            self.analyzers['vader'] = SentimentIntensityAnalyzer()
            
            # RoBERTa-based sentiment analysis
            self.pipelines['roberta_sentiment'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            # Multilingual sentiment analysis
            self.pipelines['multilingual_sentiment'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
            )
            
            # Emotion detection
            self.pipelines['emotion'] = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None
            )
            
            logger.info("Sentiment analysis models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing sentiment models: {e}")
    
    def _initialize_ner_models(self):
        """Initialize Named Entity Recognition models"""
        try:
            # Load spaCy models
            spacy_models = {
                'en': 'en_core_web_sm',
                'de': 'de_core_news_sm',
                'es': 'es_core_news_sm',
                'fr': 'fr_core_news_sm'
            }
            
            self.models['spacy'] = {}
            for lang, model_name in spacy_models.items():
                try:
                    self.models['spacy'][lang] = spacy.load(model_name)
                    logger.info(f"Loaded spaCy model for {lang}")
                except OSError:
                    logger.warning(f"spaCy model {model_name} not found")
            
            # Transformer-based NER
            self.pipelines['ner_transformer'] = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple"
            )
            
            # Multilingual NER
            self.pipelines['multilingual_ner'] = pipeline(
                "ner",
                model="Babelscape/wikineural-multilingual-ner",
                aggregation_strategy="simple"
            )
            
            logger.info("NER models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing NER models: {e}")
    
    def _initialize_text_analytics(self):
        """Initialize text analytics tools"""
        try:
            # Keyword extraction
            self.models['keybert'] = KeyBERT('distilbert-base-nli-mean-tokens')
            
            # YAKE keyword extractor
            self.models['yake'] = yake.KeywordExtractor(
                lan="en",
                n=3,
                dedupLim=0.7,
                top=20
            )
            
            # Text summarizers
            self.models['lsa_summarizer'] = LsaSummarizer()
            self.models['textrank_summarizer'] = TextRankSummarizer()
            
            # Language translator
            self.models['translator'] = Translator()
            
            logger.info("Text analytics tools initialized")
            
        except Exception as e:
            logger.error(f"Error initializing text analytics: {e}")
    
    def analyze_sentiment_comprehensive(self, text: str, language: str = 'auto') -> Dict[str, Any]:
        """Comprehensive sentiment analysis using multiple models"""
        try:
            results = {}
            
            # Detect language if auto
            if language == 'auto':
                try:
                    language = detect(text)
                except:
                    language = 'en'
            
            results['detected_language'] = language
            
            # VADER sentiment (works well with social media text)
            vader_scores = self.analyzers['vader'].polarity_scores(text)
            results['vader'] = {
                'compound': vader_scores['compound'],
                'positive': vader_scores['pos'],
                'neutral': vader_scores['neu'],
                'negative': vader_scores['neg'],
                'classification': self._classify_vader_sentiment(vader_scores['compound'])
            }
            
            # TextBlob sentiment
            blob = TextBlob(text)
            results['textblob'] = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'classification': self._classify_textblob_sentiment(blob.sentiment.polarity)
            }
            
            # RoBERTa sentiment
            try:
                roberta_result = self.pipelines['roberta_sentiment'](text)[0]
                results['roberta'] = {
                    'label': roberta_result['label'],
                    'score': roberta_result['score'],
                    'classification': roberta_result['label'].lower()
                }
            except Exception as e:
                logger.warning(f"RoBERTa sentiment failed: {e}")
                results['roberta'] = {'error': str(e)}
            
            # Multilingual sentiment (if not English)
            if language != 'en':
                try:
                    multilingual_result = self.pipelines['multilingual_sentiment'](text)[0]
                    results['multilingual'] = {
                        'label': multilingual_result['label'],
                        'score': multilingual_result['score']
                    }
                except Exception as e:
                    logger.warning(f"Multilingual sentiment failed: {e}")
                    results['multilingual'] = {'error': str(e)}
            
            # Emotion detection
            try:
                emotions = self.pipelines['emotion'](text)
                results['emotions'] = emotions
                results['dominant_emotion'] = max(emotions, key=lambda x: x['score'])
            except Exception as e:
                logger.warning(f"Emotion detection failed: {e}")
                results['emotions'] = {'error': str(e)}
            
            # Overall sentiment consensus
            results['consensus'] = self._calculate_sentiment_consensus(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive sentiment analysis: {e}")
            return {'error': str(e)}
    
    def _classify_vader_sentiment(self, compound_score: float) -> str:
        """Classify VADER compound score"""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def _classify_textblob_sentiment(self, polarity: float) -> str:
        """Classify TextBlob polarity score"""
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _calculate_sentiment_consensus(self, results: Dict) -> Dict[str, Any]:
        """Calculate consensus sentiment from multiple models"""
        try:
            sentiments = []
            scores = []
            
            # Collect sentiment classifications
            if 'vader' in results and 'classification' in results['vader']:
                sentiments.append(results['vader']['classification'])
                scores.append(results['vader']['compound'])
            
            if 'textblob' in results and 'classification' in results['textblob']:
                sentiments.append(results['textblob']['classification'])
                scores.append(results['textblob']['polarity'])
            
            if 'roberta' in results and 'classification' in results['roberta']:
                sentiments.append(results['roberta']['classification'])
                # Convert RoBERTa score to -1 to 1 scale
                roberta_score = results['roberta']['score']
                if results['roberta']['label'] == 'LABEL_2':  # Positive
                    scores.append(roberta_score)
                elif results['roberta']['label'] == 'LABEL_0':  # Negative
                    scores.append(-roberta_score)
                else:  # Neutral
                    scores.append(0)
            
            # Calculate consensus
            if sentiments:
                # Most common sentiment
                consensus_sentiment = max(set(sentiments), key=sentiments.count)
                
                # Average score
                consensus_score = np.mean(scores) if scores else 0
                
                # Confidence based on agreement
                agreement_ratio = sentiments.count(consensus_sentiment) / len(sentiments)
                
                return {
                    'sentiment': consensus_sentiment,
                    'score': consensus_score,
                    'confidence': agreement_ratio,
                    'model_agreement': f"{sentiments.count(consensus_sentiment)}/{len(sentiments)}"
                }
            
            return {'sentiment': 'neutral', 'score': 0, 'confidence': 0}
            
        except Exception as e:
            logger.error(f"Error calculating sentiment consensus: {e}")
            return {'error': str(e)}
    
    def extract_entities_comprehensive(self, text: str, language: str = 'auto') -> Dict[str, Any]:
        """Comprehensive named entity recognition"""
        try:
            results = {}
            
            # Detect language
            if language == 'auto':
                try:
                    language = detect(text)
                except:
                    language = 'en'
            
            results['detected_language'] = language
            
            # spaCy NER (if model available)
            if language in self.models.get('spacy', {}):
                try:
                    doc = self.models['spacy'][language](text)
                    results['spacy'] = {
                        'entities': [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents],
                        'tokens': [(token.text, token.pos_, token.tag_) for token in doc],
                        'noun_phrases': [chunk.text for chunk in doc.noun_chunks]
                    }
                except Exception as e:
                    logger.warning(f"spaCy NER failed: {e}")
                    results['spacy'] = {'error': str(e)}
            
            # Transformer-based NER
            try:
                transformer_entities = self.pipelines['ner_transformer'](text)
                results['transformer'] = {
                    'entities': [(ent['word'], ent['entity_group'], ent['score']) for ent in transformer_entities]
                }
            except Exception as e:
                logger.warning(f"Transformer NER failed: {e}")
                results['transformer'] = {'error': str(e)}
            
            # Multilingual NER (if not English)
            if language != 'en':
                try:
                    multilingual_entities = self.pipelines['multilingual_ner'](text)
                    results['multilingual'] = {
                        'entities': [(ent['word'], ent['entity_group'], ent['score']) for ent in multilingual_entities]
                    }
                except Exception as e:
                    logger.warning(f"Multilingual NER failed: {e}")
                    results['multilingual'] = {'error': str(e)}
            
            # NLTK NER (basic)
            try:
                tokens = nltk.word_tokenize(text)
                pos_tags = nltk.pos_tag(tokens)
                named_entities = nltk.ne_chunk(pos_tags)
                
                nltk_entities = []
                for chunk in named_entities:
                    if hasattr(chunk, 'label'):
                        entity_text = ' '.join([token for token, pos in chunk.leaves()])
                        nltk_entities.append((entity_text, chunk.label()))
                
                results['nltk'] = {'entities': nltk_entities}
                
            except Exception as e:
                logger.warning(f"NLTK NER failed: {e}")
                results['nltk'] = {'error': str(e)}
            
            # Consolidate entities
            results['consolidated'] = self._consolidate_entities(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive entity extraction: {e}")
            return {'error': str(e)}
    
    def _consolidate_entities(self, ner_results: Dict) -> Dict[str, List]:
        """Consolidate entities from different NER models"""
        try:
            consolidated = {
                'PERSON': [],
                'ORGANIZATION': [],
                'LOCATION': [],
                'MISCELLANEOUS': [],
                'DATE': [],
                'MONEY': [],
                'PERCENT': []
            }
            
            # Process spaCy entities
            if 'spacy' in ner_results and 'entities' in ner_results['spacy']:
                for entity, label, start, end in ner_results['spacy']['entities']:
                    category = self._map_entity_label(label)
                    if category in consolidated:
                        consolidated[category].append({
                            'text': entity,
                            'label': label,
                            'source': 'spacy',
                            'confidence': 0.9
                        })
            
            # Process transformer entities
            if 'transformer' in ner_results and 'entities' in ner_results['transformer']:
                for entity, label, score in ner_results['transformer']['entities']:
                    category = self._map_entity_label(label)
                    if category in consolidated:
                        consolidated[category].append({
                            'text': entity,
                            'label': label,
                            'source': 'transformer',
                            'confidence': score
                        })
            
            # Remove duplicates and sort by confidence
            for category in consolidated:
                # Remove duplicates based on text similarity
                unique_entities = []
                seen_texts = set()
                
                for entity in consolidated[category]:
                    entity_text = entity['text'].lower().strip()
                    if entity_text not in seen_texts:
                        seen_texts.add(entity_text)
                        unique_entities.append(entity)
                
                # Sort by confidence
                unique_entities.sort(key=lambda x: x['confidence'], reverse=True)
                consolidated[category] = unique_entities
            
            return consolidated
            
        except Exception as e:
            logger.error(f"Error consolidating entities: {e}")
            return {}
    
    def _map_entity_label(self, label: str) -> str:
        """Map various entity labels to standard categories"""
        label_mapping = {
            # spaCy labels
            'PERSON': 'PERSON',
            'ORG': 'ORGANIZATION',
            'GPE': 'LOCATION',
            'LOC': 'LOCATION',
            'DATE': 'DATE',
            'TIME': 'DATE',
            'MONEY': 'MONEY',
            'PERCENT': 'PERCENT',
            'MISC': 'MISCELLANEOUS',
            
            # Transformer labels
            'PER': 'PERSON',
            'ORG': 'ORGANIZATION',
            'LOC': 'LOCATION',
            'MISC': 'MISCELLANEOUS'
        }
        
        return label_mapping.get(label, 'MISCELLANEOUS')
    
    def extract_keywords_comprehensive(self, text: str) -> Dict[str, Any]:
        """Comprehensive keyword extraction using multiple methods"""
        try:
            results = {}
            
            # YAKE keyword extraction
            try:
                yake_keywords = self.models['yake'].extract_keywords(text)
                results['yake'] = [
                    {'keyword': kw[1], 'score': kw[0], 'method': 'yake'}
                    for kw in yake_keywords[:15]
                ]
            except Exception as e:
                logger.warning(f"YAKE extraction failed: {e}")
                results['yake'] = {'error': str(e)}
            
            # KeyBERT extraction
            try:
                keybert_keywords = self.models['keybert'].extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 3),
                    stop_words='english',
                    top_k=15
                )
                results['keybert'] = [
                    {'keyword': kw[0], 'score': kw[1], 'method': 'keybert'}
                    for kw in keybert_keywords
                ]
            except Exception as e:
                logger.warning(f"KeyBERT extraction failed: {e}")
                results['keybert'] = {'error': str(e)}
            
            # TF-IDF based extraction
            try:
                # Simple TF-IDF for single document
                from sklearn.feature_extraction.text import TfidfVectorizer
                from collections import Counter
                
                # Tokenize and get word frequencies
                words = nltk.word_tokenize(text.lower())
                words = [word for word in words if word.isalnum() and len(word) > 2]
                
                # Remove stopwords
                stop_words = set(nltk.corpus.stopwords.words('english'))
                words = [word for word in words if word not in stop_words]
                
                # Get most frequent words
                word_freq = Counter(words)
                results['frequency'] = [
                    {'keyword': word, 'score': freq, 'method': 'frequency'}
                    for word, freq in word_freq.most_common(15)
                ]
                
            except Exception as e:
                logger.warning(f"Frequency extraction failed: {e}")
                results['frequency'] = {'error': str(e)}
            
            # Consolidate keywords
            results['consolidated'] = self._consolidate_keywords(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive keyword extraction: {e}")
            return {'error': str(e)}
    
    def _consolidate_keywords(self, keyword_results: Dict) -> List[Dict]:
        """Consolidate keywords from different extraction methods"""
        try:
            all_keywords = []
            
            # Collect all keywords
            for method, keywords in keyword_results.items():
                if method != 'consolidated' and isinstance(keywords, list):
                    all_keywords.extend(keywords)
            
            # Group by keyword text
            keyword_groups = {}
            for kw in all_keywords:
                text = kw['keyword'].lower()
                if text not in keyword_groups:
                    keyword_groups[text] = []
                keyword_groups[text].append(kw)
            
            # Calculate consolidated scores
            consolidated = []
            for text, group in keyword_groups.items():
                # Average score across methods
                avg_score = np.mean([kw['score'] for kw in group])
                methods = [kw['method'] for kw in group]
                
                consolidated.append({
                    'keyword': group[0]['keyword'],  # Use original case
                    'score': avg_score,
                    'methods': methods,
                    'method_count': len(set(methods))
                })
            
            # Sort by score and method agreement
            consolidated.sort(key=lambda x: (x['method_count'], x['score']), reverse=True)
            
            return consolidated[:20]  # Return top 20
            
        except Exception as e:
            logger.error(f"Error consolidating keywords: {e}")
            return []
    
    def summarize_text(self, text: str, method: str = 'auto', sentences: int = 3) -> Dict[str, Any]:
        """Text summarization using multiple methods"""
        try:
            results = {}
            
            # Prepare text
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            
            # LSA summarization
            try:
                lsa_summary = self.models['lsa_summarizer'](parser.document, sentences)
                results['lsa'] = ' '.join([str(sentence) for sentence in lsa_summary])
            except Exception as e:
                logger.warning(f"LSA summarization failed: {e}")
                results['lsa'] = {'error': str(e)}
            
            # TextRank summarization
            try:
                textrank_summary = self.models['textrank_summarizer'](parser.document, sentences)
                results['textrank'] = ' '.join([str(sentence) for sentence in textrank_summary])
            except Exception as e:
                logger.warning(f"TextRank summarization failed: {e}")
                results['textrank'] = {'error': str(e)}
            
            # Simple extractive summarization
            try:
                sentences_list = nltk.sent_tokenize(text)
                if len(sentences_list) <= sentences:
                    results['extractive'] = text
                else:
                    # Score sentences by keyword frequency
                    word_freq = {}
                    words = nltk.word_tokenize(text.lower())
                    for word in words:
                        if word.isalnum():
                            word_freq[word] = word_freq.get(word, 0) + 1
                    
                    sentence_scores = {}
                    for sentence in sentences_list:
                        words_in_sentence = nltk.word_tokenize(sentence.lower())
                        score = sum(word_freq.get(word, 0) for word in words_in_sentence)
                        sentence_scores[sentence] = score
                    
                    # Get top sentences
                    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:sentences]
                    results['extractive'] = ' '.join([sentence for sentence, score in top_sentences])
                    
            except Exception as e:
                logger.warning(f"Extractive summarization failed: {e}")
                results['extractive'] = {'error': str(e)}
            
            # Choose best summary
            if method == 'auto':
                # Prefer LSA if available, then TextRank, then extractive
                if 'lsa' in results and isinstance(results['lsa'], str):
                    results['best'] = results['lsa']
                    results['best_method'] = 'lsa'
                elif 'textrank' in results and isinstance(results['textrank'], str):
                    results['best'] = results['textrank']
                    results['best_method'] = 'textrank'
                elif 'extractive' in results and isinstance(results['extractive'], str):
                    results['best'] = results['extractive']
                    results['best_method'] = 'extractive'
                else:
                    results['best'] = text[:500] + "..." if len(text) > 500 else text
                    results['best_method'] = 'truncation'
            else:
                results['best'] = results.get(method, text)
                results['best_method'] = method
            
            return results
            
        except Exception as e:
            logger.error(f"Error in text summarization: {e}")
            return {'error': str(e)}
    
    def analyze_text_comprehensive(self, text: str, language: str = 'auto') -> Dict[str, Any]:
        """Comprehensive text analysis combining all NLP capabilities"""
        try:
            analysis = {}
            
            # Basic statistics
            analysis['statistics'] = {
                'character_count': len(text),
                'word_count': len(text.split()),
                'sentence_count': len(nltk.sent_tokenize(text)),
                'paragraph_count': len(text.split('\n\n')),
                'average_word_length': np.mean([len(word) for word in text.split()]),
                'average_sentence_length': len(text.split()) / len(nltk.sent_tokenize(text)) if nltk.sent_tokenize(text) else 0
            }
            
            # Language detection
            if language == 'auto':
                try:
                    language = detect(text)
                except:
                    language = 'en'
            analysis['language'] = language
            
            # Readability metrics
            analysis['readability'] = {
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                'gunning_fog': textstat.gunning_fog(text),
                'automated_readability_index': textstat.automated_readability_index(text),
                'coleman_liau_index': textstat.coleman_liau_index(text),
                'reading_time_minutes': textstat.reading_time(text, ms_per_char=14.69)
            }
            
            # Sentiment analysis
            analysis['sentiment'] = self.analyze_sentiment_comprehensive(text, language)
            
            # Entity recognition
            analysis['entities'] = self.extract_entities_comprehensive(text, language)
            
            # Keyword extraction
            analysis['keywords'] = self.extract_keywords_comprehensive(text)
            
            # Text summarization (if text is long enough)
            if len(text.split()) > 50:
                analysis['summary'] = self.summarize_text(text)
            
            # Text classification (topic modeling would go here)
            analysis['classification'] = self._classify_text_topic(text)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive text analysis: {e}")
            return {'error': str(e)}
    
    def _classify_text_topic(self, text: str) -> Dict[str, Any]:
        """Basic text topic classification"""
        try:
            # Simple keyword-based topic classification
            topics = {
                'technology': ['computer', 'software', 'AI', 'artificial intelligence', 'machine learning', 'programming', 'code', 'algorithm', 'data', 'digital'],
                'science': ['research', 'study', 'experiment', 'hypothesis', 'theory', 'scientific', 'discovery', 'analysis', 'method'],
                'business': ['company', 'market', 'profit', 'revenue', 'customer', 'business', 'economy', 'financial', 'investment'],
                'health': ['medical', 'health', 'doctor', 'patient', 'treatment', 'disease', 'medicine', 'hospital', 'therapy'],
                'politics': ['government', 'political', 'election', 'policy', 'law', 'congress', 'president', 'vote', 'democracy'],
                'sports': ['game', 'team', 'player', 'sport', 'match', 'score', 'championship', 'tournament', 'athlete'],
                'entertainment': ['movie', 'music', 'celebrity', 'film', 'show', 'entertainment', 'actor', 'artist', 'performance']
            }
            
            text_lower = text.lower()
            topic_scores = {}
            
            for topic, keywords in topics.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    topic_scores[topic] = score
            
            if topic_scores:
                primary_topic = max(topic_scores, key=topic_scores.get)
                return {
                    'primary_topic': primary_topic,
                    'topic_scores': topic_scores,
                    'confidence': topic_scores[primary_topic] / len(topics[primary_topic])
                }
            else:
                return {
                    'primary_topic': 'general',
                    'topic_scores': {},
                    'confidence': 0.0
                }
                
        except Exception as e:
            logger.error(f"Error in topic classification: {e}")
            return {'error': str(e)}

# Global instance
advanced_nlp = None

def get_advanced_nlp():
    """Get or create the global advanced NLP instance"""
    global advanced_nlp
    if advanced_nlp is None:
        advanced_nlp = AdvancedNLP()
    return advanced_nlp
