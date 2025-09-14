#!/usr/bin/env python3
"""
Advanced Knowledge Engine with Vector Database and Real-time Information Retrieval
Implements RAG (Retrieval-Augmented Generation) for global question answering
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp

# Vector database and embeddings
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Web scraping and information retrieval
import requests
from bs4 import BeautifulSoup
import feedparser
import newspaper
from newspaper import Article

# Data processing
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Text processing
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeEngine:
    """Advanced knowledge retrieval and management system"""
    
    def __init__(self):
        self.embeddings_model = None
        self.vector_db = None
        self.chroma_client = None
        self.knowledge_cache = {}
        self.web_cache = {}
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_vector_db()
        self._initialize_knowledge_sources()
        
        logger.info("Knowledge Engine initialized successfully!")
    
    def _initialize_embeddings(self):
        """Initialize embedding models"""
        try:
            logger.info("Loading sentence transformer for embeddings...")
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Test embedding generation
            test_embedding = self.embeddings_model.encode(["test sentence"])
            self.embedding_dimension = test_embedding.shape[1]
            logger.info(f"Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
    
    def _initialize_vector_db(self):
        """Initialize vector database (ChromaDB and FAISS)"""
        try:
            # Initialize ChromaDB
            self.chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chroma_db"
            ))
            
            # Create or get collection
            try:
                self.vector_db = self.chroma_client.get_collection("knowledge_base")
                logger.info("Loaded existing ChromaDB collection")
            except:
                self.vector_db = self.chroma_client.create_collection("knowledge_base")
                logger.info("Created new ChromaDB collection")
            
            # Initialize FAISS index as backup
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dimension)
            self.faiss_texts = []
            
        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
    
    def _initialize_knowledge_sources(self):
        """Initialize various knowledge sources"""
        self.knowledge_sources = {
            'news_feeds': [
                'https://feeds.bbci.co.uk/news/world/rss.xml',
                'https://rss.cnn.com/rss/edition.rss',
                'https://feeds.reuters.com/reuters/topNews',
                'https://feeds.npr.org/1001/rss.xml',
                'https://feeds.washingtonpost.com/rss/world',
                'https://www.theguardian.com/world/rss'
            ],
            'tech_feeds': [
                'https://feeds.feedburner.com/TechCrunch',
                'https://feeds.arstechnica.com/arstechnica/index',
                'https://www.wired.com/feed/rss',
                'https://feeds.mashable.com/Mashable'
            ],
            'science_feeds': [
                'https://feeds.nature.com/nature/rss/current',
                'https://feeds.sciencedaily.com/sciencedaily/top_news',
                'https://www.science.org/rss/news_current.xml'
            ],
            'search_engines': [
                'https://duckduckgo.com/html/?q=',
                'https://www.bing.com/search?q='
            ]
        }
        
        # Load initial knowledge base
        self._load_initial_knowledge()
    
    def _load_initial_knowledge(self):
        """Load initial knowledge base with common facts"""
        initial_knowledge = [
            {
                'text': 'Python is a high-level, interpreted programming language with dynamic semantics. Its high-level built-in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development.',
                'category': 'programming',
                'source': 'python.org',
                'timestamp': datetime.now().isoformat()
            },
            {
                'text': 'Artificial Intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of intelligent agents.',
                'category': 'technology',
                'source': 'academic',
                'timestamp': datetime.now().isoformat()
            },
            {
                'text': 'Climate change refers to long-term shifts in global temperatures and weather patterns. While climate changes are natural, since the 1800s, human activities have been the main driver of climate change.',
                'category': 'environment',
                'source': 'scientific_consensus',
                'timestamp': datetime.now().isoformat()
            },
            {
                'text': 'Machine Learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.',
                'category': 'technology',
                'source': 'academic',
                'timestamp': datetime.now().isoformat()
            },
            {
                'text': 'Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.',
                'category': 'technology',
                'source': 'academic',
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        # Add to vector database
        for item in initial_knowledge:
            self.add_knowledge(item['text'], item['category'], item['source'])
    
    def add_knowledge(self, text: str, category: str = 'general', source: str = 'unknown') -> bool:
        """Add knowledge to the vector database"""
        try:
            # Generate embedding
            embedding = self.embeddings_model.encode([text])
            
            # Add to ChromaDB
            doc_id = f"{category}_{len(self.faiss_texts)}_{datetime.now().timestamp()}"
            
            self.vector_db.add(
                embeddings=embedding.tolist(),
                documents=[text],
                metadatas=[{
                    'category': category,
                    'source': source,
                    'timestamp': datetime.now().isoformat()
                }],
                ids=[doc_id]
            )
            
            # Add to FAISS as backup
            self.faiss_index.add(embedding.astype('float32'))
            self.faiss_texts.append({
                'text': text,
                'category': category,
                'source': source,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Added knowledge: {text[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return False
    
    def search_knowledge(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search knowledge base using vector similarity"""
        try:
            # Generate query embedding
            query_embedding = self.embeddings_model.encode([query])
            
            # Search in ChromaDB
            results = self.vector_db.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else 0.0,
                    'similarity': 1 / (1 + results['distances'][0][i]) if 'distances' in results else 1.0
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []
    
    async def get_real_time_info(self, query: str) -> Dict[str, Any]:
        """Get real-time information from web sources"""
        try:
            results = {
                'news': await self._get_news_info(query),
                'web_search': await self._web_search(query),
                'articles': await self._get_articles(query)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting real-time info: {e}")
            return {}
    
    async def _get_news_info(self, query: str) -> List[Dict[str, Any]]:
        """Get relevant news information"""
        try:
            news_items = []
            query_words = query.lower().split()
            
            for feed_url in self.knowledge_sources['news_feeds']:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:10]:
                        # Check relevance
                        title_words = entry.title.lower().split()
                        summary_words = entry.summary.lower().split() if hasattr(entry, 'summary') else []
                        
                        relevance_score = sum(1 for word in query_words 
                                            if any(word in title_word or title_word in word 
                                                 for title_word in title_words + summary_words))
                        
                        if relevance_score > 0:
                            news_items.append({
                                'title': entry.title,
                                'summary': entry.summary if hasattr(entry, 'summary') else '',
                                'link': entry.link,
                                'published': entry.published if hasattr(entry, 'published') else '',
                                'source': feed_url,
                                'relevance_score': relevance_score
                            })
                            
                except Exception as e:
                    logger.warning(f"Error fetching from {feed_url}: {e}")
            
            # Sort by relevance and return top results
            news_items.sort(key=lambda x: x['relevance_score'], reverse=True)
            return news_items[:5]
            
        except Exception as e:
            logger.error(f"Error getting news info: {e}")
            return []
    
    async def _web_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search and extract information"""
        try:
            search_results = []
            
            # Use DuckDuckGo for privacy-friendly search
            search_url = f"https://duckduckgo.com/html/?q={query}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(search_url, headers=headers, timeout=10) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            # Extract search results
                            for result in soup.find_all('a', class_='result__a')[:5]:
                                title = result.get_text().strip()
                                url = result.get('href', '')
                                
                                if title and url:
                                    search_results.append({
                                        'title': title,
                                        'url': url,
                                        'source': 'DuckDuckGo'
                                    })
                                    
                except Exception as e:
                    logger.warning(f"Web search error: {e}")
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return []
    
    async def _get_articles(self, query: str) -> List[Dict[str, Any]]:
        """Get and process full articles related to query"""
        try:
            articles = []
            
            # Get URLs from web search
            search_results = await self._web_search(query)
            
            for result in search_results[:3]:  # Process top 3 results
                try:
                    article = Article(result['url'])
                    article.download()
                    article.parse()
                    article.nlp()
                    
                    if article.text and len(article.text) > 100:
                        articles.append({
                            'title': article.title,
                            'text': article.text[:1000] + "..." if len(article.text) > 1000 else article.text,
                            'summary': article.summary,
                            'keywords': article.keywords,
                            'url': result['url'],
                            'publish_date': article.publish_date.isoformat() if article.publish_date else None
                        })
                        
                except Exception as e:
                    logger.warning(f"Error processing article {result['url']}: {e}")
            
            return articles
            
        except Exception as e:
            logger.error(f"Error getting articles: {e}")
            return []
    
    def update_knowledge_from_web(self, query: str) -> Dict[str, Any]:
        """Update knowledge base with real-time web information"""
        try:
            # Get real-time information
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            real_time_info = loop.run_until_complete(self.get_real_time_info(query))
            loop.close()
            
            added_count = 0
            
            # Add news articles to knowledge base
            for news_item in real_time_info.get('news', []):
                if news_item.get('summary'):
                    success = self.add_knowledge(
                        news_item['summary'],
                        'news',
                        news_item['source']
                    )
                    if success:
                        added_count += 1
            
            # Add article content to knowledge base
            for article in real_time_info.get('articles', []):
                if article.get('text'):
                    success = self.add_knowledge(
                        article['text'],
                        'article',
                        article['url']
                    )
                    if success:
                        added_count += 1
            
            return {
                'added_items': added_count,
                'real_time_info': real_time_info,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating knowledge from web: {e}")
            return {'error': str(e)}
    
    def get_comprehensive_answer(self, query: str) -> Dict[str, Any]:
        """Get comprehensive answer using all available knowledge sources"""
        try:
            # Search existing knowledge base
            kb_results = self.search_knowledge(query)
            
            # Get real-time information
            web_update = self.update_knowledge_from_web(query)
            
            # Search again after update
            updated_kb_results = self.search_knowledge(query)
            
            # Combine and rank results
            all_results = kb_results + updated_kb_results
            
            # Remove duplicates and sort by relevance
            unique_results = []
            seen_texts = set()
            
            for result in all_results:
                text_hash = hash(result['text'][:100])  # Use first 100 chars as identifier
                if text_hash not in seen_texts:
                    seen_texts.add(text_hash)
                    unique_results.append(result)
            
            # Sort by similarity score
            unique_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            # Generate comprehensive answer
            answer_parts = []
            sources = []
            
            for result in unique_results[:3]:  # Use top 3 results
                answer_parts.append(result['text'])
                sources.append({
                    'category': result['metadata'].get('category', 'unknown'),
                    'source': result['metadata'].get('source', 'unknown'),
                    'similarity': result.get('similarity', 0)
                })
            
            comprehensive_answer = " ".join(answer_parts)
            
            return {
                'query': query,
                'answer': comprehensive_answer,
                'sources': sources,
                'knowledge_base_results': len(kb_results),
                'web_update_info': web_update,
                'total_results': len(unique_results),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive answer: {e}")
            return {
                'query': query,
                'answer': f"I apologize, but I encountered an error: {str(e)}",
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        try:
            return {
                'total_documents': len(self.faiss_texts),
                'embedding_dimension': self.embedding_dimension,
                'knowledge_sources': len(self.knowledge_sources['news_feeds']) + len(self.knowledge_sources['tech_feeds']),
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}

# Global instance
knowledge_engine = None

def get_knowledge_engine():
    """Get or create the global knowledge engine instance"""
    global knowledge_engine
    if knowledge_engine is None:
        knowledge_engine = KnowledgeEngine()
    return knowledge_engine
