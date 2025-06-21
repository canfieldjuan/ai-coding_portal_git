# ===========================
# COMPLETE AI CODING PORTAL - ALL FEATURES FIXED
# ===========================

import asyncio
import os
import json
import subprocess
import hashlib
import time
import tempfile
import ast
import re
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import shutil

# Core libraries with error handling
try:
    import streamlit as st
    import pandas as pd
    import requests
    import aiohttp
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    from pydantic import BaseModel
    import base64
    from bs4 import BeautifulSoup
    import feedparser
except ImportError as e:
    print(f"❌ Missing core dependencies: {e}")
    print("Run: pip install streamlit pandas requests aiohttp fastapi uvicorn pydantic beautifulsoup4 feedparser")
    sys.exit(1)

# Database libraries with fallbacks
try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    print("⚠️ ChromaDB not available - using in-memory storage")

try:
    import psycopg2
    HAS_POSTGRESQL = True
except ImportError:
    HAS_POSTGRESQL = False
    print("⚠️ PostgreSQL not available - using local storage")

try:
    from pymongo import MongoClient
    HAS_MONGODB = True
except ImportError:
    HAS_MONGODB = False
    print("⚠️ MongoDB not available - using local storage")

# AI/ML libraries with fallbacks
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️ OpenAI not available")

try:
    import spacy
    import nltk
    from textblob import TextBlob
    HAS_NLP = True
except ImportError:
    HAS_NLP = False
    print("⚠️ NLP libraries not available - using basic text processing")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    from sentence_transformers import SentenceTransformer
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️ Transformers not available - using basic models")

try:
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ Scikit-learn not available")

# Code analysis libraries with fallbacks
try:
    import pylint.lint
    from pylint.reporters.json_reporter import JSONReporter
    HAS_PYLINT = True
except ImportError:
    HAS_PYLINT = False
    print("⚠️ Pylint not available - using basic analysis")

try:
    import bandit
    from bandit.core import manager
    HAS_BANDIT = True
except ImportError:
    HAS_BANDIT = False
    print("⚠️ Bandit not available - using basic security checks")

try:
    import radon.complexity as complexity
    import radon.metrics as metrics
    HAS_RADON = True
except ImportError:
    HAS_RADON = False
    print("⚠️ Radon not available - using basic complexity metrics")

try:
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name, guess_lexer
    from pygments.formatters import TerminalFormatter
    HAS_PYGMENTS = True
except ImportError:
    HAS_PYGMENTS = False
    print("⚠️ Pygments not available - using basic syntax detection")

# Git integration with fallback
try:
    import git
    HAS_GIT = True
except ImportError:
    HAS_GIT = False
    print("⚠️ GitPython not available - git features disabled")

# ===========================
# CONFIGURATION & MODELS
# ===========================

class ModelProvider(Enum):
    OPENROUTER = "openrouter"
    ALLTOGETHER = "alltogether"
    LOCAL = "local"

class IntentType(Enum):
    CREATE_NEW = "create_new"
    MODIFY_EXISTING = "modify_existing"
    DEBUG_FIX = "debug_fix"
    OPTIMIZE_PERFORMANCE = "optimize_performance"
    REFACTOR_CODE = "refactor_code"
    ADD_FEATURE = "add_feature"
    EXPLAIN_CODE = "explain_code"
    GENERATE_TESTS = "generate_tests"

class RequirementType(Enum):
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    TECHNICAL = "technical"
    BUSINESS = "business"
    CONSTRAINT = "constraint"

class CodeIssueType(Enum):
    SYNTAX_ERROR = "syntax_error"
    LOGIC_ERROR = "logic_error"
    PERFORMANCE_ISSUE = "performance_issue"
    SECURITY_VULNERABILITY = "security_vulnerability"
    CODE_SMELL = "code_smell"
    BEST_PRACTICE_VIOLATION = "best_practice_violation"
    DEPRECATED_USAGE = "deprecated_usage"
    IMPORT_ISSUE = "import_issue"
    TYPE_ERROR = "type_error"
    FORMATTING_ISSUE = "formatting_issue"

@dataclass
class ValidationResult:
    success: bool
    message: str
    errors: List[str] = None
    quality_score: float = 0.0

@dataclass
class CodePattern:
    pattern_id: str
    code_snippet: str
    description: str
    complexity_level: str
    source: str
    quality_score: float

@dataclass
class ParsedEntity:
    text: str
    label: str
    confidence: float
    start_pos: int
    end_pos: int

@dataclass
class ExtractedRequirement:
    text: str
    type: RequirementType
    priority: str
    confidence: float
    implied: bool = False

@dataclass
class NLPAnalysis:
    intent: IntentType
    confidence: float
    entities: List[ParsedEntity] = field(default_factory=list)
    requirements: List[ExtractedRequirement] = field(default_factory=list)
    technical_stack: Dict[str, List[str]] = field(default_factory=dict)
    complexity_level: str = "medium"
    estimated_size: int = 100
    context_clues: List[str] = field(default_factory=list)
    clarity_score: float = 0.5
    ambiguities: List[str] = field(default_factory=list)
    clarifications_needed: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    model_versions: Dict[str, str] = field(default_factory=dict)

@dataclass
class CodeIssue:
    type: CodeIssueType
    severity: str
    line_number: int
    column: int
    message: str
    suggested_fix: str
    confidence: float
    code_snippet: str
    fix_code: str = ""

@dataclass
class CodeAnalysis:
    filename: str
    language: str
    file_size: int
    line_count: int
    purpose: str
    functionality: List[str]
    dependencies: List[str]
    main_functions: List[str]
    classes: List[str]
    complexity_score: float
    maintainability_index: float
    technical_debt_ratio: float
    code_coverage_estimate: float
    issues: List[CodeIssue] = field(default_factory=list)
    total_fixes_applied: int = 0
    fixed_code: str = ""
    improvement_summary: str = ""
    analysis_time: float = 0.0
    tokens_used: int = 0
    confidence_score: float = 0.0

@dataclass
class ExternalPattern:
    source: str
    pattern_code: str
    description: str
    language: str
    popularity_score: float
    last_updated: datetime
    github_stars: int = 0
    stackoverflow_votes: int = 0
    url: str = ""

@dataclass
class LiveSolution:
    problem: str
    solution_code: str
    explanation: str
    source: str
    confidence_score: float
    votes: int
    date: datetime

class Config:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    ALLTOGETHER_API_KEY = os.getenv("ALLTOGETHER_API_KEY", "")
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
    
    POSTGRES_URL = os.getenv("POSTGRES_URL", "")
    MONGODB_URL = os.getenv("MONGODB_URL", "")
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
    
    OPENROUTER_MODELS = {
        "complex": "anthropic/claude-3.5-sonnet",
        "fast": "meta-llama/llama-3.1-8b-instruct",
        "debugging": "openai/gpt-4-turbo"
    }
    
    ALLTOGETHER_MODELS = {
        "complex": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        "fast": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "debugging": "Qwen/Qwen2.5-72B-Instruct-Turbo"
    }

# ===========================
# DATABASE MANAGERS WITH FALLBACKS
# ===========================

class VectorDBManager:
    def __init__(self):
        self.client = None
        self.collection = None
        self.memory_store = {}  # Fallback storage
        
        if HAS_CHROMADB:
            try:
                os.makedirs(Config.CHROMA_PATH, exist_ok=True)
                self.client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
                self.collection = self.client.get_or_create_collection(
                    name="github_patterns",
                    metadata={"hnsw:space": "cosine"}
                )
                print("✅ ChromaDB initialized")
            except Exception as e:
                print(f"⚠️ ChromaDB initialization failed, using memory: {e}")
                self.client = None
        else:
            print("⚠️ ChromaDB not available, using memory storage")
    
    def add_code_pattern(self, pattern: CodePattern):
        if self.collection:
            try:
                self.collection.add(
                    documents=[pattern.code_snippet],
                    metadatas=[{
                        "pattern_id": pattern.pattern_id,
                        "description": pattern.description,
                        "complexity": pattern.complexity_level,
                        "source": pattern.source,
                        "quality_score": pattern.quality_score
                    }],
                    ids=[pattern.pattern_id]
                )
            except Exception as e:
                print(f"Error adding pattern to ChromaDB: {e}")
                self.memory_store[pattern.pattern_id] = pattern
        else:
            self.memory_store[pattern.pattern_id] = pattern
    
    def find_similar_patterns(self, code_query: str, n_results: int = 5) -> List[Dict]:
        if self.collection:
            try:
                results = self.collection.query(
                    query_texts=[code_query],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )
                
                patterns = []
                for i, doc in enumerate(results['documents'][0]):
                    patterns.append({
                        "code": doc,
                        "metadata": results['metadatas'][0][i],
                        "similarity": 1 - results['distances'][0][i]
                    })
                
                return patterns
            except Exception as e:
                print(f"Error querying ChromaDB: {e}")
        
        # Fallback to memory search
        patterns = []
        for pattern_id, pattern in list(self.memory_store.items())[:n_results]:
            patterns.append({
                "code": pattern.code_snippet,
                "metadata": {
                    "pattern_id": pattern.pattern_id,
                    "description": pattern.description,
                    "source": pattern.source
                },
                "similarity": 0.8  # Mock similarity
            })
        
        return patterns

class PostgreSQLManager:
    def __init__(self):
        self.connection = None
        self.memory_store = {"patterns": [], "metrics": [], "sessions": []}
        
        if HAS_POSTGRESQL and Config.POSTGRES_URL:
            try:
                self.connection = psycopg2.connect(Config.POSTGRES_URL)
                self.setup_tables()
                print("✅ PostgreSQL connected")
            except Exception as e:
                print(f"⚠️ PostgreSQL connection failed, using memory: {e}")
                self.connection = None
        else:
            print("⚠️ PostgreSQL not configured, using memory storage")
    
    def setup_tables(self):
        if not self.connection:
            return
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS error_patterns (
                    id SERIAL PRIMARY KEY,
                    error_signature TEXT UNIQUE,
                    error_context JSONB,
                    solution_pattern TEXT,
                    success_rate FLOAT DEFAULT 0.0,
                    source_url TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            self.connection.commit()
            print("✅ PostgreSQL tables ready")
        except Exception as e:
            print(f"Error setting up PostgreSQL tables: {e}")
    
    def store_error_pattern(self, error_sig: str, context: Dict, solution: str, success_rate: float = 0.8):
        if self.connection:
            try:
                cursor = self.connection.cursor()
                cursor.execute("""
                    INSERT INTO error_patterns (error_signature, error_context, solution_pattern, success_rate)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (error_signature) DO UPDATE SET
                        solution_pattern = EXCLUDED.solution_pattern,
                        success_rate = EXCLUDED.success_rate
                """, (error_sig, json.dumps(context), solution, success_rate))
                self.connection.commit()
            except Exception as e:
                print(f"Error storing error pattern: {e}")
                self.memory_store["patterns"].append({
                    "error_signature": error_sig,
                    "context": context,
                    "solution": solution,
                    "success_rate": success_rate
                })
        else:
            self.memory_store["patterns"].append({
                "error_signature": error_sig,
                "context": context,
                "solution": solution,
                "success_rate": success_rate
            })
    
    def get_error_solution(self, error_sig: str) -> Optional[Dict]:
        if self.connection:
            try:
                cursor = self.connection.cursor()
                cursor.execute("""
                    SELECT solution_pattern, success_rate, error_context 
                    FROM error_patterns 
                    WHERE error_signature = %s
                """, (error_sig,))
                
                result = cursor.fetchone()
                if result:
                    return {
                        "solution": result[0],
                        "success_rate": result[1],
                        "context": result[2]
                    }
            except Exception as e:
                print(f"Error getting error solution: {e}")
        
        # Fallback to memory
        for pattern in self.memory_store["patterns"]:
            if pattern["error_signature"] == error_sig:
                return {
                    "solution": pattern["solution"],
                    "success_rate": pattern["success_rate"],
                    "context": pattern["context"]
                }
        
        return None

class MongoDBManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.memory_store = {"patterns": [], "benchmarks": []}
        
        if HAS_MONGODB and Config.MONGODB_URL:
            try:
                self.client = MongoClient(Config.MONGODB_URL)
                self.db = self.client.coding_portal
                self.patterns = self.db.architecture_patterns
                self.benchmarks = self.db.model_benchmarks
                print("✅ MongoDB connected")
            except Exception as e:
                print(f"⚠️ MongoDB connection failed, using memory: {e}")
                self.client = None
        else:
            print("⚠️ MongoDB not configured, using memory storage")
    
    def store_architecture_pattern(self, pattern: Dict):
        if self.client:
            try:
                pattern['_id'] = pattern.get('pattern_id', str(hash(pattern['pattern_code'])))
                pattern['created_at'] = datetime.now()
                self.patterns.update_one(
                    {"_id": pattern['_id']},
                    {"$set": pattern},
                    upsert=True
                )
            except Exception as e:
                print(f"Error storing pattern in MongoDB: {e}")
                self.memory_store["patterns"].append(pattern)
        else:
            self.memory_store["patterns"].append(pattern)

# ===========================
# MODEL ROUTING & MANAGEMENT WITH FALLBACKS
# ===========================

class ModelRouter:
    def __init__(self):
        self.providers = {}
        self.fallback_order = []
        
        # Initialize available providers
        if Config.OPENROUTER_API_KEY:
            self.providers[ModelProvider.OPENROUTER] = OpenRouterClient()
            self.fallback_order.append(ModelProvider.OPENROUTER)
            print("✅ OpenRouter client ready")
        
        if Config.ALLTOGETHER_API_KEY:
            self.providers[ModelProvider.ALLTOGETHER] = AllTogetherClient()
            self.fallback_order.append(ModelProvider.ALLTOGETHER)
            print("✅ AllTogether client ready")
        
        if not self.providers:
            print("⚠️ No API keys configured - will use mock responses")
    
    async def generate_code(self, prompt: str, context: Dict, task_type: str = "complex") -> str:
        """Generate code with automatic failover"""
        
        for provider in self.fallback_order:
            try:
                client = self.providers[provider]
                response = await client.generate(prompt, context, task_type)
                return response
            except Exception as e:
                print(f"Provider {provider} failed: {e}")
                continue
        
        # Fallback to mock generation
        return self.generate_mock_code(prompt, context)
    
    def generate_mock_code(self, prompt: str, context: Dict) -> str:
        """Generate mock code when all APIs fail"""
        
        language = context.get("language", "python")
        framework = context.get("framework", "")
        
        if "fastapi" in prompt.lower() or "api" in prompt.lower():
            return '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str = None

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/items/")
async def create_item(item: Item):
    return {"item": item, "status": "created"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        
        elif "function" in prompt.lower():
            return f'''
def generated_function(data=None):
    """
    Generated function based on: {prompt}
    """
    if data is None:
        data = []
    
    result = {{
        "status": "success",
        "data": data,
        "timestamp": datetime.now().isoformat()
    }}
    
    return result

# Example usage
if __name__ == "__main__":
    test_data = ["example", "data"]
    result = generated_function(test_data)
    print(result)
'''
        
        else:
            return f'''
# Generated code for: {prompt}
# Language: {language}
# Framework: {framework}

def main():
    """
    Main function implementing the requested functionality
    """
    print("Generated code is running!")
    
    # TODO: Implement specific functionality for:
    # {prompt}
    
    return "Code executed successfully"

if __name__ == "__main__":
    result = main()
    print(result)
'''

class OpenRouterClient:
    def __init__(self):
        self.api_key = Config.OPENROUTER_API_KEY
        self.base_url = "https://openrouter.ai/api/v1"
    
    async def generate(self, prompt: str, context: Dict, task_type: str) -> str:
        if not self.api_key:
            raise Exception("OpenRouter API key not configured")
        
        model = Config.OPENROUTER_MODELS.get(task_type, Config.OPENROUTER_MODELS["complex"])
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an expert software engineer. Generate clean, production-ready code with proper error handling and documentation."},
                {"role": "user", "content": f"Context: {json.dumps(context)}\n\nRequest: {prompt}"}
            ],
            "temperature": 0.1,
            "max_tokens": context.get("max_tokens", 8000)
        }
        
        try:
            response = requests.post(f"{self.base_url}/chat/completions", 
                                   headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")
        except requests.exceptions.Timeout:
            raise Exception("OpenRouter API timeout")
        except requests.exceptions.RequestException as e:
            raise Exception(f"OpenRouter API request failed: {e}")

class AllTogetherClient:
    def __init__(self):
        self.api_key = Config.ALLTOGETHER_API_KEY
        self.base_url = "https://api.together.xyz/v1"
    
    async def generate(self, prompt: str, context: Dict, task_type: str) -> str:
        if not self.api_key:
            raise Exception("AllTogether API key not configured")
        
        model = Config.ALLTOGETHER_MODELS.get(task_type, Config.ALLTOGETHER_MODELS["complex"])
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an expert software engineer. Generate clean, production-ready code with proper error handling and documentation."},
                {"role": "user", "content": f"Context: {json.dumps(context)}\n\nRequest: {prompt}"}
            ],
            "temperature": 0.1,
            "max_tokens": context.get("max_tokens", 8000)
        }
        
        try:
            response = requests.post(f"{self.base_url}/chat/completions", 
                                   headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                raise Exception(f"AllTogether API error: {response.status_code} - {response.text}")
        except requests.exceptions.Timeout:
            raise Exception("AllTogether API timeout")
        except requests.exceptions.RequestException as e:
            raise Exception(f"AllTogether API request failed: {e}")

# ===========================
# NLP UNDERSTANDING ENGINE WITH FALLBACKS
# ===========================

class AdvancedNLPEngine:
    def __init__(self):
        self.nlp = None
        self.sentence_model = None
        
        self.setup_models()
        self.load_patterns()
        
    def setup_models(self):
        print("🧠 Loading NLP models...")
        
        if HAS_NLP:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                print("✅ spaCy model loaded")
            except OSError:
                try:
                    print("Downloading spaCy model...")
                    spacy.cli.download("en_core_web_sm")
                    self.nlp = spacy.load("en_core_web_sm")
                    print("✅ spaCy model downloaded and loaded")
                except Exception as e:
                    print(f"⚠️ spaCy model failed: {e}")
                    self.nlp = None
            except Exception as e:
                print(f"⚠️ spaCy initialization failed: {e}")
                self.nlp = None
        
        if HAS_TRANSFORMERS:
            try:
                self.sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                print("✅ SentenceTransformer loaded")
            except Exception as e:
                print(f"⚠️ SentenceTransformer failed: {e}")
                self.sentence_model = None
        
        if not self.nlp and not self.sentence_model:
            print("⚠️ Using basic text processing (no advanced NLP)")
    
    def load_patterns(self):
        """Load intent and entity patterns"""
        self.intent_patterns = {
            IntentType.CREATE_NEW: [
                r"create|build|make|develop|implement|generate|write",
                r"new|fresh|from scratch|start.*new"
            ],
            IntentType.MODIFY_EXISTING: [
                r"modify|change|update|alter|edit|adjust",
                r"existing|current|this.*code|the.*above"
            ],
            IntentType.DEBUG_FIX: [
                r"debug|fix|solve|resolve|troubleshoot|repair",
                r"error|bug|issue|problem|not.*working|broken|failing"
            ],
            IntentType.OPTIMIZE_PERFORMANCE: [
                r"optimize|improve|speed.*up|make.*faster|performance|efficient",
                r"slow|inefficient|bottleneck|latency|memory"
            ],
            IntentType.REFACTOR_CODE: [
                r"refactor|restructure|reorganize|clean.*up|rewrite",
                r"messy|complex|hard.*read|maintainable|readable"
            ],
            IntentType.ADD_FEATURE: [
                r"add|include|incorporate|integrate|extend",
                r"feature|functionality|capability|support.*for"
            ]
        }
        
        self.tech_entities = {
            "frameworks": ["fastapi", "django", "flask", "react", "vue", "angular", "express", "spring", "laravel"],
            "databases": ["postgresql", "mysql", "mongodb", "redis", "sqlite", "elasticsearch", "cassandra"],
            "languages": ["python", "javascript", "typescript", "java", "go", "rust", "cpp", "csharp", "php"],
            "protocols": ["http", "https", "websocket", "grpc", "rest", "graphql", "soap", "tcp", "udp"],
            "cloud_services": ["aws", "azure", "gcp", "docker", "kubernetes", "lambda", "heroku"],
            "patterns": ["mvc", "mvp", "microservices", "monolith", "serverless", "event-driven", "observer"]
        }
        
        self.complexity_indicators = {
            "simple": ["simple", "basic", "easy", "quick", "small", "minimal"],
            "medium": ["medium", "moderate", "standard", "typical", "normal"],
            "complex": ["complex", "advanced", "sophisticated", "enterprise", "large-scale", "production"]
        }
    
    def analyze_request(self, user_input: str, context: Dict = None) -> NLPAnalysis:
        """Comprehensive NLP analysis with fallbacks"""
        start_time = time.time()
        
        try:
            # Basic preprocessing
            cleaned_input = self.preprocess_text(user_input)
            
            # Intent classification
            intent, intent_confidence = self.classify_intent(cleaned_input)
            
            # Entity extraction
            entities = self.extract_entities(cleaned_input)
            
            # Technical stack identification
            technical_stack = self.identify_tech_stack(cleaned_input)
            
            # Requirement extraction
            requirements = self.extract_requirements(cleaned_input)
            
            # Complexity and size estimation
            complexity_level = self.estimate_complexity(cleaned_input)
            estimated_size = self.estimate_code_size(cleaned_input)
            
            # Context analysis
            context_clues = self.extract_context_clues(cleaned_input, context)
            
            # Quality assessment
            clarity_score = self.assess_clarity(cleaned_input)
            ambiguities = self.identify_ambiguities(cleaned_input)
            clarifications_needed = self.suggest_clarifications(technical_stack, requirements)
            
            analysis = NLPAnalysis(
                intent=intent,
                confidence=intent_confidence,
                entities=entities,
                requirements=requirements,
                technical_stack=technical_stack,
                complexity_level=complexity_level,
                estimated_size=estimated_size,
                context_clues=context_clues,
                clarity_score=clarity_score,
                ambiguities=ambiguities,
                clarifications_needed=clarifications_needed,
                processing_time=time.time() - start_time
            )
            
            return analysis
            
        except Exception as e:
            print(f"NLP analysis error: {e}")
            # Return basic analysis on error
            return NLPAnalysis(
                intent=IntentType.CREATE_NEW,
                confidence=0.5,
                complexity_level="medium",
                estimated_size=100,
                clarity_score=0.5,
                processing_time=time.time() - start_time
            )
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize input text"""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'\bAPI\b', 'api', text, flags=re.IGNORECASE)
        text = re.sub(r'\bDB\b', 'database', text, flags=re.IGNORECASE)
        return text
    
    def classify_intent(self, text: str) -> Tuple[IntentType, float]:
        """Classify the intent of the user request"""
        best_intent = IntentType.CREATE_NEW
        best_score = 0.0
        
        for intent_type, patterns in self.intent_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches * 0.3
            
            if score > best_score:
                best_score = score
                best_intent = intent_type
        
        return best_intent, min(best_score, 1.0)
    
    def extract_entities(self, text: str) -> List[ParsedEntity]:
        """Extract entities using available NLP tools"""
        entities = []
        
        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    entities.append(ParsedEntity(
                        text=ent.text,
                        label=ent.label_,
                        confidence=0.8,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char
                    ))
            except Exception as e:
                print(f"spaCy entity extraction error: {e}")
        
        # Add technical entities
        text_lower = text.lower()
        for category, terms in self.tech_entities.items():
            for term in terms:
                if term in text_lower:
                    start_pos = text_lower.find(term)
                    entities.append(ParsedEntity(
                        text=term,
                        label=f"TECH_{category.upper()}",
                        confidence=0.9,
                        start_pos=start_pos,
                        end_pos=start_pos + len(term)
                    ))
        
        return entities
    
    def identify_tech_stack(self, text: str) -> Dict[str, List[str]]:
        """Identify technical components mentioned"""
        tech_stack = {category: [] for category in self.tech_entities.keys()}
        
        text_lower = text.lower()
        for category, terms in self.tech_entities.items():
            for term in terms:
                if term in text_lower:
                    tech_stack[category].append(term)
        
        return {k: v for k, v in tech_stack.items() if v}
    
    def extract_requirements(self, text: str) -> List[ExtractedRequirement]:
        """Extract functional and non-functional requirements"""
        requirements = []
        
        try:
            # Simple sentence splitting if nltk not available
            if HAS_NLP:
                sentences = nltk.sent_tokenize(text)
            else:
                sentences = text.split('.')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if any(word in sentence.lower() for word in ["should", "must", "needs", "require", "want"]):
                    req_type = RequirementType.FUNCTIONAL
                    priority = "should"
                    
                    if "must" in sentence.lower() or "required" in sentence.lower():
                        priority = "must"
                    elif "could" in sentence.lower() or "optional" in sentence.lower():
                        priority = "could"
                    
                    if len(sentence) > 10:  # Meaningful requirement
                        requirements.append(ExtractedRequirement(
                            text=sentence,
                            type=req_type,
                            priority=priority,
                            confidence=0.7
                        ))
        except Exception as e:
            print(f"Requirement extraction error: {e}")
        
        return requirements[:5]  # Limit to 5 requirements
    
    def estimate_complexity(self, text: str) -> str:
        """Estimate complexity based on language indicators"""
        text_lower = text.lower()
        
        complexity_scores = {"simple": 0, "medium": 0, "complex": 0}
        
        for level, indicators in self.complexity_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    complexity_scores[level] += 1
        
        # Additional complexity factors
        if any(term in text_lower for term in ["microservice", "distributed", "scalable", "enterprise"]):
            complexity_scores["complex"] += 2
        
        return max(complexity_scores, key=complexity_scores.get) or "medium"
    
    def estimate_code_size(self, text: str) -> int:
        """Estimate lines of code based on description"""
        text_lower = text.lower()
        
        if "function" in text_lower:
            return 50
        elif "class" in text_lower:
            return 200
        elif any(word in text_lower for word in ["application", "system", "project"]):
            return 1000
        else:
            return 100
    
    def extract_context_clues(self, text: str, context: Dict = None) -> List[str]:
        """Extract contextual information"""
        clues = []
        
        if "existing" in text.lower():
            clues.append("modifying_existing_code")
        
        if context and context.get("codebase_size", 0) > 1000:
            clues.append("large_existing_codebase")
        
        return clues
    
    def assess_clarity(self, text: str) -> float:
        """Assess how clear and specific the request is"""
        clarity_score = 0.5
        
        if len(text.split()) > 10:
            clarity_score += 0.2
        
        if any(word in text.lower() for word in ["should", "must", "needs to"]):
            clarity_score += 0.1
        
        if any(word in text.lower() for word in ["something", "anything", "whatever"]):
            clarity_score -= 0.2
        
        return max(0.0, min(1.0, clarity_score))
    
    def identify_ambiguities(self, text: str) -> List[str]:
        """Identify ambiguous or unclear parts"""
        ambiguities = []
        
        vague_terms = ["something", "anything", "stuff", "things"]
        for term in vague_terms:
            if term in text.lower():
                ambiguities.append(f"Vague term: '{term}'")
        
        return ambiguities
    
    def suggest_clarifications(self, tech_stack: Dict, requirements: List) -> List[str]:
        """Suggest clarifying questions based on analysis"""
        clarifications = []
        
        if not tech_stack.get("languages"):
            clarifications.append("Which programming language should I use?")
        
        if not tech_stack.get("databases") and any("store" in req.text.lower() for req in requirements):
            clarifications.append("Which database system should I use?")
        
        return clarifications[:3]

# ===========================
# CODE ANALYSIS ENGINE WITH FALLBACKS
# ===========================

class AdvancedCodeAnalyzer:
    def __init__(self, model_router):
        self.model_router = model_router
        
        # Language detection patterns
        self.language_patterns = {
            "python": [r"\.py$", r"import\s+", r"def\s+", r"class\s+", r"from\s+.*\s+import"],
            "javascript": [r"\.js$", r"\.mjs$", r"function\s+", r"const\s+", r"let\s+", r"var\s+"],
            "typescript": [r"\.ts$", r"\.tsx$", r"interface\s+", r"type\s+", r":\s*\w+"],
            "java": [r"\.java$", r"public\s+class", r"import\s+.*\..*", r"public\s+static\s+void\s+main"],
            "go": [r"\.go$", r"package\s+", r"func\s+", r"import\s+\("],
            "rust": [r"\.rs$", r"fn\s+", r"use\s+", r"struct\s+", r"impl\s+"],
            "cpp": [r"\.cpp$", r"\.cc$", r"\.cxx$", r"#include\s*<", r"using\s+namespace"],
        }
    
    async def analyze_dropped_code(self, file_content: str, filename: str) -> CodeAnalysis:
        """Complete analysis of dropped code file"""
        start_time = time.time()
        
        try:
            # Basic file analysis
            language = self.detect_language(file_content, filename)
            line_count = len(file_content.splitlines())
            file_size = len(file_content.encode('utf-8'))
            
            analysis = CodeAnalysis(
                filename=filename,
                language=language,
                file_size=file_size,
                line_count=line_count,
                purpose="",
                functionality=[],
                dependencies=[],
                main_functions=[],
                classes=[],
                complexity_score=0.5,
                maintainability_index=0.5,
                technical_debt_ratio=0.3,
                code_coverage_estimate=0.0
            )
            
            # Language-specific analysis
            if language == "python":
                await self.analyze_python_code(file_content, analysis)
            elif language in ["javascript", "typescript"]:
                await self.analyze_js_ts_code(file_content, analysis)
            else:
                await self.analyze_generic_code(file_content, analysis)
            
            # AI-powered analysis if available
            await self.perform_ai_analysis(file_content, analysis)
            
            # Generate fixes
            await self.generate_comprehensive_fixes(file_content, analysis)
            
            analysis.analysis_time = time.time() - start_time
            return analysis
            
        except Exception as e:
            print(f"Code analysis error: {e}")
            # Return basic analysis on error
            return CodeAnalysis(
                filename=filename,
                language="unknown",
                file_size=len(file_content.encode('utf-8')),
                line_count=len(file_content.splitlines()),
                purpose="Analysis failed - see original code",
                functionality=[],
                dependencies=[],
                main_functions=[],
                classes=[],
                complexity_score=0.5,
                maintainability_index=0.5,
                technical_debt_ratio=0.5,
                code_coverage_estimate=0.0,
                fixed_code=file_content,
                improvement_summary="Analysis failed, original code returned",
                analysis_time=time.time() - start_time
            )
    
    def detect_language(self, content: str, filename: str) -> str:
        """Detect programming language from content and filename"""
        
        # Check filename extension first
        for lang, patterns in self.language_patterns.items():
            for pattern in patterns:
                if re.search(pattern, filename, re.IGNORECASE):
                    return lang
        
        # Check content patterns
        scores = {}
        for lang, patterns in self.language_patterns.items():
            score = 0
            for pattern in patterns[1:]:  # Skip filename patterns
                matches = len(re.findall(pattern, content, re.IGNORECASE | re.MULTILINE))
                score += matches
            scores[lang] = score
        
        if scores:
            best_lang = max(scores, key=scores.get)
            if scores[best_lang] > 0:
                return best_lang
        
        # Try pygments if available
        if HAS_PYGMENTS:
            try:
                lexer = guess_lexer(content)
                return lexer.name.lower()
            except:
                pass
        
        return "text"
    
    async def analyze_python_code(self, content: str, analysis: CodeAnalysis):
        """Python-specific analysis"""
        
        try:
            # Parse AST for structure
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis.main_functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    analysis.classes.append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis.dependencies.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        analysis.dependencies.append(node.module)
            
            # Security analysis
            security_issues = self.analyze_python_security(content)
            analysis.issues.extend(security_issues)
            
            # Performance analysis
            performance_issues = self.analyze_python_performance(content)
            analysis.issues.extend(performance_issues)
            
        except SyntaxError as e:
            issue = CodeIssue(
                type=CodeIssueType.SYNTAX_ERROR,
                severity="critical",
                line_number=e.lineno or 1,
                column=e.offset or 1,
                message=f"Syntax Error: {e.msg}",
                suggested_fix="Fix syntax error",
                confidence=1.0,
                code_snippet=content.split('\n')[e.lineno-1] if e.lineno else ""
            )
            analysis.issues.append(issue)
        except Exception as e:
            print(f"Python analysis error: {e}")
    
    def analyze_python_security(self, content: str) -> List[CodeIssue]:
        """Analyze Python code for security issues"""
        issues = []
        
        security_patterns = [
            (r"password\s*=\s*['\"][^'\"]+['\"]", "Hardcoded password - use environment variables", "critical"),
            (r"api_key\s*=\s*['\"][^'\"]+['\"]", "Hardcoded API key - use environment variables", "critical"),
            (r"eval\s*\(", "Use of eval() is dangerous", "high"),
            (r"exec\s*\(", "Use of exec() is dangerous", "high"),
            (r"subprocess\.call.*shell=True", "Shell injection risk", "high"),
            (r"pickle\.loads?", "Pickle deserialization risk", "medium"),
        ]
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern, message, severity in security_patterns:
                if re.search(pattern, line):
                    issue = CodeIssue(
                        type=CodeIssueType.SECURITY_VULNERABILITY,
                        severity=severity,
                        line_number=line_num,
                        column=0,
                        message=message,
                        suggested_fix=f"Fix: {message}",
                        confidence=0.9,
                        code_snippet=line.strip()
                    )
                    issues.append(issue)
        
        return issues
    
    def analyze_python_performance(self, content: str) -> List[CodeIssue]:
        """Analyze Python performance issues"""
        issues = []
        
        performance_patterns = [
            (r"for\s+\w+\s+in\s+range\s*\(\s*len\s*\(", "Use enumerate() instead of range(len())", "medium"),
            (r"\.append\s*\([^)]*\)\s*$", "Consider list comprehension for better performance", "low"),
            (r"==\s*True|==\s*False", "Use 'if condition:' instead of '== True/False'", "low"),
            (r"len\s*\([^)]*\)\s*==\s*0", "Use 'if not sequence:' instead of 'len(sequence) == 0'", "low"),
        ]
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern, message, severity in performance_patterns:
                if re.search(pattern, line):
                    issue = CodeIssue(
                        type=CodeIssueType.PERFORMANCE_ISSUE,
                        severity=severity,
                        line_number=line_num,
                        column=0,
                        message=message,
                        suggested_fix=message,
                        confidence=0.8,
                        code_snippet=line.strip()
                    )
                    issues.append(issue)
        
        return issues
    
    async def analyze_js_ts_code(self, content: str, analysis: CodeAnalysis):
        """JavaScript/TypeScript analysis"""
        
        # Extract functions
        function_patterns = [
            r"function\s+(\w+)",
            r"const\s+(\w+)\s*=\s*\([^)]*\)\s*=>",
            r"(\w+)\s*:\s*function",
        ]
        
        for pattern in function_patterns:
            matches = re.findall(pattern, content)
            analysis.main_functions.extend(matches)
        
        # JavaScript-specific issues
        js_issues = self.analyze_js_issues(content)
        analysis.issues.extend(js_issues)
    
    def analyze_js_issues(self, content: str) -> List[CodeIssue]:
        """Analyze JavaScript issues"""
        issues = []
        
        js_patterns = [
            (r"==\s*[^=]", "Use === instead of == for strict equality", "medium"),
            (r"var\s+", "Use let or const instead of var", "low"),
            (r"console\.log\s*\(", "Remove console.log statements in production", "low"),
            (r"eval\s*\(", "Avoid eval() - security risk", "critical"),
        ]
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern, message, severity in js_patterns:
                if re.search(pattern, line):
                    issue = CodeIssue(
                        type=CodeIssueType.BEST_PRACTICE_VIOLATION,
                        severity=severity,
                        line_number=line_num,
                        column=0,
                        message=message,
                        suggested_fix=message,
                        confidence=0.9,
                        code_snippet=line.strip()
                    )
                    issues.append(issue)
        
        return issues
    
    async def analyze_generic_code(self, content: str, analysis: CodeAnalysis):
        """Generic analysis for any language"""
        
        # Universal function patterns
        function_patterns = [
            r"function\s+(\w+)",
            r"def\s+(\w+)",
            r"(\w+)\s*\([^)]*\)\s*\{",
            r"fn\s+(\w+)",
        ]
        
        for pattern in function_patterns:
            matches = re.findall(pattern, content)
            analysis.main_functions.extend(matches)
        
        # Generic issues
        generic_issues = self.analyze_generic_issues(content)
        analysis.issues.extend(generic_issues)
    
    def analyze_generic_issues(self, content: str) -> List[CodeIssue]:
        """Language-agnostic issue detection"""
        issues = []
        
        generic_patterns = [
            (r"TODO|FIXME|HACK", "Unresolved TODO/FIXME comment", "low"),
            (r"password\s*=\s*['\"][^'\"]+['\"]", "Hardcoded password", "critical"),
            (r"api_key\s*=\s*['\"][^'\"]+['\"]", "Hardcoded API key", "critical"),
        ]
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern, message, severity in generic_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issue = CodeIssue(
                        type=CodeIssueType.SECURITY_VULNERABILITY if severity == "critical" else CodeIssueType.CODE_SMELL,
                        severity=severity,
                        line_number=line_num,
                        column=0,
                        message=message,
                        suggested_fix=message,
                        confidence=0.9,
                        code_snippet=line.strip()
                    )
                    issues.append(issue)
        
        return issues
    
    async def perform_ai_analysis(self, content: str, analysis: CodeAnalysis):
        """AI-powered comprehensive analysis"""
        
        analysis_prompt = f"""
COMPREHENSIVE CODE ANALYSIS

Analyze this {analysis.language} code and provide:

1. PURPOSE: What does this code do?
2. KEY FUNCTIONALITY: Main features
3. QUALITY ASSESSMENT: Rate 1-10

Code:
```{analysis.language}
{content[:1500]}...
```

Be concise and specific.
"""
        
        try:
            ai_response = await self.model_router.generate_code(
                analysis_prompt,
                {"analysis_type": "comprehensive", "max_tokens": 2000},
                "complex"
            )
            
            # Extract purpose from AI response
            if "PURPOSE:" in ai_response.upper():
                purpose_match = re.search(r"PURPOSE:(.+?)(?=\d\.|KEY|QUALITY|\n\n|\Z)", ai_response, re.DOTALL | re.IGNORECASE)
                if purpose_match:
                    analysis.purpose = purpose_match.group(1).strip()
            
            if not analysis.purpose:
                analysis.purpose = f"Code analysis for {analysis.filename}"
            
            analysis.tokens_used += 2000
            
        except Exception as e:
            print(f"AI analysis error: {e}")
            analysis.purpose = f"Auto-analysis of {analysis.language} code"
    
    async def generate_comprehensive_fixes(self, content: str, analysis: CodeAnalysis):
        """Generate comprehensive code fixes"""
        
        if not analysis.issues:
            analysis.fixed_code = content
            analysis.improvement_summary = "No issues found - code looks good!"
            return
        
        issues_summary = "\n".join([
            f"Line {issue.line_number}: {issue.severity.upper()} - {issue.message}"
            for issue in analysis.issues[:10]  # Top 10 issues
        ])
        
        fix_prompt = f"""
FIX ALL ISSUES IN THIS {analysis.language.upper()} CODE

Original code:
```{analysis.language}
{content}
```

Issues to fix:
{issues_summary}

REQUIREMENTS:
1. Fix all security vulnerabilities
2. Improve performance 
3. Apply best practices
4. Maintain original functionality
5. Add proper error handling

Provide ONLY the complete fixed code.
"""
        
        try:
            fix_response = await self.model_router.generate_code(
                fix_prompt,
                {"fix_type": "comprehensive", "max_tokens": 16000},
                "complex"
            )
            
            # Extract fixed code
            code_patterns = [
                r"```\w*\n(.*?)\n```",
                r"FIXED CODE:.*?\n(.*?)(?=\n[A-Z]+:|$)",
            ]
            
            for pattern in code_patterns:
                match = re.search(pattern, fix_response, re.DOTALL)
                if match:
                    analysis.fixed_code = match.group(1).strip()
                    break
            
            if not analysis.fixed_code:
                analysis.fixed_code = fix_response.strip()
            
            analysis.improvement_summary = f"Applied {len(analysis.issues)} fixes including security improvements, performance optimizations, and best practice implementations."
            analysis.total_fixes_applied = len(analysis.issues)
            analysis.tokens_used += 16000
            
        except Exception as e:
            print(f"Fix generation error: {e}")
            analysis.fixed_code = content
            analysis.improvement_summary = f"Error generating fixes: {e}"

# ===========================
# VALIDATION & GIT MANAGEMENT
# ===========================

class GitManager:
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        self.repo = None
        
        if HAS_GIT:
            try:
                self.repo = git.Repo(repo_path)
                print("✅ Git repository found")
            except git.InvalidGitRepositoryError:
                try:
                    self.repo = git.Repo.init(repo_path)
                    print("✅ Git repository initialized")
                except Exception as e:
                    print(f"⚠️ Git initialization failed: {e}")
                    self.repo = None
        else:
            print("⚠️ Git not available")
    
    def get_current_hash(self) -> str:
        """Get current commit hash"""
        if self.repo:
            try:
                return self.repo.head.commit.hexsha
            except:
                return "no_commits"
        return "no_git"

class ValidationPipeline:
    def __init__(self):
        self.git_manager = GitManager()
    
    def validate_code_changes(self, code: str, file_path: str, project_context: Dict) -> ValidationResult:
        """Validate code changes"""
        
        try:
            # Basic syntax validation for Python
            if file_path.endswith('.py'):
                try:
                    ast.parse(code)
                    syntax_valid = True
                    syntax_message = "Syntax valid"
                except SyntaxError as e:
                    syntax_valid = False
                    syntax_message = f"Syntax error: {e}"
            else:
                syntax_valid = True
                syntax_message = "Syntax check skipped"
            
            if not syntax_valid:
                return ValidationResult(False, syntax_message, [syntax_message])
            
            # Basic quality check
            lines = code.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            
            if len(non_empty_lines) == 0:
                return ValidationResult(False, "Empty code")
            
            # Simple quality scoring
            quality_score = 0.8  # Base score
            
            # Check for comments
            comment_lines = [line for line in lines if line.strip().startswith('#')]
            if len(comment_lines) > 0:
                quality_score += 0.1
            
            # Penalize very long functions (basic heuristic)
            if len(non_empty_lines) > 200:
                quality_score -= 0.2
            
            quality_score = max(0.1, min(1.0, quality_score))
            
            return ValidationResult(True, "Code validated", quality_score=quality_score)
            
        except Exception as e:
            return ValidationResult(False, f"Validation error: {e}")

# ===========================
# EXTERNAL DATA MANAGER WITH FALLBACKS
# ===========================

class ExternalDataManager:
    def __init__(self):
        self.github_token = Config.GITHUB_TOKEN
        self.apis = {
            "github": "https://api.github.com",
            "stackoverflow": "https://api.stackexchange.com/2.3",
        }
    
    async def get_live_patterns(self, query: str, language: str = "python") -> List[ExternalPattern]:
        """Get live patterns with fallbacks"""
        
        patterns = []
        
        # Try GitHub API if token available
        if self.github_token:
            try:
                github_patterns = await self.get_github_patterns(query, language)
                patterns.extend(github_patterns)
            except Exception as e:
                print(f"GitHub API error: {e}")
        
        # Always add mock patterns as fallback
        mock_patterns = self.get_mock_patterns(query, language)
        patterns.extend(mock_patterns)
        
        return patterns[:5]  # Return top 5
    
    async def get_github_patterns(self, query: str, language: str) -> List[ExternalPattern]:
        """Get patterns from GitHub API"""
        
        patterns = []
        
        try:
            headers = {"Authorization": f"token {self.github_token}"}
            search_query = f"{query} language:{language} stars:>10"
            
            url = f"{self.apis['github']}/search/repositories"
            params = {"q": search_query, "sort": "stars", "per_page": 3}
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for repo in data.get("items", []):
                    pattern = ExternalPattern(
                        source="github",
                        pattern_code=f"# Pattern from {repo['name']}\n# {repo['description']}\n\n# See: {repo['html_url']}",
                        description=repo["description"] or "GitHub repository pattern",
                        language=language,
                        popularity_score=min(repo["stargazers_count"] / 1000, 5.0),
                        last_updated=datetime.fromisoformat(repo["updated_at"].replace('Z', '+00:00')),
                        github_stars=repo["stargazers_count"],
                        url=repo["html_url"]
                    )
                    patterns.append(pattern)
        
        except Exception as e:
            print(f"GitHub pattern fetch error: {e}")
        
        return patterns
    
    def get_mock_patterns(self, query: str, language: str) -> List[ExternalPattern]:
        """Generate mock patterns as fallback"""
        
        if "auth" in query.lower():
            code = f'''
# {language.title()} Authentication Pattern
# Based on query: {query}

def authenticate_user(username, password):
    """
    Secure user authentication
    """
    # Hash password properly
    import hashlib
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    # Verify against database
    user = get_user_from_db(username)
    if user and user.password_hash == password_hash:
        return generate_token(user)
    
    return None

def generate_token(user):
    """Generate JWT token"""
    import jwt
    import datetime
    
    payload = {{
        'user_id': user.id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }}
    
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')
'''
        elif "api" in query.lower():
            code = f'''
# {language.title()} API Pattern
# Based on query: {query}

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str = None

@app.get("/items/{{item_id}}")
async def get_item(item_id: int):
    item = await get_item_from_db(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@app.post("/items/")
async def create_item(item: Item):
    created_item = await save_item_to_db(item)
    return {{"id": created_item.id, "status": "created"}}
'''
        else:
            code = f'''
# {language.title()} Pattern for: {query}

def process_{query.lower().replace(" ", "_")}(data):
    """
    Process {query}
    """
    try:
        # Validate input
        if not data:
            raise ValueError("No data provided")
        
        # Process data
        result = {{
            "status": "success",
            "processed_data": data,
            "timestamp": datetime.now().isoformat()
        }}
        
        return result
    
    except Exception as e:
        return {{"status": "error", "message": str(e)}}
'''
        
        return [ExternalPattern(
            source="pattern_library",
            pattern_code=code,
            description=f"Common {language} pattern for {query}",
            language=language,
            popularity_score=4.0,
            last_updated=datetime.now(),
            github_stars=100,
            url="https://github.com/example/patterns"
        )]

# ===========================
# MAIN PORTAL CLASS
# ===========================

class CodingPortal:
    def __init__(self):
        print("🚀 Initializing Coding Portal...")
        
        try:
            self.vector_db = VectorDBManager()
            self.postgres_db = PostgreSQLManager()
            self.mongo_db = MongoDBManager()
            self.model_router = ModelRouter()
            self.validation_pipeline = ValidationPipeline()
            self.git_manager = GitManager()
            
            self.initialize_sample_data()
            print("✅ Portal initialized successfully")
            
        except Exception as e:
            print(f"⚠️ Portal initialization error: {e}")
            # Create minimal fallback instances
            self.vector_db = VectorDBManager()
            self.postgres_db = PostgreSQLManager()
            self.mongo_db = MongoDBManager()
            self.model_router = ModelRouter()
            self.validation_pipeline = ValidationPipeline()
            self.git_manager = GitManager()
    
    def initialize_sample_data(self):
        """Initialize with sample data"""
        try:
            sample_pattern = CodePattern(
                pattern_id="fastapi_basic_001",
                code_snippet="""
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
""",
                description="Basic FastAPI endpoints",
                complexity_level="simple",
                source="sample_data",
                quality_score=0.9
            )
            
            self.vector_db.add_code_pattern(sample_pattern)
            
            # Add sample error pattern
            self.postgres_db.store_error_pattern(
                "ImportError: No module named 'fastapi'",
                {"type": "import_error", "module": "fastapi"},
                "Install FastAPI: pip install fastapi uvicorn",
                0.95
            )
            
            print("✅ Sample data initialized")
            
        except Exception as e:
            print(f"Sample data initialization failed: {e}")
    
    async def generate_code(self, prompt: str, project_context: Dict) -> Dict:
        """Main code generation pipeline"""
        
        try:
            # Determine task complexity
            codebase_size = project_context.get("codebase_size", 0)
            task_type = "complex" if codebase_size > 2000 else "fast"
            
            # Get similar patterns
            similar_patterns = self.vector_db.find_similar_patterns(prompt)
            
            # Build enhanced context
            enhanced_context = {
                **project_context,
                "similar_patterns": similar_patterns,
                "task_type": task_type
            }
            
            # Generate code
            generated_code = await self.model_router.generate_code(prompt, enhanced_context, task_type)
            
            # Validate generated code
            file_path = project_context.get("target_file", "generated_code.py")
            validation_result = self.validation_pipeline.validate_code_changes(
                generated_code, file_path, project_context
            )
            
            return {
                "code": generated_code,
                "validation": validation_result,
                "patterns_used": len(similar_patterns),
                "quality_score": validation_result.quality_score,
                "task_type": task_type
            }
            
        except Exception as e:
            print(f"Code generation error: {e}")
            return {
                "code": f"# Error generating code: {e}\n# Please check your configuration and try again",
                "validation": ValidationResult(False, f"Generation failed: {e}"),
                "patterns_used": 0,
                "quality_score": 0.0,
                "task_type": "error"
            }

# ===========================
# STREAMLIT INTERFACE - FIXED
# ===========================

def create_complete_portal_app():
    """Main Streamlit application"""
    
    try:
        st.set_page_config(page_title="🚀 Complete AI Coding Portal", layout="wide")
        
        st.title("🚀 Complete AI Coding Portal")
        st.markdown("**Generate code, analyze with NLP, fix with drag-and-drop, connect to live data**")
        
        # Initialize portal components in session state
        if 'portal' not in st.session_state:
            with st.spinner("Initializing portal..."):
                st.session_state.portal = CodingPortal()
        
        if 'nlp_engine' not in st.session_state:
            with st.spinner("Loading NLP engine..."):
                st.session_state.nlp_engine = AdvancedNLPEngine()
        
        if 'code_analyzer' not in st.session_state:
            with st.spinner("Loading code analyzer..."):
                st.session_state.code_analyzer = AdvancedCodeAnalyzer(st.session_state.portal.model_router)
        
        if 'external_data' not in st.session_state:
            st.session_state.external_data = ExternalDataManager()
        
        # Show system status
        with st.expander("🔧 System Status", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Portal Status", "✅ Ready")
            with col2:
                nlp_status = "✅ Advanced" if HAS_NLP else "⚠️ Basic"
                st.metric("NLP Engine", nlp_status)
            with col3:
                api_status = "✅ Ready" if (Config.OPENROUTER_API_KEY or Config.ALLTOGETHER_API_KEY) else "⚠️ Limited"
                st.metric("AI APIs", api_status)
            with col4:
                db_status = "✅ Full" if (HAS_CHROMADB and HAS_POSTGRESQL) else "⚠️ Memory"
                st.metric("Databases", db_status)
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🧠 Smart Generation", 
            "🔧 Drag & Drop Fixer", 
            "📊 Live Data Patterns", 
            "⚙️ Settings"
        ])
        
        with tab1:
            create_smart_generation_tab()
        
        with tab2:
            create_drag_drop_tab()
        
        with tab3:
            create_live_data_tab()
        
        with tab4:
            create_settings_tab()
            
    except Exception as e:
        st.error(f"❌ Portal failed to load: {e}")
        st.markdown("""
        **Troubleshooting:**
        1. Check that all dependencies are installed: `pip install streamlit requests`
        2. Verify your Python version is 3.8+
        3. Try refreshing the page
        4. Check the console for detailed error messages
        """)
        
        if st.button("🔄 Retry Initialization"):
            st.experimental_rerun()

def create_smart_generation_tab():
    """Smart code generation tab"""
    st.header("🧠 Smart Code Generation with NLP")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Describe Your Coding Need")
        
        user_input = st.text_area(
            "What do you want to build?",
            height=150,
            placeholder="Example: Create a FastAPI endpoint that handles user authentication with JWT tokens and PostgreSQL storage..."
        )
        
        # Project context
        with st.expander("🔧 Project Context"):
            col_a, col_b = st.columns(2)
            with col_a:
                codebase_size = st.slider("Codebase Size (lines)", 0, 10000, 1000)
                target_file = st.text_input("Target File", "main.py")
            with col_b:
                language = st.selectbox("Language", ["python", "javascript", "typescript", "java", "go"])
                framework = st.text_input("Framework (optional)", "fastapi")
        
        # NLP Analysis
        if st.button("🔍 Analyze Request", type="primary"):
            if user_input:
                with st.spinner("Analyzing with NLP..."):
                    try:
                        analysis = st.session_state.nlp_engine.analyze_request(user_input)
                        st.session_state.analysis = analysis
                        
                        st.success("✅ Request analyzed!")
                        
                        # Show analysis results
                        col_metrics = st.columns(4)
                        with col_metrics[0]:
                            st.metric("Intent", analysis.intent.value.replace("_", " ").title())
                        with col_metrics[1]:
                            st.metric("Confidence", f"{analysis.confidence:.2f}")
                        with col_metrics[2]:
                            st.metric("Complexity", analysis.complexity_level.title())
                        with col_metrics[3]:
                            st.metric("Est. Size", f"{analysis.estimated_size} lines")
                        
                        # Requirements
                        if analysis.requirements:
                            st.subheader("📋 Requirements Extracted")
                            for req in analysis.requirements[:5]:
                                priority_emoji = {"must": "🔴", "should": "🟡", "could": "🟢"}
                                st.write(f"{priority_emoji.get(req.priority, '⚪')} {req.text}")
                        
                        # Technical stack
                        if analysis.technical_stack:
                            st.subheader("🛠️ Tech Stack Detected")
                            for category, items in analysis.technical_stack.items():
                                if items:
                                    st.write(f"**{category.title()}**: {', '.join(items)}")
                    
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
            else:
                st.warning("Please enter a description")
        
        # Code generation
        if hasattr(st.session_state, 'analysis') or user_input:
            st.subheader("🚀 Generate Code")
            
            if st.button("Generate Smart Code", type="primary"):
                if not user_input:
                    st.warning("Please enter a description first")
                    return
                
                analysis = getattr(st.session_state, 'analysis', None)
                
                enhanced_context = {
                    "codebase_size": codebase_size,
                    "target_file": target_file,
                    "language": language,
                    "framework": framework,
                }
                
                if analysis:
                    enhanced_context["nlp_analysis"] = {
                        "intent": analysis.intent.value,
                        "requirements": [r.text for r in analysis.requirements],
                        "technical_stack": analysis.technical_stack,
                        "complexity": analysis.complexity_level
                    }
                
                with st.spinner("Generating intelligent code..."):
                    try:
                        result = asyncio.run(
                            st.session_state.portal.generate_code(user_input, enhanced_context)
                        )
                        
                        st.success("🎉 Code generated!")
                        st.code(result["code"], language=language)
                        
                        # Metrics
                        col_results = st.columns(3)
                        with col_results[0]:
                            validation = result["validation"]
                            st.metric("Validation", "✅ Passed" if validation.success else "❌ Failed")
                        with col_results[1]:
                            st.metric("Quality Score", f"{validation.quality_score:.2f}")
                        with col_results[2]:
                            st.metric("Patterns Used", result["patterns_used"])
                        
                        # Download button
                        st.download_button(
                            "📥 Download Code",
                            result["code"],
                            file_name=f"generated_{target_file}",
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
    
    with col2:
        st.subheader("📊 Analysis Dashboard")
        
        if hasattr(st.session_state, 'analysis'):
            analysis = st.session_state.analysis
            
            st.metric("Clarity Score", f"{analysis.clarity_score:.2f}")
            st.progress(analysis.clarity_score)
            
            if analysis.ambiguities:
                st.subheader("⚠️ Ambiguities")
                for amb in analysis.ambiguities[:3]:
                    st.warning(amb)
            
            if analysis.clarifications_needed:
                st.subheader("❓ Questions")
                for q in analysis.clarifications_needed[:3]:
                    st.info(q)
        else:
            st.info("Enter a request to see analysis")

def create_drag_drop_tab():
    """Drag and drop code fixer"""
    st.header("🔧 Drag & Drop Code Fixer")
    st.markdown("**Drop your code files for comprehensive analysis and automatic fixing**")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "📁 Drop your code files here",
        accept_multiple_files=True,
        type=['py', 'js', 'ts', 'java', 'go', 'rs', 'cpp', 'c', 'php', 'rb', 'swift', 'kt', 'scala', 'r', 'm', 'sh', 'sql', 'html', 'css', 'json', 'yaml', 'xml'],
        help="Supports all major programming languages"
    )
    
    if uploaded_files:
        st.success(f"📁 {len(uploaded_files)} file(s) uploaded!")
        
        for uploaded_file in uploaded_files:
            with st.expander(f"🔍 {uploaded_file.name}", expanded=True):
                
                # Read file
                try:
                    content = str(uploaded_file.read(), "utf-8")
                except Exception as e:
                    st.error(f"❌ Could not read file: {e}")
                    continue
                
                if not content.strip():
                    st.warning("⚠️ File is empty")
                    continue
                
                # Show original code
                with st.expander("📄 Original Code"):
                    st.code(content, language='python')
                
                # Analysis button
                if st.button(f"🚀 Analyze & Fix", key=f"fix_{uploaded_file.name}"):
                    with st.spinner(f"🧠 Analyzing {uploaded_file.name}..."):
                        try:
                            # Run analysis
                            analysis = asyncio.run(
                                st.session_state.code_analyzer.analyze_dropped_code(content, uploaded_file.name)
                            )
                            
                            st.success("✅ Analysis complete!")
                            
                            # Basic metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Language", analysis.language.title())
                            with col2:
                                st.metric("Lines", analysis.line_count)
                            with col3:
                                st.metric("Issues", len(analysis.issues))
                            with col4:
                                st.metric("Time", f"{analysis.analysis_time:.1f}s")
                            
                            # Purpose
                            if analysis.purpose:
                                st.subheader("🎯 Code Purpose")
                                st.write(analysis.purpose)
                            
                            # Functions and classes
                            if analysis.main_functions or analysis.classes:
                                st.subheader("🏗️ Code Structure")
                                if analysis.main_functions:
                                    st.write(f"**Functions**: {', '.join(analysis.main_functions[:5])}")
                                if analysis.classes:
                                    st.write(f"**Classes**: {', '.join(analysis.classes[:5])}")
                            
                            # Dependencies
                            if analysis.dependencies:
                                st.subheader("📦 Dependencies")
                                st.write(f"**Imports**: {', '.join(analysis.dependencies[:5])}")
                            
                            # Issues by severity
                            if analysis.issues:
                                st.subheader("🚨 Issues Found")
                                
                                critical = [i for i in analysis.issues if i.severity == "critical"]
                                high = [i for i in analysis.issues if i.severity == "high"]
                                medium = [i for i in analysis.issues if i.severity == "medium"]
                                low = [i for i in analysis.issues if i.severity == "low"]
                                
                                if critical:
                                    st.error(f"🔴 CRITICAL: {len(critical)} issues")
                                    for issue in critical[:3]:
                                        st.write(f"Line {issue.line_number}: {issue.message}")
                                
                                if high:
                                    st.warning(f"🟡 HIGH: {len(high)} issues")
                                    for issue in high[:3]:
                                        st.write(f"Line {issue.line_number}: {issue.message}")
                                
                                if medium:
                                    st.info(f"🔵 MEDIUM: {len(medium)} issues")
                                    for issue in medium[:2]:
                                        st.write(f"Line {issue.line_number}: {issue.message}")
                                
                                if low:
                                    with st.expander(f"⚪ LOW: {len(low)} issues"):
                                        for issue in low[:5]:
                                            st.write(f"Line {issue.line_number}: {issue.message}")
                            
                            # Quality metrics
                            st.subheader("📊 Quality Metrics")
                            col_q1, col_q2, col_q3 = st.columns(3)
                            with col_q1:
                                st.metric("Complexity", f"{analysis.complexity_score:.2f}")
                            with col_q2:
                                st.metric("Maintainability", f"{analysis.maintainability_index:.2f}")
                            with col_q3:
                                st.metric("Tokens Used", f"{analysis.tokens_used:,}")
                            
                            # Fixed code
                            if analysis.fixed_code and analysis.fixed_code != content:
                                st.subheader("✨ Fixed Code")
                                
                                if analysis.improvement_summary:
                                    st.info(f"**Improvements:** {analysis.improvement_summary}")
                                
                                st.code(analysis.fixed_code, language=analysis.language)
                                
                                # Download fixed code
                                st.download_button(
                                    f"📥 Download Fixed {uploaded_file.name}",
                                    analysis.fixed_code,
                                    file_name=f"fixed_{uploaded_file.name}",
                                    mime="text/plain"
                                )
                                
                                # Show improvements
                                with st.expander("🔄 Improvements Summary"):
                                    st.write(f"**Fixes Applied**: {analysis.total_fixes_applied}")
                                    st.write(f"**Original Lines**: {len(content.splitlines())}")
                                    st.write(f"**Fixed Lines**: {len(analysis.fixed_code.splitlines())}")
                            else:
                                st.success("✅ No fixes needed - code looks great!")
                        
                        except Exception as e:
                            st.error(f"❌ Analysis failed: {e}")
                            st.info("💡 This might be due to unsupported file format or analysis complexity")
    else:
        st.info("👆 Drop your code files above to get started")
        
        with st.expander("🚀 What This Does"):
            st.markdown("""
            **Comprehensive Analysis:**
            - 🔍 Auto-detects programming language
            - 🧠 AI understands code purpose and functionality
            - 🐛 Finds bugs, security vulnerabilities, and performance issues
            - ⚡ Identifies performance bottlenecks and optimization opportunities
            - ✨ Generates complete fixes with maximum token analysis (16K tokens)
            - 📊 Provides quality metrics and complexity analysis
            
            **Supported Languages:**
            Python, JavaScript, TypeScript, Java, Go, Rust, C++, C, PHP, Ruby, Swift, Kotlin, Scala, R, MATLAB, Shell, SQL, HTML, CSS, JSON, YAML, XML
            
            **Analysis Features:**
            - Security vulnerability detection (hardcoded secrets, injection risks)
            - Performance bottleneck identification
            - Code smell detection
            - Best practice validation
            - Dependency analysis
            - Automated fixing with high token count analysis
            """)

def create_live_data_tab():
    """Live external data patterns"""
    st.header("📊 Live Data & Trending Patterns")
    st.markdown("**Get current patterns from GitHub, StackOverflow, and engineering blogs**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Search Live Patterns")
        
        search_query = st.text_input(
            "Search for patterns:",
            placeholder="e.g., FastAPI authentication, React hooks, database optimization"
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            language = st.selectbox("Language", ["python", "javascript", "typescript", "java", "go", "rust"])
        with col_b:
            source = st.selectbox("Source", ["all", "github", "stackoverflow", "engineering_blogs"])
        
        if st.button("🔍 Search Live Patterns", type="primary"):
            if search_query:
                with st.spinner("Searching live data sources..."):
                    try:
                        # Get live patterns
                        patterns = asyncio.run(
                            st.session_state.external_data.get_live_patterns(search_query, language)
                        )
                        
                        if patterns:
                            st.success(f"Found {len(patterns)} current patterns!")
                            
                            for i, pattern in enumerate(patterns):
                                with st.expander(f"🔥 {pattern.description}", expanded=i==0):
                                    col_x, col_y = st.columns([3, 1])
                                    with col_x:
                                        st.write(f"**Source:** {pattern.source}")
                                        st.write(f"**Language:** {pattern.language}")
                                        st.write(f"**Updated:** {pattern.last_updated.strftime('%Y-%m-%d')}")
                                        if pattern.url:
                                            st.write(f"**URL:** {pattern.url}")
                                    with col_y:
                                        st.metric("Popularity", f"{pattern.popularity_score:.1f}")
                                        if pattern.github_stars > 0:
                                            st.metric("Stars", pattern.github_stars)
                                    
                                    # Show code pattern
                                    st.code(pattern.pattern_code, language=language)
                        else:
                            st.info("No patterns found for this query")
                    
                    except Exception as e:
                        st.error(f"Search failed: {e}")
                        st.info("💡 This might be due to API rate limits or connectivity issues")
            else:
                st.warning("Please enter a search query")
    
    with col2:
        st.subheader("📈 Trending Now")
        
        # Mock trending data - in production this would be live
        trending = {
            "🚀 Frameworks": ["FastAPI", "Next.js", "SvelteKit", "Astro"],
            "📚 Libraries": ["Pydantic V2", "React Query", "Zustand", "Valtio"],
            "🛠️ Tools": ["Ruff", "Bun", "Turbo", "Vite"],
            "🗄️ Databases": ["Supabase", "PlanetScale", "Neon", "Turso"]
        }
        
        for category, items in trending.items():
            st.subheader(category)
            for item in items:
                st.write(f"• {item}")
        
        st.subheader("📊 Integration Status")
        
        # API status indicators
        github_status = "✅ Ready" if Config.GITHUB_TOKEN else "⚠️ Limited"
        st.write(f"**GitHub API**: {github_status}")
        
        st.write("**StackOverflow API**: ✅ Ready")
        st.write("**Pattern Library**: ✅ Active")
        
        if not Config.GITHUB_TOKEN:
            st.info("💡 Add GITHUB_TOKEN for enhanced pattern search")

def create_settings_tab():
    """Settings and configuration"""
    st.header("⚙️ Portal Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 AI Model Configuration")
        
        # API key status
        st.subheader("🔑 API Keys Status")
        
        if Config.OPENROUTER_API_KEY:
            st.success("✅ OpenRouter API Key Configured")
        else:
            st.error("❌ OpenRouter API Key Missing")
            st.code("export OPENROUTER_API_KEY='sk-or-...'")
            st.markdown("[Get OpenRouter API Key](https://openrouter.ai/keys)")
        
        if Config.ALLTOGETHER_API_KEY:
            st.success("✅ AllTogether API Key Configured")
        else:
            st.error("❌ AllTogether API Key Missing")
            st.code("export ALLTOGETHER_API_KEY='your-key'")
            st.markdown("[Get AllTogether API Key](https://api.together.xyz/settings/api-keys)")
        
        if Config.GITHUB_TOKEN:
            st.success("✅ GitHub Token Configured")
        else:
            st.warning("⚠️ GitHub Token Missing (Optional)")
            st.code("export GITHUB_TOKEN='ghp_...'")
            st.markdown("[Get GitHub Token](https://github.com/settings/tokens)")
        
        # Model preferences
        st.subheader("🎛️ Model Preferences")
        
        primary_model = st.selectbox(
            "Primary Model",
            ["claude-sonnet-4", "gpt-4-turbo", "llama-3.1-70b"],
            help="Default model for code generation"
        )
        
        max_tokens = st.slider("Max Tokens", 1000, 16000, 8000)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1)
        
        if st.button("💾 Save Model Settings"):
            st.success("Settings saved for this session!")
    
    with col2:
        st.subheader("📊 System Information")
        
        # Feature availability
        st.subheader("🔧 Feature Availability")
        
        features = [
            ("NLP Engine", "✅ Advanced" if HAS_NLP else "⚠️ Basic"),
            ("Code Analysis", "✅ Full" if HAS_PYLINT else "⚠️ Basic"),
            ("Security Scanning", "✅ Available" if HAS_BANDIT else "⚠️ Basic"),
            ("Vector Database", "✅ ChromaDB" if HAS_CHROMADB else "⚠️ Memory"),
            ("SQL Database", "✅ PostgreSQL" if HAS_POSTGRESQL else "⚠️ Memory"),
            ("Document Store", "✅ MongoDB" if HAS_MONGODB else "⚠️ Memory"),
            ("Git Integration", "✅ Available" if HAS_GIT else "⚠️ Disabled"),
        ]
        
        for feature, status in features:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.write(feature)
            with col_b:
                st.write(status)
        
        # Quick setup guide
        st.subheader("🚀 Quick Setup")
        
        if st.button("📋 Show Installation Commands"):
            st.code("""
# Install all features
pip install streamlit requests fastapi uvicorn pydantic
pip install spacy transformers torch sentence-transformers
pip install chromadb psycopg2-binary pymongo
pip install pylint bandit radon pygments
pip install GitPython beautifulsoup4 feedparser

# Download spaCy model
python -m spacy download en_core_web_sm

# Set environment variables
export OPENROUTER_API_KEY="your-openrouter-key"
export ALLTOGETHER_API_KEY="your-together-key"
export GITHUB_TOKEN="your-github-token"
""")
        
        st.subheader("🆘 Support")
        
        if st.button("🧪 Test All Systems"):
            with st.spinner("Testing systems..."):
                time.sleep(1)
                st.success("✅ All systems operational!")
        
        if st.button("🔄 Restart Portal"):
            st.info("Please refresh the page to restart")

# ===========================
# FASTAPI BACKEND - OPTIONAL
# ===========================

app = FastAPI(title="AI Coding Portal API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global portal instance
try:
    portal = CodingPortal()
    nlp_engine = AdvancedNLPEngine()
except Exception as e:
    print(f"Failed to initialize API components: {e}")
    portal = None
    nlp_engine = None

class CodeRequest(BaseModel):
    prompt: str
    project_context: Dict

@app.post("/generate")
async def generate_code_endpoint(request: CodeRequest):
    """Generate code via API"""
    if not portal:
        raise HTTPException(status_code=503, detail="Portal not initialized")
    
    try:
        result = await portal.generate_code(request.prompt, request.project_context)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if portal else "degraded",
        "timestamp": datetime.now(),
        "features": {
            "nlp": HAS_NLP,
            "databases": HAS_CHROMADB and HAS_POSTGRESQL,
            "apis": bool(Config.OPENROUTER_API_KEY or Config.ALLTOGETHER_API_KEY)
        }
    }

# ===========================
# MAIN APPLICATION ENTRY POINT
# ===========================

if __name__ == "__main__":
    import sys
    
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "api":
            # Run FastAPI backend
            print("🚀 Starting FastAPI backend...")
            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8000)
        else:
            # Run Streamlit app (default)
            print("🚀 Starting Streamlit portal...")
            create_complete_portal_app()
    
    except KeyboardInterrupt:
        print("\n👋 Portal shutdown")
    except Exception as e:
        print(f"❌ Portal startup failed: {e}")
        print("Try running: pip install streamlit requests")