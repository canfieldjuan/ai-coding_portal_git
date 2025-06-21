#!/usr/bin/env python3
"""
AI Coding Portal - Safe Batch Creator
Creates all LOW-RISK components that can't break anything
"""

import os
from pathlib import Path
import sys

def create_safe_files():
    """Create all safe components in batch"""
    
    print("🔒 AI Coding Portal - Creating SAFE Components")
    print("=" * 60)
    print("Creating: Models, Utils, Config, Tests")
    print("Risk Level: LOW (Can't break anything)")
    print("=" * 60)
    
    safe_files = {
        # =======================================
        # CONFIGURATION (Safe - Just settings)
        # =======================================
        
        "app/config.py": '''"""
Configuration management for AI Coding Portal
"""

import os
from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, validator
from pathlib import Path

class Settings(BaseSettings):
    """Application settings with validation"""
    
    # API Configuration
    openrouter_api_key: str = ""
    default_model: str = "anthropic/claude-3.5-sonnet"
    api_timeout: int = 60
    
    # Application Configuration
    debug: bool = False
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    
    # File Handling
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: str = "py,js,ts,java,go,rs,cpp,c,php,rb,swift,kt,html,css,sql,yaml,json"
    upload_dir: str = "./data/uploads"
    generated_dir: str = "./data/generated"
    cache_dir: str = "./data/cache"
    
    # Database Configuration  
    database_url: str = "sqlite:///./data/coding_portal.db"
    
    # Cache Configuration
    cache_ttl: int = 3600  # 1 hour
    max_cache_size: int = 1000
    
    # AI Model Configuration
    models: Dict[str, str] = {
        "fast": "meta-llama/llama-3.1-8b-instruct",
        "balanced": "anthropic/claude-3.5-sonnet", 
        "powerful": "openai/gpt-4-turbo",
        "code_focused": "deepseek/deepseek-coder-33b-instruct"
    }
    
    @validator("allowed_extensions")
    def validate_extensions(cls, v):
        """Convert comma-separated string to list"""
        return [ext.strip() for ext in v.split(",")]
    
    @validator("openrouter_api_key")
    def validate_api_key(cls, v):
        """Warn if API key is missing"""
        if not v:
            print("⚠️  WARNING: OPENROUTER_API_KEY not set - AI features will use mock responses")
        return v
    
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.upload_dir,
            self.generated_dir, 
            self.cache_dir,
            "./data"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Create directories on import
settings.create_directories()

# Model configurations for different use cases
MODEL_CONFIGS = {
    "code_generation": {
        "model": settings.models["balanced"],
        "temperature": 0.1,
        "max_tokens": 4000,
        "system_prompt": "You are an expert software engineer. Generate clean, production-ready code with proper error handling and documentation."
    },
    "code_analysis": {
        "model": settings.models["code_focused"], 
        "temperature": 0.0,
        "max_tokens": 2000,
        "system_prompt": "You are a code analysis expert. Identify issues, suggest improvements, and provide detailed explanations."
    },
    "code_fixing": {
        "model": settings.models["powerful"],
        "temperature": 0.1, 
        "max_tokens": 6000,
        "system_prompt": "You are a debugging expert. Fix code issues while maintaining original functionality and improving code quality."
    },
    "quick_help": {
        "model": settings.models["fast"],
        "temperature": 0.3,
        "max_tokens": 1000,
        "system_prompt": "You are a helpful programming assistant. Provide quick, accurate answers to coding questions."
    }
}

# Supported programming languages and their configurations
LANGUAGE_CONFIGS = {
    "python": {
        "extensions": [".py"],
        "comment_style": "#",
        "execution_supported": True,
        "analysis_tools": ["pylint", "bandit", "mypy"]
    },
    "javascript": {
        "extensions": [".js", ".mjs"],
        "comment_style": "//", 
        "execution_supported": True,
        "analysis_tools": ["eslint", "jshint"]
    },
    "typescript": {
        "extensions": [".ts", ".tsx"],
        "comment_style": "//",
        "execution_supported": True, 
        "analysis_tools": ["tslint", "typescript"]
    },
    "java": {
        "extensions": [".java"],
        "comment_style": "//",
        "execution_supported": False,
        "analysis_tools": ["checkstyle", "spotbugs"]
    },
    "go": {
        "extensions": [".go"],
        "comment_style": "//",
        "execution_supported": True,
        "analysis_tools": ["golint", "go vet"]
    },
    "rust": {
        "extensions": [".rs"],
        "comment_style": "//", 
        "execution_supported": True,
        "analysis_tools": ["clippy", "rustfmt"]
    },
    "cpp": {
        "extensions": [".cpp", ".cc", ".cxx"],
        "comment_style": "//",
        "execution_supported": False,
        "analysis_tools": ["cppcheck", "clang-tidy"]
    },
    "c": {
        "extensions": [".c"],
        "comment_style": "//",
        "execution_supported": False,
        "analysis_tools": ["cppcheck", "splint"]
    }
}

def get_model_config(task_type: str) -> Dict[str, Any]:
    """Get model configuration for specific task type"""
    return MODEL_CONFIGS.get(task_type, MODEL_CONFIGS["code_generation"])

def detect_language(filename: str) -> Optional[str]:
    """Detect programming language from filename"""
    extension = Path(filename).suffix.lower()
    
    for language, config in LANGUAGE_CONFIGS.items():
        if extension in config["extensions"]:
            return language
    
    return None

def is_supported_file(filename: str) -> bool:
    """Check if file type is supported"""
    extension = Path(filename).suffix.lstrip(".")
    return extension in settings.allowed_extensions
''',

        # =======================================
        # DATABASE MODELS (Safe - Just data definitions)
        # =======================================
        
        "app/models/base.py": '''"""
Base database model classes
"""

from sqlalchemy import Column, Integer, DateTime, String, Text, Float, Boolean, JSON
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base, declared_attr

Base = declarative_base()

class BaseModel(Base):
    """Base model with common fields"""
    __abstract__ = True
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def __repr__(self):
        """String representation"""
        return f"<{self.__class__.__name__}(id={self.id})>"
''',

        "app/models/code.py": '''"""
Code-related database models
"""

from sqlalchemy import Column, Integer, String, Text, Float, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from .base import BaseModel

class CodeProject(BaseModel):
    """Code project/session management"""
    __tablename__ = "code_projects"
    
    name = Column(String(255), nullable=False)
    description = Column(Text)
    language = Column(String(50))
    framework = Column(String(100))
    status = Column(String(50), default="active")  # active, completed, archived
    metadata = Column(JSON, default=dict)
    
    # Relationships
    files = relationship("CodeFile", back_populates="project", cascade="all, delete-orphan")
    generations = relationship("CodeGeneration", back_populates="project")

class CodeFile(BaseModel):
    """Individual code files"""
    __tablename__ = "code_files"
    
    project_id = Column(Integer, ForeignKey("code_projects.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    original_content = Column(Text)
    processed_content = Column(Text)
    file_size = Column(Integer)
    language = Column(String(50))
    file_hash = Column(String(64))  # For deduplication
    
    # Analysis results
    analysis_status = Column(String(50), default="pending")
    issues_found = Column(Integer, default=0)
    security_score = Column(Float, default=0.0)
    quality_score = Column(Float, default=0.0)
    analysis_data = Column(JSON, default=dict)
    
    # Relationships
    project = relationship("CodeProject", back_populates="files")
    analyses = relationship("CodeAnalysis", back_populates="file")

class CodeGeneration(BaseModel):
    """AI code generation requests and results"""
    __tablename__ = "code_generations"
    
    project_id = Column(Integer, ForeignKey("code_projects.id"))
    prompt = Column(Text, nullable=False)
    context = Column(JSON, default=dict)
    
    # AI Configuration
    model_used = Column(String(100))
    task_type = Column(String(50))  # generation, analysis, fixing, etc.
    
    # Request details
    max_tokens = Column(Integer)
    temperature = Column(Float)
    
    # Results
    generated_code = Column(Text)
    status = Column(String(50), default="pending")  # pending, completed, failed
    tokens_used = Column(Integer)
    processing_time = Column(Float)
    error_message = Column(Text)
    
    # Quality metrics
    validation_passed = Column(Boolean, default=False)
    quality_score = Column(Float)
    
    # Relationships
    project = relationship("CodeProject", back_populates="generations")

class CodeAnalysis(BaseModel):
    """Detailed code analysis results"""
    __tablename__ = "code_analyses"
    
    file_id = Column(Integer, ForeignKey("code_files.id"), nullable=False)
    analysis_type = Column(String(50))  # security, performance, style, etc.
    
    # Analysis results
    issues = Column(JSON, default=list)  # List of found issues
    suggestions = Column(JSON, default=list)  # Improvement suggestions
    metrics = Column(JSON, default=dict)  # Various code metrics
    
    # Scores
    overall_score = Column(Float, default=0.0)
    security_score = Column(Float, default=0.0)
    performance_score = Column(Float, default=0.0)
    maintainability_score = Column(Float, default=0.0)
    
    # Processing info
    analyzer_version = Column(String(50))
    processing_time = Column(Float)
    
    # Relationships
    file = relationship("CodeFile", back_populates="analyses")

class APIUsage(BaseModel):
    """Track API usage for monitoring and rate limiting"""
    __tablename__ = "api_usage"
    
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10))
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    
    # Request details
    request_size = Column(Integer)
    response_size = Column(Integer)
    processing_time = Column(Float)
    status_code = Column(Integer)
    
    # AI Usage (if applicable)
    model_used = Column(String(100))
    tokens_used = Column(Integer)
    cost_estimate = Column(Float)
    
    # Error tracking
    error_type = Column(String(100))
    error_message = Column(Text)
''',

        "app/models/user.py": '''"""
User and session management models
"""

from sqlalchemy import Column, String, Boolean, DateTime, JSON, Integer
from .base import BaseModel

class Session(BaseModel):
    """User session management (for demo purposes)"""
    __tablename__ = "sessions"
    
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    
    # Session data
    preferences = Column(JSON, default=dict)
    last_activity = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Usage tracking
    requests_count = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)
    files_processed = Column(Integer, default=0)

class UserPreferences(BaseModel):
    """User preferences and settings"""
    __tablename__ = "user_preferences"
    
    session_id = Column(String(255), nullable=False, index=True)
    
    # Code generation preferences
    preferred_language = Column(String(50), default="python")
    preferred_framework = Column(String(100))
    code_style = Column(String(50), default="standard")
    
    # AI model preferences
    preferred_model_tier = Column(String(50), default="balanced")
    temperature_preference = Column(Float, default=0.1)
    max_tokens_preference = Column(Integer, default=4000)
    
    # UI preferences
    theme = Column(String(20), default="dark")
    editor_font_size = Column(Integer, default=14)
    auto_save = Column(Boolean, default=True)
    
    # Analysis preferences
    security_analysis = Column(Boolean, default=True)
    performance_analysis = Column(Boolean, default=True)
    style_analysis = Column(Boolean, default=False)
''',

        # =======================================
        # UTILITY FUNCTIONS (Safe - Pure functions)
        # =======================================
        
        "app/utils/exceptions.py": '''"""
Custom exception classes for the AI Coding Portal
"""

class CodingPortalException(Exception):
    """Base exception for the coding portal"""
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code

class AIServiceError(CodingPortalException):
    """AI service related errors"""
    pass

class RateLimitError(AIServiceError):
    """Rate limiting errors"""
    pass

class ModelNotAvailableError(AIServiceError):
    """Model availability errors"""
    pass

class FileProcessingError(CodingPortalException):
    """File processing errors"""
    pass

class ValidationError(CodingPortalException):
    """Input validation errors"""
    pass

class DatabaseError(CodingPortalException):
    """Database operation errors"""
    pass

class ConfigurationError(CodingPortalException):
    """Configuration errors"""
    pass

class AuthenticationError(CodingPortalException):
    """Authentication errors"""
    pass

class AuthorizationError(CodingPortalException):
    """Authorization errors"""
    pass

class ResourceNotFoundError(CodingPortalException):
    """Resource not found errors"""
    pass

class ServiceUnavailableError(CodingPortalException):
    """Service unavailable errors"""
    pass
''',

        "app/utils/validators.py": '''"""
Input validation utilities
"""

import os
import mimetypes
import re
from typing import List, Optional, Tuple
from pathlib import Path

from ..config import settings, is_supported_file, detect_language
from .exceptions import ValidationError

class FileValidator:
    """File upload validation"""
    
    @staticmethod
    def validate_file_size(file_size: int) -> bool:
        """Validate file size"""
        if file_size > settings.max_file_size:
            raise ValidationError(
                f"File size ({file_size} bytes) exceeds maximum allowed size ({settings.max_file_size} bytes)",
                error_code="FILE_TOO_LARGE"
            )
        return True
    
    @staticmethod
    def validate_file_type(filename: str, content: bytes = None) -> Tuple[bool, str]:
        """Validate file type and detect language"""
        
        # Check extension
        if not is_supported_file(filename):
            raise ValidationError(
                f"File type not supported. Allowed extensions: {', '.join(settings.allowed_extensions)}",
                error_code="UNSUPPORTED_FILE_TYPE"
            )
        
        # Detect programming language
        language = detect_language(filename)
        if not language and content:
            # Try to detect from content
            language = FileValidator._detect_language_from_content(content, filename)
        
        return True, language or "text"
    
    @staticmethod
    def _detect_language_from_content(content: bytes, filename: str) -> Optional[str]:
        """Detect programming language from file content"""
        try:
            text_content = content.decode('utf-8', errors='ignore')
            
            # Simple heuristics for language detection
            patterns = {
                "python": [r'#!/usr/bin/env python', r'import \\w+', r'from \\w+ import', r'def \\w+\\(', r'class \\w+:'],
                "javascript": [r'function \\w+', r'const \\w+', r'let \\w+', r'var \\w+', r'=> {'],
                "typescript": [r'interface \\w+', r'type \\w+', r': \\w+\\[\\]', r'extends \\w+'],
                "java": [r'public class', r'import java\\.', r'public static void main', r'@Override'],
                "go": [r'package main', r'func main\\(\\)', r'import \\(', r'func \\w+\\('],
                "rust": [r'fn main\\(\\)', r'use std::', r'impl \\w+', r'struct \\w+'],
                "cpp": [r'#include <', r'using namespace', r'int main\\(', r'std::'],
                "c": [r'#include <stdio\\.h>', r'int main\\(', r'printf\\('],
                "php": [r'<\\?php', r'function \\w+\\(', r'\\$\\w+'],
                "ruby": [r'def \\w+', r'class \\w+', r'require \\'', r'puts '],
                "swift": [r'import Swift', r'func \\w+\\(', r'var \\w+:', r'let \\w+:']
            }
            
            scores = {}
            for language, lang_patterns in patterns.items():
                score = 0
                for pattern in lang_patterns:
                    matches = len(re.findall(pattern, text_content, re.IGNORECASE))
                    score += matches
                scores[language] = score
            
            if scores and max(scores.values()) > 0:
                return max(scores, key=scores.get)
            
            # Fallback to extension-based detection
            return detect_language(filename)
            
        except Exception:
            return None
    
    @staticmethod
    def validate_filename(filename: str) -> bool:
        """Validate filename"""
        if not filename or len(filename.strip()) == 0:
            raise ValidationError("Filename cannot be empty", error_code="EMPTY_FILENAME")
        
        if len(filename) > 255:
            raise ValidationError("Filename too long (max 255 characters)", error_code="FILENAME_TOO_LONG")
        
        # Check for dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\\0']
        if any(char in filename for char in dangerous_chars):
            raise ValidationError("Filename contains invalid characters", error_code="INVALID_FILENAME")
        
        # Check for dangerous names
        dangerous_names = ['CON', 'PRN', 'AUX', 'NUL'] + [f'COM{i}' for i in range(1, 10)] + [f'LPT{i}' for i in range(1, 10)]
        if Path(filename).stem.upper() in dangerous_names:
            raise ValidationError("Filename not allowed", error_code="DANGEROUS_FILENAME")
        
        return True
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage"""
        # Remove dangerous characters
        safe_chars = []
        for char in filename:
            if char.isalnum() or char in '.-_':
                safe_chars.append(char)
            else:
                safe_chars.append('_')
        
        sanitized = ''.join(safe_chars)
        
        # Ensure it's not empty
        if not sanitized.strip('._'):
            sanitized = "unknown_file"
        
        # Ensure it's not too long
        if len(sanitized) > 100:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:100-len(ext)] + ext
        
        return sanitized

class CodeValidator:
    """Code content validation"""
    
    @staticmethod
    def validate_code_content(content: str, language: str) -> bool:
        """Validate code content"""
        
        if not content or len(content.strip()) == 0:
            raise ValidationError("Code content cannot be empty", error_code="EMPTY_CODE")
        
        if len(content) > 1_000_000:  # 1MB text limit
            raise ValidationError("Code content too large (max 1MB)", error_code="CODE_TOO_LARGE")
        
        # Language-specific validation
        if language == "python":
            return CodeValidator._validate_python_code(content)
        elif language in ["javascript", "typescript"]:
            return CodeValidator._validate_javascript_code(content)
        elif language == "java":
            return CodeValidator._validate_java_code(content)
        
        return True
    
    @staticmethod
    def _validate_python_code(content: str) -> bool:
        """Basic Python code validation"""
        try:
            # Try to parse the AST
            import ast
            ast.parse(content)
            return True
        except SyntaxError as e:
            raise ValidationError(f"Python syntax error: {e}", error_code="PYTHON_SYNTAX_ERROR")
        except Exception:
            # Allow through if we can't parse (might be incomplete code)
            return True
    
    @staticmethod
    def _validate_javascript_code(content: str) -> bool:
        """Basic JavaScript code validation"""
        # Simple checks for common syntax issues
        lines = content.split('\\n')
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('/*'):
                continue
            
            # Check for unterminated strings
            if line.count('"') % 2 != 0 or line.count("'") % 2 != 0:
                if not line.endswith('\\\\'):  # Not escaped
                    pass  # Could be multiline string, allow it
        
        return True
    
    @staticmethod
    def _validate_java_code(content: str) -> bool:
        """Basic Java code validation"""
        # Check for basic Java structure
        if 'public class' not in content and 'class' not in content:
            # Might be just a method or fragment, allow it
            pass
        
        return True

class PromptValidator:
    """AI prompt validation"""
    
    @staticmethod
    def validate_prompt(prompt: str) -> bool:
        """Validate AI prompt"""
        
        if not prompt or len(prompt.strip()) == 0:
            raise ValidationError("Prompt cannot be empty", error_code="EMPTY_PROMPT")
        
        if len(prompt) > 10_000:
            raise ValidationError("Prompt too long (max 10,000 characters)", error_code="PROMPT_TOO_LONG")
        
        # Check for potentially harmful content
        harmful_patterns = [
            "ignore previous instructions",
            "forget everything above",
            "system:",
            "jailbreak",
            "DAN mode",
            "roleplay as",
            "pretend to be"
        ]
        
        prompt_lower = prompt.lower()
        for pattern in harmful_patterns:
            if pattern in prompt_lower:
                raise ValidationError(
                    "Prompt contains potentially harmful content", 
                    error_code="HARMFUL_PROMPT"
                )
        
        return True
    
    @staticmethod
    def sanitize_prompt(prompt: str) -> str:
        """Sanitize prompt for safe processing"""
        # Remove excessive whitespace
        sanitized = ' '.join(prompt.split())
        
        # Limit length
        if len(sanitized) > 10_000:
            sanitized = sanitized[:10_000] + "..."
        
        # Remove potential injection attempts
        sanitized = re.sub(r'\\n\\s*system:', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized

class RequestValidator:
    """API request validation"""
    
    @staticmethod
    def validate_task_type(task_type: str) -> bool:
        """Validate task type"""
        valid_types = ["code_generation", "code_analysis", "code_fixing", "quick_help"]
        
        if task_type not in valid_types:
            raise ValidationError(
                f"Invalid task type. Must be one of: {', '.join(valid_types)}",
                error_code="INVALID_TASK_TYPE"
            )
        
        return True
    
    @staticmethod
    def validate_language(language: str) -> bool:
        """Validate programming language"""
        from ..config import LANGUAGE_CONFIGS
        
        if language not in LANGUAGE_CONFIGS:
            raise ValidationError(
                f"Unsupported language. Supported: {', '.join(LANGUAGE_CONFIGS.keys())}",
                error_code="UNSUPPORTED_LANGUAGE"
            )
        
        return True
''',

        "app/utils/helpers.py": '''"""
General helper utilities
"""

import hashlib
import time
import uuid
import re
from typing import Any, Dict, List, Optional
from pathlib import Path
import json

def generate_session_id() -> str:
    """Generate unique session ID"""
    return str(uuid.uuid4())

def generate_file_hash(content: bytes) -> str:
    """Generate SHA-256 hash of file content"""
    return hashlib.sha256(content).hexdigest()

def create_safe_filename(original_name: str, prefix: str = None) -> str:
    """Create a safe filename with timestamp"""
    timestamp = int(time.time())
    
    # Sanitize the original name
    safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', original_name)
    
    if prefix:
        return f"{prefix}_{timestamp}_{safe_name}"
    else:
        return f"{timestamp}_{safe_name}"

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """Extract code blocks from markdown text"""
    
    # Pattern to match code blocks
    pattern = r'```(\\w+)?\\n(.*?)\\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    code_blocks = []
    for language, code in matches:
        code_blocks.append({
            "language": language or "text",
            "code": code.strip()
        })
    
    return code_blocks

def extract_inline_code(text: str) -> List[str]:
    """Extract inline code segments from text"""
    pattern = r'`([^`]+)`'
    return re.findall(pattern, text)

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def ensure_directory_exists(path: str) -> bool:
    """Ensure directory exists, create if it doesn't"""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False

def clean_json(data: Any) -> Any:
    """Clean data for JSON serialization"""
    if isinstance(data, dict):
        return {k: clean_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json(item) for item in data]
    elif hasattr(data, 'isoformat'):  # datetime
        return data.isoformat()
    elif hasattr(data, '__dict__'):  # custom objects
        return clean_json(data.__dict__)
    else:
        return data

def parse_requirements_from_text(text: str) -> List[Dict[str, str]]:
    """Parse requirements from natural language text"""
    requirements = []
    
    # Look for requirement keywords
    requirement_patterns = [
        r"(must|should|shall|will|need to|require|want to)\\s+(.+?)(?:\\.|$)",
        r"it\\s+(must|should|shall|will|needs to)\\s+(.+?)(?:\\.|$)"
    ]
    
    for pattern in requirement_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for priority, requirement in matches:
            requirements.append({
                "text": requirement.strip(),
                "priority": priority.lower(),
                "type": "functional"
            })
    
    return requirements

def estimate_code_complexity(code: str, language: str) -> Dict[str, Any]:
    """Estimate code complexity metrics"""
    lines = code.split('\\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    # Basic metrics
    metrics = {
        "total_lines": len(lines),
        "code_lines": len(non_empty_lines),
        "comment_lines": 0,
        "blank_lines": len(lines) - len(non_empty_lines),
        "estimated_complexity": "low"
    }
    
    # Count comments based on language
    comment_chars = {
        "python": "#",
        "javascript": "//",
        "typescript": "//", 
        "java": "//",
        "go": "//",
        "rust": "//",
        "cpp": "//",
        "c": "//",
        "php": "//",
        "ruby": "#",
        "swift": "//"
    }
    
    comment_char = comment_chars.get(language, "#")
    for line in lines:
        if line.strip().startswith(comment_char):
            metrics["comment_lines"] += 1
    
    # Estimate complexity
    if metrics["code_lines"] > 500:
        metrics["estimated_complexity"] = "high"
    elif metrics["code_lines"] > 100:
        metrics["estimated_complexity"] = "medium"
    
    return metrics

def format_duration(seconds: float) -> str:
    """Format duration in human readable format"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:.0f}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

def extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from text"""
    url_pattern = r'https?://[^\\s<>"{}|\\\\^`\\[\\]]+'
    return re.findall(url_pattern, text)

def validate_json_structure(data: str, expected_keys: List[str] = None) -> bool:
    """Validate JSON structure"""
    try:
        parsed = json.loads(data)
        
        if expected_keys:
            missing_keys = set(expected_keys) - set(parsed.keys())
            if missing_keys:
                return False
        
        return True
    except (json.JSONDecodeError, AttributeError):
        return False

def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries"""
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result
''',

        "app/utils/cache.py": '''"""
Caching utilities for AI responses and data
"""

import json
import time
import hashlib
from typing import Any, Optional, Dict
from cachetools import TTLCache
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """In-memory cache manager with TTL support"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = TTLCache(maxsize=max_size, ttl=ttl)
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
            "errors": 0
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if key in self.cache:
                self.stats["hits"] += 1
                value = self.cache[key]
                logger.debug(f"Cache hit: {key}")
                return value
            else:
                self.stats["misses"] += 1
                logger.debug(f"Cache miss: {key}")
                return None
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            # For TTLCache, we can't set per-item TTL easily
            # We'll store with timestamp and check manually if needed
            cache_entry = {
                "value": value,
                "timestamp": time.time(),
                "ttl": ttl
            }
            
            self.cache[key] = cache_entry
            self.stats["sets"] += 1
            logger.debug(f"Cache set: {key}")
            return True
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            if key in self.cache:
                del self.cache[key]
                logger.debug(f"Cache delete: {key}")
                return True
            return False
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            self.cache.clear()
            logger.info("Cache cleared")
            return True
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache clear error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / max(total_requests, 1)) * 100
        
        return {
            **self.stats,
            "size": len(self.cache),
            "max_size": self.cache.maxsize,
            "hit_rate": round(hit_rate, 2),
            "total_requests": total_requests
        }
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()

class TimedCache:
    """Simple time-based cache"""
    
    def __init__(self, default_ttl: int = 300):
        self.cache = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value if not expired"""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < entry["ttl"]:
                return entry["value"]
            else:
                # Expired, remove it
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value with TTL"""
        self.cache[key] = {
            "value": value,
            "timestamp": time.time(),
            "ttl": ttl or self.default_ttl
        }
    
    def clear_expired(self) -> int:
        """Clear expired entries and return count"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if current_time - entry["timestamp"] >= entry["ttl"]:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)

class LRUCache:
    """Simple LRU cache implementation"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get value and update access order"""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value and manage size"""
        if key in self.cache:
            # Update existing
            self.cache[key] = value
            self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # Add new
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)
''',

        # =======================================
        # TEST FILES (Safe - Just test templates)
        # =======================================
        
        "tests/test_models.py": '''"""
Test database models
"""

import pytest
from app.models.base import BaseModel
from app.models.code import CodeProject, CodeFile, CodeGeneration, CodeAnalysis
from app.models.user import Session, UserPreferences

def test_base_model():
    """Test base model functionality"""
    # This would test the base model if we had a test database
    pass

def test_code_project_model():
    """Test CodeProject model"""
    # Test model creation and validation
    pass

def test_code_file_model():
    """Test CodeFile model"""
    # Test file model functionality
    pass

def test_session_model():
    """Test Session model"""
    # Test session management
    pass

# More tests would be added here
''',

        "tests/test_utils.py": '''"""
Test utility functions
"""

import pytest
from app.utils.validators import FileValidator, CodeValidator, PromptValidator
from app.utils.helpers import (
    generate_session_id, format_file_size, extract_code_blocks,
    estimate_code_complexity
)
from app.utils.exceptions import ValidationError

def test_file_validator():
    """Test file validation"""
    # Test valid filename
    assert FileValidator.validate_filename("test.py") == True
    
    # Test invalid filename
    with pytest.raises(ValidationError):
        FileValidator.validate_filename("")

def test_code_validator():
    """Test code validation"""
    # Test valid Python code
    valid_python = "def hello(): return 'world'"
    assert CodeValidator.validate_code_content(valid_python, "python") == True
    
    # Test invalid Python code
    with pytest.raises(ValidationError):
        CodeValidator.validate_code_content("def invalid(", "python")

def test_prompt_validator():
    """Test prompt validation"""
    # Test valid prompt
    assert PromptValidator.validate_prompt("Create a hello world function") == True
    
    # Test empty prompt
    with pytest.raises(ValidationError):
        PromptValidator.validate_prompt("")

def test_helpers():
    """Test helper functions"""
    # Test session ID generation
    session_id = generate_session_id()
    assert len(session_id) > 0
    
    # Test file size formatting
    assert format_file_size(1024) == "1.0 KB"
    assert format_file_size(1048576) == "1.0 MB"
    
    # Test code block extraction
    text = "Here is code: ```python\\nprint('hello')\\n```"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0]["language"] == "python"

def test_complexity_estimation():
    """Test code complexity estimation"""
    simple_code = "print('hello')"
    metrics = estimate_code_complexity(simple_code, "python")
    assert metrics["estimated_complexity"] == "low"
    
    complex_code = "\\n".join([f"line_{i} = {i}" for i in range(600)])
    metrics = estimate_code_complexity(complex_code, "python")
    assert metrics["estimated_complexity"] == "high"
''',

        "tests/test_config.py": '''"""
Test configuration management
"""

import pytest
from app.config import Settings, get_model_config, detect_language, is_supported_file

def test_settings_validation():
    """Test settings validation"""
    settings = Settings()
    
    # Test default values
    assert settings.port == 8000
    assert settings.debug == False
    assert settings.max_file_size > 0

def test_model_config():
    """Test model configuration retrieval"""
    config = get_model_config("code_generation")
    
    assert "model" in config
    assert "temperature" in config
    assert "max_tokens" in config
    assert "system_prompt" in config

def test_language_detection():
    """Test programming language detection"""
    assert detect_language("test.py") == "python"
    assert detect_language("test.js") == "javascript"
    assert detect_language("test.java") == "java"
    assert detect_language("unknown.xyz") is None

def test_file_support():
    """Test file type support checking"""
    assert is_supported_file("test.py") == True
    assert is_supported_file("test.js") == True
    assert is_supported_file("test.exe") == False
'''
    }
    
    total_files = 0
    successful_files = 0
    
    for file_path, content in safe_files.items():
        try:
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Created: {file_path}")
            successful_files += 1
            
        except Exception as e:
            print(f"❌ Failed: {file_path} - {e}")
        
        total_files += 1
    
    print("=" * 60)
    print(f"🔒 SAFE BATCH CREATION COMPLETE!")
    print(f"✅ Successfully created: {successful_files}/{total_files} files")
    print("=" * 60)
    
    return successful_files == total_files

def validate_safe_components():
    """Validate that all safe components were created correctly"""
    
    print("\n🔍 Validating safe components...")
    
    required_files = [
        "app/config.py",
        "app/models/base.py",
        "app/models/code.py", 
        "app/models/user.py",
        "app/utils/exceptions.py",
        "app/utils/validators.py",
        "app/utils/helpers.py",
        "app/utils/cache.py",
        "tests/test_models.py",
        "tests/test_utils.py",
        "tests/test_config.py"
    ]
    
    missing_files = []
    syntax_errors = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            # Check for basic syntax errors
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Basic Python syntax check
                if file_path.endswith('.py'):
                    compile(content, file_path, 'exec')
                    
            except SyntaxError as e:
                syntax_errors.append(f"{file_path}: {e}")
            except Exception:
                pass  # Skip other errors for now
    
    if missing_files:
        print("❌ Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
    
    if syntax_errors:
        print("❌ Syntax errors:")
        for error in syntax_errors:
            print(f"   - {error}")
    
    if not missing_files and not syntax_errors:
        print("✅ All safe components validated successfully!")
        return True
    else:
        return False

def main():
    """Main function"""
    
    print("🔒 AI Coding Portal - Safe Batch Creator")
    print("Creating LOW-RISK components that can't break anything")
    print("=" * 60)
    
    # Check if app directory exists
    if not Path("app").exists():
        print("❌ Error: 'app' directory not found")
        print("   Please run create_project_structure.py first")
        return False
    
    try:
        # Create safe files
        success = create_safe_files()
        
        if success:
            # Validate components
            if validate_safe_components():
                print("\n" + "=" * 60)
                print("🎉 SAFE BATCH CREATION SUCCESSFUL!")
                print("=" * 60)
                print("\n📋 What was created:")
                print("✅ Configuration management (app/config.py)")
                print("✅ Database models (app/models/)")
                print("✅ Utility functions (app/utils/)")
                print("✅ Test scaffolding (tests/)")
                print("\n🔄 Next Steps:")
                print("1. 🔧 Test safe components: python -m pytest tests/ -v")
                print("2. ⚠️  Create risky components: python scripts/batch_create_risky.py")
                print("3. 🚀 Run complete system: python run.py")
                print("\n🎯 Ready for risky component creation!")
                return True
            else:
                print("❌ Validation failed")
                return False
        else:
            print("❌ Safe file creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during safe batch creation: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)