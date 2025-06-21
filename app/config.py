"""
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
