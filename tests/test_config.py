"""
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
