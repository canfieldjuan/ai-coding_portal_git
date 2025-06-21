"""
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
    text = "Here is code: ```python\nprint('hello')\n```"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0]["language"] == "python"

def test_complexity_estimation():
    """Test code complexity estimation"""
    simple_code = "print('hello')"
    metrics = estimate_code_complexity(simple_code, "python")
    assert metrics["estimated_complexity"] == "low"
    
    complex_code = "\n".join([f"line_{i} = {i}" for i in range(600)])
    metrics = estimate_code_complexity(complex_code, "python")
    assert metrics["estimated_complexity"] == "high"
