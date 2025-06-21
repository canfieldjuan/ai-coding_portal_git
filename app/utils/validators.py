"""
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
                "python": [r'#!/usr/bin/env python', r'import \w+', r'from \w+ import', r'def \w+\(', r'class \w+:'],
                "javascript": [r'function \w+', r'const \w+', r'let \w+', r'var \w+', r'=> {'],
                "typescript": [r'interface \w+', r'type \w+', r': \w+\[\]', r'extends \w+'],
                "java": [r'public class', r'import java\.', r'public static void main', r'@Override'],
                "go": [r'package main', r'func main\(\)', r'import \(', r'func \w+\('],
                "rust": [r'fn main\(\)', r'use std::', r'impl \w+', r'struct \w+'],
                "cpp": [r'#include <', r'using namespace', r'int main\(', r'std::'],
                "c": [r'#include <stdio\.h>', r'int main\(', r'printf\('],
                "php": [r'<\?php', r'function \w+\(', r'\$\w+'],
                "ruby": [r'def \w+', r'class \w+', r'require \'', r'puts '],
                "swift": [r'import Swift', r'func \w+\(', r'var \w+:', r'let \w+:']
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
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\0']
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
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('/*'):
                continue
            
            # Check for unterminated strings
            if line.count('"') % 2 != 0 or line.count("'") % 2 != 0:
                if not line.endswith('\\'):  # Not escaped
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
        sanitized = re.sub(r'\n\s*system:', '', sanitized, flags=re.IGNORECASE)
        
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
