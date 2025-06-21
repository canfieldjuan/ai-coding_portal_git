"""
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
    pattern = r'```(\w+)?\n(.*?)\n```'
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
        r"(must|should|shall|will|need to|require|want to)\s+(.+?)(?:\.|$)",
        r"it\s+(must|should|shall|will|needs to)\s+(.+?)(?:\.|$)"
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
    lines = code.split('\n')
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
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
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
