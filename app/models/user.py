"""
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
