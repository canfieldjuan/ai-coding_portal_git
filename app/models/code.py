"""
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
