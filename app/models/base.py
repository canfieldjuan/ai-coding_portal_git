"""
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
