"""
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
