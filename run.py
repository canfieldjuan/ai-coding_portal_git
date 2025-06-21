#!/usr/bin/env python3
"""
Development server runner for AI Coding Portal
"""

import uvicorn
import sys
from pathlib import Path

# Add app directory to path
sys.path.append(str(Path(__file__).parent))

if __name__ == "__main__":
    print("🚀 Starting AI Coding Portal Backend")
    print("   📍 Server: http://localhost:8000")
    print("   📚 API Docs: http://localhost:8000/docs")
    print("   🔧 Debug mode enabled")
    print()
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
