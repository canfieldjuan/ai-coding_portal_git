# AI Coding Portal Backend

Production-grade AI coding assistant with multi-provider support.

## Quick Start

1. `python setup.py` - Setup project
2. Edit `.env` with your `OPENROUTER_API_KEY`  
3. `python run.py` - Start server
4. `python test_api.py` - Test endpoints

## API Endpoints

- `GET /health` - Health check
- `POST /api/generate` - Generate code
- `POST /api/analyze` - Analyze code  
- `POST /api/fix` - Fix code issues
- `POST /api/upload` - Upload files

## Project Structure

```
app/
├── models/          # Database models
├── services/        # Business logic  
├── utils/           # Utilities
└── api/routes/      # API endpoints
```

For detailed documentation, see `docs/` folder.
