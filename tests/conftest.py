import pytest
import asyncio
from httpx import AsyncClient
from app.main import app

@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def sample_code():
    return '''
def hello_world():
    print("Hello, World!")
    return "success"
'''
