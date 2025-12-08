# Contributing to Papr Memory 🤝

Thank you for your interest in contributing to Papr Memory! This guide will help you get started with contributing to our intelligent memory management system.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Adding New Features](#adding-new-features)
- [Documentation](#documentation)
- [Community](#community)

## 📜 Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- **Be respectful**: Treat everyone with respect and kindness
- **Be inclusive**: Welcome contributors of all backgrounds and experience levels
- **Be collaborative**: Work together towards common goals
- **Be constructive**: Provide helpful feedback and suggestions
- **Be patient**: Remember that everyone is learning and growing

## 🚀 Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Python 3.8+** installed
- **Docker & Docker Compose** (recommended for development)
- **Git** for version control
- **OpenAI API Key** for core functionality testing

### Quick Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/memory.git
   cd memory
   ```
3. **Set up the development environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   docker-compose up -d
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🛠️ Development Setup

### Option 1: Docker Development (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f web

# Run tests inside container
docker-compose exec web pytest

# Access shell
docker-compose exec web bash
```

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start services (MongoDB, Neo4j, Redis, Qdrant, Parse Server)
docker-compose up -d mongodb neo4j redis qdrant parse-server

# Run the application
python main.py
```

### Vision Architecture (for Contributors)

For a broader, forward-looking view of what we are building across applications, SDKs, intelligence layer, memory tiers, and storage backends, see the Vision Architecture diagram in `README.md`.

```mermaid
graph TB
    subgraph Vision
        I1[Memory Cortex]
        I2[Durable Execution]
        T0[Tier-0: Working]
        T1[Tier-1: Episodic]
        T2[Tier-2: Vector]
        T3[Tier-3: Graph]
        B1[MongoDB]
        B2[Qdrant]
        B3[Neo4j]
    end

    I1 --> I2
    I1 --> T0 --> B1
    I1 --> T1 --> B1
    I1 --> T2 --> B2
    I1 --> T3 --> B3
```

### Environment Configuration

1. **Copy environment file**: `cp .env.example .env`
2. **Required variables** for development:
   ```bash
   OPENAI_API_KEY=your-openai-api-key
   PARSE_SERVER_APPLICATION_ID=papr-app-id
   PARSE_SERVER_MASTER_KEY=papr-master-key
   ```
3. **Database URLs** (when using Docker):
   ```bash
   MONGODB_URL=mongodb://admin:password@localhost:27017/papr_memory?authSource=admin
   NEO4J_URL=bolt://localhost:7687
   REDIS_URL=redis://:password@localhost:6379/0
   QDRANT_URL=http://localhost:6333
   ```

## 🎨 Code Style Guidelines

### Python Code Style

We follow **PEP 8** with some modifications:

```python
# Use Black formatting
black .

# Line length: 88 characters (Black default)
# Use type hints for function parameters and return values
def process_memory(content: str, user_id: str) -> Dict[str, Any]:
    """Process a memory item and return metadata."""
    pass

# Use descriptive variable names
memory_content = "Important meeting notes"
user_session_token = "abc123"

# Use docstrings for functions and classes
class MemoryManager:
    """Manages memory storage and retrieval operations."""

    def add_memory(self, content: str) -> str:
        """Add a new memory and return its ID.

        Args:
            content: The memory content to store

        Returns:
            The unique memory ID
        """
        pass
```

### Code Organization

```
├── main.py                 # FastAPI application entry point
├── app_factory.py         # Application factory and configuration
├── routes/                # API route definitions
│   ├── memory_routes.py   # Memory-related endpoints
│   └── user_routes.py     # User management endpoints
├── models/                # Pydantic models and data structures
├── services/              # Business logic and external integrations
├── datastore/            # Database connection and utilities
├── utils/                # Utility functions and helpers
└── tests/                # Test files
```

### Import Organization

```python
# Standard library imports
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Third-party imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Local imports
from models.memory_models import MemoryRequest
from services.memory_service import MemoryService
```

### Error Handling

```python
# Use specific exception types
try:
    result = await memory_service.process_memory(content)
except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))
except MemoryNotFoundError as e:
    raise HTTPException(status_code=404, detail="Memory not found")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise HTTPException(status_code=500, detail="Internal server error")
```

## 🔄 Pull Request Process

### Before Submitting

1. **Test your changes**:
   ```bash
   # Run all tests
   pytest

   # Run specific test categories
   pytest tests/unit/
   pytest tests/integration/

   # Check code coverage
   pytest --cov=./ --cov-report=html
   ```

2. **Format your code**:
   ```bash
   # Format with Black
   black .

   # Sort imports
   isort .

   # Run linting
   flake8 .
   ```

3. **Update documentation** if needed

### Pull Request Checklist

- [ ] **Branch naming**: Use descriptive names like `feature/add-vector-search` or `fix/memory-deletion-bug`
- [ ] **Commit messages**: Write clear, descriptive commit messages
- [ ] **Tests**: Add tests for new functionality
- [ ] **Documentation**: Update documentation for new features
- [ ] **Breaking changes**: Note any breaking changes in the PR description
- [ ] **Environment variables**: Update `.env.example` if new variables are added

### Commit Message Format

```
type(scope): short description

Longer description if needed

Fixes #123
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Examples**:
```
feat(memory): add vector similarity search
fix(auth): resolve session token validation issue
docs(api): update endpoint documentation
test(memory): add unit tests for memory creation
```

### PR Description Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## 🧪 Testing

### Test Structure

```
tests/
├── unit/                  # Unit tests for individual functions
│   ├── test_memory_service.py
│   └── test_auth_utils.py
├── integration/          # Integration tests for API endpoints
│   ├── test_memory_api.py
│   └── test_user_api.py
├── fixtures/             # Test data and fixtures
└── conftest.py          # Pytest configuration
```

### Writing Tests

```python
# Unit test example
import pytest
from services.memory_service import MemoryService

@pytest.fixture
def memory_service():
    return MemoryService()

def test_process_memory_valid_input(memory_service):
    """Test memory processing with valid input."""
    content = "Test memory content"
    result = memory_service.process_memory(content)

    assert result is not None
    assert "memory_id" in result
    assert result["content"] == content

def test_process_memory_empty_content(memory_service):
    """Test memory processing with empty content."""
    with pytest.raises(ValueError, match="Content cannot be empty"):
        memory_service.process_memory("")
```

```python
# Integration test example
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_create_memory_endpoint():
    """Test memory creation via API."""
    response = client.post(
        "/v1/memories",
        json={"content": "Test memory", "type": "text"},
        headers={"X-API-Key": "test-key"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "memory_id" in data
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_memory_service.py

# Run tests with coverage
pytest --cov=./ --cov-report=html

# Run only fast tests (exclude integration)
pytest -m "not integration"
```

## ➕ Adding New Features

### Feature Development Process

1. **Create an issue** describing the feature
2. **Discuss the approach** with maintainers
3. **Create a feature branch**: `git checkout -b feature/feature-name`
4. **Implement the feature**:
   - Add models (if needed)
   - Implement business logic
   - Add API endpoints
   - Write tests
   - Update documentation
5. **Test thoroughly**
6. **Submit a pull request**

### Feature Guidelines

- **Keep features focused**: One feature per PR
- **Maintain backward compatibility** when possible
- **Add comprehensive tests**
- **Update documentation**
- **Consider performance implications**
- **Follow existing patterns**

### API Design Guidelines

```python
# RESTful endpoint design
@app.post("/v1/memories")           # Create memory
@app.get("/v1/memories/{id}")       # Get specific memory
@app.put("/v1/memories/{id}")       # Update memory
@app.delete("/v1/memories/{id}")    # Delete memory
@app.get("/v1/memories/search")     # Search memories

# Use Pydantic models for request/response
class MemoryRequest(BaseModel):
    content: str
    type: MemoryType = MemoryType.TEXT
    metadata: Optional[MemoryMetadata] = None

class MemoryResponse(BaseModel):
    memory_id: str
    content: str
    created_at: datetime
    updated_at: Optional[datetime] = None
```

## 📚 Documentation

### Types of Documentation

1. **Code comments**: Explain complex logic
2. **Docstrings**: Document functions and classes
3. **README updates**: Keep setup instructions current
4. **API documentation**: Auto-generated from Pydantic models
5. **Architecture docs**: High-level system design

### Documentation Standards

```python
def search_memories(
    query: str,
    user_id: str,
    limit: int = 10,
    offset: int = 0
) -> List[MemorySearchResult]:
    """Search memories using semantic similarity.

    Performs vector similarity search across user's memories
    and returns ranked results based on semantic relevance.

    Args:
        query: Search query string
        user_id: ID of the user performing the search
        limit: Maximum number of results to return (default: 10)
        offset: Number of results to skip for pagination (default: 0)

    Returns:
        List of MemorySearchResult objects ordered by relevance

    Raises:
        ValueError: If query is empty or user_id is invalid
        MemoryServiceError: If search operation fails

    Example:
        >>> results = search_memories("meeting notes", "user123", limit=5)
        >>> print(f"Found {len(results)} memories")
    """
    pass
```

## 🏗️ Architecture Considerations

### When Adding New Components

1. **Database changes**: Consider migration scripts
2. **API versioning**: Use `/v1/`, `/v2/` for breaking changes
3. **Configuration**: Add new environment variables to `.env.example`
4. **Dependencies**: Update `requirements.txt` and Docker files
5. **Monitoring**: Add logging and health checks
6. **Security**: Review authentication and authorization needs

### Performance Guidelines

- **Database queries**: Use pagination and indexing
- **Memory usage**: Be mindful of large data processing
- **Async operations**: Use async/await for I/O operations
- **Caching**: Leverage Redis for frequently accessed data
- **Vector operations**: Optimize embedding and search operations

## 🐛 Bug Reports

### Before Reporting a Bug

1. **Search existing issues** to avoid duplicates
2. **Try to reproduce** the issue consistently
3. **Check the latest version** to see if it's already fixed

### Bug Report Template

```markdown
## Bug Description
Clear description of what the bug is.

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g. Ubuntu 20.04]
- Python Version: [e.g. 3.9.7]
- Docker Version: [e.g. 20.10.8]

## Additional Context
Any other context about the problem.
```

## 🌟 Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Discord**: Real-time chat and community support (https://discord.gg/sWpR5a3H)

### Getting Help

1. **Check the documentation** first
2. **Search existing issues** and discussions
3. **Ask in discussions** for general questions
4. **Create an issue** for bugs or feature requests

### Recognition

We appreciate all contributions! Contributors are recognized in:
- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **GitHub contributor graphs**
- **Community highlights** in our documentation

## 📄 License

By contributing to Papr Memory, you agree that your contributions will be licensed under the GNU Affero General Public License v3.0.

---

**Thank you for contributing to Papr Memory! 🚀**

Your contributions help build a better memory management system for everyone. If you have any questions, don't hesitate to ask in our discussions or reach out to the maintainers.