# Development Guide

## Overview

This guide covers the development workflow, coding standards, and best practices for contributing to the SDG Pipeline project.

## 🛠️ Development Environment

### Prerequisites

- **Python 3.8+**: Core runtime environment
- **Docker & Docker Compose**: Containerized development
- **Git**: Version control
- **Make**: Build automation (optional)
- **VS Code**: Recommended IDE with Python extensions

### Environment Setup

1. **Clone Repository**
```bash
git clone <repository-url>
cd sdg-pipeline
```

2. **Install Development Tools**
```bash
pip install -r requirements-dev.txt
```

3. **Set Up Environment**
```bash
# Copy environment template
cp .env.template .env

# Edit with your local configuration
nano .env
```

4. **Initialize Database**
```bash
# Start infrastructure services
docker-compose up -d database_service redis

# Run migrations
python scripts/migrate.py migrate
```

## 🧪 Testing

### Test Structure

```
src/[service]/tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── api/           # API endpoint tests
└── fixtures/      # Test data and mocks
```

### Running Tests

```bash
# All tests
pytest

# Specific service
pytest src/api/tests/ -v

# With coverage
pytest --cov=src --cov-report=html

# Specific test file
pytest src/api/tests/unit/test_auth.py -v
```

### Writing Tests

```python
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

## 📝 Coding Standards

### Code Formatting

The project uses automated code formatting tools:

```bash
# Format code (requires black and isort)
python scripts/format_code.py format

# Check formatting without changes
python scripts/format_code.py format --dry-run
```

### Import Organization

- **Standard Library**: First, alphabetical
- **Third-party**: Second, alphabetical
- **Local imports**: Last, with blank lines between groups

```python
import os
import sys
from typing import List, Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel

from .config import settings
from ..core.logging_config import get_logger
```

### Naming Conventions

- **Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Modules**: `snake_case`
- **Packages**: `snake_case`

### Type Hints

Use type hints for all function parameters and return values:

```python
from typing import List, Optional, Dict, Any

def process_articles(articles: List[Dict[str, Any]]) -> Optional[List[str]]:
    """Process a list of articles and return processed content."""
    pass
```

## 🔧 Code Quality Tools

### Linting

```bash
# Run all linting checks
python scripts/format_code.py lint

# Check specific issues
python scripts/format_code.py check-imports
```

### Type Checking

```bash
# Run mypy type checking
mypy src/

# Check specific module
mypy src/api/main.py
```

### Pre-commit Hooks

Set up pre-commit hooks to automatically check code quality:

```bash
python scripts/format_code.py setup-hooks
```

## 🏗️ Architecture Guidelines

### Service Design

Each service should follow these principles:

1. **Single Responsibility**: One primary function
2. **Dependency Injection**: Use FastAPI dependency system
3. **Error Handling**: Standardized error responses
4. **Logging**: Centralized logging configuration
5. **Health Checks**: Comprehensive health endpoints

### API Design

- **RESTful**: Follow REST conventions
- **Versioning**: Include API versioning in URLs
- **Documentation**: Auto-generated OpenAPI docs
- **Validation**: Pydantic models for all inputs/outputs
- **Authentication**: JWT tokens for user sessions

### Database Design

- **Migrations**: Use Alembic for schema changes
- **Models**: SQLAlchemy ORM with proper relationships
- **Connections**: Connection pooling and proper cleanup
- **Indexing**: Appropriate indexes for performance

## 🔄 Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/add-user-profile
```

### 2. Write Tests First (TDD)

```bash
# Create test file
touch src/api/tests/unit/test_user_profile.py

# Write failing test
pytest src/api/tests/unit/test_user_profile.py -v
```

### 3. Implement Feature

```bash
# Implement the feature
nano src/api/user_profile.py

# Run tests to verify
pytest src/api/tests/unit/test_user_profile.py -v
```

### 4. Code Quality Checks

```bash
# Format code
python scripts/format_code.py format

# Run all checks
python scripts/format_code.py check-all
```

### 5. Update Documentation

```bash
# Update API docs if needed
# Update README files
# Update docstrings
```

### 6. Commit Changes

```bash
git add .
git commit -m "feat: add user profile management

- Add user profile API endpoints
- Implement profile data validation
- Add comprehensive tests
- Update API documentation"
```

### 7. Create Pull Request

```bash
git push origin feature/add-user-profile
# Create PR on GitHub/GitLab
```

## 🚀 Deployment

### Local Deployment

```bash
# Start all services
docker-compose up -d

# Check health
curl http://localhost:80/health
```

### Production Deployment

See [Deployment Guide](../DEPLOYMENT_GUIDE.md) for detailed production deployment instructions.

## 🔍 Debugging

### Logging

All services use centralized logging:

```python
from src.core.logging_config import get_logger

logger = get_logger("api")

def my_function():
    logger.info("Processing request")
    logger.error("Something went wrong", extra={"user_id": 123})
```

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
# Restart services
docker-compose restart
```

### Database Debugging

```bash
# Connect to database
docker-compose exec database_service psql -U postgres -d sdg_pipeline

# Check active connections
SELECT * FROM pg_stat_activity;

# View recent queries
SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;
```

## 📊 Performance Monitoring

### Application Metrics

Services expose Prometheus metrics at `/metrics`:

```bash
curl http://localhost:8000/metrics
```

### Database Performance

```bash
# Query performance
EXPLAIN ANALYZE SELECT * FROM articles WHERE created_at > '2025-01-01';

# Index usage
SELECT * FROM pg_stat_user_indexes ORDER BY idx_scan DESC;
```

## 🤝 Contributing

### Pull Request Guidelines

1. **Title**: Use conventional commit format
2. **Description**: Clear explanation of changes
3. **Tests**: Include tests for new features
4. **Documentation**: Update docs for API changes
5. **Breaking Changes**: Clearly mark breaking changes

### Code Review Checklist

- [ ] Tests pass
- [ ] Code formatted correctly
- [ ] No linting errors
- [ ] Type hints included
- [ ] Documentation updated
- [ ] Security considerations addressed
- [ ] Performance impact assessed

## 📚 Resources

### Documentation

- [API Documentation](./api.md)
- [Architecture Guide](./architecture.md)
- [Testing Guide](./testing.md)
- [Security Guide](./security.md)

### External Links

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy Documentation](https://sqlalchemy.org/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

## 🆘 Getting Help

### Common Issues

1. **Import Errors**: Check Python path and virtual environment
2. **Database Issues**: Verify connection strings and service status
3. **Docker Problems**: Check logs with `docker-compose logs`
4. **Test Failures**: Run individual tests with `-v` flag

### Support Channels

- **Issues**: GitHub Issues for bugs and feature requests
- **Discussions**: GitHub Discussions for questions
- **Slack**: Team communication channel
- **Documentation**: Check docs first for common questions
