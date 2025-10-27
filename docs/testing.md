# Testing Guide

## Overview

Comprehensive testing strategy for the SDG Pipeline project, covering unit tests, integration tests, API tests, and performance tests.

## 🧪 Test Structure

### Test Organization

```
src/[service]/tests/
├── __init__.py
├── conftest.py          # Shared fixtures and configuration
├── unit/               # Unit tests
│   ├── test_models.py
│   ├── test_services.py
│   └── test_utils.py
├── integration/        # Integration tests
│   ├── test_database.py
│   ├── test_external_apis.py
│   └── test_service_integration.py
├── api/               # API endpoint tests
│   ├── test_auth.py
│   ├── test_articles.py
│   └── test_health.py
└── fixtures/          # Test data and mocks
    ├── sample_data.json
    └── mock_responses.py
```

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **API Tests**: Test REST API endpoints
4. **End-to-End Tests**: Test complete user workflows
5. **Performance Tests**: Test system performance under load

## 🛠️ Testing Tools

### Core Testing Framework

```bash
# Install testing dependencies
pip install -r requirements-dev.txt

# Core testing tools
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
httpx>=0.25.0
```

### Test Configuration

`pytest.ini`:
```ini
[tool:pytest]
testpaths = src
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --cov=src --cov-report=html
markers =
    unit: Unit tests
    integration: Integration tests
    api: API tests
    slow: Slow running tests
    external: Tests requiring external services
```

## 🧪 Writing Tests

### Unit Tests

```python
import pytest
from src.api.services.user_service import UserService

class TestUserService:
    @pytest.fixture
    def user_service(self):
        return UserService()

    def test_create_user_success(self, user_service):
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "securepass123"
        }

        user = user_service.create_user(user_data)

        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.id is not None

    def test_create_user_duplicate_email(self, user_service):
        # Test duplicate email handling
        pass
```

### Integration Tests

```python
import pytest
from sqlalchemy.orm import sessionmaker
from src.core.db_utils import get_database_engine

class TestDatabaseIntegration:
    @pytest.fixture
    def db_session(self):
        engine = get_database_engine()
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()

    def test_user_creation_persistence(self, db_session):
        # Test that user creation persists to database
        from src.api.models import User

        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashedpass"
        )

        db_session.add(user)
        db_session.commit()

        # Verify user was saved
        saved_user = db_session.query(User).filter_by(username="testuser").first()
        assert saved_user is not None
        assert saved_user.email == "test@example.com"
```

### API Tests

```python
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

class TestAuthAPI:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_login_success(self, client):
        login_data = {
            "username": "testuser",
            "password": "testpass"
        }

        response = client.post("/auth/login", json=login_data)

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data

    def test_login_invalid_credentials(self, client):
        login_data = {
            "username": "testuser",
            "password": "wrongpass"
        }

        response = client.post("/auth/login", json=login_data)

        assert response.status_code == 401
        data = response.json()
        assert data["error_code"] == "AUTHENTICATION_ERROR"
```

### Async Tests

```python
import pytest
from httpx import AsyncClient

class TestAsyncAPI:
    @pytest.mark.asyncio
    async def test_async_endpoint(self):
        async with AsyncClient(base_url="http://testserver") as client:
            response = await client.get("/async-endpoint")
            assert response.status_code == 200
```

## 🏃 Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test file
pytest src/api/tests/unit/test_user_service.py

# Run tests in specific directory
pytest src/api/tests/unit/

# Run tests with specific marker
pytest -m unit

# Run tests with verbose output
pytest -v

# Run tests and stop on first failure
pytest -x
```

### Coverage Reporting

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Generate XML coverage report (for CI/CD)
pytest --cov=src --cov-report=xml

# Show coverage in terminal
pytest --cov=src --cov-report=term-missing

# Minimum coverage requirement
pytest --cov=src --cov-fail-under=80
```

### Test Selection and Filtering

```bash
# Run tests matching pattern
pytest -k "user and create"

# Run tests in class
pytest -k "TestUserService"

# Run slow tests only
pytest -m slow

# Skip integration tests
pytest -m "not integration"
```

## 🧰 Test Fixtures and Mocks

### Database Fixtures

```python
# conftest.py
import pytest
from sqlalchemy.orm import sessionmaker
from src.core.db_utils import get_database_engine, get_db

@pytest.fixture(scope="session")
def db_engine():
    return get_database_engine()

@pytest.fixture(scope="function")
def db_session(db_engine):
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from src.api.main import app

    # Override dependencies for testing
    app.dependency_overrides[get_db] = lambda: next(db_session)
    return TestClient(app)
```

### Mocking External Services

```python
import pytest
from unittest.mock import Mock, patch

class TestExternalService:
    @patch('src.api.services.external_service.requests.get')
    def test_external_api_call(self, mock_get):
        # Mock the external API response
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_get.return_value = mock_response

        service = ExternalService()
        result = service.call_external_api()

        assert result["status"] == "success"
        mock_get.assert_called_once()
```

## 🚀 Continuous Integration

### GitHub Actions Example

```yaml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
      redis:
        image: redis:7

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.8'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run tests
      run: pytest --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## 📊 Test Metrics and Reporting

### Coverage Goals

- **Unit Tests**: 80%+ coverage
- **Integration Tests**: 70%+ coverage
- **API Tests**: 90%+ coverage
- **Overall**: 75%+ coverage

### Test Performance

```bash
# Run tests with timing
pytest --durations=10

# Profile slow tests
pytest --profile
```

### Test Results Analysis

```bash
# Generate JUnit XML report
pytest --junitxml=test-results.xml

# Generate HTML test report
pytest --html=test-report.html
```

## 🔍 Testing Best Practices

### Test Naming

```python
# Good test names
def test_user_creation_with_valid_data()
def test_login_fails_with_wrong_password()
def test_article_search_returns_relevant_results()

# Bad test names
def test_user()          # Too vague
def test_function()      # Not descriptive
def test_stuff()         # Meaningless
```

### Test Structure

```python
def test_feature_scenario_expected_result():
    # Arrange: Set up test data and mocks
    user_data = create_test_user_data()

    # Act: Perform the action being tested
    result = user_service.create_user(user_data)

    # Assert: Verify the expected outcome
    assert result.success is True
    assert result.user_id is not None
```

### Test Data Management

```python
# Use factories for test data
def create_test_user(overrides=None):
    data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "securepass123"
    }
    if overrides:
        data.update(overrides)
    return data

# Clean up after tests
@pytest.fixture(autouse=True)
def cleanup_database(db_session):
    yield
    # Clean up test data
    db_session.query(User).filter(User.username.like("test%")).delete()
    db_session.commit()
```

## 🐛 Debugging Tests

### Common Issues

1. **Database State**: Tests affecting each other due to shared database state
2. **Async Tests**: Forgetting `@pytest.mark.asyncio`
3. **Fixture Scoping**: Using wrong fixture scope (`function` vs `session`)
4. **Mock Setup**: Not properly configuring mocks

### Debugging Tools

```bash
# Run test with detailed output
pytest -v -s --pdb

# Stop on first failure
pytest -x --tb=long

# Run specific failing test
pytest src/api/tests/test_auth.py::TestAuth::test_login -v
```

## 🎯 Performance Testing

### Load Testing with Locust

```python
# tests/load/api_load.py
from locust import HttpUser, task, between

class ApiUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def test_health_endpoint(self):
        self.client.get("/health")

    @task(3)  # Higher weight
    def test_search_endpoint(self):
        self.client.post("/articles/search", json={
            "query": "sustainable development",
            "limit": 10
        })

# Run load test
# locust -f tests/load/api_load.py --host=http://localhost:8000
```

### Benchmarking

```python
import time
import pytest

def benchmark_function(func, iterations=100):
    times = []
    for _ in range(iterations):
        start = time.time()
        func()
        end = time.time()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    print(f"Average time: {avg_time:.4f}s")
    print(f"Min time: {min(times):.4f}s")
    print(f"Max time: {max(times):.4f}s")

def test_api_performance(client):
    def search_request():
        response = client.post("/articles/search", json={"query": "test"})
        assert response.status_code == 200

    benchmark_function(search_request, 50)
```

## 📋 Testing Checklist

### Before Committing

- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] API tests pass
- [ ] Code coverage meets requirements
- [ ] No linting errors
- [ ] Type checking passes
- [ ] Documentation updated

### Pull Request Requirements

- [ ] New features have corresponding tests
- [ ] Bug fixes include regression tests
- [ ] Test coverage maintained or improved
- [ ] Performance tests for critical paths
- [ ] Integration tests for service interactions

## 🔧 Maintenance

### Test Data Management

```bash
# Clean up test databases
python scripts/clean_test_data.py

# Refresh test fixtures
python scripts/update_test_fixtures.py
```

### Test Suite Health

```bash
# Check for flaky tests
pytest --count=5 --tb=no -q

# Find slowest tests
pytest --durations=20

# Check test dependencies
python scripts/check_test_dependencies.py
```

## 📚 Resources

### Testing Libraries

- [pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [SQLAlchemy Testing](https://docs.sqlalchemy.org/en/14/orm/session_transaction.html#testing)

### Best Practices

- [Testing Pyramid](https://martinfowler.com/bliki/TestPyramid.html)
- [Given-When-Then](https://martinfowler.com/bliki/GivenWhenThen.html)
- [Test-Driven Development](https://en.wikipedia.org/wiki/Test-driven_development)
