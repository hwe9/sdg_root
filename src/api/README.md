# API Service

## Overview

The API Service is the main REST API gateway for the SDG Pipeline project. It provides the primary interface for users and external systems to interact with the SDG content processing pipeline.

## Architecture

### Dependencies

This service depends on:
- **Database**: PostgreSQL for user data and SDG metadata
- **Cache**: Redis for session management and caching
- **Auth Service**: JWT token validation and user authentication
- **Data Processing**: Content processing pipeline
- **Vectorization**: Semantic search capabilities

### External Services

- **Weaviate**: Vector database for content similarity search
- **Content Extraction**: AI-powered content analysis
- **Data Retrieval**: Content collection from various sources

## API Endpoints

### Authentication

```
POST /auth/login          # User login
POST /auth/register       # User registration
POST /auth/refresh        # Token refresh
POST /auth/logout         # User logout
```

### SDG Management

```
GET    /sdgs             # List all SDGs
GET    /sdgs/{id}        # Get specific SDG
POST   /sdgs             # Create new SDG
PUT    /sdgs/{id}        # Update SDG
DELETE /sdgs/{id}        # Delete SDG
```

### Content Management

```
GET    /articles         # List articles
GET    /articles/{id}    # Get specific article
POST   /articles         # Create article
PUT    /articles/{id}    # Update article
DELETE /articles/{id}    # Delete article
POST   /articles/search  # Semantic search
```

### User Management

```
GET    /users            # List users (admin only)
GET    /users/{id}       # Get user profile
PUT    /users/{id}       # Update user profile
DELETE /users/{id}       # Delete user (admin only)
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection | Auto-generated | Yes |
| `REDIS_URL` | Redis connection | `redis://redis:6379` | No |
| `SECRET_KEY` | JWT signing key | - | Yes |
| `API_PORT` | Service port | `8000` | No |
| `CORS_ORIGINS` | Allowed CORS origins | `*` | No |

### Service Configuration

The API service uses centralized configuration management with support for:
- Environment-based configuration
- Configuration validation
- Hot-reloading for development

## Data Flow

1. **Request Reception**: API receives HTTP requests with authentication
2. **Authentication**: JWT tokens validated via Auth Service
3. **Authorization**: Role-based access control applied
4. **Business Logic**: Request processed with database/cache operations
5. **Response**: Formatted JSON response returned

## Error Handling

### Error Response Format

```json
{
  "error_code": "VALIDATION_ERROR",
  "message": "Invalid input data",
  "details": {
    "field": "email",
    "reason": "Invalid email format"
  },
  "timestamp": "2025-10-27T12:48:34.404Z",
  "request_id": "req-12345",
  "service": "api"
}
```

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `VALIDATION_ERROR` | Invalid input data | 400 |
| `AUTHENTICATION_ERROR` | Invalid credentials | 401 |
| `AUTHORIZATION_ERROR` | Insufficient permissions | 403 |
| `NOT_FOUND` | Resource not found | 404 |
| `CONFLICT` | Resource conflict | 409 |
| `SERVICE_UNAVAILABLE` | External service unavailable | 503 |

## Monitoring

### Health Checks

The service provides comprehensive health checks:

```json
{
  "status": "healthy",
  "service": "api",
  "version": "2.0.0",
  "timestamp": "2025-10-27T12:48:34.404Z",
  "components": {
    "database": "connected",
    "redis": "connected",
    "auth_service": "available"
  },
  "dependencies": {
    "auth_service": "healthy",
    "data_processing": "healthy"
  }
}
```

### Metrics

- **Request Metrics**: Count, duration, error rates by endpoint
- **Database Metrics**: Connection pool status, query performance
- **Cache Metrics**: Hit rates, eviction rates
- **Authentication Metrics**: Login attempts, token validation

## Development

### Local Setup

```bash
# Install dependencies
pip install -r src/api/requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://user:pass@localhost:5432/sdg"
export SECRET_KEY="your-secret-key"

# Run database migrations
python scripts/migrate.py migrate

# Start service
uvicorn src.api.main:app --reload --port 8000
```

### Testing

```bash
# Unit tests
pytest src/api/tests/unit/ -v

# Integration tests
pytest src/api/tests/integration/ -v

# API tests
pytest src/api/tests/api/ -v
```

## Deployment

### Docker Configuration

```yaml
api_service:
  build:
    context: .
    dockerfile: src/api/Dockerfile
  ports:
    - "8000:8000"
  environment:
    - DATABASE_URL=${DATABASE_URL}
    - REDIS_URL=${REDIS_URL}
    - SECRET_KEY=${SECRET_KEY}
  depends_on:
    - database_service
    - redis
    - auth_service
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval: 30s
    timeout: 15s
    retries: 5
```

## Performance

### Benchmarks

- **Average Response Time**: <50ms for simple queries
- **Throughput**: 1000+ requests/second
- **Concurrent Users**: 1000+ simultaneous connections
- **Database Query Time**: <10ms average

### Optimization Features

- **Connection Pooling**: Efficient database connection management
- **Caching**: Redis-based response caching for frequent queries
- **Async Processing**: Non-blocking I/O for external service calls
- **Request Batching**: Optimized bulk operations

## Security

### Authentication

- JWT-based stateless authentication
- Refresh token rotation
- Secure password hashing (bcrypt)
- Session management with Redis

### Authorization

- Role-based access control (RBAC)
- Resource-level permissions
- API key authentication for service accounts

### Data Protection

- Input validation and sanitization
- SQL injection prevention via SQLAlchemy
- XSS protection via content validation
- Rate limiting to prevent abuse

## API Documentation

When running, the API service provides interactive documentation at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check database service
   docker-compose logs database_service

   # Test connection
   python -c "import psycopg2; psycopg2.connect('${DATABASE_URL}')"
   ```

2. **Authentication Issues**
   ```bash
   # Check auth service health
   curl http://localhost:8005/health

   # Verify JWT secret
   echo $SECRET_KEY
   ```

3. **High Latency**
   ```bash
   # Check Redis connectivity
   docker-compose exec redis redis-cli ping

   # Monitor database performance
   docker-compose exec database_service pg_stat_activity
   ```

## Contributing

### Adding New Endpoints

1. Define Pydantic models in `schemas.py`
2. Implement business logic in main service file
3. Add route in `main.py` with proper authentication
4. Write comprehensive tests
5. Update API documentation

### Best Practices

- Use dependency injection for database sessions
- Implement proper error handling with standardized responses
- Add request/response logging for debugging
- Include input validation for all endpoints
- Write unit tests for all business logic
