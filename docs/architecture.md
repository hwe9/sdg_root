# Architecture Guide

## Overview

The SDG Pipeline is a comprehensive microservices architecture designed for processing, analyzing, and retrieving Sustainable Development Goals (SDG) related content. This document outlines the system architecture, design decisions, and technical patterns used throughout the project.

## 🏗️ System Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │  Auth Service   │    │   User Client   │
│    (FastAPI)    │◄──►│    (FastAPI)    │◄──►│                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│ Data Processing │    │     Database    │
│    (FastAPI)    │────►│  (PostgreSQL)  │
└─────────────────┘    └─────────────────┘
         │                       ▲
         ▼                       │
┌─────────────────┐             │
│ Vectorization   │             │
│    (FastAPI)    │◄────────────┘
└─────────────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐
│ Content Extract │    │   Weaviate      │
│    (FastAPI)    │────►│  (Vector DB)   │
└─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│ Data Retrieval  │
│    (FastAPI)    │
└─────────────────┘
```

### Service Responsibilities

| Service | Responsibility | Technology |
|---------|----------------|------------|
| **API Gateway** | REST API, request routing, response formatting | FastAPI, Pydantic |
| **Auth Service** | User authentication, authorization, JWT management | FastAPI, JWT |
| **Data Processing** | Content analysis, AI processing, data transformation | FastAPI, AI/ML libraries |
| **Data Retrieval** | Content collection from external sources | FastAPI, HTTP clients |
| **Vectorization** | Text embedding, vector database operations | FastAPI, Sentence Transformers |
| **Content Extraction** | Content extraction using AI models | FastAPI, AI APIs |

### Infrastructure Services

| Service | Purpose | Technology |
|---------|---------|------------|
| **PostgreSQL** | Primary data storage | PostgreSQL 16 |
| **Redis** | Caching, session storage | Redis 7 |
| **Weaviate** | Vector similarity search | Weaviate 1.24 |
| **Nginx** | Reverse proxy, load balancing | Nginx |

## 🏛️ Architectural Patterns

### Microservices Pattern

**Why Microservices?**
- **Scalability**: Each service can scale independently
- **Technology Diversity**: Different services can use different tech stacks
- **Fault Isolation**: Failure in one service doesn't affect others
- **Team Autonomy**: Teams can work on different services independently

**Service Boundaries:**
- Each service owns its data and business logic
- Services communicate via well-defined APIs
- Services are independently deployable
- Services have their own databases (when needed)

### API Gateway Pattern

**Responsibilities:**
- Request routing and load balancing
- Authentication and authorization
- Rate limiting and throttling
- Request/response transformation
- Service discovery

**Benefits:**
- Single entry point for clients
- Centralized cross-cutting concerns
- Simplified client code
- Better security control

### CQRS Pattern

**Command Query Responsibility Segregation:**
- **Commands**: Write operations (create, update, delete)
- **Queries**: Read operations (search, retrieve)
- **Separation**: Different models for read and write operations

```python
# Command (Write)
class CreateArticleCommand:
    title: str
    content: str
    author_id: int

# Query (Read)
class ArticleSearchQuery:
    query: str
    filters: Dict[str, Any]
    pagination: PaginationParams
```

### Event-Driven Architecture

**Event Types:**
- **Domain Events**: Business logic events (ArticleCreated, UserLoggedIn)
- **Integration Events**: Cross-service communication
- **System Events**: Infrastructure events (ServiceStarted, DatabaseError)

**Event Flow:**
```
User Action → Command → Domain Event → Event Handler → Side Effects
```

## 🗄️ Data Architecture

### Database Design

#### Primary Database (PostgreSQL)

```
sdg_pipeline (database)
├── users (table)
│   ├── id (PK)
│   ├── username (unique)
│   ├── email (unique)
│   ├── hashed_password
│   ├── role
│   └── created_at
├── sdgs (table)
│   ├── id (PK)
│   ├── code (unique)
│   ├── title
│   ├── description
│   └── targets (jsonb)
├── articles (table)
│   ├── id (PK)
│   ├── title
│   ├── content
│   ├── author_id (FK)
│   ├── sdg_ids (array)
│   ├── created_at
│   └── updated_at
└── user_sessions (table)
    ├── id (PK)
    ├── user_id (FK)
    ├── token_hash
    └── expires_at
```

#### Vector Database (Weaviate)

```
weaviate (vector database)
├── Article (class)
│   ├── title (text)
│   ├── content (text)
│   ├── embedding (vector[768])
│   └── metadata (object)
└── SDG (class)
    ├── code (text)
    ├── title (text)
    ├── embedding (vector[768])
    └── targets (object)
```

### Data Flow Patterns

#### Write Path (Command)
```
Client → API Gateway → Auth Service → Business Service → Database
    ↓         ↓             ↓          ↓             ↓
 Validate → Authorize → Process → Persist → Confirm
```

#### Read Path (Query)
```
Client → API Gateway → Cache → Database/Vector DB → Response
    ↓         ↓           ↓          ↓               ↓
 Validate → Route → Check → Query → Format
```

## 🔐 Security Architecture

### Authentication & Authorization

#### JWT Token Flow
```
1. User Login → Auth Service validates credentials
2. Auth Service generates JWT token pair
3. Access token returned to client
4. Refresh token stored securely
5. Client includes access token in requests
6. API Gateway validates token
7. User permissions checked per request
```

#### Security Layers
```
┌─────────────────┐
│   Client App    │
└─────────────────┘
         │
    ┌────────────┐
    │ API Gateway│ ← Rate Limiting, Input Validation
    └────────────┘
         │
    ┌────────────┐
    │ Auth Service│ ← JWT Validation, User Management
    └────────────┘
         │
    ┌────────────┐
    │  Services  │ ← Business Logic Validation
    └────────────┘
         │
    ┌────────────┐
    │  Database  │ ← SQL Injection Prevention, Encryption
    └────────────┘
```

### Data Protection

- **Encryption at Rest**: Sensitive data encrypted in database
- **Encryption in Transit**: TLS 1.3 for all communications
- **Password Hashing**: bcrypt with salt rounds
- **Token Security**: Short-lived access tokens, secure refresh tokens

## 📊 Performance Architecture

### Caching Strategy

#### Multi-Level Caching
```
L1: Application Cache (Redis) - User sessions, frequent queries
L2: Database Cache (PostgreSQL) - Query result caching
L3: CDN Cache - Static assets, API responses
```

#### Cache Keys
```
user:{user_id}:profile
articles:search:{query_hash}:{page}
sdg:{sdg_id}:details
session:{session_id}
```

### Database Optimization

#### Indexing Strategy
```sql
-- Primary keys (automatic)
-- Foreign key indexes (automatic)
-- Unique constraints (automatic)

-- Performance indexes
CREATE INDEX idx_articles_author_id ON articles(author_id);
CREATE INDEX idx_articles_created_at ON articles(created_at DESC);
CREATE INDEX idx_articles_sdg_ids ON articles USING GIN(sdg_ids);
CREATE INDEX idx_user_sessions_expires_at ON user_sessions(expires_at);

-- Full-text search indexes
CREATE INDEX idx_articles_content_fts ON articles
USING GIN(to_tsvector('english', title || ' ' || content));
```

#### Connection Pooling
```python
# SQLAlchemy engine configuration
engine = create_engine(
    database_url,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=10,
    max_overflow=20,
    echo=False
)
```

### Asynchronous Processing

#### Task Queue Architecture
```
FastAPI Endpoint → Redis Queue → Worker Process → Result Cache
     ↓                ↓              ↓              ↓
  Immediate Response → Enqueue → Process → Store Result
```

#### Worker Services
- **Data Processing Worker**: Heavy AI/ML computations
- **Content Extraction Worker**: External API calls
- **Indexing Worker**: Search index updates

## 🔍 Monitoring & Observability

### Metrics Collection

#### Application Metrics
```python
# Request metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests',
                       ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds',
                           'HTTP request duration', ['method', 'endpoint'])

# Business metrics
ARTICLES_PROCESSED = Counter('articles_processed_total',
                           'Total articles processed')
SEARCH_QUERIES = Counter('search_queries_total',
                        'Total search queries')
```

#### System Metrics
- CPU utilization per service
- Memory usage patterns
- Database connection pools
- Redis cache hit rates
- External API response times

### Logging Architecture

#### Structured Logging
```json
{
  "timestamp": "2025-10-27T12:48:34.404Z",
  "service": "api",
  "level": "INFO",
  "request_id": "req-12345",
  "user_id": "user-678",
  "operation": "article_search",
  "duration_ms": 45.2,
  "message": "Article search completed successfully"
}
```

#### Log Aggregation
```
Service Logs → Fluentd → Elasticsearch → Kibana
     ↓           ↓          ↓           ↓
  Structured → Collect → Index → Visualize
```

### Health Checks

#### Service Health
```json
{
  "status": "healthy",
  "service": "api",
  "version": "2.0.0",
  "timestamp": "2025-10-27T12:48:34.404Z",
  "checks": {
    "database": "healthy",
    "redis": "healthy",
    "external_apis": "healthy"
  },
  "metrics": {
    "uptime": "2h 34m",
    "requests_per_second": 12.5,
    "error_rate": 0.01
  }
}
```

## 🚀 Deployment Architecture

### Container Orchestration

#### Docker Compose (Development)
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - database
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
```

#### Kubernetes (Production)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-service
  template:
    spec:
      containers:
      - name: api
        image: sdg/api:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
```

### Service Mesh

#### Istio Integration
```
API Gateway → Istio Gateway → Virtual Services → Service Entries
     ↓             ↓               ↓              ↓
  External → Routing → Load Balance → Circuit Breaker
```

## 🔧 Development Architecture

### Code Organization

```
src/
├── core/                    # Shared components
│   ├── db_utils.py         # Database utilities
│   ├── logging_config.py   # Logging configuration
│   ├── error_handler.py    # Error handling
│   └── config_manager.py   # Configuration management
├── api/                     # API Gateway service
│   ├── main.py             # FastAPI application
│   ├── models.py           # Database models
│   ├── schemas.py          # Pydantic schemas
│   └── tests/              # Service tests
├── auth/                    # Authentication service
└── [other services...]
```

### Dependency Management

#### Centralized Requirements
```
requirements.txt          # Production dependencies
requirements-dev.txt      # Development tools
requirements-test.txt     # Testing dependencies
requirements-security.txt # Security updates
```

#### Dependency Injection
```python
# FastAPI dependency injection
def get_database_session() -> Session:
    """Database session dependency"""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

@app.get("/articles/{article_id}")
def get_article(
    article_id: int,
    db: Session = Depends(get_database_session)
):
    return db.query(Article).filter(Article.id == article_id).first()
```

## 📈 Scalability Patterns

### Horizontal Scaling

#### Stateless Services
- All services are stateless by design
- Session data stored in Redis
- Database handles concurrent access
- Load balancer distributes requests

#### Database Scaling
```sql
-- Read replicas for read-heavy operations
-- Sharding by tenant/user for multi-tenant
-- Connection pooling for high concurrency
-- Query optimization and indexing
```

#### Cache Scaling
```
Client → CDN → Load Balancer → Redis Cluster → Database
   ↓       ↓          ↓              ↓           ↓
 Cache → Cache → Distribute → Cluster → Scale Out
```

### Vertical Scaling

#### Resource Allocation
```yaml
# Kubernetes resource limits
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

#### Performance Monitoring
- APM tools (New Relic, DataDog)
- Custom performance metrics
- Database query monitoring
- Memory leak detection

## 🧪 Testing Architecture

### Test Pyramid
```
End-to-End Tests (10%)     ┌─────────────┐
                           │   Manual    │
Integration Tests (20%)    │   Testing   │
                           │             │
Unit Tests (70%)          └─────────────┘
```

### Testing Infrastructure

#### Test Environments
```
Development → Staging → Production
     ↓           ↓          ↓
   Local     Integration  Live System
  Testing    Testing     User Testing
```

#### Test Data Management
```python
# Test fixtures
@pytest.fixture
def sample_user():
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "securepass123"
    }

# Database test setup
@pytest.fixture
def test_db():
    # Create test database
    # Run migrations
    # Load test data
    yield db
    # Clean up
```

## 🔄 CI/CD Architecture

### Pipeline Stages
```
Source Code → Build → Test → Security Scan → Deploy → Monitor
     ↓           ↓       ↓         ↓            ↓         ↓
   Commit → Docker → Unit → Dependency → Staging → Alerts
            Build   Tests   Check       Deploy
```

### Deployment Strategies

#### Blue-Green Deployment
```
Load Balancer
     │
     ├── Blue Environment (v1.0)
     │    ├── API Service v1.0
     │    ├── Auth Service v1.0
     │    └── Database v1.0
     │
     └── Green Environment (v2.0) ← Traffic switches here
          ├── API Service v2.0
          ├── Auth Service v2.0
          └── Database v2.0
```

#### Canary Deployment
```
User Traffic → Load Balancer → 95% v1.0, 5% v2.0
     ↓              ↓              ↓
  Monitor → Metrics → Gradual rollout or rollback
```

## 📚 Documentation Architecture

### Documentation Types

- **API Documentation**: OpenAPI/Swagger auto-generated
- **Architecture Docs**: System design and decisions
- **Developer Guides**: Setup, testing, deployment
- **User Guides**: API usage, integration examples

### Documentation Tools

- **MkDocs**: Static site generation
- **Swagger UI**: Interactive API documentation
- **PlantUML**: Architecture diagrams
- **Markdown**: All documentation format

## 🎯 Design Principles

### SOLID Principles

- **Single Responsibility**: Each service/class has one reason to change
- **Open/Closed**: Open for extension, closed for modification
- **Liskov Substitution**: Subtypes are substitutable for their base types
- **Interface Segregation**: Clients depend only on methods they use
- **Dependency Inversion**: Depend on abstractions, not concretions

### Twelve-Factor App

1. **Codebase**: One codebase tracked in revision control
2. **Dependencies**: Explicitly declare and isolate dependencies
3. **Config**: Store config in environment
4. **Backing Services**: Treat backing services as attached resources
5. **Build, Release, Run**: Strictly separate build and run stages
6. **Processes**: Execute app as one or more stateless processes
7. **Port Binding**: Export services via port binding
8. **Concurrency**: Scale out via process model
9. **Disposability**: Maximize robustness with fast startup/shutdown
10. **Dev/Prod Parity**: Keep development/staging/production similar
11. **Logs**: Treat logs as event streams
12. **Admin Processes**: Run admin tasks as one-off processes

### Performance Principles

- **Caching**: Cache at multiple levels (application, database, CDN)
- **Async Processing**: Handle I/O-bound operations asynchronously
- **Connection Pooling**: Reuse database and external connections
- **Indexing**: Proper database indexing for query performance
- **Pagination**: Limit result sets and implement cursor-based pagination
- **Compression**: Compress responses and cache data

This architecture provides a scalable, maintainable, and robust foundation for the SDG content processing pipeline while following industry best practices and modern software design patterns.
