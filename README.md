# SDG Pipeline Project

A comprehensive microservices architecture for Sustainable Development Goals (SDG) content processing, analysis, and retrieval.

## 🏗️ Architecture Overview

The SDG Pipeline is built as a microservices architecture consisting of the following core services:

### Core Services

| Service | Port | Description |
|---------|------|-------------|
| **API Service** | `8000` | Main REST API gateway and user interface |
| **Auth Service** | `8005` | Authentication and authorization service |
| **Data Processing** | `8001` | Content processing and AI analysis pipeline |
| **Data Retrieval** | `8002` | Data collection from various sources |
| **Vectorization** | `8003` | Text embedding and vector database management |
| **Content Extraction** | `8004` | Content extraction using AI models |

### Infrastructure Services

| Service | Description |
|---------|-------------|
| **PostgreSQL** | Primary relational database |
| **Redis** | Caching and session storage |
| **Weaviate** | Vector database for semantic search |
| **Nginx** | Reverse proxy and load balancer |

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- 8GB+ RAM recommended
- 20GB+ disk space

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd sdg-pipeline
```

2. **Set up environment variables**
```bash
cp .env.template .env
# Edit .env with your configuration
```

3. **Run database migrations**
```bash
python scripts/migrate.py migrate
```

4. **Start all services**
```bash
# Development
docker-compose up -d

# Production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Health Check

```bash
# Check all services
curl http://localhost:80/health

# Individual service health checks
curl http://localhost:8000/health  # API Service
curl http://localhost:8005/health  # Auth Service
curl http://localhost:8001/health  # Data Processing
```

## 📚 Documentation

### For Developers

- [Development Setup](./docs/development.md)
- [API Documentation](./docs/api.md)
- [Architecture Guide](./docs/architecture.md)
- [Testing Guide](./docs/testing.md)

### For Operators

- [Deployment Guide](./DEPLOYMENT_GUIDE.md)
- [Monitoring Guide](./docs/monitoring.md)
- [Troubleshooting](./docs/troubleshooting.md)
- [Security Guide](./docs/security.md)

### Service Documentation

- [API Service](./src/api/README.md)
- [Auth Service](./src/auth/README.md)
- [Data Processing](./src/data_processing/README.md)
- [Data Retrieval](./src/data_retrieval/README.md)
- [Vectorization](./src/vectorization/README.md)
- [Content Extraction](./src/content_extraction/README.md)

## 🛠️ Development

### Code Quality

```bash
# Format code
python scripts/format_code.py format

# Run linting
python scripts/format_code.py lint

# Check imports
python scripts/format_code.py check-imports

# Run all checks
python scripts/format_code.py check-all
```

### Testing

```bash
# Run all tests
python scripts/test_consistency.py

# Run specific service tests
pytest src/api/tests/ -v

# Run with coverage
pytest --cov=src --cov-report=html
```

### Management Tools

```bash
# Requirements management
python scripts/manage_requirements.py update
python scripts/manage_requirements.py validate

# Environment management
python scripts/manage_env.py check-missing
python scripts/manage_env.py generate-template

# Docker management
python scripts/manage_docker.py generate-compose
python scripts/manage_docker.py validate

# Logging management
python scripts/manage_logging.py show
python scripts/manage_logging.py check
```

## 🔧 Configuration

### Environment Variables

The project uses centralized environment variable management. Key configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Environment (development/staging/production) | `development` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `DATABASE_URL` | PostgreSQL connection URL | Auto-generated |
| `REDIS_URL` | Redis connection URL | `redis://redis:6379` |
| `WEAVIATE_URL` | Weaviate database URL | `http://weaviate_service:8080` |

### Service URLs

All services communicate via internal Docker network:

```
API Service: http://api_service:8000
Auth Service: http://auth_service:8005
Data Processing: http://data_processing_service:8001
Data Retrieval: http://data_retrieval_service:8002
Vectorization: http://vectorization_service:8003
Content Extraction: http://content_extraction_service:8004
```

## 📊 Monitoring

### Health Endpoints

All services expose standardized health check endpoints:

```bash
GET /health          # Service health status
GET /ready           # Readiness probe
GET /metrics         # Prometheus metrics
```

### Logging

Centralized logging with structured JSON output:

```json
{
  "timestamp": "2025-10-27T12:48:34.404Z",
  "service": "api",
  "level": "INFO",
  "name": "src.api.main",
  "message": "API request completed"
}
```

### Metrics

- **Prometheus** metrics available at `/metrics`
- **Health checks** with dependency status
- **Performance monitoring** with request duration tracking

## 🔒 Security

### Authentication & Authorization

- JWT-based authentication
- Role-based access control (RBAC)
- API key authentication for service-to-service calls

### Data Protection

- Encrypted sensitive data storage
- Secure password hashing (bcrypt)
- HTTPS everywhere in production

### Security Features

- Rate limiting
- Input validation
- SQL injection prevention
- XSS protection
- CORS configuration

## 🤝 Contributing

### Development Workflow

1. **Fork and clone** the repository
2. **Create a feature branch** (`git checkout -b feature/my-feature`)
3. **Run tests** (`python scripts/format_code.py check-all`)
4. **Make changes** following the coding standards
5. **Update documentation** if needed
6. **Submit a pull request**

### Coding Standards

- **Formatting**: Black code formatter with 88 character line length
- **Imports**: isort with black-compatible sorting
- **Linting**: flake8 with relaxed rules for string formatting
- **Type hints**: MyPy for static type checking
- **Logging**: Centralized logging configuration

### Commit Messages

Follow conventional commit format:

```
feat: add new API endpoint
fix: resolve authentication bug
docs: update deployment guide
refactor: simplify database queries
test: add unit tests for user service
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋 Support

### Getting Help

- **Documentation**: Check the [docs](./docs/) directory
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Use GitHub Discussions for questions

### Community

- **Contributing Guide**: See [CONTRIBUTING.md](./CONTRIBUTING.md)
- **Code of Conduct**: See [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)

---

**Built with ❤️ for Sustainable Development Goals**
