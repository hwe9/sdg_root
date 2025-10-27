# DEPLOYMENT_GUIDE.md
# SDG Project Deployment Guide

## 🚀 Pre-Deployment Checklist

### 1. Environment Setup
- [ ] Ensure all required environment variables are set
- [ ] Verify database credentials and connectivity
- [ ] Check Redis configuration
- [ ] Validate Weaviate configuration
- [ ] Confirm all secrets are properly configured

### 2. Database Preparation
- [ ] Backup existing database
- [ ] Run database migrations
- [ ] Validate schema consistency
- [ ] Test database connectivity

### 3. Service Dependencies
- [ ] Ensure PostgreSQL is running
- [ ] Verify Redis is accessible
- [ ] Confirm Weaviate service is ready
- [ ] Check transformer service availability

## 📋 Deployment Order

### Phase 1: Infrastructure Services
Deploy these services first as they are dependencies for all others:

```bash
# 1. Database Service
docker-compose up -d database_service
docker-compose logs -f database_service

# Wait for database to be ready
docker-compose exec database_service pg_isready -U postgres

# 2. Redis Service
docker-compose up -d redis
docker-compose logs -f redis

# 3. Weaviate Transformer Service
docker-compose up -d weaviate_transformer_service
docker-compose logs -f weaviate_transformer_service

# Wait for transformer to be ready (can take 5-10 minutes)
curl -f http://localhost:8081/.well-known/ready

# 4. Weaviate Service
docker-compose up -d weaviate_service
docker-compose logs -f weaviate_service

# Wait for Weaviate to be ready
curl -f http://localhost:8080/v1/.well-known/ready
```

### Phase 2: Core Services
Deploy core application services:

```bash
# 5. Database Bootstrap (one-time setup)
docker-compose up db_bootstrap
docker-compose logs db_bootstrap

# 6. Auth Service
docker-compose up -d auth_service
docker-compose logs -f auth_service

# Wait for auth service to be healthy
curl -f http://localhost:8005/health

# 7. API Service
docker-compose up -d api_service
docker-compose logs -f api_service

# Wait for API service to be ready
curl -f http://localhost:8000/ready
```

### Phase 3: Processing Services
Deploy data processing services:

```bash
# 8. Data Retrieval Service
docker-compose up -d data_retrieval_service
docker-compose logs -f data_retrieval_service

# Wait for data retrieval to be healthy
curl -f http://localhost:8002/health

# 9. Content Extraction Service
docker-compose up -d content_extraction_service
docker-compose logs -f content_extraction_service

# Wait for content extraction to be healthy
curl -f http://localhost:8004/health

# 10. Data Processing Service
docker-compose up -d data_processing_service
docker-compose logs -f data_processing_service

# Wait for data processing to be healthy
curl -f http://localhost:8001/health

# 11. Vectorization Service
docker-compose up -d vectorization_service
docker-compose logs -f vectorization_service

# Wait for vectorization to be healthy
curl -f http://localhost:8003/health
```

### Phase 4: Proxy and Monitoring
Deploy final services:

```bash
# 12. Nginx Proxy
docker-compose up -d nginx_proxy
docker-compose logs -f nginx_proxy

# 13. Optional Services (if needed)
docker-compose --profile aux up -d monitoring_service
docker-compose --profile aux up -d pgadmin_service
docker-compose --profile aux up -d weaviate_console
```

## 🔧 Database Migration Process

### Before Deployment
1. **Backup existing database:**
```bash
docker-compose exec database_service pg_dump -U postgres sdg_pipeline > backup_$(date +%Y%m%d_%H%M%S).sql
```

2. **Run migrations:**
```bash
# Using the migration CLI tool
python src/migration/migrate.py migrate

# Or manually
docker-compose exec api_service python src/migration/migrate.py migrate
```

3. **Validate schema:**
```bash
python src/migration/migrate.py validate
```

### Migration Rollback (if needed)
```bash
# Rollback specific migration
python src/migration/migrate.py rollback 001_consolidate_users

# Restore from backup
docker-compose exec -T database_service psql -U postgres sdg_pipeline < backup_YYYYMMDD_HHMMSS.sql
```

## 🧪 Testing Deployment

### 1. Run Consistency Tests
```bash
# Run all consistency tests
python src/testing/test_consistency.py

# Or using pytest
pytest src/testing/test_consistency.py -v
```

### 2. Manual Health Checks
```bash
# Check all service health endpoints
curl -f http://localhost:8000/health  # API Service
curl -f http://localhost:8001/health  # Data Processing
curl -f http://localhost:8002/health  # Data Retrieval
curl -f http://localhost:8003/health  # Vectorization
curl -f http://localhost:8004/health  # Content Extraction
curl -f http://localhost:8005/health  # Auth Service
```

### 3. Integration Tests
```bash
# Test API endpoints
curl -f http://localhost:8000/
curl -f http://localhost:8000/users/
curl -f http://localhost:8000/sdgs/

# Test authentication
curl -X POST http://localhost:8005/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}'
```

## 📊 Monitoring and Validation

### Service Status Dashboard
Access these URLs to monitor services:
- **API Service**: http://localhost:8000/docs
- **Auth Service**: http://localhost:8005/docs
- **Data Processing**: http://localhost:8001/docs
- **Data Retrieval**: http://localhost:8002/docs
- **Vectorization**: http://localhost:8003/docs
- **Content Extraction**: http://localhost:8004/docs
- **Weaviate Console**: http://localhost:3001
- **PgAdmin**: http://localhost:5050

### Health Check Script
```bash
#!/bin/bash
# health_check.sh

services=(
    "http://localhost:8000/health:API Service"
    "http://localhost:8001/health:Data Processing"
    "http://localhost:8002/health:Data Retrieval"
    "http://localhost:8003/health:Vectorization"
    "http://localhost:8004/health:Content Extraction"
    "http://localhost:8005/health:Auth Service"
)

echo "🔍 Checking service health..."
for service in "${services[@]}"; do
    url=$(echo $service | cut -d: -f1-2)
    name=$(echo $service | cut -d: -f3)
    
    if curl -sf "$url" > /dev/null; then
        echo "✅ $name: Healthy"
    else
        echo "❌ $name: Unhealthy"
    fi
done
```

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Check PostgreSQL is running: `docker-compose ps database_service`
   - Verify connection string: `echo $DATABASE_URL`
   - Check logs: `docker-compose logs database_service`

2. **Service Dependency Issues**
   - Check service startup order
   - Verify health check endpoints
   - Review dependency manager logs

3. **Migration Failures**
   - Check database permissions
   - Verify migration SQL syntax
   - Review migration logs

4. **Authentication Issues**
   - Verify SECRET_KEY is set
   - Check Redis connectivity
   - Review auth service logs

### Log Analysis
```bash
# View all service logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api_service
docker-compose logs -f auth_service

# View recent logs only
docker-compose logs --tail=100 -f
```

## 🔄 Rollback Procedures

### Service Rollback
```bash
# Stop all services
docker-compose down

# Rollback to previous version
git checkout previous-stable-tag
docker-compose up -d
```

### Database Rollback
```bash
# Stop services
docker-compose stop api_service auth_service

# Rollback migrations
python src/migration/migrate.py rollback 001_consolidate_users

# Restart services
docker-compose start api_service auth_service
```

## 📈 Performance Optimization

### Resource Limits
Monitor resource usage and adjust limits in `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: "2.0"
      memory: 4G
    reservations:
      cpus: "1.0"
      memory: 2G
```

### Database Optimization
```sql
-- Check database performance
SELECT * FROM pg_stat_activity;
SELECT * FROM pg_stat_database;

-- Optimize queries
EXPLAIN ANALYZE SELECT * FROM users WHERE username = 'admin';
```

## 🔐 Security Considerations

1. **Environment Variables**
   - Use strong passwords for all services
   - Rotate secrets regularly
   - Use encrypted secret storage

2. **Network Security**
   - Configure proper firewall rules
   - Use internal networks for service communication
   - Enable SSL/TLS for external access

3. **Database Security**
   - Use non-default database credentials
   - Enable connection encryption
   - Regular security updates

## 📝 Post-Deployment Checklist

- [ ] All services are healthy and responding
- [ ] Database migrations completed successfully
- [ ] Authentication is working
- [ ] API endpoints are accessible
- [ ] Data processing pipeline is functional
- [ ] Monitoring is active
- [ ] Backup procedures are in place
- [ ] Documentation is updated
- [ ] Team is notified of deployment completion

