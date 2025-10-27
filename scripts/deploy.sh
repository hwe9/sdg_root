#!/bin/bash
# scripts/deploy.sh
# SDG Project Deployment Script

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
MIGRATION_SCRIPT="src/migration/migrate.py"
TEST_SCRIPT="src/testing/test_consistency.py"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    
    # Check if docker-compose is available
    if ! command -v docker-compose > /dev/null 2>&1; then
        log_error "docker-compose is not installed. Please install docker-compose and try again."
        exit 1
    fi
    
    # Check if compose file exists
    if [ ! -f "$COMPOSE_FILE" ]; then
        log_error "Docker compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

backup_database() {
    log_info "Creating database backup..."
    
    # Create backup directory if it doesn't exist
    mkdir -p backups
    
    # Create backup filename with timestamp
    BACKUP_FILE="backups/sdg_backup_$(date +%Y%m%d_%H%M%S).sql"
    
    # Check if database service is running
    if docker-compose ps database_service | grep -q "Up"; then
        docker-compose exec -T database_service pg_dump -U postgres sdg_pipeline > "$BACKUP_FILE"
        log_success "Database backup created: $BACKUP_FILE"
    else
        log_warning "Database service not running, skipping backup"
    fi
}

run_migrations() {
    log_info "Running database migrations..."
    
    # Check if migration script exists
    if [ ! -f "$MIGRATION_SCRIPT" ]; then
        log_error "Migration script not found: $MIGRATION_SCRIPT"
        exit 1
    fi
    
    # Run migrations
    if python3 "$MIGRATION_SCRIPT" migrate; then
        log_success "Database migrations completed successfully"
    else
        log_error "Database migrations failed"
        exit 1
    fi
}

deploy_infrastructure() {
    log_info "Deploying infrastructure services..."
    
    # Start database service
    log_info "Starting database service..."
    docker-compose up -d database_service
    
    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker-compose exec database_service pg_isready -U postgres > /dev/null 2>&1; then
            log_success "Database is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        log_error "Database failed to start within 60 seconds"
        exit 1
    fi
    
    # Start Redis
    log_info "Starting Redis service..."
    docker-compose up -d redis
    
    # Start Weaviate Transformer
    log_info "Starting Weaviate Transformer service..."
    docker-compose up -d weaviate_transformer_service
    
    # Wait for transformer to be ready (can take 5-10 minutes)
    log_info "Waiting for Weaviate Transformer to be ready (this may take 5-10 minutes)..."
    timeout=600
    while [ $timeout -gt 0 ]; do
        if curl -sf http://localhost:8081/.well-known/ready > /dev/null 2>&1; then
            log_success "Weaviate Transformer is ready"
            break
        fi
        sleep 5
        timeout=$((timeout - 5))
    done
    
    if [ $timeout -le 0 ]; then
        log_error "Weaviate Transformer failed to start within 10 minutes"
        exit 1
    fi
    
    # Start Weaviate
    log_info "Starting Weaviate service..."
    docker-compose up -d weaviate_service
    
    # Wait for Weaviate to be ready
    log_info "Waiting for Weaviate to be ready..."
    timeout=120
    while [ $timeout -gt 0 ]; do
        if curl -sf http://localhost:8080/v1/.well-known/ready > /dev/null 2>&1; then
            log_success "Weaviate is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        log_error "Weaviate failed to start within 2 minutes"
        exit 1
    fi
}

deploy_core_services() {
    log_info "Deploying core services..."
    
    # Run database bootstrap
    log_info "Running database bootstrap..."
    docker-compose up db_bootstrap
    
    # Start Auth service
    log_info "Starting Auth service..."
    docker-compose up -d auth_service
    
    # Wait for auth service to be healthy
    log_info "Waiting for Auth service to be healthy..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -sf http://localhost:8005/health > /dev/null 2>&1; then
            log_success "Auth service is healthy"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        log_error "Auth service failed to start within 60 seconds"
        exit 1
    fi
    
    # Start API service
    log_info "Starting API service..."
    docker-compose up -d api_service
    
    # Wait for API service to be ready
    log_info "Waiting for API service to be ready..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -sf http://localhost:8000/ready > /dev/null 2>&1; then
            log_success "API service is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        log_error "API service failed to start within 60 seconds"
        exit 1
    fi
}

deploy_processing_services() {
    log_info "Deploying processing services..."
    
    # Start Data Retrieval service
    log_info "Starting Data Retrieval service..."
    docker-compose up -d data_retrieval_service
    
    # Start Content Extraction service
    log_info "Starting Content Extraction service..."
    docker-compose up -d content_extraction_service
    
    # Start Data Processing service
    log_info "Starting Data Processing service..."
    docker-compose up -d data_processing_service
    
    # Start Vectorization service
    log_info "Starting Vectorization service..."
    docker-compose up -d vectorization_service
    
    # Wait for all services to be healthy
    log_info "Waiting for processing services to be healthy..."
    services=("data_retrieval_service:8002" "content_extraction_service:8004" "data_processing_service:8001" "vectorization_service:8003")
    
    for service in "${services[@]}"; do
        service_name=$(echo $service | cut -d: -f1)
        port=$(echo $service | cut -d: -f2)
        
        timeout=120
        while [ $timeout -gt 0 ]; do
            if curl -sf http://localhost:$port/health > /dev/null 2>&1; then
                log_success "$service_name is healthy"
                break
            fi
            sleep 2
            timeout=$((timeout - 2))
        done
        
        if [ $timeout -le 0 ]; then
            log_error "$service_name failed to start within 2 minutes"
            exit 1
        fi
    done
}

deploy_proxy() {
    log_info "Deploying Nginx proxy..."
    docker-compose up -d nginx_proxy
    
    # Wait for proxy to be ready
    log_info "Waiting for Nginx proxy to be ready..."
    timeout=30
    while [ $timeout -gt 0 ]; do
        if curl -sf http://localhost:80/health > /dev/null 2>&1; then
            log_success "Nginx proxy is ready"
            break
        fi
        sleep 1
        timeout=$((timeout - 1))
    done
    
    if [ $timeout -le 0 ]; then
        log_error "Nginx proxy failed to start within 30 seconds"
        exit 1
    fi
}

run_tests() {
    log_info "Running consistency tests..."
    
    # Check if test script exists
    if [ ! -f "$TEST_SCRIPT" ]; then
        log_warning "Test script not found: $TEST_SCRIPT"
        return
    fi
    
    # Run tests
    if python3 "$TEST_SCRIPT"; then
        log_success "All consistency tests passed"
    else
        log_error "Some consistency tests failed"
        exit 1
    fi
}

show_status() {
    log_info "Deployment Status:"
    echo ""
    echo "Service URLs:"
    echo "  - API Service: http://localhost:8000"
    echo "  - Auth Service: http://localhost:8005"
    echo "  - Data Processing: http://localhost:8001"
    echo "  - Data Retrieval: http://localhost:8002"
    echo "  - Vectorization: http://localhost:8003"
    echo "  - Content Extraction: http://localhost:8004"
    echo "  - Nginx Proxy: http://localhost:80"
    echo ""
    echo "Monitoring:"
    echo "  - Weaviate Console: http://localhost:3001"
    echo "  - PgAdmin: http://localhost:5050"
    echo ""
    echo "API Documentation:"
    echo "  - API Docs: http://localhost:8000/docs"
    echo "  - Auth Docs: http://localhost:8005/docs"
    echo ""
    echo "Health Checks:"
    echo "  - API Health: http://localhost:8000/health"
    echo "  - Auth Health: http://localhost:8005/health"
    echo ""
    log_success "Deployment completed successfully!"
}

# Main deployment function
deploy() {
    log_info "Starting SDG Project deployment..."
    
    check_prerequisites
    backup_database
    deploy_infrastructure
    run_migrations
    deploy_core_services
    deploy_processing_services
    deploy_proxy
    run_tests
    show_status
}

# Rollback function
rollback() {
    log_info "Rolling back deployment..."
    
    # Stop all services
    docker-compose down
    
    # Restore database backup if available
    LATEST_BACKUP=$(ls -t backups/sdg_backup_*.sql 2>/dev/null | head -n1)
    if [ -n "$LATEST_BACKUP" ]; then
        log_info "Restoring database from backup: $LATEST_BACKUP"
        docker-compose up -d database_service
        sleep 10
        docker-compose exec -T database_service psql -U postgres sdg_pipeline < "$LATEST_BACKUP"
        log_success "Database restored from backup"
    else
        log_warning "No backup found, skipping database restore"
    fi
    
    log_success "Rollback completed"
}

# Main script logic
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "rollback")
        rollback
        ;;
    "status")
        docker-compose ps
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "test")
        run_tests
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|status|logs|test}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Deploy the entire SDG project (default)"
        echo "  rollback - Rollback the deployment"
        echo "  status   - Show service status"
        echo "  logs     - Show service logs"
        echo "  test     - Run consistency tests"
        exit 1
        ;;
esac

