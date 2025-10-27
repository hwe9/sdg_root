#!/bin/bash
# scripts/health_check.sh
# SDG Project Health Check Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Service endpoints
services=(
    "http://localhost:8000/health:API Service"
    "http://localhost:8001/health:Data Processing"
    "http://localhost:8002/health:Data Retrieval"
    "http://localhost:8003/health:Vectorization"
    "http://localhost:8004/health:Content Extraction"
    "http://localhost:8005/health:Auth Service"
)

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

check_service_health() {
    local url=$1
    local name=$2
    
    if curl -sf "$url" > /dev/null 2>&1; then
        log_success "$name: Healthy"
        return 0
    else
        log_error "$name: Unhealthy"
        return 1
    fi
}

check_all_services() {
    log_info "Checking service health..."
    echo ""
    
    local healthy_count=0
    local total_count=${#services[@]}
    
    for service in "${services[@]}"; do
        url=$(echo $service | cut -d: -f1-2)
        name=$(echo $service | cut -d: -f3)
        
        if check_service_health "$url" "$name"; then
            ((healthy_count++))
        fi
    done
    
    echo ""
    log_info "Health Check Summary:"
    echo "  Healthy: $healthy_count/$total_count"
    
    if [ $healthy_count -eq $total_count ]; then
        log_success "All services are healthy!"
        return 0
    else
        log_error "Some services are unhealthy!"
        return 1
    fi
}

check_detailed_health() {
    log_info "Detailed health check..."
    echo ""
    
    for service in "${services[@]}"; do
        url=$(echo $service | cut -d: -f1-2)
        name=$(echo $service | cut -d: -f3)
        
        echo "=== $name ==="
        if response=$(curl -sf "$url" 2>/dev/null); then
            echo "$response" | jq . 2>/dev/null || echo "$response"
        else
            log_error "Failed to get health status"
        fi
        echo ""
    done
}

main() {
    case "${1:-check}" in
        "check")
            check_all_services
            ;;
        "detailed")
            check_detailed_health
            ;;
        *)
            echo "Usage: $0 {check|detailed}"
            echo ""
            echo "Commands:"
            echo "  check     - Quick health check (default)"
            echo "  detailed  - Detailed health check with JSON output"
            exit 1
            ;;
    esac
}

main "$@"

