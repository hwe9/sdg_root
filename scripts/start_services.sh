#!/bin/bash
# Service startup script

echo "🚀 Starting SDG Pipeline Services..."

# Build and start services
docker-compose build
docker-compose up -d database_service
sleep 10

# Initialize database
docker-compose run --rm api_service python -c "from src.core.db_init import initialize_database; initialize_database()"

# Start remaining services
docker-compose up -d

echo "✅ All services started successfully!"
echo "📊 Access points:"
echo "  - API Service: http://localhost:8000/docs"
echo "  - Vectorization: http://localhost:8003/docs"
echo "  - Content Extraction: http://localhost:8004/docs"
echo "  - PgAdmin: http://localhost:5050"
echo "  - Weaviate Console: http://localhost:3001"
