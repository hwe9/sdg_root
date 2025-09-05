# Integration tests for SDG pipeline
import pytest
import httpx
import asyncio

BASE_URLS = {
    "api": "http://localhost:8000",
    "vectorization": "http://localhost:8003", 
    "content_extraction": "http://localhost:8004",
    "data_processing": "http://localhost:8001"
}

@pytest.mark.asyncio
async def test_all_services_healthy():
    """Test all services are running and healthy"""
    async with httpx.AsyncClient() as client:
        for service, base_url in BASE_URLS.items():
            try:
                response = await client.get(f"{base_url}/health", timeout=10.0)
                assert response.status_code == 200
                health_data = response.json()
                assert health_data["status"] == "healthy"
                print(f"✅ {service} service healthy")
            except Exception as e:
                pytest.fail(f"❌ {service} service failed: {e}")

@pytest.mark.asyncio
async def test_end_to_end_pipeline():
    """Test complete pipeline flow"""
    test_url = "https://www.un.org/sustainabledevelopment/poverty/"
    
    async with httpx.AsyncClient() as client:
        # 1. Extract content
        extract_response = await client.post(
            f"{BASE_URLS['content_extraction']}/extract",
            json={"url": test_url}
        )
        assert extract_response.status_code == 200
        
        # 2. Process content  
        process_response = await client.post(
            f"{BASE_URLS['data_processing']}/process-content",
            json={"content_items": extract_response.json()["content"]}
        )
        assert process_response.status_code == 200
        
        # 3. Search content
        search_response = await client.post(
            f"{BASE_URLS['vectorization']}/search", 
            json={"query": "poverty reduction", "limit": 5}
        )
        assert search_response.status_code == 200
        assert len(search_response.json()["results"]) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
