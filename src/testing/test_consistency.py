# src/testing/test_consistency.py
import pytest
import asyncio
import httpx
import os
from typing import Dict, Any, List
from datetime import datetime

class ConsistencyTester:
    """Comprehensive testing framework for SDG project consistency"""
    
    def __init__(self):
        self.base_urls = {
            "api": "http://localhost:8000",
            "auth": "http://localhost:8005", 
            "data_processing": "http://localhost:8001",
            "data_retrieval": "http://localhost:8002",
            "vectorization": "http://localhost:8003",
            "content_extraction": "http://localhost:8004"
        }
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def test_all_services_health(self) -> Dict[str, Any]:
        """Test health endpoints of all services"""
        results = {
            "passed": [],
            "failed": [],
            "details": {}
        }
        
        for service_name, base_url in self.base_urls.items():
            try:
                response = await self.client.get(f"{base_url}/health")
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check for standardized response format
                    required_fields = ["status", "service", "version", "timestamp"]
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if missing_fields:
                        results["failed"].append(f"{service_name}: Missing fields {missing_fields}")
                    elif data["status"] != "healthy":
                        results["failed"].append(f"{service_name}: Status is {data['status']}")
                    else:
                        results["passed"].append(service_name)
                        results["details"][service_name] = data
                else:
                    results["failed"].append(f"{service_name}: HTTP {response.status_code}")
                    
            except Exception as e:
                results["failed"].append(f"{service_name}: {str(e)}")
        
        return results
    
    async def test_api_version_consistency(self) -> Dict[str, Any]:
        """Test that all services have consistent API versions"""
        results = {
            "passed": True,
            "versions": {},
            "inconsistent_services": []
        }
        
        for service_name, base_url in self.base_urls.items():
            try:
                response = await self.client.get(f"{base_url}/health")
                if response.status_code == 200:
                    data = response.json()
                    version = data.get("version", "unknown")
                    results["versions"][service_name] = version
                    
                    if version != "2.0.0":
                        results["passed"] = False
                        results["inconsistent_services"].append(f"{service_name}: {version}")
            except Exception as e:
                results["passed"] = False
                results["inconsistent_services"].append(f"{service_name}: Error - {str(e)}")
        
        return results
    
    async def test_database_connectivity(self) -> Dict[str, Any]:
        """Test database connectivity across services"""
        results = {
            "passed": [],
            "failed": [],
            "details": {}
        }
        
        for service_name, base_url in self.base_urls.items():
            try:
                response = await self.client.get(f"{base_url}/health")
                if response.status_code == 200:
                    data = response.json()
                    components = data.get("components", {})
                    db_status = components.get("database", "unknown")
                    
                    if db_status == "connected":
                        results["passed"].append(service_name)
                    else:
                        results["failed"].append(f"{service_name}: Database {db_status}")
                    
                    results["details"][service_name] = db_status
            except Exception as e:
                results["failed"].append(f"{service_name}: {str(e)}")
        
        return results
    
    async def test_service_dependencies(self) -> Dict[str, Any]:
        """Test service dependency configurations"""
        results = {
            "passed": [],
            "failed": [],
            "details": {}
        }
        
        for service_name, base_url in self.base_urls.items():
            try:
                response = await self.client.get(f"{base_url}/health")
                if response.status_code == 200:
                    data = response.json()
                    dependencies = data.get("dependencies", {})
                    
                    # Check if dependencies are properly configured
                    if isinstance(dependencies, dict):
                        results["passed"].append(service_name)
                        results["details"][service_name] = dependencies
                    else:
                        results["failed"].append(f"{service_name}: Invalid dependencies format")
            except Exception as e:
                results["failed"].append(f"{service_name}: {str(e)}")
        
        return results
    
    async def test_user_model_consistency(self) -> Dict[str, Any]:
        """Test user model consistency across services"""
        results = {
            "passed": True,
            "issues": [],
            "details": {}
        }
        
        # Test API service user endpoints
        try:
            response = await self.client.get(f"{self.base_urls['api']}/users/")
            if response.status_code == 200:
                results["details"]["api_users"] = "accessible"
            else:
                results["passed"] = False
                results["issues"].append(f"API users endpoint: HTTP {response.status_code}")
        except Exception as e:
            results["passed"] = False
            results["issues"].append(f"API users endpoint: {str(e)}")
        
        # Test Auth service user endpoints
        try:
            response = await self.client.get(f"{self.base_urls['auth']}/users/")
            if response.status_code == 200:
                results["details"]["auth_users"] = "accessible"
            else:
                results["passed"] = False
                results["issues"].append(f"Auth users endpoint: HTTP {response.status_code}")
        except Exception as e:
            results["passed"] = False
            results["issues"].append(f"Auth users endpoint: {str(e)}")
        
        return results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all consistency tests"""
        print("🧪 Running SDG Project Consistency Tests...")
        
        test_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "tests": {}
        }
        
        # Run all test suites
        test_results["tests"]["health_checks"] = await self.test_all_services_health()
        test_results["tests"]["api_versions"] = await self.test_api_version_consistency()
        test_results["tests"]["database_connectivity"] = await self.test_database_connectivity()
        test_results["tests"]["service_dependencies"] = await self.test_service_dependencies()
        test_results["tests"]["user_model_consistency"] = await self.test_user_model_consistency()
        
        # Calculate overall results
        total_tests = len(test_results["tests"])
        passed_tests = sum(1 for test in test_results["tests"].values() 
                          if isinstance(test, dict) and test.get("passed", False))
        
        test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "overall_status": "PASSED" if passed_tests == total_tests else "FAILED"
        }
        
        return test_results
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# Pytest fixtures and test functions
@pytest.fixture
async def consistency_tester():
    tester = ConsistencyTester()
    yield tester
    await tester.close()

@pytest.mark.asyncio
async def test_all_services_health(consistency_tester):
    """Test all services health endpoints"""
    results = await consistency_tester.test_all_services_health()
    assert len(results["failed"]) == 0, f"Health check failures: {results['failed']}"

@pytest.mark.asyncio
async def test_api_version_consistency(consistency_tester):
    """Test API version consistency"""
    results = await consistency_tester.test_api_version_consistency()
    assert results["passed"], f"Version inconsistencies: {results['inconsistent_services']}"

@pytest.mark.asyncio
async def test_database_connectivity(consistency_tester):
    """Test database connectivity"""
    results = await consistency_tester.test_database_connectivity()
    assert len(results["failed"]) == 0, f"Database connectivity failures: {results['failed']}"

@pytest.mark.asyncio
async def test_user_model_consistency(consistency_tester):
    """Test user model consistency"""
    results = await consistency_tester.test_user_model_consistency()
    assert results["passed"], f"User model issues: {results['issues']}"

if __name__ == "__main__":
    async def main():
        tester = ConsistencyTester()
        try:
            results = await tester.run_all_tests()
            
            print(f"\n📊 Test Results Summary:")
            print(f"Overall Status: {results['summary']['overall_status']}")
            print(f"Tests Passed: {results['summary']['passed_tests']}/{results['summary']['total_tests']}")
            
            if results['summary']['overall_status'] == 'FAILED':
                print("\n❌ Failed Tests:")
                for test_name, test_result in results['tests'].items():
                    if isinstance(test_result, dict) and not test_result.get('passed', True):
                        print(f"  - {test_name}: {test_result}")
            
            return results['summary']['overall_status'] == 'PASSED'
        finally:
            await tester.close()
    
    # Run the tests
    success = asyncio.run(main())
    exit(0 if success else 1)

