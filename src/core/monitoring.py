import logging
import time as time_module
import aiohttp
import psutil
import asyncio
from typing import Dict
from typing import Any
from typing import List
from typing import Optional
from datetime import datetime
from datetime import timedelta
from collections import defaultdict
from collections import deque
import json
from dataclasses import dataclass
from dataclasses import asdict

@dataclass
class PerformanceMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    active_connections: int
    processed_items: int
    error_count: int
    processing_time: float

class HealthChecker:
    """Comprehensive health checking for all services"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 data points
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "error_rate": 5.0,
            "response_time": 30.0
        }
        self.service_status = {}
        
    async def collect_system_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive system metrics"""
        try:
            # CPU and Memory
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
            # Process info
            active_connections = len(psutil.net_connections())
            
            metrics = PerformanceMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_io=network_io,
                active_connections=active_connections,
                processed_items=0,  # To be updated by services
                error_count=0,  # To be updated by services
                processing_time=0.0  # To be updated by services
            )
            
            self.metrics_history.append(metrics)
            return metrics
            
        except Exception as e:
            logging.error(f"Error collecting system metrics: {e}")
            return None
    
    async def check_service_health(self, service_name: str, 
                                 check_function) -> Dict[str, Any]:
        """Check health of individual service"""
        start_time = time_module.time()
        
        try:
            if asyncio.iscoroutinefunction(check_function):
                result = await check_function()
            else:
                result = check_function()
            
            response_time = time_module.time() - start_time
            
            health_status = {
                "service": service_name,
                "status": "healthy" if result.get("status") == "healthy" else "unhealthy",
                "response_time": response_time,
                "details": result,
                "timestamp": datetime.utcnow().isoformat(),
                "alerts": self._check_alerts(service_name, result, response_time)
            }
            
            self.service_status[service_name] = health_status
            return health_status
            
        except Exception as e:
            error_status = {
                "service": service_name,
                "status": "error",
                "error": str(e),
                "response_time": time_module.time() - start_time,
                "timestamp": datetime.utcnow().isoformat(),
                "alerts": ["Service check failed"]
            }
            
            self.service_status[service_name] = error_status
            return error_status
    
    def _check_alerts(self, service_name: str, result: Dict, response_time: float) -> List[str]:
        """Check for alert conditions"""
        alerts = []
        
        # Response time check
        if response_time > self.alert_thresholds["response_time"]:
            alerts.append(f"High response time: {response_time:.2f}s")
        
        # Service-specific checks
        if "error_rate" in result and result["error_rate"] > self.alert_thresholds["error_rate"]:
            alerts.append(f"High error rate: {result['error_rate']:.2f}%")
        
        return alerts
    
    async def get_comprehensive_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        system_metrics = await self.collect_system_metrics()
        
        # Calculate trends
        trends = self._calculate_trends()
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": self._determine_overall_status(),
            "system_metrics": asdict(system_metrics) if system_metrics else None,
            "service_status": self.service_status,
            "trends": trends,
            "alerts": self._get_active_alerts(),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate performance trends"""
        if len(self.metrics_history) < 10:
            return {"status": "insufficient_data"}
        
        recent_metrics = list(self.metrics_history)[-10:]
        older_metrics = list(self.metrics_history)[-20:-10] if len(self.metrics_history) >= 20 else []
        
        if not older_metrics:
            return {"status": "insufficient_historical_data"}
        
        # Calculate averages
        recent_avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        older_avg_cpu = sum(m.cpu_usage for m in older_metrics) / len(older_metrics)
        
        recent_avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        older_avg_memory = sum(m.memory_usage for m in older_metrics) / len(older_metrics)
        
        return {
            "cpu_trend": "increasing" if recent_avg_cpu > older_avg_cpu * 1.1 else "stable",
            "memory_trend": "increasing" if recent_avg_memory > older_avg_memory * 1.1 else "stable",
            "cpu_change": ((recent_avg_cpu - older_avg_cpu) / older_avg_cpu) * 100,
            "memory_change": ((recent_avg_memory - older_avg_memory) / older_avg_memory) * 100
        }
    
    def _determine_overall_status(self) -> str:
        """Determine overall system status"""
        if not self.service_status:
            return "unknown"
        
        statuses = [service["status"] for service in self.service_status.values()]
        
        if "error" in statuses:
            return "critical"
        elif "unhealthy" in statuses:
            return "degraded"
        elif all(status == "healthy" for status in statuses):
            return "healthy"
        else:
            return "unknown"
    
    def _get_active_alerts(self) -> List[str]:
        """Get all active alerts"""
        alerts = []
        for service_name, status in self.service_status.items():
            if status.get("alerts"):
                for alert in status["alerts"]:
                    alerts.append(f"{service_name}: {alert}")
        return alerts
    
    def _generate_recommendations(self) -> List[str]:
        """Generate system improvement recommendations"""
        recommendations = []
        
        if len(self.metrics_history) > 0:
            latest_metrics = self.metrics_history[-1]
            
            if latest_metrics.cpu_usage > 80:
                recommendations.append("Consider scaling up CPU resources or optimizing processes")
            
            if latest_metrics.memory_usage > 85:
                recommendations.append("Memory usage is high - consider increasing RAM or optimizing memory usage")
            
            if latest_metrics.disk_usage > 90:
                recommendations.append("Disk space is critical - clean up old files or increase storage")
        
        error_services = [name for name, status in self.service_status.items() 
                         if status["status"] == "error"]
        if error_services:
            recommendations.append(f"Investigate and fix errors in: {', '.join(error_services)}")
        
        return recommendations

# Global health checker instance
health_checker = HealthChecker()

async def start_monitoring():
    """Start continuous monitoring"""
    while True:
        try:
            await health_checker.collect_system_metrics()
            await asyncio.sleep(60)  # Collect metrics every minute
        except Exception as e:
            logging.error(f"Monitoring error: {e}")
            await asyncio.sleep(60)

# Health check endpoints für FastAPI services
async def get_health_status():
    """Get current health status for API endpoints"""
    return await health_checker.get_comprehensive_health_report()
