import asyncio
import logging
from datetime import datetime
from datetime import timedelta
from typing import Dict
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(
    self,
    global_rps: float = 10.0,
    per_domain_rps: float = 1.5,
    robots_respect: bool = True,
    robots_default_delay: float = 1.0,
    jitter_ms: int = 200
    ):
        self.domain_delays = {}
        self.robots_cache = {}
        self.default_delay = max(robots_default_delay, 1.0 / max(per_domain_rps, 0.001))
        self.max_delay = 60.0
        self.cache_ttl = 3600
        self.session = None
        self.global_rps = global_rps
        self.per_domain_rps = per_domain_rps
        self.robots_respect = robots_respect
        self.jitter_ms = jitter_ms

        logger.info(
            f"RateLimiter configured: global_rps={global_rps}, per_domain_rps={per_domain_rps}, "
            f"robots_respect={robots_respect}, default_delay={self.default_delay:.2f}s"
        )

    async def initialize(self):
        """Initialize rate limiter"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={'User-Agent': 'SDG-Pipeline-Bot/2.0 (+https://sdg-pipeline.org/bot)'}
        )
        logger.info("✅ Rate limiter initialized")

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()

    async def wait_for_rate_limit(self, url: str):
        """Wait for appropriate delay based on domain and robots.txt"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Get delay for this domain
            delay = await self._get_domain_delay(domain, url)

            # Check if we need to wait
            if domain in self.domain_delays:
                last_request, _ = self.domain_delays[domain]
                time_since_last = (datetime.utcnow() - last_request).total_seconds()

                if time_since_last < delay:
                    wait_time = delay - time_since_last
                    logger.info(f"⏱️ Rate limiting: waiting {wait_time:.1f}s for {domain}")
                    await asyncio.sleep(wait_time)

            # Update last request time
            self.domain_delays[domain] = (datetime.utcnow(), delay)

        except Exception as e:
            logger.error(f"Error in rate limiting: {e}")
            # Default delay on error
            await asyncio.sleep(self.default_delay)

    async def _get_domain_delay(self, domain: str, url: str) -> float:
        """Get appropriate delay for domain based on robots.txt"""
        try:
            # Check cache
            if domain in self.robots_cache:
                robots_data, cached_time = self.robots_cache[domain]
                if (datetime.utcnow() - cached_time).seconds < self.cache_ttl:
                    return self._extract_delay_from_robots(robots_data, url)

            # Fetch and parse robots.txt
            robots_url = f"https://{domain}/robots.txt"
            robots_data = await self._fetch_robots_txt(robots_url)

            if robots_data:
                self.robots_cache[domain] = (robots_data, datetime.utcnow())
                return self._extract_delay_from_robots(robots_data, url)

            # Fallback to default delay
            return self.default_delay

        except Exception as e:
            logger.warning(f"Error getting robots.txt for {domain}: {e}")
            return self.default_delay

    async def _fetch_robots_txt(self, robots_url: str) -> Optional[str]:
        """Fetch robots.txt content"""
        try:
            async with self.session.get(robots_url) as response:
                if response.status == 200:
                    content = await response.text()
                    logger.debug(f"✅ Fetched robots.txt from {robots_url}")
                    return content
                else:
                    logger.debug(f"No robots.txt found at {robots_url} (status: {response.status})")
                    return None

        except Exception as e:
            logger.debug(f"Error fetching robots.txt: {e}")
            return None

    def _extract_delay_from_robots(self, robots_content: str, url: str) -> float:
        """Extract crawl delay from robots.txt content"""
        try:
            rp = RobotFileParser()
            rp.set_url("dummy") # Required but not used for our purposes
            rp.read()

            # Parse robots.txt content
            lines = robots_content.strip().split('\n')
            current_user_agent = None
            delay = self.default_delay

            for line in lines:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue

                if line.lower().startswith('user-agent:'):
                    current_user_agent = line.split(':', 1)[1].strip().lower()
                elif line.lower().startswith('crawl-delay:') and current_user_agent:
                    if current_user_agent == '*' or 'sdg-pipeline-bot' in current_user_agent:
                        try:
                            delay_value = float(line.split(':', 1)[1].strip())
                            delay = min(delay_value, self.max_delay) # Cap at max_delay
                            break
                        except ValueError:
                            continue

            logger.debug(f"Using delay {delay}s for robots.txt compliance")
            return delay

        except Exception as e:
            logger.warning(f"Error parsing robots.txt: {e}")
            return self.default_delay

    def get_domain_stats(self) -> Dict[str, Dict]:
        """Get rate limiting statistics"""
        stats = {}
        for domain, (last_request, delay) in self.domain_delays.items():
            stats[domain] = {
                "last_request": last_request.isoformat(),
                "delay_seconds": delay,
                "robots_cached": domain in self.robots_cache
            }
        return stats

    def health_check(self) -> Dict[str, any]:
        """Health check for rate limiter"""
        return {
            "status": "healthy",
            "tracked_domains": len(self.domain_delays),
            "robots_cache_size": len(self.robots_cache),
            "default_delay": self.default_delay,
            "max_delay": self.max_delay
        }
