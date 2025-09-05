"""
Enhanced Security Validator
SSRF protection and comprehensive URL validation
"""

import logging
import socket
import ipaddress
from urllib.parse import urlparse
from typing import Tuple, Dict, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SecurityValidationResult:
    is_valid: bool
    reason: str = ""
    risk_level: str = "low" # low, medium, high, critical

class SecurityValidator:
    """
    Enhanced security validation with SSRF protection
    """

    def __init__(self):
        # Blocked IP ranges (RFC 1918 private networks, localhost, etc.)
        self.blocked_networks = [
            ipaddress.ip_network('127.0.0.0/8'), # Localhost
            ipaddress.ip_network('10.0.0.0/8'), # Private Class A
            ipaddress.ip_network('172.16.0.0/12'), # Private Class B
            ipaddress.ip_network('192.168.0.0/16'), # Private Class C
            ipaddress.ip_network('169.254.0.0/16'), # Link-local
            ipaddress.ip_network('224.0.0.0/4'), # Multicast
            ipaddress.ip_network('::1/128'), # IPv6 localhost
            ipaddress.ip_network('fc00::/7'), # IPv6 unique local
            ipaddress.ip_network('fe80::/10'), # IPv6 link-local
        ]

        # Blocked hostnames/domains
        self.blocked_hostnames = {
            'localhost',
            'metadata.google.internal',
            '169.254.169.254', # AWS/GCP metadata
            '100.100.100.200', # Alibaba Cloud metadata
        }

        # Suspicious patterns
        self.suspicious_patterns = [
            'file://',
            'ftp://',
            'gopher://',
            'dict://',
            'ldap://',
            'jar://'
        ]

    async def validate_url(self, url: str) -> SecurityValidationResult:
        """Comprehensive URL security validation"""
        try:
            # Basic URL parsing
            try:
                parsed = urlparse(url)
            except Exception:
                return SecurityValidationResult(
                    is_valid=False,
                    reason="Invalid URL format",
                    risk_level="medium"
                )

            # Scheme validation
            if parsed.scheme.lower() not in ['http', 'https']:
                return SecurityValidationResult(
                    is_valid=False,
                    reason=f"Disallowed scheme: {parsed.scheme}",
                    risk_level="high"
                )

            # Check for suspicious patterns
            url_lower = url.lower()
            for pattern in self.suspicious_patterns:
                if pattern in url_lower:
                    return SecurityValidationResult(
                        is_valid=False,
                        reason=f"Suspicious URL pattern: {pattern}",
                        risk_level="high"
                    )

            # Hostname validation
            hostname = parsed.hostname
            if not hostname:
                return SecurityValidationResult(
                    is_valid=False,
                    reason="No hostname in URL",
                    risk_level="medium"
                )

            # Check blocked hostnames
            if hostname.lower() in self.blocked_hostnames:
                return SecurityValidationResult(
                    is_valid=False,
                    reason=f"Blocked hostname: {hostname}",
                    risk_level="critical"
                )

            # DNS resolution and IP validation
            try:
                resolved_ips = socket.getaddrinfo(hostname, None)
                for ip_info in resolved_ips:
                    ip_str = ip_info[4][0]
                    try:
                        ip_addr = ipaddress.ip_address(ip_str)
                        # Check against blocked networks
                        for network in self.blocked_networks:
                            if ip_addr in network:
                                return SecurityValidationResult(
                                    is_valid=False,
                                    reason=f"IP {ip_str} in blocked network {network}",
                                    risk_level="critical"
                                )
                    except ValueError:
                        # Invalid IP address format
                        continue

            except socket.gaierror:
                return SecurityValidationResult(
                    is_valid=False,
                    reason=f"Cannot resolve hostname: {hostname}",
                    risk_level="medium"
                )

            # Port validation
            port = parsed.port
            if port and port in [22, 23, 25, 53, 110, 143, 993, 995]: # Common service ports
                return SecurityValidationResult(
                    is_valid=False,
                    reason=f"Potentially unsafe port: {port}",
                    risk_level="high"
                )

            # URL length validation
            if len(url) > 2048:
                return SecurityValidationResult(
                    is_valid=False,
                    reason="URL too long",
                    risk_level="medium"
                )

            # Path traversal check
            if '../' in parsed.path or '..\\' in parsed.path:
                return SecurityValidationResult(
                    is_valid=False,
                    reason="Path traversal detected",
                    risk_level="high"
                )

            return SecurityValidationResult(
                is_valid=True,
                reason="URL passed security validation",
                risk_level="low"
            )

        except Exception as e:
            logger.error(f"Error in security validation: {e}")
            return SecurityValidationResult(
                is_valid=False,
                reason=f"Security validation error: {str(e)}",
                risk_level="high"
            )

    def health_check(self) -> Dict[str, Any]:
        """Health check for security validator"""
        return {
            "status": "healthy",
            "blocked_networks_count": len(self.blocked_networks),
            "blocked_hostnames_count": len(self.blocked_hostnames),
            "suspicious_patterns_count": len(self.suspicious_patterns)
        }
