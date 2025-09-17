# src/core/url_validator.py
"""
Central URL validator used across services.

Behavior:
- Allows only http/https schemes.
- Denies direct IP literals and hostnames resolving to private/link-local/multicast ranges.
- Enforces a domain whitelist (configurable by services wrapping this validator).
- Denies a set of sensitive infra ports, including DB/Redis/SMB etc.
- Returns (ok: bool, reason: str) with stable messages used by tests.

Note: SecurityValidator may choose to NOT enforce the domain whitelist in compatibility mode.
"""

import re
import socket
import ipaddress
from urllib.parse import urlparse
from typing import Set, List, Tuple
import logging

logger = logging.getLogger(__name__)


class URLValidator:
    def __init__(self):
        # Blocked IP ranges (RFC 1918 private networks, localhost, etc.)
        self.blocked_networks = [
            ipaddress.ip_network("127.0.0.0/8"),     # Localhost
            ipaddress.ip_network("10.0.0.0/8"),      # Private A
            ipaddress.ip_network("172.16.0.0/12"),   # Private B
            ipaddress.ip_network("192.168.0.0/16"),  # Private C
            ipaddress.ip_network("169.254.0.0/16"),  # Link-local
            ipaddress.ip_network("224.0.0.0/4"),     # Multicast
            ipaddress.ip_network("::1/128"),         # IPv6 localhost
            ipaddress.ip_network("fc00::/7"),        # IPv6 private
        ]

        # Allowed schemes
        self.allowed_schemes = {"http", "https"}

        # Blocked ports (infra/services)
        self.blocked_ports = {
            22,    # SSH
            23,    # Telnet
            25,    # SMTP
            53,    # DNS
            135,   # RPC
            139,   # NetBIOS
            445,   # SMB
            1433,  # SQL Server
            3306,  # MySQL
            5432,  # PostgreSQL
            6379,  # Redis
            8080,  # Weaviate (internal)
        }

        # Allowed domains (whitelist)
        self.allowed_domains: Set[str] = {
            "un.org", "who.int", "worldbank.org", "oecd.org",
            "europa.eu", "unicef.org", "undp.org", "unesco.org",
            "doi.org", "crossref.org", "googleapis.com",
        }

    def validate_url(self, url: str) -> Tuple[bool, str]:
        try:
            parsed = urlparse(url)
            scheme = (parsed.scheme or "").lower()
            if scheme not in self.allowed_schemes:
                return False, f"Scheme '{scheme}' not allowed"

            hostname = (parsed.hostname or "").lower()
            if not hostname:
                return False, "No hostname provided"

            # 1) Direct IP literals: always reject (IPv4/IPv6)
            try:
                ipaddress.ip_address(hostname)
                return False, "Direct IP addresses not allowed"
            except ValueError:
                pass  # hostname is not a direct IP literal

            # 2) Whitelist (exact or subdomain match)
            if not self._is_domain_allowed(hostname):
                return False, f"Domain '{hostname}' not in whitelist"

            # 3) DNS resolution and block private/link-local/multicast
            try:
                infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
                for family, _, _, _, sockaddr in infos:
                    ip_str = sockaddr[0]
                    try:
                        ip = ipaddress.ip_address(ip_str)
                        if self._is_ip_blocked(ip):
                            return False, f"IP address {ip} is in blocked range"
                    except ValueError:
                        continue
            except Exception as e:
                return False, f"Cannot resolve hostname: {e}"

            # 4) Ports
            port = parsed.port or (443 if scheme == "https" else 80)
            if port in self.blocked_ports:
                return False, f"Port {port} is blocked"

            return True, ""
        except Exception as e:
            logger.error(f"URL validation error: {e}")
            return False, f"URL validation failed: {e}"

    def _is_domain_allowed(self, hostname: str) -> bool:
        """Check if domain is in whitelist; exact or subdomain match."""
        hostname = hostname.lower()
        if hostname in self.allowed_domains:
            return True
        for allowed_domain in self.allowed_domains:
            if hostname.endswith(f".{allowed_domain}"):
                return True
        return False

    def _is_ip_blocked(self, ip: ipaddress.ip_address) -> bool:
        """Check if IP is in blocked ranges."""
        for network in self.blocked_networks:
            if ip in network:
                return True
        return False


# Global validator instance (optional convenience)
url_validator = URLValidator()
