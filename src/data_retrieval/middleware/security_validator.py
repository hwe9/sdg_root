# src/data_retrieval/middleware/security_validator.py
"""
Enhanced Security Validator with optional integration of the central URLValidator.

Key points:
- Preserves existing SSRF protections and adds optional whitelist enforcement via URLValidator.
- Keeps explicit blocks for sensitive ports used in mail services (110/143/993/995), which are
  not blocked in the core URLValidator by default.
- Adds checks for URL credentials, unicode/control chars, and configurable whitelist enforcement.
- Keeps message wording compatible with existing expectations.
"""

import os
import logging
import socket
import ipaddress
import re
from urllib.parse import urlparse
from typing import Dict, Any, Optional
from dataclasses import dataclass
from ...core.url_validator import URLValidator  # path aligned with src layout

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SecurityValidationResult:
    is_valid: bool
    reason: str = ""
    risk_level: str = "low"  # low, medium, high, critical


class SecurityValidator:
    """
    Enhanced security validation with SSRF protection and integration of the core URLValidator.

    Backward compatibility:
    - Continues to block RFC1918, link-local, loopback ranges, and metadata hosts.
    - Keeps explicit denial for mail-related ports (110/143/993/995) and other risky service ports.
    - Optionally enforces the URLValidator's domain whitelist; disabled by default to avoid breaking
      existing retrieval on non-whitelisted public domains.
    """

    def __init__(self):
        # Reuse central validator
        self.url_validator = URLValidator()

        # Optional enforcement of URLValidator's domain whitelist (default off for compatibility)
        self.enforce_whitelist = bool(int(os.getenv("URL_VALIDATOR_ENFORCE_WHITELIST", "0")))
        # Allow adding extra allowed domains without changing core config
        extra_domains = os.getenv("URL_VALIDATOR_EXTRA_ALLOWED_DOMAINS", "")
        if extra_domains:
            for d in [x.strip().lower() for x in extra_domains.split(",") if x.strip()]:
                self.url_validator.allowed_domains.add(d)

        # Blocked IP ranges (RFC 1918, localhost, link-local, multicast, IPv6 local)
        self.blocked_networks = [
            ipaddress.ip_network("127.0.0.0/8"),
            ipaddress.ip_network("10.0.0.0/8"),
            ipaddress.ip_network("172.16.0.0/12"),
            ipaddress.ip_network("192.168.0.0/16"),
            ipaddress.ip_network("169.254.0.0/16"),
            ipaddress.ip_network("224.0.0.0/4"),
            ipaddress.ip_network("::1/128"),
            ipaddress.ip_network("fc00::/7"),
            ipaddress.ip_network("fe80::/10"),
        ]

        # Blocked hostnames/domains (metadata endpoints etc.)
        self.blocked_hostnames = {
            "localhost",
            "metadata.google.internal",
            "169.254.169.254",  # AWS/GCP metadata
            "100.100.100.200",  # Alibaba Cloud metadata
        }

        # Suspicious patterns (beyond scheme)
        self.suspicious_patterns = [
            "file://",
            "ftp://",
            "gopher://",
            "dict://",
            "ldap://",
            "jar://",
            "data:",         
            "smb://",        
            "mailto:",       
            "ssh://",        
        ]

        # Additional risky service ports (kept from original security validator)
        self.extra_blocked_ports = {22, 23, 25, 53, 110, 143, 993, 995}

        # Max URL length hard limit
        self.max_url_length = int(os.getenv("URL_MAX_LENGTH", "2048"))

        # Precompiled control-char check
        self._control_chars_rx = re.compile(r"[\x00-\x1F\x7F]")

    def _classify_risk(self, reason: str) -> str:
        txt = reason.lower()
        if any(k in txt for k in ["metadata", "blocked network", "direct ip", "localhost", "path traversal"]):
            return "critical"
        if any(k in txt for k in ["port", "scheme", "suspicious", "credentials"]):
            return "high"
        if any(k in txt for k in ["resolve", "long", "unknown"]):
            return "medium"
        return "low"

    def _has_credentials(self, parsed) -> bool:
        # Disallow credentials in URL to mitigate credential-leak vectors
        return bool(getattr(parsed, "username", None) or getattr(parsed, "password", None))

    def _contains_control_chars(self, url: str) -> bool:
        return bool(self._control_chars_rx.search(url))

    async def validate_url(self, url: str) -> SecurityValidationResult:
        """Comprehensive URL security validation with layered checks."""
        try:
            # Basic parsing
            try:
                parsed = urlparse(url)
            except Exception:
                return SecurityValidationResult(
                    is_valid=False,
                    reason="Invalid URL format",
                    risk_level="medium",
                )

            # Scheme check (limit to HTTP/S)
            scheme = (parsed.scheme or "").lower()
            if scheme not in {"http", "https"}:
                return SecurityValidationResult(
                    is_valid=False,
                    reason=f"Disallowed scheme: {parsed.scheme}",
                    risk_level="high",
                )

            # Quick rejects: suspicious patterns beyond scheme, control chars, credentials
            url_lower = url.lower()
            for pattern in self.suspicious_patterns:
                if pattern in url_lower:
                    return SecurityValidationResult(
                        is_valid=False,
                        reason=f"Suspicious URL pattern: {pattern}",
                        risk_level="high",
                    )

            if self._contains_control_chars(url):
                return SecurityValidationResult(
                    is_valid=False,
                    reason="Control characters detected in URL",
                    risk_level="high",
                )

            if self._has_credentials(parsed):
                return SecurityValidationResult(
                    is_valid=False,
                    reason="Credentials in URL are not allowed",
                    risk_level="high",
                )

            # Hostname presence
            hostname = (parsed.hostname or "").strip()
            if not hostname:
                return SecurityValidationResult(
                    is_valid=False,
                    reason="No hostname in URL",
                    risk_level="medium",
                )

            # Block explicit dangerous hostnames
            if hostname.lower() in self.blocked_hostnames:
                return SecurityValidationResult(
                    is_valid=False,
                    reason=f"Blocked hostname: {hostname}",
                    risk_level="critical",
                )

            ok, reason = self.url_validator.validate_url(url)
            if not ok:
                whitelist_msg = "not in whitelist"
                if (not self.enforce_whitelist) and (whitelist_msg in reason):
                    logger.warning(f"[SecurityValidator] Whitelist not enforced, continuing: {reason}")
                else:
                    return SecurityValidationResult(
                        is_valid=False,
                        reason=reason,
                        risk_level=self._classify_risk(reason),
                    )

            # Additional risky service ports not covered by core URLValidator
            # (kept for backward compatibility)
            port = parsed.port or (443 if scheme == "https" else 80)
            if port in self.extra_blocked_ports:
                return SecurityValidationResult(
                    is_valid=False,
                    reason=f"Potentially unsafe port: {port}",
                    risk_level="high",
                )

            # URL length guardrail
            if len(url) > self.max_url_length:
                return SecurityValidationResult(
                    is_valid=False,
                    reason="URL too long",
                    risk_level="medium",
                )

            # Path traversal
            if "../" in parsed.path or "..\\" in parsed.path:
                return SecurityValidationResult(
                    is_valid=False,
                    reason="Path traversal detected",
                    risk_level="high",
                )

            # Defensive re-resolution (redundant with URLValidator, but kept to preserve behavior)
            try:
                resolved = socket.getaddrinfo(hostname, None)
                for ip_info in resolved:
                    ip_str = ip_info[4][0]
                    try:
                        ip_addr = ipaddress.ip_address(ip_str)
                        for network in self.blocked_networks:
                            if ip_addr in network:
                                return SecurityValidationResult(
                                    is_valid=False,
                                    reason=f"IP {ip_str} in blocked network {network}",
                                    risk_level="critical",
                                )
                    except ValueError:
                        continue
            except socket.gaierror:
                return SecurityValidationResult(
                    is_valid=False,
                    reason=f"Cannot resolve hostname: {hostname}",
                    risk_level="medium",
                )

            # All checks passed
            return SecurityValidationResult(
                is_valid=True,
                reason="URL passed security validation",
                risk_level="low",
            )

        except Exception as e:
            logger.error(f"Error in security validation: {e}")
            return SecurityValidationResult(
                is_valid=False,
                reason=f"Security validation error: {str(e)}",
                risk_level="high",
            )

    def health_check(self) -> Dict[str, Any]:
        """Health check for security validator with URLValidator details."""
        try:
            return {
                "status": "healthy",
                "blocked_networks_count": len(self.blocked_networks),
                "blocked_hostnames_count": len(self.blocked_hostnames),
                "suspicious_patterns_count": len(self.suspicious_patterns),
                "url_validator": {
                    "allowed_domains_count": len(self.url_validator.allowed_domains),
                    "blocked_ports_core_count": len(self.url_validator.blocked_ports),
                    "extra_blocked_ports_count": len(self.extra_blocked_ports),
                    "enforce_whitelist": self.enforce_whitelist,
                    "max_url_length": self.max_url_length,
                },
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
