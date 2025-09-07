# src/tests/test_url_validator.py
import pytest
from src.core.url_validator import URLValidator

def test_reject_loopback_ip_literal():
    v = URLValidator()
    ok, reason = v.validate_url("http://127.0.0.1:8080")
    assert ok is False
    assert "Direct IP addresses not allowed" in reason

def test_reject_private_rfc1918_ip_literal():
    v = URLValidator()
    ok, reason = v.validate_url("http://10.0.0.1")
    assert ok is False
    assert "Direct IP addresses not allowed" in reason

def test_blocked_port_postgres_on_whitelisted_domain():
    v = URLValidator()
    ok, reason = v.validate_url("https://sdgs.un.org:5432")
    assert ok is False
    assert "Port 5432 is blocked" in reason

def test_blocked_port_redis_on_whitelisted_domain():
    v = URLValidator()
    ok, reason = v.validate_url("https://sdgs.un.org:6379")
    assert ok is False
    assert "Port 6379 is blocked" in reason
