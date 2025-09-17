import pytest
import httpx
import asyncio
import os

BASE_URL = os.getenv("DATA_RETRIEVAL_URL", "http://localhost:8002")

@pytest.mark.asyncio
async def test_start_retrieval_json_and_pdf():
    # JSON-API-Quelle und PDF-Quelle (kleine Testdatei)
    sources = [
        "https://httpbin.org/json",
        "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    ]
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{BASE_URL}/retrieve",
            json={"sources": sources, "force_refresh": True, "max_concurrent": 2}
        )
        assert resp.status_code == 200
