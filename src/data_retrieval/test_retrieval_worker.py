# test_retrieval_worker.py

import os
import pytest
from pathlib import Path
from retrieval_worker import RetrievalWorker

tmp_path = Path("/app/raw_data")

def test_load_sources(tmp_path):
    testfile = tmp_path / "quelle.txt"
    test_link = "https://wwwcdn.imo.org/localresources/en/MediaCentre/HotTopics/Documents/IMO%20SDG%20Brochure.pdf"
    testfile.write_text(test_link)
    worker = RetrievalWorker(str(testfile), str(tmp_path), str(tmp_path / "out.json"))
    urls = worker.load_sources()
    assert test_link in urls

def test_download_generic_content(tmp_path):
    worker = RetrievalWorker("", str(tmp_path), "")
    # Die URL muss auf eine kleine, existierende Datei im Netz zeigen!
    result = worker.download_generic_content("https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf")
    assert result is not None
    assert result["file_path"].endswith(".pdf")
    assert os.path.exists(result["file_path"])
