# Backwards-compatibility shim for legacy tests
import os
from pathlib import Path
from .core.download_strategies import HTTPDownloadStrategy

class RetrievalWorker:
    def __init__(self, sources_file: str, data_dir: str, out_file: str):
        self.sources_file = sources_file
        self.data_dir = data_dir
        self.out_file = out_file

    def load_sources(self):
        urls = set()
        if self.sources_file and os.path.exists(self.sources_file):
            with open(self.sources_file, "r", encoding="utf-8") as f:
                for line in f:
                    u = line.strip()
                    if u and not u.startswith("#"):
                        urls.add(u)
        return urls

    def download_generic_content(self, url: str):
        import asyncio
        strat = HTTPDownloadStrategy()
        async def _run():
            await strat.initialize()
            try:
                return await strat.download(url, self.data_dir)
            finally:
                await strat.cleanup()
        return asyncio.run(_run())
