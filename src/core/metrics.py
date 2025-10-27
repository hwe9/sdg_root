# src/core/metrics.py
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict
from typing import Sequence
from typing import Tuple
from typing import Optional
from typing import Any

from prometheus_client import REGISTRY
from prometheus_client import Histogram
from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import Summary


# --- interne Caches, damit wir jeden Collector nur einmal anlegen ---
_HIST: Dict[str, Histogram] = {}
_CNT: Dict[str, Counter] = {}
_GAUGE: Dict[str, Gauge] = {}
_SUM: Dict[str, Summary] = {}


def _key(prefix: str, name: str, labelnames: Sequence[str] = (), extra: Optional[Tuple[Any, ...]] = None) -> str:
    """Eindeutiger Key für den Cache (pro Metriktyp + Name + Labels + Extras)."""
    parts = [prefix, name, ",".join(labelnames)]
    if extra is not None:
        parts.append(repr(extra))
    return "|".join(parts)


def _find_existing_collector(name: str):
    mapping = getattr(REGISTRY, "_names_to_collectors", None)
    if isinstance(mapping, dict):
        return mapping.get(name)
    return None


def get_histogram(
    name: str,
    documentation: str,
    *,
    labelnames: Sequence[str] = (),
    buckets: Optional[Sequence[float]] = None,
    registry=REGISTRY,
    **kwargs
) -> Histogram:
    key = _key("H", name, labelnames, tuple(buckets) if buckets is not None else None)
    if key in _HIST:
        return _HIST[key]

    try:
        h = Histogram(
            name,
            documentation,
            labelnames=labelnames,
            buckets=buckets,          # None => Default-Buckets
            registry=registry,
            **kwargs
        )
    except ValueError:
        # Vermutlich schon registriert → versuche vorhandenen Collector zu holen
        existing = _find_existing_collector(name)
        if isinstance(existing, Histogram):
            h = existing
        else:
            raise

    _HIST[key] = h
    return h


def get_counter(
    name: str,
    documentation: str,
    *,
    labelnames: Sequence[str] = (),
    registry=REGISTRY,
    **kwargs
) -> Counter:
    key = _key("C", name, labelnames)
    if key in _CNT:
        return _CNT[key]

    try:
        c = Counter(
            name,
            documentation,
            labelnames=labelnames,
            registry=registry,
            **kwargs
        )
    except ValueError:
        existing = _find_existing_collector(name)
        if isinstance(existing, Counter):
            c = existing
        else:
            raise

    _CNT[key] = c
    return c


def get_gauge(
    name: str,
    documentation: str,
    *,
    labelnames: Sequence[str] = (),
    registry=REGISTRY,
    **kwargs
) -> Gauge:
    key = _key("G", name, labelnames)
    if key in _GAUGE:
        return _GAUGE[key]

    try:
        g = Gauge(
            name,
            documentation,
            labelnames=labelnames,
            registry=registry,
            **kwargs
        )
    except ValueError:
        existing = _find_existing_collector(name)
        if isinstance(existing, Gauge):
            g = existing
        else:
            raise

    _GAUGE[key] = g
    return g


def get_summary(
    name: str,
    documentation: str,
    *,
    labelnames: Sequence[str] = (),
    registry=REGISTRY,
    **kwargs
) -> Summary:
    key = _key("S", name, labelnames)
    if key in _SUM:
        return _SUM[key]

    try:
        s = Summary(
            name,
            documentation,
            labelnames=labelnames,
            registry=registry,
            **kwargs
        )
    except ValueError:
        existing = _find_existing_collector(name)
        if isinstance(existing, Summary):
            s = existing
        else:
            raise

    _SUM[key] = s
    return s



@contextmanager
def time_block(histogram: Histogram, **labels):
    """
    Context-Manager zum Stoppen/Umschließen von Blöcken:
        with time_block(HIST, service="db"): ...
    """
    timer = histogram.labels(**labels).time() if labels else histogram.time()
    with timer:
        yield


def timeit(histogram: Histogram, **labels):
    """
    Decorator zum Messen von Funktionslaufzeiten:
        @timeit(HIST, service="db")
        def foo(): ...
    """
    def deco(fn):
        def wrapper(*args, **kwargs):
            with time_block(histogram, **labels):
                return fn(*args, **kwargs)
        return wrapper
    return deco
