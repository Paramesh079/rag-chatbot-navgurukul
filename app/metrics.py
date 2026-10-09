"""Latency & quality metrics tracking."""

from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class QueryMetric:
    """Metrics for a single query."""

    query: str
    total_latency_s: float
    retrieval_latency_s: float
    generation_latency_s: float
    num_chunks_retrieved: int
    model_used: str
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """In-memory metrics collector with rolling window."""

    def __init__(self, max_history: int = 1000):
        self._history: Deque[QueryMetric] = deque(maxlen=max_history)

    def record(self, metric: QueryMetric) -> None:
        self._history.append(metric)

    def get_summary(self) -> Dict:
        """Compute summary statistics over recorded queries."""
        if not self._history:
            return {
                "total_queries": 0,
                "avg_latency_s": 0,
                "p50_latency_s": 0,
                "p95_latency_s": 0,
                "p99_latency_s": 0,
                "avg_chunks_retrieved": 0,
            }

        latencies = [m.total_latency_s for m in self._history]
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        def percentile(data: List[float], p: float) -> float:
            idx = int(p / 100 * (len(data) - 1))
            return data[min(idx, len(data) - 1)]

        return {
            "total_queries": n,
            "avg_latency_s": round(statistics.mean(latencies), 3),
            "p50_latency_s": round(percentile(latencies_sorted, 50), 3),
            "p95_latency_s": round(percentile(latencies_sorted, 95), 3),
            "p99_latency_s": round(percentile(latencies_sorted, 99), 3),
            "min_latency_s": round(min(latencies), 3),
            "max_latency_s": round(max(latencies), 3),
            "avg_chunks_retrieved": round(
                statistics.mean(m.num_chunks_retrieved for m in self._history), 1
            ),
            "avg_retrieval_latency_s": round(
                statistics.mean(m.retrieval_latency_s for m in self._history), 3
            ),
            "avg_generation_latency_s": round(
                statistics.mean(m.generation_latency_s for m in self._history), 3
            ),
        }

    def get_recent(self, n: int = 20) -> List[Dict]:
        """Return the last n query metrics as dicts."""
        items = list(self._history)[-n:]
        return [
            {
                "query": m.query[:100],
                "total_latency_s": round(m.total_latency_s, 3),
                "retrieval_latency_s": round(m.retrieval_latency_s, 3),
                "generation_latency_s": round(m.generation_latency_s, 3),
                "num_chunks": m.num_chunks_retrieved,
                "model": m.model_used,
                "timestamp": m.timestamp,
            }
            for m in items
        ]


# Global singleton
metrics_collector = MetricsCollector()
