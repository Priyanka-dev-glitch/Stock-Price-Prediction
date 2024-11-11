from prometheus_fastapi_instrumentator.metrics import Info
from typing import Callable
from prometheus_client import Counter



def http_requests_total(metric_namespace='', metric_subsystem='') -> Callable[[Info], None]:
    total = Counter(
        name="http_requests_total",
        documentation="Total number of requests by method, status and handler.",
        labelnames=(
            "method",
            "status",
            "handler",
        ),
        namespace=metric_namespace,
        subsystem=metric_subsystem,
    )

    def instrumentation(info: Info) -> None:
        total.labels(info.method, info.modified_status, info.modified_handler).inc()

    return instrumentation