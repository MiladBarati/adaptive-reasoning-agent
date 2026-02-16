"""Centralized OpenTelemetry tracing configuration.

Tracing is opt-in via the OTEL_ENABLED environment variable.
When disabled (default), all tracers return no-op spans with zero overhead.
"""

import os

from opentelemetry import metrics, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

_initialized: bool = False


def setup_telemetry() -> None:
    """Initialize the OpenTelemetry TracerProvider.

    Call this once at application startup (e.g., in FastAPI lifespan).
    Reads OTEL_ENABLED from the environment; defaults to "false".
    """
    global _initialized
    if _initialized:
        return

    enabled = os.getenv("OTEL_ENABLED", "false").lower() == "true"

    if enabled:
        resource = Resource.create({"service.name": "rag-agent"})
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        # Initialize Prometheus metrics exporter
        from opentelemetry import metrics
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        from opentelemetry.sdk.metrics import MeterProvider

        # PrometheusMetricReader automatically exposes a WSGI app to serve metrics
        # We will mount this in FastAPI later
        reader = PrometheusMetricReader()
        meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(meter_provider)

    _initialized = True


def get_tracer(name: str) -> trace.Tracer:
    """Return a tracer for the given module name.

    If telemetry was not set up or is disabled, this returns a no-op tracer.
    """
    return trace.get_tracer(name)


def get_meter(name: str) -> "metrics.Meter":
    """Return a meter for the given module name.

    If telemetry was not set up or is disabled, this returns a no-op meter.
    """
    from opentelemetry import metrics

    return metrics.get_meter(name)
