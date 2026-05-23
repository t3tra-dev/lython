from __future__ import annotations

from .callable_parts import (
    CallableResolutionMixin,
    CallableReturnSummaryMixin,
    CallableValueMixin,
)


class CallableMetadataMixin(
    CallableValueMixin,
    CallableResolutionMixin,
    CallableReturnSummaryMixin,
):
    """Facade for callable metadata, resolution, and return summary tracking."""
