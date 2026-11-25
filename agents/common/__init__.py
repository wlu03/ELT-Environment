"""Shared helpers for ELT-Bench agent runners."""

from .warehouse import (
    DESTINATIONS,
    DestinationPreparationError,
    DestinationSpec,
    adapt_prompt,
    destination_choices,
    get_destination,
    prepare_destination,
    resolve_benchmark_path,
    resolve_inputs_path,
    snowflake_identifier,
)

__all__ = [
    "DESTINATIONS",
    "DestinationPreparationError",
    "DestinationSpec",
    "adapt_prompt",
    "destination_choices",
    "get_destination",
    "prepare_destination",
    "resolve_benchmark_path",
    "resolve_inputs_path",
    "snowflake_identifier",
]
