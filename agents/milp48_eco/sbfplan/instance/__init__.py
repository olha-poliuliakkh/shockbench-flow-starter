"""Instances: frozen dataclasses, canonical JSON and hash, the `tiny` fixture (design §2; Q70, Q85)."""

from sbfplan.instance.io import (
    InstanceError,
    canonical_json,
    load_instance,
    missing_provenance,
    placeholder_leaves,
    sha256_hex,
)
from sbfplan.instance.schema import SCHEMA_VERSION, Instance


__all__ = [
    "SCHEMA_VERSION",
    "Instance",
    "InstanceError",
    "canonical_json",
    "load_instance",
    "missing_provenance",
    "placeholder_leaves",
    "sha256_hex",
]
