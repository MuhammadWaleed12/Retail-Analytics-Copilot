"""Validation helpers for structured agent answers."""

import ast
import json
import math
import re
from typing import Any


def _parse_collection(value: Any) -> Any:
    """Parse a JSON or Python-literal collection without executing code."""
    if not isinstance(value, str):
        return value

    text = value.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (ValueError, SyntaxError):
            continue
    return None


def matches_format_hint(value: Any, format_hint: str) -> bool:
    """Return whether an answer satisfies the requested output shape."""
    hint = format_hint.strip()

    if hint == "int":
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        return isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()) is not None

    if hint == "float":
        if isinstance(value, bool):
            return False
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    if hint.startswith("{") and hint.endswith("}"):
        return isinstance(_parse_collection(value), dict)

    if hint.startswith("list["):
        return isinstance(_parse_collection(value), list)

    return True
