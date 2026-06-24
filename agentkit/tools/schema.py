"""Infer a JSON Schema (draft 2020-12) for a tool from a function's signature.

We build a Pydantic model from the function's typed parameters and emit its
JSON schema. Parameter descriptions are pulled from the docstring when present.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable

from pydantic import create_model
from pydantic.fields import FieldInfo


def _split_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Return (summary, {param_name: description}) from a docstring.

    Supports the common ``Args:`` / ``Parameters:`` block where each line is
    ``name: description`` or ``name (type): description``.
    """
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()
    summary_lines: list[str] = []
    params: dict[str, str] = {}
    in_args = False
    for line in lines:
        header = line.strip().lower().rstrip(":")
        if header in ("args", "arguments", "parameters", "params"):
            in_args = True
            continue
        if in_args:
            if not line.strip():
                continue
            m = re.match(r"\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)", line)
            if m:
                params[m.group(1)] = m.group(2).strip()
            else:
                # A continuation or a new section ends the args block.
                if line.strip().lower().rstrip(":") in ("returns", "raises", "examples"):
                    in_args = False
        else:
            summary_lines.append(line)
    return " ".join(s.strip() for s in summary_lines if s.strip()).strip(), params


def build_schema(fn: Callable) -> tuple[str, dict[str, Any]]:
    """Build ``(description, json_schema)`` for ``fn``.

    The RunContext parameter (named ``ctx`` or annotated RunContext) is excluded
    from the model-facing schema; it is injected by the tool runner.
    """
    sig = inspect.signature(fn)
    summary, param_docs = _split_docstring(fn.__doc__)

    fields: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name in ("self", "cls", "ctx", "context"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else Any
        default = param.default if param.default is not inspect.Parameter.empty else ...
        field = FieldInfo(default=default, description=param_docs.get(name))
        fields[name] = (annotation, field)

    model = create_model(f"{fn.__name__}_Args", **fields)  # type: ignore[call-overload]
    schema = model.model_json_schema()
    # JSON Schema draft 2020-12 dialect marker.
    schema.setdefault("$schema", "https://json-schema.org/draft/2020-12/schema")
    return summary or (fn.__name__.replace("_", " ")), schema
