import hashlib
import json
import re
from typing import Annotated, Any, Final, Literal

from pydantic import AfterValidator, Field, TypeAdapter, field_validator

from .common import NonBlankLabel, StrictModel, require_unique

MAX_SCHEMA_BYTES: Final = 64_000
MAX_SCHEMA_DEPTH: Final = 16
MAX_REGEX_CHARS: Final = 1_000
MAX_CHOICES: Final = 256
MAX_GRAMMAR_CHARS: Final = 32_000


def canonical_hash(obj: Any) -> str:
    # Compact separators: whitespace in the original request doesn't matter.
    # No sort_keys: field order is meaningful, so it stays part of the key.
    blob = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class _ConstraintBase(StrictModel):
    def cache_key(self) -> str:
        return canonical_hash(self.model_dump(mode="json", by_alias=True))


class JsonSchemaConstraint(_ConstraintBase):
    type: Literal["json_schema"] = "json_schema"
    json_schema: dict[str, Any] = Field(alias="schema")

    @field_validator("json_schema")
    @classmethod
    def _check_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        blob = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        size_bytes = len(blob.encode("utf-8"))
        if size_bytes > MAX_SCHEMA_BYTES:
            raise ValueError(
                f"schema is {size_bytes} bytes; the limit is {MAX_SCHEMA_BYTES}"
            )

        depth = _max_depth(value)
        if depth > MAX_SCHEMA_DEPTH:
            raise ValueError(
                f"schema depth is {depth}; the limit is {MAX_SCHEMA_DEPTH}"
            )

        refs = _iter_refs(value)
        for ref in refs:
            if not isinstance(ref, str) or not ref.startswith("#"):
                raise ValueError(
                    f"only local $ref values starting with '#' are allowed, got {ref!r}"
                )

        if value.get("type") != "object":
            raise ValueError('the root schema must have "type": "object"')

        return value


def _max_depth(node: Any, level: int = 1) -> int:
    if level > MAX_SCHEMA_DEPTH:
        return level

    if isinstance(node, dict):
        return max((_max_depth(v, level + 1) for v in node.values()), default=level)
    elif isinstance(node, list):
        return max((_max_depth(item, level + 1) for item in node), default=level)
    else:
        return level


def _iter_refs(node: Any) -> list[Any]:
    refs = []

    if isinstance(node, dict):
        if "$ref" in node:
            refs.append(node["$ref"])
        for v in node.values():
            refs.extend(_iter_refs(v))
    elif isinstance(node, list):
        for item in node:
            refs.extend(_iter_refs(item))

    return refs


class RegexConstraint(_ConstraintBase):
    type: Literal["regex"] = "regex"
    pattern: str = Field(min_length=1, max_length=MAX_REGEX_CHARS)

    @field_validator("pattern")
    @classmethod
    def _check_pattern(cls, value: str) -> str:
        # Reject lookarounds
        if "(?=" in value or "(?!" in value or "(?<=" in value or "(?<!" in value:
            raise ValueError("pattern contains lookarounds, which are not supported")

        # Reject backreferences \1...\9
        if re.search(r"\\[1-9]", value):
            raise ValueError("pattern contains backreferences, which are not supported")

        # Reject named backreferences \k<...>
        if "\\k<" in value:
            raise ValueError(
                "pattern contains named backreferences, which are not supported"
            )

        # Compile to catch syntax errors
        try:
            re.compile(value)
        except re.error as e:
            raise ValueError(f"invalid regex pattern: {e}") from e

        return value


class ChoiceConstraint(_ConstraintBase):
    type: Literal["choice"] = "choice"
    choices: Annotated[
        tuple[NonBlankLabel, ...],
        Field(min_length=2, max_length=MAX_CHOICES),
        AfterValidator(require_unique),
    ]


class GrammarConstraint(_ConstraintBase):
    type: Literal["grammar"] = "grammar"
    grammar: str = Field(min_length=1, max_length=MAX_GRAMMAR_CHARS)

    @field_validator("grammar")
    @classmethod
    def _check_grammar(cls, value: str) -> str:
        if not re.search(r"^\s*root\s*::=", value, re.MULTILINE):
            raise ValueError("grammar must define a root rule (e.g., 'root ::= ...')")
        return value


Constraint = Annotated[
    JsonSchemaConstraint | RegexConstraint | ChoiceConstraint | GrammarConstraint,
    Field(discriminator="type"),
]

CONSTRAINT_ADAPTER: Final = TypeAdapter(Constraint)  # build once, at import
