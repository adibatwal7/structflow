from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from structflow.schemas.constraints import (
    CONSTRAINT_ADAPTER,
    MAX_SCHEMA_DEPTH,
    ChoiceConstraint,
    GrammarConstraint,
    JsonSchemaConstraint,
    RegexConstraint,
)


class Address(BaseModel):
    city: str


class Person(BaseModel):
    name: str
    address: Address


def test_real_pydantic_schema_passes() -> None:
    schema = Person.model_json_schema()  # contains "$defs" and a local "$ref"
    c = CONSTRAINT_ADAPTER.validate_python({"type": "json_schema", "schema": schema})
    assert isinstance(c, JsonSchemaConstraint)
    assert c.json_schema == schema


class TestJsonSchemaConstraint:
    def test_accepts_simple_object_schema(self) -> None:
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        c = CONSTRAINT_ADAPTER.validate_python(
            {"type": "json_schema", "schema": schema}
        )
        assert isinstance(c, JsonSchemaConstraint)
        assert c.json_schema == schema

    def test_rejects_too_large_schema(self) -> None:
        large_schema = {
            "type": "object",
            "properties": {"x" * 70000: {"type": "string"}},
        }
        with pytest.raises(ValidationError) as exc_info:
            CONSTRAINT_ADAPTER.validate_python(
                {"type": "json_schema", "schema": large_schema}
            )
        assert "bytes" in str(exc_info.value)

    def test_rejects_too_deep_schema(self) -> None:
        node: dict[str, Any] = {"type": "object"}
        for _ in range(MAX_SCHEMA_DEPTH):
            node = {"type": "object", "properties": {"x": node}}
        with pytest.raises(ValidationError) as exc_info:
            CONSTRAINT_ADAPTER.validate_python({"type": "json_schema", "schema": node})
        assert "depth" in str(exc_info.value)

    def test_rejects_external_ref(self) -> None:
        schema = {"type": "object", "$ref": "http://example.com/schema"}
        with pytest.raises(ValidationError) as exc_info:
            CONSTRAINT_ADAPTER.validate_python(
                {"type": "json_schema", "schema": schema}
            )
        assert "only local $ref" in str(exc_info.value)

    def test_accepts_local_ref(self) -> None:
        schema = {
            "type": "object",
            "$defs": {"Address": {"type": "object"}},
            "$ref": "#/$defs/Address",
        }
        c = CONSTRAINT_ADAPTER.validate_python(
            {"type": "json_schema", "schema": schema}
        )
        assert isinstance(c, JsonSchemaConstraint)

    def test_rejects_non_object_root(self) -> None:
        schema = {"type": "array", "items": {"type": "string"}}
        with pytest.raises(ValidationError) as exc_info:
            CONSTRAINT_ADAPTER.validate_python(
                {"type": "json_schema", "schema": schema}
            )
        assert 'must have "type": "object"' in str(exc_info.value)

    def test_cache_key_is_consistent(self) -> None:
        schema1 = {"type": "object", "properties": {"name": {"type": "string"}}}
        schema2 = {"type": "object", "properties": {"name": {"type": "string"}}}
        c1 = CONSTRAINT_ADAPTER.validate_python(
            {"type": "json_schema", "schema": schema1}
        )
        c2 = CONSTRAINT_ADAPTER.validate_python(
            {"type": "json_schema", "schema": schema2}
        )
        assert c1.cache_key() == c2.cache_key()

    def test_cache_key_ignores_spacing(self) -> None:
        # Same schema from JSON with different spacing should have same key
        json1 = '{"type":"json_schema","schema":{"type":"object"}}'
        json2 = '{ "type" : "json_schema" , "schema" : { "type" : "object" } }'
        c1 = CONSTRAINT_ADAPTER.validate_json(json1)
        c2 = CONSTRAINT_ADAPTER.validate_json(json2)
        assert c1.cache_key() == c2.cache_key()

    def test_cache_key_respects_field_order(self) -> None:
        # Swapping property order should change the key
        schema1 = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "number"}},
        }
        schema2 = {
            "type": "object",
            "properties": {"b": {"type": "number"}, "a": {"type": "string"}},
        }
        c1 = CONSTRAINT_ADAPTER.validate_python(
            {"type": "json_schema", "schema": schema1}
        )
        c2 = CONSTRAINT_ADAPTER.validate_python(
            {"type": "json_schema", "schema": schema2}
        )
        assert c1.cache_key() != c2.cache_key()


class TestRegexConstraint:
    def test_accepts_valid_regex(self) -> None:
        c = CONSTRAINT_ADAPTER.validate_python({"type": "regex", "pattern": r"\d+"})
        assert isinstance(c, RegexConstraint)
        assert c.pattern == r"\d+"

    def test_rejects_lookahead(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            CONSTRAINT_ADAPTER.validate_python({"type": "regex", "pattern": r"a(?=b)"})
        assert "lookarounds" in str(exc_info.value)

    def test_rejects_lookbehind(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            CONSTRAINT_ADAPTER.validate_python({"type": "regex", "pattern": r"(?<=a)b"})
        assert "lookarounds" in str(exc_info.value)

    def test_rejects_backreference(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            CONSTRAINT_ADAPTER.validate_python({"type": "regex", "pattern": r"(a)\1"})
        assert "backreferences" in str(exc_info.value)

    def test_rejects_named_backreference(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            CONSTRAINT_ADAPTER.validate_python(
                {"type": "regex", "pattern": r"(?P<name>a)\k<name>"}
            )
        assert "backreferences" in str(exc_info.value)

    def test_rejects_invalid_regex_syntax(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            CONSTRAINT_ADAPTER.validate_python(
                {"type": "regex", "pattern": r"(?unclosed"}
            )
        assert "invalid regex" in str(exc_info.value)

    def test_rejects_empty_pattern(self) -> None:
        with pytest.raises(ValidationError):
            CONSTRAINT_ADAPTER.validate_python({"type": "regex", "pattern": ""})

    def test_rejects_too_long_pattern(self) -> None:
        with pytest.raises(ValidationError):
            CONSTRAINT_ADAPTER.validate_python({"type": "regex", "pattern": "a" * 2000})


class TestChoiceConstraint:
    def test_accepts_valid_choices(self) -> None:
        c = CONSTRAINT_ADAPTER.validate_python(
            {"type": "choice", "choices": ["a", "b", "c"]}
        )
        assert isinstance(c, ChoiceConstraint)
        assert c.choices == ("a", "b", "c")

    def test_rejects_single_choice(self) -> None:
        with pytest.raises(ValidationError):
            CONSTRAINT_ADAPTER.validate_python({"type": "choice", "choices": ["a"]})

    def test_rejects_duplicate_choices(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            CONSTRAINT_ADAPTER.validate_python(
                {"type": "choice", "choices": ["a", "b", "a"]}
            )
        assert "unique" in str(exc_info.value)

    def test_rejects_too_many_choices(self) -> None:
        with pytest.raises(ValidationError):
            CONSTRAINT_ADAPTER.validate_python(
                {"type": "choice", "choices": [str(i) for i in range(300)]}
            )

    def test_rejects_blank_choice(self) -> None:
        with pytest.raises(ValidationError):
            CONSTRAINT_ADAPTER.validate_python(
                {"type": "choice", "choices": ["a", "   ", "b"]}
            )


class TestGrammarConstraint:
    def test_accepts_valid_grammar(self) -> None:
        grammar = "root ::= expr\nexpr ::= term\n"
        c = CONSTRAINT_ADAPTER.validate_python({"type": "grammar", "grammar": grammar})
        assert isinstance(c, GrammarConstraint)
        assert c.grammar == grammar

    def test_rejects_missing_root_rule(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            CONSTRAINT_ADAPTER.validate_python(
                {"type": "grammar", "grammar": "expr ::= term\n"}
            )
        assert "root rule" in str(exc_info.value)

    def test_rejects_empty_grammar(self) -> None:
        with pytest.raises(ValidationError):
            CONSTRAINT_ADAPTER.validate_python({"type": "grammar", "grammar": ""})

    def test_rejects_too_long_grammar(self) -> None:
        with pytest.raises(ValidationError):
            CONSTRAINT_ADAPTER.validate_python(
                {"type": "grammar", "grammar": "a" * 40000}
            )


class TestDiscriminator:
    def test_invalid_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CONSTRAINT_ADAPTER.validate_python({"type": "invalid", "pattern": "test"})

    def test_field_name_rejected(self) -> None:
        # Using "json_schema" instead of "schema" should fail
        with pytest.raises(ValidationError):
            CONSTRAINT_ADAPTER.validate_python(
                {"type": "json_schema", "json_schema": {"type": "object"}}
            )

    def test_tag_error_mentions_options(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            CONSTRAINT_ADAPTER.validate_python({"type": "invalid", "pattern": "test"})
        errors = exc_info.value.errors()
        assert errors[0]["type"] == "union_tag_invalid"
        assert "json_schema" in str(exc_info.value)

    def test_missing_type_gives_union_tag_not_found(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            CONSTRAINT_ADAPTER.validate_python({"pattern": "test"})
        errors = exc_info.value.errors()
        assert errors[0]["type"] == "union_tag_not_found"

    def test_regex_serializes_correctly(self) -> None:
        c = CONSTRAINT_ADAPTER.validate_python({"type": "regex", "pattern": r"\d+"})
        assert c.model_dump(mode="json") == {"type": "regex", "pattern": r"\d+"}
