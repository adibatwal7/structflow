# API Contracts

This directory contains JSON Schema definitions for StructFlow's public API.

## Files

- `inference_request.schema.json` - Schema for inference requests (classify, extract_spans, structured, generate)
- `inference_response.schema.json` - Schema for inference responses
- `problem_details.schema.json` - Schema for RFC 9457 error responses

## Regenerating Contracts

To regenerate the schema files after making changes to the Pydantic models:

```bash
python -m structflow.schemas.export
```

The contract tests in `tests/contract/test_contracts.py` will fail if the committed schemas are out of date.

## Server-Only Validation Rules

Some validation rules are enforced by the server but cannot be expressed in JSON Schema. Clients must rely on server validation for these:

- **Non-blank text** - Text fields must contain at least one non-whitespace character
- **Label count constraints** - `top_k` must not exceed the number of labels
- **Threshold constraint** - `threshold` can only be used when `multi_label` is true
- **Conversation order** - System messages only allowed as first message; last message must be from user
- **JSON Schema constraints** - Local-only `$ref` references and object root type are enforced
- **Regex restrictions** - Lookarounds and backreferences are not supported
- **Label sorting** - Classification labels must be sorted by score (highest first)
