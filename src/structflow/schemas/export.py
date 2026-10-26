import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from .errors import ProblemDetails
from .requests import INFERENCE_REQUEST_ADAPTER
from .responses import INFERENCE_RESPONSE_ADAPTER

CONTRACTS: Final[dict[str, Callable[[], dict[str, Any]]]] = {
    "inference_request": lambda: INFERENCE_REQUEST_ADAPTER.json_schema(
        mode="validation"
    ),
    "inference_response": lambda: INFERENCE_RESPONSE_ADAPTER.json_schema(
        mode="serialization"
    ),
    "problem_details": lambda: ProblemDetails.model_json_schema(mode="serialization"),
}


def render(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> int:
    output_dir = Path("docs/contracts")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, schema_fn in CONTRACTS.items():
        schema = schema_fn()
        output_path = output_dir / f"{name}.schema.json"
        output_path.write_text(render(schema), encoding="utf-8")
        print(output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
