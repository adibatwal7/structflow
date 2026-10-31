"""Step 3.5 measurements: TTFT, TPOT and tokens for masked vs free JSON.

Run it while StructFlow (port 8080) is connected to vLLM:
    python measure35.py
"""

import json
import statistics
import sys
import time

import httpx2

URL = "http://localhost:8080/v1/infer"
RUNS = 5
TEXT = (
    "Maria Lopez is 34 and lives in Madrid. Kenji Sato is 51 and lives in Osaka. "
    "Amara Okafor is 27 and lives in Lagos."
)
PERSON = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"},
    },
    "required": ["name", "age", "city"],
    "additionalProperties": False,
}
SCHEMA = {
    "type": "object",
    "properties": {"people": {"type": "array", "items": PERSON}},
    "required": ["people"],
    "additionalProperties": False,
}
ASK = [{"role": "user", "content": f"List every person in this text.\n\n{TEXT}"}]
SAMPLING = {"temperature": 0, "max_tokens": 300}

MASKED = {
    "task": "structured",
    "messages": ASK,
    "constraint": {"type": "json_schema", "schema": SCHEMA},
    "sampling": SAMPLING,
}
FREE = {  # the same instruction, but no mask: the model is only asked nicely
    "task": "generate",
    "messages": [
        {
            "role": "system",
            "content": "Reply with only a JSON object that matches this JSON Schema:\n"
            + json.dumps(SCHEMA),
        },
        *ASK,
    ],
    "sampling": SAMPLING,
}


def run(name: str, body: dict[str, object], client: httpx2.Client) -> None:
    ttft, tpot, tokens = [], [], []
    for i in range(RUNS + 1):
        r = client.post(URL, json=body)
        if r.status_code != 200:
            print(f"{name}: HTTP {r.status_code} {r.text[:300]}")
            return
        data = r.json()
        t = data["timings"]
        if i == 0:  # the first call may compile the grammar: show it, don't average it
            print(f"{name:<14} first call: ttft {t['ttft_ms'] or 0:7.1f} ms")
            continue
        ttft.append(t["ttft_ms"] or 0.0)
        tpot.append(t["tpot_ms"] or 0.0)
        tokens.append(data["usage"]["completion_tokens"])
        time.sleep(0.2)
    print(
        f"{name:<14} median of {RUNS}: ttft {statistics.median(ttft):7.1f} ms | "
        f"tpot {statistics.median(tpot):5.1f} ms | tokens {statistics.median(tokens):.0f}"
    )
    result = data["result"]
    print(
        f"{'':<14} sample answer: {json.dumps(result.get('raw_text') or result.get('text'))[:160]}"
    )


def main() -> int:
    with httpx2.Client(timeout=120) as client:
        run("masked JSON", MASKED, client)
        run("free JSON", FREE, client)
    return 0


if __name__ == "__main__":
    sys.exit(main())
