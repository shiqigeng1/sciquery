from __future__ import annotations

import json
import re
from typing import Any


def extract_json_payload(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("Empty response; expected JSON payload.")

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidates = fenced + [text]
    decoder = json.JSONDecoder()

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        for index, char in enumerate(candidate):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
                return value
            except json.JSONDecodeError:
                continue

    raise ValueError("Unable to locate a valid JSON payload in model response.")
