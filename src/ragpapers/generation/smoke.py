"""Smoke test of the real providers: one tiny call to the generator and one to the judge.

Checks that the keys in .env work and prints the exact model IDs each API reports.
Responses are cached, so running it again makes no API call.

Usage (PowerShell, from the repo root):
    python -m ragpapers.generation.smoke
"""

from ragpapers.config import load_config
from ragpapers.generation.client import make_client
from ragpapers.generation.providers import Message

PROMPT = [Message("user", "Reply with the single word: ready")]


def main() -> None:
    config = load_config()
    for role in ("generator", "judge"):
        client = make_client(config, role)
        r = client.generate(PROMPT)
        print(f"{role}: provider {r.provider}")
        print(f"  requested model  {r.requested_model}")
        print(f"  reported model   {r.model}")
        if r.system_fingerprint:
            print(f"  fingerprint      {r.system_fingerprint}")
        print(f"  reply            {r.text.strip()!r} (finish: {r.finish_reason})")
        print(
            f"  tokens           in {r.input_tokens}, out {r.output_tokens} "
            f"(reasoning {r.reasoning_tokens})"
        )
        print(f"  latency          {r.latency_ms:.0f} ms, cached {r.cached}")


if __name__ == "__main__":
    main()
