"""Demonstrate temperature effects with clearer API error handling."""

import os

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError


PROMPT = "Write a one-sentence tagline for a coffee shop."
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURES = [0.0, 0.5, 1.0, 1.5]


def main() -> int:
    """Run the temperature demo and explain common API failures clearly."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Add it to your environment or .env file.")
        return 1

    client = OpenAI()

    for temp in TEMPERATURES:
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": PROMPT}],
                temperature=temp,
                max_tokens=50,
            )
        except RateLimitError as exc:
            error = getattr(exc, "body", {}) or {}
            error_details = error.get("error", {})
            error_code = error_details.get("code")

            if error_code == "insufficient_quota":
                print(
                    "OpenAI request failed: your account has no available API quota. "
                    "Add billing or credits, or switch to a different provider/model."
                )
            else:
                print(f"OpenAI rate limit hit: {exc}")

            return 1

        content = response.choices[0].message.content or ""
        print(f"Temperature {temp}: {content.strip()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
