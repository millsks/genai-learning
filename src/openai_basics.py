# src/openai_basics.py
"""Complete guide to OpenAI API usage."""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()  # Automatically reads OPENAI_API_KEY from environment

# --- Basic Chat Completion ---
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are a concise Python tutor. Explain concepts clearly with examples.",
        },
        {
            "role": "user",
            "content": "What are Python decorators?",
        },
    ],
    temperature=0.7,
    max_tokens=500,
)

print(response.choices[0].message.content)
print(f"\nTokens used — Input: {response.usage.prompt_tokens}, "
      f"Output: {response.usage.completion_tokens}, "
      f"Total: {response.usage.total_tokens}")


# --- Streaming Chat Completion ---
def stream_chat(messages: list[dict], model: str = "gpt-4o-mini"):
    """Stream a chat completion and print tokens as they arrive."""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            print(token, end="", flush=True)
            full_response += token

    print()  # Newline after streaming completes
    return full_response

# Usage
result = stream_chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a haiku about machine learning."},
])


# --- Multi-turn Conversation Manager ---
class ChatSession:
    """Manages a multi-turn conversation with an LLM."""

    def __init__(self, model: str = "gpt-4o-mini", system_prompt: str = ""):
        self.model = model
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def chat(self, user_message: str) -> str:
        """Send a message and get a response, maintaining conversation history."""
        self.messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.7,
        )

        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message

    def get_history(self) -> list[dict]:
        """Return the full conversation history."""
        return self.messages.copy()

    def clear(self):
        """Clear conversation history (keep system prompt)."""
        self.messages = [m for m in self.messages if m["role"] == "system"]

# Usage
session = ChatSession(
    system_prompt="You are an expert Python developer. Be concise."
)
print(session.chat("What's the difference between a list and a tuple?"))
print("---")
print(session.chat("Show me a practical example of when to use each."))
print("---")
print(session.chat("Now explain which is more memory efficient and why."))


# --- Structured Output ---
import json

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You analyze text sentiment. Return JSON with keys: "
                       "sentiment (positive/negative/neutral), confidence (0-1), "
                       "key_phrases (list of strings).",
        },
        {
            "role": "user",
            "content": "I absolutely love this new AI framework! It makes development "
                       "so much easier and the documentation is fantastic.",
        },
    ],
    response_format={"type": "json_object"},
    temperature=0,
)

result = json.loads(response.choices[0].message.content)
print(json.dumps(result, indent=2))
# Output:
# {
#   "sentiment": "positive",
#   "confidence": 0.95,
#   "key_phrases": ["absolutely love", "so much easier", "documentation is fantastic"]
# }