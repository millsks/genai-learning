# Exploring tokenization with the tiktoken library (used by OpenAI models)
# pixi add tiktoken

import tiktoken

# Load the tokenizer for GPT-4
encoder = tiktoken.encoding_for_model("gpt-4")

text = "Generative AI is transforming software development!"
tokens = encoder.encode(text)
print(f"Text: {text}")
print(f"Token IDs: {tokens}")
print(f"Number of tokens: {len(tokens)}")

# Decode individual tokens to see the breakdown
for token_id in tokens:
    decoded = encoder.decode([token_id])
    print(f"  {token_id:>6} → '{decoded}'")