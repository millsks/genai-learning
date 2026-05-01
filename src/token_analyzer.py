# src/token_analyzer.py
"""Analyze token counts and estimated costs across LLM providers."""

import tiktoken
from rich.console import Console
from rich.table import Table

console = Console()

# Pricing per 1M tokens (approximate, as of early 2026)
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
}

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens for a given text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def analyze_prompt(prompt: str, expected_output_tokens: int = 500):
    """Analyze cost of a prompt across different models."""
    input_tokens = count_tokens(prompt)

    table = Table(title=f"Token Economy Analysis")
    table.add_column("Model", style="cyan")
    table.add_column("Input Tokens", justify="right")
    table.add_column("Est. Output Tokens", justify="right")
    table.add_column("Input Cost", justify="right", style="yellow")
    table.add_column("Output Cost", justify="right", style="yellow")
    table.add_column("Total Cost", justify="right", style="bold green")

    for model, prices in PRICING.items():
        input_cost = (input_tokens / 1_000_000) * prices["input"]
        output_cost = (expected_output_tokens / 1_000_000) * prices["output"]
        total = input_cost + output_cost
        table.add_row(
            model,
            str(input_tokens),
            str(expected_output_tokens),
            f"${input_cost:.6f}",
            f"${output_cost:.6f}",
            f"${total:.6f}",
        )

    console.print(f"\n[bold]Prompt preview:[/bold] {prompt[:100]}...")
    console.print(f"[bold]Character count:[/bold] {len(prompt)}")
    console.print(table)

if __name__ == "__main__":
    # Example: analyze a RAG prompt with context
    sample_prompt = """You are a helpful assistant that answers questions based on the 
    provided context. Use only the information in the context to answer.
    
    Context:
    The transformer architecture was introduced in 2017 in the paper "Attention Is All 
    You Need" by Vaswani et al. It replaced recurrent neural networks as the dominant 
    architecture for natural language processing tasks. The key innovation was the 
    self-attention mechanism, which allows the model to process all tokens in parallel 
    rather than sequentially. This led to massive speedups in training and enabled the 
    creation of much larger models. Modern LLMs like GPT-4, Claude, and Llama are all 
    based on the transformer architecture.
    
    Question: What was the key innovation of the transformer architecture?"""

    analyze_prompt(sample_prompt, expected_output_tokens=200)