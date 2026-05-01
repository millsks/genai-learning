# src/verify_setup.py
"""Verify that the development environment is properly configured."""

import sys
from rich.console import Console
from rich.table import Table

console = Console()

def check_imports():
    """Check that all required packages are importable."""
    packages = {
        "openai": "OpenAI SDK",
        "anthropic": "Anthropic SDK",
        "transformers": "HuggingFace Transformers",
        "torch": "PyTorch",
        "langchain": "LangChain",
        "chromadb": "ChromaDB",
        "sentence_transformers": "Sentence Transformers",
        "dotenv": "python-dotenv",
        "pandas": "Pandas",
        "numpy": "NumPy",
        "bs4": "BeautifulSoup4",
    }

    table = Table(title="Environment Check", show_header=True)
    table.add_column("Package", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Version", style="green")
    table.add_column("Status", style="bold")

    all_ok = True
    for package, description in packages.items():
        try:
            mod = __import__(package)
            version = getattr(mod, "__version__", "N/A")
            table.add_row(package, description, version, "✅ OK")
        except ImportError as e:
            table.add_row(package, description, "—", f"❌ {e}")
            all_ok = False

    console.print(table)

    if all_ok:
        console.print("\n[bold green]✅ All packages installed correctly![/bold green]")
    else:
        console.print("\n[bold red]❌ Some packages are missing. Run: pixi install[/bold red]")

    return all_ok

def check_api_keys():
    """Check that API keys are configured."""
    import os
    from dotenv import load_dotenv
    load_dotenv()

    console.print("\n[bold]API Key Check:[/bold]")
    keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    }
    for name, value in keys.items():
        if value and not value.startswith("sk-your"):
            console.print(f"  {name}: [green]✅ Configured[/green]")
        else:
            console.print(f"  {name}: [yellow]⚠️ Not set (optional for local models)[/yellow]")

if __name__ == "__main__":
    console.print("[bold blue]🔍 Generative AI Environment Verification[/bold blue]\n")
    console.print(f"Python: {sys.version}\n")
    check_imports()
    check_api_keys()