# src/chatbot.py
"""Interactive multi-provider CLI chatbot with session management."""

import json
import os
from datetime import datetime
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown

# Assumes MultiProviderChat is importable from the module above
# from multi_provider_chat import MultiProviderChat, ChatResponse

console = Console()

class InteractiveChatbot:
    """Full-featured interactive chatbot with session management."""

    def __init__(self):
        self.chat = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.session_start = datetime.now()
        self.conversation_log = []

    def initialize(self):
        """Set up the chatbot with user preferences."""
        console.print(Panel(
            "[bold blue]🤖 AI Chatbot — Multi-Provider Edition[/bold blue]\n"
            "Choose your AI provider and start chatting!",
            border_style="blue"
        ))

        provider = Prompt.ask(
            "Select provider",
            choices=["openai", "anthropic"],
            default="openai"
        )

        model_defaults = {"openai": "gpt-4o-mini", "anthropic": "claude-sonnet-4-20250514"}
        model = Prompt.ask("Model", default=model_defaults[provider])

        self.chat = MultiProviderChat(provider=provider, model=model)
        self.chat.set_system_prompt(
            "You are a helpful, knowledgeable AI assistant. "
            "Format your responses in markdown when appropriate."
        )
        console.print(f"\n[green]✅ Connected to {provider}/{model}[/green]")
        console.print("[dim]Commands: /save, /clear, /stats, /quit[/dim]\n")

    def handle_command(self, command: str) -> bool:
        """Handle slash commands. Returns True if should continue, False to quit."""
        if command == "/quit":
            return False
        elif command == "/clear":
            self.chat.history = [m for m in self.chat.history if m["role"] == "system"]
            console.print("[yellow]Conversation cleared.[/yellow]")
        elif command == "/stats":
            elapsed = (datetime.now() - self.session_start).seconds
            console.print(Panel(
                f"Total input tokens: {self.total_input_tokens}\n"
                f"Total output tokens: {self.total_output_tokens}\n"
                f"Messages exchanged: {len(self.conversation_log)}\n"
                f"Session duration: {elapsed // 60}m {elapsed % 60}s",
                title="Session Statistics", border_style="cyan"
            ))
        elif command == "/save":
            filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w") as f:
                json.dump(self.conversation_log, f, indent=2)
            console.print(f"[green]Conversation saved to {filename}[/green]")
        else:
            console.print(f"[red]Unknown command: {command}[/red]")
        return True

    def run(self):
        """Main chat loop."""
        self.initialize()

        while True:
            try:
                user_input = Prompt.ask("[bold green]You[/bold green]")
            except (KeyboardInterrupt, EOFError):
                break

            if not user_input.strip():
                continue

            if user_input.startswith("/"):
                if not self.handle_command(user_input.strip()):
                    break
                continue

            try:
                response = self.chat.send(user_input)
                self.total_input_tokens += response.input_tokens
                self.total_output_tokens += response.output_tokens
                self.conversation_log.append({
                    "role": "user", "content": user_input,
                    "timestamp": datetime.now().isoformat()
                })
                self.conversation_log.append({
                    "role": "assistant", "content": response.content,
                    "model": response.model, "tokens": response.output_tokens,
                    "timestamp": datetime.now().isoformat()
                })

                console.print(f"\n[bold blue]🤖 {response.provider}[/bold blue]")
                console.print(Markdown(response.content))
                console.print(f"[dim]({response.input_tokens}+{response.output_tokens} tokens)[/dim]\n")

            except Exception as e:
                console.print(f"[bold red]Error: {e}[/bold red]")

        console.print("\n[blue]Thanks for chatting! Goodbye. 👋[/blue]")

if __name__ == "__main__":
    bot = InteractiveChatbot()
    bot.run()