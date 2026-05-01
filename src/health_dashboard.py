# src/health_dashboard.py
"""Complete environment health dashboard."""

import json
import os
import sys
import time
from datetime import datetime

def get_system_ram_gb():
    """Return total system RAM in GB."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        return None


def check_gpu():
    """Check GPU availability and VRAM."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            return {
                "available": True,
                "name": gpu_name,
                "vram_gb": round(vram_gb, 1),
                "cuda_version": torch.version.cuda,
            }
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            ram_gb = get_system_ram_gb()
            return {
                "available": True,
                "name": "Apple Silicon (MPS)",
                "vram_gb": "shared",
                "unified_memory_gb": ram_gb,
            }
        return {"available": False, "name": "CPU only", "vram_gb": 0}
    except Exception as e:
        return {"available": False, "error": str(e)}

def check_api_connectivity():
    """Test connectivity to AI API providers."""
    import requests
    results = {}
    endpoints = {
        "OpenAI": "https://api.openai.com/v1/models",
        "Anthropic": "https://api.anthropic.com/v1/messages",
        "HuggingFace": "https://huggingface.co/api/models?limit=1",
    }
    for name, url in endpoints.items():
        try:
            start = time.time()
            resp = requests.get(url, timeout=5)
            latency_ms = round((time.time() - start) * 1000)
            results[name] = {
                "reachable": True,
                "status_code": resp.status_code,
                "latency_ms": latency_ms,
            }
        except Exception as e:
            results[name] = {"reachable": False, "error": str(e)}
    return results

def generate_report():
    """Generate a comprehensive system report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": sys.platform,
        "gpu": check_gpu(),
        "api_connectivity": check_api_connectivity(),
    }

    # Check recommended model sizes based on available resources
    gpu_info = report["gpu"]
    if gpu_info.get("name") == "Apple Silicon (MPS)":
        ram = gpu_info.get("unified_memory_gb") or 0
        if ram >= 64:
            report["recommended_local_models"] = [
                "Llama-3-70B (4-bit via Ollama/MLX)",
                "Mixtral-8x7B (via Ollama)",
                "Llama-3-8B (full precision)",
            ]
        elif ram >= 32:
            report["recommended_local_models"] = [
                "Llama-3-70B (4-bit via Ollama/MLX) — may be slow",
                "Llama-3-8B (full precision)",
                "Mistral-7B (full precision)",
                "Phi-3-medium",
            ]
        elif ram >= 16:
            report["recommended_local_models"] = [
                "Llama-3-8B (via Ollama/MLX)",
                "Mistral-7B (via Ollama/MLX)",
                "Gemma-7B",
                "Phi-3-mini",
            ]
        elif ram >= 8:
            report["recommended_local_models"] = [
                "Phi-3-mini (via Ollama/MLX)",
                "Gemma-2B",
                "TinyLlama-1.1B",
                "Llama-3-8B (4-bit, tight fit)",
            ]
        else:
            report["recommended_local_models"] = [
                "Phi-3-mini (4-bit)",
                "TinyLlama-1.1B",
                "Consider using cloud APIs for larger models",
            ]
        report["local_model_runtimes"] = ["Ollama", "MLX (Apple-optimized)", "llama.cpp (Metal)", "HuggingFace Transformers (device='mps')"]
    elif gpu_info.get("vram_gb", 0) and isinstance(gpu_info["vram_gb"], (int, float)):
        vram = gpu_info["vram_gb"]
        if vram >= 24:
            report["recommended_local_models"] = ["Llama-3-70B (4-bit)", "Mistral-Large"]
        elif vram >= 8:
            report["recommended_local_models"] = ["Llama-3-8B", "Mistral-7B", "Phi-3-mini"]
        elif vram >= 4:
            report["recommended_local_models"] = ["Phi-3-mini (4-bit)", "TinyLlama-1.1B"]
        else:
            report["recommended_local_models"] = ["CPU models only — use APIs instead"]
    else:
        report["recommended_local_models"] = ["CPU models only — use APIs instead"]

    # Save report
    with open("system_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report

if __name__ == "__main__":
    from rich.console import Console
    from rich.json import JSON
    console = Console()
    console.print("[bold]🏥 AI Development Environment Health Report[/bold]\n")
    report = generate_report()
    console.print(JSON(json.dumps(report, indent=2)))
    console.print(f"\n[green]Report saved to system_report.json[/green]")
