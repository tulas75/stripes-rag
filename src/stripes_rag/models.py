"""LLM model registry for the chat interface."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelEntry:
    name: str
    model_id: str
    api_base: str | None = None
    api_key: str | None = None


# Default models — can be extended via config in the future
_MODELS: list[ModelEntry] = [
    ModelEntry("Qwen3.5 4B (Ollama)", "ollama_chat/qwen3.5:4b-q8_0"),
    ModelEntry("Qwen3.5 9B (Ollama)", "ollama_chat/qwen3.5:9b-q8_0"),
    ModelEntry("Qwen3.5 0.8B (Ollama)", "ollama_chat/qwen3.5:0.8b"),
    ModelEntry("Qwen3.5 2B (Ollama)", "ollama_chat/qwen3.5:2b"),
    ModelEntry("DeepSeek Chat", "deepseek/deepseek-chat"),
    ModelEntry("Mistral Small 3.2", "deepinfra/mistralai/Mistral-Small-3.2-24B-Instruct-2506"),
    ModelEntry("Mistral Small", "deepinfra/mistralai/Mistral-Small-24B-Instruct-2501"),
    ModelEntry("Qwen3 Next 80B", "deepinfra/Qwen/Qwen3-Next-80B-A3B-Instruct"),
    ModelEntry("Qwen3 30B", "deepinfra/Qwen/Qwen3-30B-A3B"),
    ModelEntry("MiniMax M2.1", "anthropic/MiniMax-M2.1"),
    ModelEntry("Qwen3.5 35B (Ollama)", "ollama_chat/qwen3.5:35b"),
    ModelEntry("Qwen3 4B (Ollama)", "ollama_chat/qwen3:4b-instruct-2507-q8_0"),
    ModelEntry("Groq Llama 3.1 8B", "groq/llama-3.1-8b-instant"),
    ModelEntry("Mistral Small (API)", "mistral/mistral-small-latest"),
    ModelEntry("Qwen3.5 0.8B (llama.cpp)", "openai/Qwen3.5-0.8B-Q8_0", "http://192.168.1.18:8082/v1", "dummy"),
]


def list_models() -> list[dict]:
    return [
        {
            "name": m.name,
            "model_id": m.model_id,
            "api_base": m.api_base,
            "api_key": m.api_key,
        }
        for m in _MODELS
    ]


def get_model(name: str) -> ModelEntry | None:
    for m in _MODELS:
        if m.name == name:
            return m
    return None
