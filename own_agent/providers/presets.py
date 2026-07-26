"""Provider presets for common LLM vendors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from own_agent.providers.types import ProviderCapabilities


@dataclass(frozen=True, slots=True)
class ProviderPreset:
    name: str
    kind: str
    api_key_env: str
    default_model: str
    base_url_env: str | None = None
    default_base_url: str | None = None
    capabilities: ProviderCapabilities = ProviderCapabilities(supports_streaming=True)
    extra_headers: dict[str, str] | None = None
    extra_body: dict[str, Any] | None = None


def get_preset(name: str) -> ProviderPreset | None:
    return PROVIDER_PRESETS.get(name.lower())

def list_presets() -> list[ProviderPreset]:
    return list(PROVIDER_PRESETS.values())


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "openai": ProviderPreset(
        name="openai", kind="openai-compatible",
        api_key_env="OPENAI_API_KEY", default_model="gpt-4.1-mini",
    ),
    "deepseek": ProviderPreset(
        name="deepseek", kind="openai-compatible",
        api_key_env="DEEPSEEK_API_KEY", default_model="deepseek-chat",
        base_url_env="DEEPSEEK_BASE_URL", default_base_url="https://api.deepseek.com",
    ),
    "qwen": ProviderPreset(
        name="qwen", kind="openai-compatible",
        api_key_env="DASHSCOPE_API_KEY", default_model="qwen-plus",
        base_url_env="DASHSCOPE_BASE_URL",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    "ollama": ProviderPreset(
        name="ollama", kind="openai-compatible",
        api_key_env="OLLAMA_API_KEY", default_model="qwen2.5-coder:7b",
        base_url_env="OLLAMA_BASE_URL", default_base_url="http://localhost:11434/v1",
    ),
    "anthropic": ProviderPreset(
        name="anthropic", kind="anthropic",
        api_key_env="ANTHROPIC_API_KEY", default_model="claude-sonnet-4-5",
        base_url_env="ANTHROPIC_BASE_URL",
        capabilities=ProviderCapabilities(
            supports_streaming=True, supports_forced_tool_choice=True,
            supports_parallel_tool_calls=True,
        ),
    ),
}
