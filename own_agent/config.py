"""Configuration loading for own-agent."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib

from dotenv import load_dotenv


PROJECT_CONFIG_NAME = "own-agent.toml"


@dataclass(frozen=True, slots=True)
class AppConfig:
    provider_name: str
    env: dict[str, str]
    project_config: dict[str, Any] | None = None
    global_config: dict[str, Any] | None = None

    def get_env(self, name: str, default: str | None = None) -> str | None:
        return self.env.get(name, default)

    def get_provider_value(self, name: str, *, env: str | None = None, default: str | None = None) -> str | None:
        if env:
            env_value = self.get_env(env)
            if env_value:
                return env_value
        for config in (self.project_config, self.global_config):
            if not config:
                continue
            provider = config.get("provider")
            if isinstance(provider, dict):
                direct = provider.get(name)
                if direct is not None:
                    return str(direct)
                nested = provider.get(self.provider_name)
                if isinstance(nested, dict):
                    value = nested.get(name)
                    if value is not None:
                        return str(value)
        return default

    def get_provider_bool(self, name: str, *, env: str | None = None, default: bool = False) -> bool:
        if env:
            env_value = self.get_env(env)
            if env_value:
                return env_value.lower() in ("1", "true", "yes", "on")
        for config in (self.project_config, self.global_config):
            if not config:
                continue
            provider = config.get("provider")
            if isinstance(provider, dict):
                value = provider.get(name)
                if isinstance(value, bool):
                    return value
                nested = provider.get(self.provider_name)
                if isinstance(nested, dict):
                    value = nested.get(name)
                    if isinstance(value, bool):
                        return value
        return default

    def get_config_value(self, name: str, *, default: str | None = None) -> str | None:
        for config in (self.project_config, self.global_config):
            if config:
                value = config.get(name)
                if isinstance(value, str) and value:
                    return value
        return default


def load_config(
    provider_name: str | None = None,
    *,
    project_root: Path | str | None = None,
    env: dict[str, str] | None = None,
) -> AppConfig:
    load_dotenv()
    env_snapshot = dict(os.environ if env is None else env)
    root = Path(project_root or os.getcwd()).resolve()
    global_path = _default_global_config_path()
    project_path = root / PROJECT_CONFIG_NAME
    global_config = _read_toml(global_path)
    project_config = _read_toml(project_path)

    selected = (
        provider_name
        or env_snapshot.get("OWN_AGENT_PROVIDER")
        or _provider_from_config(project_config)
        or _provider_from_config(global_config)
        or "openai"
    ).lower()

    return AppConfig(
        provider_name=selected,
        env=env_snapshot,
        project_config=project_config,
        global_config=global_config,
    )


def default_global_config_path() -> Path:
    return _default_global_config_path()


def _default_global_config_path() -> Path:
    if sys.platform == "win32":
        appdata = os.getenv("APPDATA")
        if appdata:
            return Path(appdata) / "own-agent" / "config.toml"
    config_home = os.getenv("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home) / "own-agent" / "config.toml"
    return Path.home() / ".config" / "own-agent" / "config.toml"


def render_default_config() -> str:
    return """# own-agent global configuration.
model = "openai/gpt-4.1-mini"

[provider]
type = "openai"
api_key_env = "OPENAI_API_KEY"

[permissions]
mode = "standard"
"""


def _read_toml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        return tomllib.load(f)


def _provider_from_config(config: dict[str, Any] | None) -> str | None:
    if not config:
        return None
    provider = config.get("provider")
    if isinstance(provider, dict):
        ptype = provider.get("type")
        if isinstance(ptype, str) and ptype:
            return ptype
    model = config.get("model")
    if isinstance(model, str) and "/" in model:
        return model.split("/", 1)[0]
    return None
