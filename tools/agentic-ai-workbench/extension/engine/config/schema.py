"""
Configuration: loads settings.yaml from this directory, then applies
AICOMPILER_<SECTION>_<KEY>=value environment variable overrides.
Falls back silently to hardcoded defaults if PyYAML is absent.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

from core.exceptions import ConfigError

_DEFAULTS: Dict[str, Any] = {
    "cpp_generator": {
        "max_retry_attempts": 3,
        "compilation_timeout_sec": 60,
        "llm_model": "claude-sonnet-4-6",
        "llm_max_tokens": 4096,
        "include_kb_in_prompt": True,
        "kb_context_limit_chars": 50_000,
    },
    "graph_focus": {
        "llm_model": "claude-sonnet-4-6",
        "llm_max_tokens": 512,
    },
    "llm": {
        "timeout_sec": 60,
        "max_retries": 3,
        "retry_base_delay": 2.0,
    },
    "orchestrator": {
        "agent_timeout_sec": 300,
    },
}


class Config:
    """Immutable-ish config container."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._data: Dict[str, Any] = copy.deepcopy(_DEFAULTS)
        if path and path.exists():
            self._load_yaml(path)
        self._apply_env()

    # ------------------------------------------------------------------ load
    def _load_yaml(self, path: Path) -> None:
        try:
            import yaml          # optional
        except ImportError:
            return               # silently use defaults when pyyaml absent
        try:
            with open(path, encoding="utf-8") as fh:
                overrides = yaml.safe_load(fh) or {}
            _deep_merge(self._data, overrides)
        except Exception as exc:
            raise ConfigError(f"Cannot load {path}: {exc}") from exc

    def _apply_env(self) -> None:
        prefix = "AICOMPILER_"
        for raw_key, raw_val in os.environ.items():
            if not raw_key.startswith(prefix):
                continue
            parts = raw_key[len(prefix):].lower().split("_", 1)
            if len(parts) != 2:
                continue
            section, key = parts
            if section not in self._data or key not in self._data[section]:
                continue
            default = self._data[section][key]
            try:
                if isinstance(default, bool):
                    self._data[section][key] = raw_val.lower() in ("1", "true", "yes")
                elif isinstance(default, int):
                    self._data[section][key] = int(raw_val)
                elif isinstance(default, float):
                    self._data[section][key] = float(raw_val)
                else:
                    self._data[section][key] = raw_val
            except (ValueError, TypeError):
                pass   # keep default on bad env value

    # ------------------------------------------------------------------ access
    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self._data.get(section, {}).get(key, default)

    def section(self, name: str) -> Dict[str, Any]:
        return dict(self._data.get(name, {}))

    def __repr__(self) -> str:
        return f"Config({list(self._data)})"


def _deep_merge(target: dict, source: dict) -> None:
    for k, v in source.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _deep_merge(target[k], v)
        else:
            target[k] = v


# --------------------------------------------------------------- module singleton
_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config(Path(__file__).parent / "settings.yaml")
    return _config
