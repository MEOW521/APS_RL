import json
from pathlib import Path
from typing import Any


class APSConfig:
    def __init__(self, config_path: str):
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with path.open("r", encoding="utf-8") as f:
            self._config: dict[str, Any] = json.load(f)
        if not isinstance(self._config, dict):
            raise ValueError("配置文件根节点必须是 JSON 对象")

    def get(self, key: str, default: Any = None) -> Any:
        current: Any = self._config
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def require(self, key: str) -> Any:
        sentinel = object()
        value = self.get(key, sentinel)
        if value is sentinel:
            raise KeyError(f"缺少必填配置项: {key}")
        return value

    def section(self, name: str) -> dict[str, Any]:
        value = self.require(name)
        if not isinstance(value, dict):
            raise ValueError(f"配置项 {name} 必须是对象")
        return value

    def as_dict(self) -> dict[str, Any]:
        return dict(self._config)