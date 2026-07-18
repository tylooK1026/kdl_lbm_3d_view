"""Persistent named preset storage for the TIFF phase viewer."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STORE_FORMAT = "tiff-phase-viewer-presets"
STORE_SCHEMA_VERSION = 1


class PresetStoreError(ValueError):
    """Raised when a preset store is invalid or cannot be updated safely."""


def normalize_preset_name(name: str) -> str:
    """Return a safe display name while preserving normal Unicode text."""

    normalized = " ".join(str(name).strip().split())
    if not normalized:
        raise PresetStoreError("预设名称不能为空。")
    if len(normalized) > 80:
        raise PresetStoreError("预设名称不能超过 80 个字符。")
    if any(ord(character) < 32 for character in normalized):
        raise PresetStoreError("预设名称不能包含控制字符。")
    return normalized


class PresetStore:
    """Read and atomically update a JSON file containing named presets."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @staticmethod
    def _empty_document() -> dict[str, Any]:
        return {
            "format": STORE_FORMAT,
            "schema_version": STORE_SCHEMA_VERSION,
            "presets": {},
        }

    def _read_document(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty_document()
        try:
            with self.path.open("r", encoding="utf-8") as stream:
                document = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise PresetStoreError(f"无法读取设置预设：{exc}") from exc

        if not isinstance(document, dict):
            raise PresetStoreError("设置预设文件的顶层结构无效。")
        if document.get("format") != STORE_FORMAT:
            raise PresetStoreError("设置预设文件格式无法识别。")
        if document.get("schema_version") != STORE_SCHEMA_VERSION:
            raise PresetStoreError("设置预设文件版本不受当前程序支持。")
        presets = document.get("presets")
        if not isinstance(presets, dict):
            raise PresetStoreError("设置预设列表无效。")
        for name, entry in presets.items():
            if not isinstance(name, str) or not isinstance(entry, dict):
                raise PresetStoreError("设置预设中包含无效条目。")
            if not isinstance(entry.get("settings"), dict):
                raise PresetStoreError(f"预设“{name}”缺少有效设置。")
        return document

    def _write_document(self, document: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(document, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except (OSError, TypeError, ValueError) as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise PresetStoreError(f"无法保存设置预设：{exc}") from exc

    def names(self) -> list[str]:
        document = self._read_document()
        return sorted(document["presets"], key=str.casefold)

    def contains(self, name: str) -> bool:
        normalized = normalize_preset_name(name)
        return normalized in self._read_document()["presets"]

    def load(self, name: str) -> dict[str, Any]:
        normalized = normalize_preset_name(name)
        entry = self._read_document()["presets"].get(normalized)
        if entry is None:
            raise PresetStoreError(f"找不到设置预设“{normalized}”。")
        return deepcopy(entry["settings"])

    def save(self, name: str, settings: dict[str, Any]) -> str:
        normalized = normalize_preset_name(name)
        if not isinstance(settings, dict):
            raise PresetStoreError("要保存的设置必须是字典结构。")
        document = self._read_document()
        document["presets"][normalized] = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "settings": deepcopy(settings),
        }
        self._write_document(document)
        return normalized

    def delete(self, name: str) -> str:
        normalized = normalize_preset_name(name)
        document = self._read_document()
        if normalized not in document["presets"]:
            raise PresetStoreError(f"找不到设置预设“{normalized}”。")
        del document["presets"][normalized]
        self._write_document(document)
        return normalized
