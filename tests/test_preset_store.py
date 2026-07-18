from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from preset_store import PresetStore, PresetStoreError, normalize_preset_name


class PresetStoreTests(unittest.TestCase):
    def test_named_presets_persist_and_are_returned_as_copies(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "presets.json"
            first = PresetStore(path)
            first.save("  Sample   A  ", {"opacity": 0.4, "labels": [3, 4]})

            second = PresetStore(path)
            self.assertEqual(second.names(), ["Sample A"])
            loaded = second.load("Sample A")
            self.assertEqual(loaded, {"opacity": 0.4, "labels": [3, 4]})
            loaded["labels"].append(5)
            self.assertEqual(second.load("Sample A")["labels"], [3, 4])

    def test_save_overwrites_only_the_selected_name(self) -> None:
        with TemporaryDirectory() as directory:
            store = PresetStore(Path(directory) / "presets.json")
            store.save("B", {"value": 2})
            store.save("A", {"value": 1})
            store.save("A", {"value": 3})

            self.assertEqual(store.names(), ["A", "B"])
            self.assertEqual(store.load("A"), {"value": 3})
            self.assertEqual(store.load("B"), {"value": 2})

    def test_delete_persists(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "presets.json"
            store = PresetStore(path)
            store.save("temporary", {"value": True})
            self.assertEqual(store.delete("temporary"), "temporary")
            self.assertEqual(PresetStore(path).names(), [])
            with self.assertRaises(PresetStoreError):
                store.load("temporary")

    def test_invalid_name_and_malformed_document_are_rejected(self) -> None:
        with self.assertRaises(PresetStoreError):
            normalize_preset_name("   ")
        with self.assertRaises(PresetStoreError):
            normalize_preset_name("x" * 81)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "presets.json"
            path.write_text(json.dumps({"presets": []}), encoding="utf-8")
            with self.assertRaises(PresetStoreError):
                PresetStore(path).names()


if __name__ == "__main__":
    unittest.main()
