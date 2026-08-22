from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from taterbench.discovery import classify_gguf, discover_models, variants_for_model


class DiscoveryTests(unittest.TestCase):
    def test_classifies_main_projector_and_drafts(self) -> None:
        self.assertEqual(classify_gguf(Path("Qwen3.8-27B-Q4_K_M.gguf")), "main")
        self.assertEqual(classify_gguf(Path("mmproj-Qwen3.8-Q8_0.gguf")), "projector")
        self.assertEqual(classify_gguf(Path("mtp-Qwen3.8-Q8_0.gguf")), "mtp")
        self.assertEqual(classify_gguf(Path("Gemma-DFlash-Q8_0.gguf")), "dflash")

    def test_registry_groups_sidecars_with_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            snapshot = home / "agent_lab/models/llm/llama-cpp/models--org--model/snapshots/abc"
            snapshot.mkdir(parents=True)
            main = snapshot / "Model-Q4_K_M.gguf"
            mtp = snapshot / "Model-MTP-Q8_0.gguf"
            dflash = snapshot / "Model-DFlash-Q8_0.gguf"
            mmproj = snapshot / "mmproj-F16.gguf"
            for path in (main, mtp, dflash, mmproj):
                path.write_bytes(b"test")
            registry = home / "agent_lab/models/llm/downloaded_models.json"
            registry.parent.mkdir(parents=True, exist_ok=True)
            registry.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "provider": "llama_cpp",
                                "model": "org/model::Model-Q4_K_M.gguf",
                                "model_path": str(main),
                                "filename": main.name,
                                "repo_id": "org/model",
                                "supports_vision": True,
                                "supports_video": True,
                                "supports_audio": True,
                                "mmproj_path": str(mmproj),
                            },
                            {"provider": "llama_cpp", "model_path": str(mtp), "filename": mtp.name},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            models = discover_models(home)
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0].repo_id, "org/model")
            self.assertTrue(models[0].supports_vision)
            self.assertTrue(models[0].supports_video)
            self.assertTrue(models[0].supports_audio)
            self.assertEqual([variant.name for variant in variants_for_model(models[0])], ["baseline", "mtp", "dflash"])
            self.assertEqual(models[0].mmproj_path, mmproj.absolute())

    def test_snapshot_symlink_keeps_draft_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            blob = root / "blobs/deadbeef"
            blob.parent.mkdir()
            blob.write_bytes(b"draft")
            snapshot = root / "snapshots/abc"
            snapshot.mkdir(parents=True)
            draft = snapshot / "Model-MTP-Q8_0.gguf"
            draft.symlink_to(blob)
            self.assertEqual(classify_gguf(draft.absolute()), "mtp")

    def test_registry_directory_is_not_used_as_projector(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            snapshot = home / "agent_lab/models/llm/llama-cpp/models--org--model/snapshots/abc"
            snapshot.mkdir(parents=True)
            main = snapshot / "Model-Q4_K_M.gguf"
            main.write_bytes(b"test")
            registry = home / "agent_lab/models/llm/downloaded_models.json"
            registry.parent.mkdir(parents=True, exist_ok=True)
            registry.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "provider": "llama_cpp",
                                "model_path": str(main),
                                "filename": main.name,
                                "repo_id": "org/model",
                                "supports_vision": True,
                                "mmproj_path": str(home),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            models = discover_models(home)

            self.assertEqual(len(models), 1)
            self.assertIsNone(models[0].mmproj_path)

    def test_repo_marked_mtp_gets_embedded_variant(self) -> None:
        from taterbench.types import ModelCandidate

        model = ModelCandidate(
            id="test",
            label="org/Qwen-MTP-GGUF::Qwen-Q4_K_M.gguf",
            provider="llama_cpp",
            model_path=Path("Qwen-Q4_K_M.gguf"),
            repo_id="org/Qwen-MTP-GGUF",
            filename="Qwen-Q4_K_M.gguf",
        )
        variants = variants_for_model(model)
        self.assertEqual([variant.name for variant in variants], ["baseline", "mtp"])
        self.assertIsNone(variants[1].draft_path)


if __name__ == "__main__":
    unittest.main()
