import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manager import adapt_chat_end, download, precision_options, validate_source
from model_config import ModelError, metadata, validate_weights


def fixture(directory: Path):
    config = {"model_type": "qwen3", "hidden_size": 448, "num_hidden_layers": 11,
              "vocab_size": 16384, "max_position_embeddings": 32768}
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "tokenizer_config.json").write_text(json.dumps({"chat_template": "template"}))
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)


class ConfigTests(unittest.TestCase):
    def test_native_context_and_shards(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            fixture(directory)
            self.assertEqual(metadata(directory)["context_length"], 32768)
            self.assertTrue(metadata(directory)["chat_template"])
            (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "missing.safetensors"}}))
            with self.assertRaises(ModelError): validate_weights(directory)

    def test_truncated_weights_and_paths(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            fixture(directory)
            path = directory / "model.safetensors"
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ModelError): validate_weights(directory)

    def test_source_and_precision(self):
        source = {"source": "huggingface", "repository": "owner/model"}
        for changes in ({"repository": "../model"}, {"endpoint": "https://secret:password@example.org"},
                        {"endpoint": "http://example.org"}, {"token_env": "not-a-variable"},
                        {"token_env": "1TOKEN"}, {"repository": ".owner/model"}):
            with self.assertRaises(ModelError): validate_source({**source, **changes})
        self.assertTrue(precision_options({"precision": "q4"}, {"quantization": None})["quantize"])
        with self.assertRaises(ModelError): precision_options({"precision": "q8"}, {"quantization": {"bits": 4}})

    def test_chatml_boundary_preserves_eos_zero_and_context(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            fixture(directory)
            generation = directory / "generation_config.json"
            generation.write_text(json.dumps({"eos_token_id": 0}))
            with patch("transformers.AutoTokenizer.from_pretrained") as load:
                tokenizer = load.return_value
                tokenizer.chat_template = "{{message}}<|im_end|>"
                tokenizer.get_vocab.return_value = {"<|im_end|>": 2}
                tokenizer.encode.return_value = [2]
                adapt_chat_end(directory)
                adapt_chat_end(directory)
                self.assertFalse(load.call_args.kwargs["trust_remote_code"])
            self.assertEqual(json.loads(generation.read_text())["eos_token_id"], [0, 2])
            self.assertEqual(metadata(directory)["context_length"], 32768)

    def test_download_dispatches_endpoint_revision_and_patterns(self):
        request = {"source": "huggingface", "repository": "owner/model", "revision": "commit",
                   "endpoint": "https://mirror.example"}
        with patch("huggingface_hub.snapshot_download", return_value="fixture") as mocked:
            download(request, Path("fixture"), config_only=True)
            self.assertEqual(mocked.call_args.kwargs["endpoint"], request["endpoint"])
            self.assertEqual(mocked.call_args.kwargs["revision"], "commit")
            self.assertEqual(mocked.call_args.kwargs["allow_patterns"][0], "config.json")
        with patch("modelscope.hub.snapshot_download.snapshot_download", return_value="fixture") as mocked:
            download({**request, "source": "modelscope"}, Path("fixture"))
            self.assertEqual(mocked.call_args.kwargs["endpoint"], request["endpoint"])
            self.assertIn("*.safetensors", mocked.call_args.kwargs["allow_patterns"])


if __name__ == "__main__": unittest.main()
