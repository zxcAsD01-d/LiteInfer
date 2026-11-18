"""Model metadata and safetensors validation without importing GPU libraries."""
import json
import math
from pathlib import Path
import struct


class ModelError(ValueError):
    """An actionable, public-safe error code."""


def read_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ModelError("missing_or_oversized_config")
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ModelError("config_must_be_object")
    return result


def metadata(directory: Path, weights: bool = True) -> dict:
    config = read_json(directory / "config.json")
    text = config.get("text_config", config)
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise ModelError("missing_model_type")
    for key in ("hidden_size", "num_hidden_layers", "vocab_size"):
        value = text.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ModelError("invalid_" + key)
    context = text.get("max_position_embeddings", text.get("n_positions"))
    if context is not None and (isinstance(context, bool) or not isinstance(context, int) or context <= 0):
        raise ModelError("invalid_context_length")
    tokenizer_file = directory / "tokenizer_config.json"
    tokenizer = read_json(tokenizer_file) if tokenizer_file.exists() else {}
    has_chat = bool(tokenizer.get("chat_template")) or (directory / "chat_template.jinja").is_file()
    quantization = config.get("quantization", config.get("quantization_config"))
    return {"model_type": model_type, "context_length": context,
            "vocab_size": text["vocab_size"], "layers": text["num_hidden_layers"],
            "hidden_size": text["hidden_size"], "quantization": quantization,
            "vision": bool(config.get("vision_config") or config.get("optiq_vision")),
            "chat_template": has_chat, "remote_code_declared": bool(config.get("auto_map")),
            "weight_bytes": validate_weights(directory) if weights else None,
            "recommended_output_tokens": min(1024, max(1, context - 1)) if context else 1024}


def safe_child(directory: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ModelError("invalid_shard_path")
    path = directory / name
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ModelError("shard_outside_model_directory")
    return path


def tensor_header(path: Path) -> dict:
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ModelError("truncated_safetensors")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= min(16 * 1024 * 1024, size - 8):
            raise ModelError("invalid_safetensors_header")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ModelError("invalid_safetensors_header")
    intervals = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(tensor, dict):
            raise ModelError("invalid_tensor_metadata")
        offsets, shape = tensor.get("data_offsets"), tensor.get("shape")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(v) is int for v in offsets):
            raise ModelError("invalid_tensor_offsets")
        if not isinstance(shape, list) or not all(type(v) is int and v >= 0 for v in shape):
            raise ModelError("invalid_tensor_shape")
        start, end = offsets
        if not 0 <= start <= end <= size - 8 - length:
            raise ModelError("truncated_tensor_data")
        widths = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "U64": 8,
                  "I32": 4, "U32": 4, "I16": 2, "U16": 2, "I8": 1, "U8": 1, "BOOL": 1}
        width = widths.get(tensor.get("dtype"))
        if width is not None and end - start != math.prod(shape) * width:
            raise ModelError("tensor_size_mismatch")
        intervals.append((start, end))
    ordered = sorted(intervals)
    if not ordered or any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ModelError("empty_or_overlapping_tensor_data")
    return header


def validate_weights(directory: Path) -> int:
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise ModelError("safetensors_weights_required")
    headers = {file.name: tensor_header(safe_child(directory, file.name)) for file in files}
    for index_file in directory.glob("*.safetensors.index.json"):
        weight_map = read_json(index_file).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ModelError("invalid_weight_index")
        for tensor, shard in weight_map.items():
            safe_child(directory, shard)
            if shard not in headers or tensor not in headers[shard]:
                raise ModelError("missing_weight_shard_or_tensor")
    return sum(file.stat().st_size for file in files)
