"""Convert an unquantized, indexed Hugging Face safetensors checkpoint to MXFP8.

The source must contain ``model.safetensors.index.json`` and BF16, FP16, or
FP32 weight shards. The output is a serialized MXFP8 checkpoint for SGLang.
"""

import argparse
import gc
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import safetensors
import safetensors.torch
import torch
from tqdm import tqdm

from slime.utils.mxfp8 import (
    MXFP8_GROUP_SIZE,
    is_mxfp8_weight_excluded,
    mxfp8_quantize,
    normalize_mxfp8_exclusions,
)

TARGET_MXFP8_BLOCK_SIZE = [1, MXFP8_GROUP_SIZE]

_DEFAULT_EXCLUDED_WEIGHT_SUBSTRINGS = (
    "layernorm",
    "embed",
    "router",
    "mlp.gate.",
    "shared_expert_gate",
    "norm",
    "lm_head",
    "eh_proj",
    "weights_proj",
    "head.",
    "wo_a",
    "ffn.gate.",
    "compressor.",
    "conv1d",
)

# These projections use torch.nn.Linear in Slime's Qwen3.5 training model and
# therefore do not participate in Transformer Engine MXFP8 autocast.
_QWEN35_GDN_BF16_MODULES = (
    ".linear_attn.in_proj_qkv",
    ".linear_attn.in_proj_z",
    ".linear_attn.in_proj_b",
    ".linear_attn.in_proj_a",
    ".linear_attn.out_proj",
)


class ConversionResult:
    def __init__(self) -> None:
        self.weight_map: dict[str, str] = {}
        self.total_size = 0

    def add_file(self, filename: str, tensors: Mapping[str, torch.Tensor]) -> None:
        for name, tensor in tensors.items():
            self.weight_map[name] = filename
            self.total_size += tensor.numel() * tensor.element_size()


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _num_hidden_layers(config: Mapping[str, object]) -> int:
    text_config = config.get("text_config", config)
    if not isinstance(text_config, Mapping) or "num_hidden_layers" not in text_config:
        raise ValueError("config.json must define num_hidden_layers, directly or under text_config.")
    return int(text_config["num_hidden_layers"])


def _validate_source_config(config: Mapping[str, object]) -> None:
    quantization_config = config.get("quantization_config")
    if quantization_config in (None, {}):
        return
    if not isinstance(quantization_config, Mapping):
        raise ValueError("Source quantization_config must be a mapping.")
    if quantization_config.get("quant_method") not in (None, "", "bf16"):
        raise ValueError("Only unquantized BF16, FP16, or FP32 source checkpoints are supported.")


def _decoder_layer_prefix(weight_name: str) -> tuple[str, int] | None:
    parts = weight_name.split(".")
    try:
        layers_index = parts.index("layers")
        layer_index = int(parts[layers_index + 1])
    except (ValueError, IndexError):
        return None
    prefix = ".".join(parts[: layers_index + 2])
    if prefix.startswith("model.layers.") or prefix.startswith("model.language_model.layers."):
        return prefix, layer_index
    return None


def _build_exclusions(
    weight_names: Sequence[str],
    num_hidden_layers: int,
    num_layers_at_start_in_bf16: int,
    num_layers_at_end_in_bf16: int,
    extra_high_precision_layers: Sequence[str],
) -> tuple[str, ...]:
    if min(num_layers_at_start_in_bf16, num_layers_at_end_in_bf16) < 0:
        raise ValueError("The number of BF16 layers cannot be negative.")
    if num_layers_at_start_in_bf16 + num_layers_at_end_in_bf16 > num_hidden_layers:
        raise ValueError("The requested BF16 head and tail exceed num_hidden_layers.")

    exclusions = set()
    tail_start = num_hidden_layers - num_layers_at_end_in_bf16
    extra_patterns = tuple(pattern for pattern in extra_high_precision_layers if pattern)

    for weight_name in weight_names:
        if not weight_name.endswith(".weight"):
            continue
        module_name = weight_name.removesuffix(".weight")
        layer = _decoder_layer_prefix(weight_name)
        if module_name.startswith("model.visual."):
            exclusions.add("model.visual")
        elif layer is None:
            exclusions.add(module_name)
        else:
            layer_prefix, layer_index = layer
            if layer_index < num_layers_at_start_in_bf16 or layer_index >= tail_start:
                exclusions.add(layer_prefix)

        if any(pattern in weight_name for pattern in _DEFAULT_EXCLUDED_WEIGHT_SUBSTRINGS):
            exclusions.add(module_name)
        if any(pattern in module_name for pattern in _QWEN35_GDN_BF16_MODULES):
            exclusions.add(module_name)
        if any(pattern in weight_name for pattern in extra_patterns):
            exclusions.add(module_name)

    return normalize_mxfp8_exclusions(tuple(exclusions))


def _process_file(
    input_path: Path,
    output_path: Path,
    filename: str,
    device: str,
    exclusions: Sequence[str],
    result: ConversionResult,
) -> None:
    output_tensors: dict[str, torch.Tensor] = {}
    with safetensors.safe_open(input_path / filename, framework="pt", device=device) as file:
        for name in file.keys():
            tensor = file.get_tensor(name)
            if not name.endswith(".weight") or is_mxfp8_weight_excluded(name, exclusions):
                output_tensors[name] = tensor
                continue
            if tensor.ndim < 2:
                raise ValueError(f"{name} has rank {tensor.ndim}; exclude it explicitly to keep it in high precision.")
            if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(f"{name} has unsupported source dtype {tensor.dtype}.")
            if tensor.shape[-1] % MXFP8_GROUP_SIZE:
                raise ValueError(
                    f"{name} has last dim {tensor.shape[-1]}, which is not divisible by {MXFP8_GROUP_SIZE}; "
                    "exclude it explicitly to keep it in high precision."
                )

            qweight, scale = mxfp8_quantize(tensor)
            output_tensors[name] = qweight
            output_tensors[name.removesuffix(".weight") + ".weight_scale_inv"] = scale

    safetensors.torch.save_file(
        output_tensors,
        output_path / filename,
        metadata={"format": "pt"},
    )
    result.add_file(filename, output_tensors)


def _copy_auxiliary_files(input_path: Path, output_path: Path) -> None:
    generated_files = {"config.json", "model.safetensors.index.json"}
    for source in input_path.iterdir():
        if source.name in generated_files or source.suffix == ".safetensors":
            continue
        destination = output_path / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        elif source.is_file():
            shutil.copy2(source, destination)


def convert_mxfp8(
    model_dir: str | Path,
    save_dir: str | Path,
    device: str,
    *,
    num_layers_at_start_in_bf16: int = 0,
    num_layers_at_end_in_bf16: int = 0,
    extra_high_precision_layers: Sequence[str] = (),
) -> None:
    input_path = Path(model_dir).expanduser().resolve()
    output_path = Path(save_dir).expanduser().resolve()
    if input_path == output_path or input_path in output_path.parents:
        raise ValueError("The output directory must be outside the source checkpoint directory.")
    if not input_path.is_dir():
        raise ValueError(f"Source checkpoint directory does not exist: {input_path}")
    if output_path.exists():
        if not output_path.is_dir():
            raise ValueError(f"Output path must be a directory: {output_path}")
        if any(output_path.iterdir()):
            raise ValueError(f"Output directory must be empty: {output_path}")

    config_path = input_path / "config.json"
    index_path = input_path / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise ValueError("The source must contain config.json and model.safetensors.index.json.")

    config = _load_json(config_path)
    _validate_source_config(config)
    index = _load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("model.safetensors.index.json must contain a non-empty weight_map.")
    if any(not isinstance(name, str) or not isinstance(filename, str) for name, filename in weight_map.items()):
        raise ValueError("The checkpoint weight_map must map tensor names to shard filenames.")

    shard_filenames = sorted(set(weight_map.values()))
    missing_shards = [filename for filename in shard_filenames if not (input_path / filename).is_file()]
    if missing_shards:
        raise ValueError(f"Missing checkpoint shards: {missing_shards}")

    exclusions = _build_exclusions(
        tuple(weight_map),
        _num_hidden_layers(config),
        num_layers_at_start_in_bf16,
        num_layers_at_end_in_bf16,
        extra_high_precision_layers,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    _copy_auxiliary_files(input_path, output_path)
    result = ConversionResult()
    exclusion_set = frozenset(exclusions)
    for filename in tqdm(shard_filenames, desc="Converting checkpoint shards"):
        _process_file(input_path, output_path, filename, device, exclusion_set, result)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_config = dict(config)
    output_config["quantization_config"] = {
        "quant_method": "mxfp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": TARGET_MXFP8_BLOCK_SIZE,
        "scale_fmt": "ue8m0",
        "modules_to_not_convert": list(exclusions),
    }
    with (output_path / "config.json").open("w", encoding="utf-8") as file:
        json.dump(output_config, file, indent=2)
        file.write("\n")

    output_index = {
        "metadata": {"total_size": result.total_size},
        "weight_map": result.weight_map,
    }
    with (output_path / "model.safetensors.index.json").open("w", encoding="utf-8") as file:
        json.dump(output_index, file, indent=2)
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Source Hugging Face checkpoint directory.")
    parser.add_argument("--save-dir", required=True, help="Empty output directory for the MXFP8 checkpoint.")
    parser.add_argument("--device", default="cuda", help="CUDA device used for conversion (default: cuda:0).")
    parser.add_argument("--num-layers-at-start-in-bf16", type=int, default=0)
    parser.add_argument("--num-layers-at-end-in-bf16", type=int, default=0)
    parser.add_argument(
        "--extra-high-precision-layers",
        nargs="*",
        default=(),
        help="Additional weight-name substrings to keep in their source precision.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("MXFP8 checkpoint conversion requires a CUDA device.")
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    if device.type != "cuda":
        raise ValueError("MXFP8 checkpoint conversion requires a CUDA device.")
    if device.index is None:
        device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    convert_mxfp8(
        args.model_dir,
        args.save_dir,
        str(device),
        num_layers_at_start_in_bf16=args.num_layers_at_start_in_bf16,
        num_layers_at_end_in_bf16=args.num_layers_at_end_in_bf16,
        extra_high_precision_layers=args.extra_high_precision_layers,
    )


if __name__ == "__main__":
    main()
