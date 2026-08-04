import re
from collections.abc import Collection, Mapping, Sequence

import torch

MXFP8_GROUP_SIZE = 32
TE_MXFP8_ROW_ALIGNMENT = 32

_PACKED_MODULE_GROUPS = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
    "in_proj_qkvz": ("in_proj_qkv", "in_proj_z"),
    "in_proj_ba": ("in_proj_b", "in_proj_a"),
}


def validate_mxfp8_config(quantization_config: Mapping[str, object]) -> None:
    """Validate the serialized MXFP8 checkpoint contract used for rollout updates."""
    expected = {
        "quant_method": "mxfp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [1, MXFP8_GROUP_SIZE],
        "scale_fmt": "ue8m0",
    }
    if not isinstance(quantization_config, Mapping):
        raise ValueError("MXFP8 quantization_config must be a mapping.")
    for key, value in expected.items():
        if quantization_config.get(key) != value:
            raise ValueError(f"MXFP8 quantization_config requires {key}={value!r}.")

    exclusions = quantization_config.get("modules_to_not_convert", ())
    if not isinstance(exclusions, Sequence) or isinstance(exclusions, (str, bytes)):
        raise ValueError("MXFP8 modules_to_not_convert must be a list of module names.")
    if any(not isinstance(module, str) or not module.strip() for module in exclusions):
        raise ValueError("MXFP8 modules_to_not_convert entries must be non-empty strings.")


def validate_mxfp8_rollout_config(
    checkpoint_quantization_config: Mapping[str, object] | None,
    requested_quantization: str | None,
) -> None:
    """Reject ambiguous BF16-to-MXFP8 startup and conflicting rollout overrides."""
    checkpoint_method = (
        checkpoint_quantization_config.get("quant_method")
        if isinstance(checkpoint_quantization_config, Mapping)
        else None
    )
    if requested_quantization == "mxfp8" and checkpoint_method != "mxfp8":
        raise ValueError(
            "--sglang-quantization=mxfp8 requires a serialized MXFP8 checkpoint; "
            "convert --hf-checkpoint first and let SGLang read its quantization_config."
        )
    if checkpoint_method != "mxfp8":
        return

    validate_mxfp8_config(checkpoint_quantization_config)
    if requested_quantization not in (None, "mxfp8"):
        raise ValueError(
            f"The serialized MXFP8 checkpoint conflicts with --sglang-quantization={requested_quantization}."
        )


def normalize_mxfp8_exclusions(modules: Sequence[str]) -> tuple[str, ...]:
    """Expand packed-module exclusions and Qwen3.5 runtime path aliases."""
    normalized = {module.rstrip(".") for module in modules if module.rstrip(".")}

    changed = True
    while changed:
        changed = False
        for module in tuple(normalized):
            parent, _, leaf = module.rpartition(".")
            if not parent:
                continue
            for packed_name, shard_names in _PACKED_MODULE_GROUPS.items():
                if leaf not in shard_names and leaf != packed_name:
                    continue
                additions = {f"{parent}.{name}" for name in (*shard_names, packed_name)}
                if not additions.issubset(normalized):
                    normalized.update(additions)
                    changed = True

    for module in tuple(normalized):
        if module.startswith("model.language_model."):
            normalized.add("model." + module.removeprefix("model.language_model."))

    def natural_key(value: str) -> list[tuple[bool, int | str]]:
        return [(token.isdigit(), int(token) if token.isdigit() else token) for token in re.findall(r"\d+|\D+", value)]

    return tuple(sorted(normalized, key=natural_key))


def is_mxfp8_weight_excluded(weight_name: str, modules: Collection[str]) -> bool:
    """Match a weight against normalized exclusions on module-path boundaries."""
    module = weight_name.removesuffix(".weight").rstrip(".")
    variants = {module}
    if module.startswith("model.language_model."):
        variants.add("model." + module.removeprefix("model.language_model."))

    excluded_modules = modules if isinstance(modules, (set, frozenset)) else frozenset(modules)
    for variant in variants:
        parts = variant.split(".")
        candidates = {
            ".".join(parts[start:end]) for start in range(len(parts)) for end in range(start + 1, len(parts) + 1)
        }
        if not candidates.isdisjoint(excluded_modules):
            return True
    return False


def mxfp8_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to rowwise MXFP8 with compact, unswizzled UE8M0 scales."""
    if weight.ndim < 2:
        raise ValueError(f"MXFP8 requires a tensor with at least two dimensions, got {weight.ndim}.")
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"MXFP8 requires FP16, BF16, or FP32 input, got {weight.dtype}.")

    weight = weight.contiguous()
    k = weight.shape[-1]
    if k % MXFP8_GROUP_SIZE != 0:
        raise ValueError(f"Last dim {k} must be divisible by {MXFP8_GROUP_SIZE} for MXFP8.")

    weight_flat = weight.view(-1, k)
    num_rows = weight_flat.shape[0]
    pad_rows = (-num_rows) % TE_MXFP8_ROW_ALIGNMENT
    if pad_rows:
        padding = torch.zeros((pad_rows, k), device=weight.device, dtype=weight.dtype)
        weight_flat = torch.cat((weight_flat, padding), dim=0)

    from transformer_engine.pytorch import MXFP8Quantizer
    from transformer_engine.pytorch.constants import TE_DType

    quantizer = MXFP8Quantizer(
        fp8_dtype=TE_DType[torch.float8_e4m3fn],
        rowwise=True,
        columnwise=False,
    )
    quantized = quantizer.quantize(weight_flat)
    metadata = quantized.get_metadata()
    qweight = metadata["rowwise_data"][:num_rows, :k].contiguous()
    qweight = qweight.view(torch.float8_e4m3fn).view_as(weight)
    scale = metadata["rowwise_scale_inv"][:num_rows, : k // MXFP8_GROUP_SIZE]
    scale = scale.contiguous().view(*weight.shape[:-1], k // MXFP8_GROUP_SIZE)

    if qweight.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"Transformer Engine returned unexpected MXFP8 weight dtype {qweight.dtype}.")
    if scale.dtype != torch.uint8:
        raise RuntimeError(f"Transformer Engine returned unexpected MXFP8 scale dtype {scale.dtype}.")
    return qweight, scale
