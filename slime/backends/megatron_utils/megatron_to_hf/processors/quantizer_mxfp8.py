import re
from functools import lru_cache

from slime.utils.mxfp8 import (
    is_mxfp8_weight_excluded,
    mxfp8_quantize,
    normalize_mxfp8_exclusions,
    validate_mxfp8_config,
)

# Qwen3.5 Gated DeltaNet projections are intentionally absent: they are plain
# torch.nn.Linear modules in the training model and remain BF16 during updates.
_MXFP8_LINEAR_WEIGHTS = {
    "self_attention.linear_proj.weight",
    "self_attention.linear_qkv.weight",
    "mlp.linear_fc1.weight",
    "mlp.linear_fc2.weight",
    # Multi-head latent attention.
    "self_attention.linear_q_proj.weight",
    "self_attention.linear_q_down_proj.weight",
    "self_attention.linear_q_up_proj.weight",
    "self_attention.linear_kv_down_proj.weight",
    "self_attention.linear_kv_up_proj.weight",
    # DeepSeek sparse attention indexer.
    "self_attention.wq_b.weight",
    "self_attention.wk.weight",
}


def quantize_params_mxfp8(args, megatron_name, converted_named_params, quantization_config):
    """Convert eligible Megatron linear weights to canonical serialized MXFP8 tensors."""
    validate_mxfp8_config(quantization_config)
    exclusions = _normalized_exclusions(tuple(quantization_config.get("modules_to_not_convert", ())))

    match = re.search(r"decoder\.layers\.(\d+)\.(.+)", megatron_name)
    if not match:
        match = re.search(r"mtp\.layers\.(\d+)\.(.+)", megatron_name)
        if not match:
            return converted_named_params
        _, rest = match.groups()
        rest = rest.replace("transformer_layer.", "")
    else:
        _, rest = match.groups()

    expert_match = re.match(r"mlp\.experts\.(.+)\.weight\d+", rest)
    if expert_match and expert_match.group(1) in {"linear_fc1", "linear_fc2"}:
        return _quantize_converted_params(converted_named_params, exclusions)

    shared_expert_match = re.match(r"mlp\.shared_experts\.(.+)", rest)
    if shared_expert_match and shared_expert_match.group(1) in {
        "linear_fc1.weight",
        "linear_fc2.weight",
    }:
        return _quantize_converted_params(converted_named_params, exclusions)

    if rest in _MXFP8_LINEAR_WEIGHTS:
        return _quantize_converted_params(converted_named_params, exclusions)

    return converted_named_params


@lru_cache(maxsize=16)
def _normalized_exclusions(modules):
    return frozenset(normalize_mxfp8_exclusions(modules))


def _quantize_converted_params(converted_named_params, exclusions):
    quantized_params = []
    for name, weight in converted_named_params:
        if name.endswith("_scale"):
            continue
        if is_mxfp8_weight_excluded(name, exclusions):
            quantized_params.append((name, weight))
            continue
        quantized_params.extend(_quantize_param(name, weight))
    return quantized_params


def _quantize_param(name, weight):
    if not name.endswith(".weight"):
        raise ValueError(f"Expected weight parameter, got {name}.")
    qweight, scale = mxfp8_quantize(weight)
    return [(name, qweight), (name.removesuffix(".weight") + ".weight_scale_inv", scale)]
