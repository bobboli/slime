import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

NUM_GPUS = 0
REPO_ROOT = Path(__file__).resolve().parents[1]

# CPU CI does not install Megatron. Mount this source package without running
# megatron_utils' runtime patch initialization.
try:
    _has_megatron = importlib.util.find_spec("megatron.core") is not None
except ModuleNotFoundError:
    _has_megatron = False
if not _has_megatron:
    _megatron_utils = types.ModuleType("slime.backends.megatron_utils")
    _megatron_utils.__path__ = [str(REPO_ROOT / "slime/backends/megatron_utils")]
    sys.modules["slime.backends.megatron_utils"] = _megatron_utils

from slime.backends.megatron_utils.megatron_to_hf import processors
from slime.backends.megatron_utils.megatron_to_hf.processors import quantizer_mxfp8
from slime.utils import mxfp8

MXFP8_CONFIG = {
    "quant_method": "mxfp8",
    "fmt": "e4m3",
    "activation_scheme": "dynamic",
    "weight_block_size": [1, 32],
    "scale_fmt": "ue8m0",
    "modules_to_not_convert": [],
}


def _load_converter_module():
    module_name = "test_convert_hf_to_mxfp8_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / "tools/convert_hf_to_mxfp8.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_te_quantizer_flattens_pads_and_crops(monkeypatch):
    calls = {}

    class FakeQuantizer:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

        def quantize(self, tensor):
            calls["input"] = tensor.clone()
            rows, columns = tensor.shape
            rowwise_data = torch.arange(rows * columns, dtype=torch.uint8).view(rows, columns)
            rowwise_scale_inv = torch.arange(128 * 4, dtype=torch.uint8).view(128, 4)
            return types.SimpleNamespace(
                get_metadata=lambda: {
                    "rowwise_data": rowwise_data,
                    "rowwise_scale_inv": rowwise_scale_inv,
                }
            )

    te = types.ModuleType("transformer_engine")
    te.__path__ = []
    te_pytorch = types.ModuleType("transformer_engine.pytorch")
    te_pytorch.__path__ = []
    te_constants = types.ModuleType("transformer_engine.pytorch.constants")
    te_pytorch.MXFP8Quantizer = FakeQuantizer
    te_constants.TE_DType = {torch.float8_e4m3fn: "e4m3"}
    monkeypatch.setitem(sys.modules, "transformer_engine", te)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", te_pytorch)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.constants", te_constants)

    weight = torch.arange(6 * 64, dtype=torch.float32).view(64, 6).t()
    qweight, scale = mxfp8.mxfp8_quantize(weight)

    assert calls["kwargs"] == {"fp8_dtype": "e4m3", "rowwise": True, "columnwise": False}
    assert calls["input"].shape == (32, 64)
    assert torch.equal(calls["input"][:6], weight.contiguous())
    assert torch.count_nonzero(calls["input"][6:]) == 0
    assert qweight.shape == weight.shape
    assert qweight.dtype == torch.float8_e4m3fn
    assert scale.shape == (6, 2)
    assert scale.dtype == torch.uint8
    expected_scale = torch.arange(128 * 4, dtype=torch.uint8).view(128, 4)[:6, :2]
    assert torch.equal(scale, expected_scale)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("weight", "message"),
    [
        (torch.ones(64), "at least two dimensions"),
        (torch.ones(2, 48), "divisible by 32"),
        (torch.ones(2, 64, dtype=torch.int32), "requires FP16, BF16, or FP32"),
    ],
)
def test_te_quantizer_rejects_invalid_inputs(weight, message):
    with pytest.raises(ValueError, match=message):
        mxfp8.mxfp8_quantize(weight)


@pytest.mark.unit
def test_mxfp8_config_and_rollout_override_validation():
    mxfp8.validate_mxfp8_rollout_config(MXFP8_CONFIG, None, "full", "nccl")
    mxfp8.validate_mxfp8_rollout_config(MXFP8_CONFIG, "mxfp8", "full", "nccl")

    with pytest.raises(ValueError, match="serialized MXFP8 checkpoint"):
        mxfp8.validate_mxfp8_rollout_config(None, "mxfp8", "full", "nccl")
    with pytest.raises(ValueError, match="conflicts"):
        mxfp8.validate_mxfp8_rollout_config(MXFP8_CONFIG, "fp8", "full", "nccl")
    with pytest.raises(ValueError, match="requires --update-weight-mode=full"):
        mxfp8.validate_mxfp8_rollout_config(MXFP8_CONFIG, None, "delta", "disk")
    with pytest.raises(ValueError, match="weight_block_size"):
        mxfp8.validate_mxfp8_config({**MXFP8_CONFIG, "weight_block_size": [128, 128]})


@pytest.mark.unit
def test_qwen35_mxfp8_requires_every_gdn_projection_in_high_precision():
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "num_hidden_layers": 4,
            "full_attention_interval": 2,
        },
    }
    exclusions = mxfp8.normalize_mxfp8_exclusions(
        [
            f"model.language_model.layers.{layer}.linear_attn.{module}"
            for layer in (0, 2)
            for module in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
        ]
    )
    quantization_config = {**MXFP8_CONFIG, "modules_to_not_convert": list(exclusions)}

    mxfp8.validate_qwen35_mxfp8_exclusions(quantization_config, config)

    quantization_config["modules_to_not_convert"].remove("model.layers.2.linear_attn.in_proj_qkv")
    with pytest.raises(ValueError, match="in_proj_qkv"):
        mxfp8.validate_qwen35_mxfp8_exclusions(quantization_config, config)
    with pytest.raises(ValueError, match="only Qwen3.5"):
        mxfp8.validate_qwen35_mxfp8_exclusions(quantization_config, {"model_type": "llama"})


@pytest.mark.unit
def test_exclusions_expand_packed_modules_and_qwen35_aliases():
    exclusions = mxfp8.normalize_mxfp8_exclusions(
        [
            "model.language_model.layers.2.linear_attn.in_proj_qkv",
            "model.layers.3.mlp.gate_proj",
        ]
    )

    for name in (
        "model.language_model.layers.2.linear_attn.in_proj_z",
        "model.language_model.layers.2.linear_attn.in_proj_qkvz",
        "model.layers.2.linear_attn.in_proj_qkv",
        "model.layers.2.linear_attn.in_proj_z",
        "model.layers.2.linear_attn.in_proj_qkvz",
        "model.layers.3.mlp.up_proj",
        "model.layers.3.mlp.gate_up_proj",
    ):
        assert name in exclusions

    assert mxfp8.is_mxfp8_weight_excluded("model.language_model.layers.2.linear_attn.in_proj_z.weight", exclusions)
    assert not mxfp8.is_mxfp8_weight_excluded(
        "model.layers.3.mlp.gate_up_projection.weight", ["model.layers.3.mlp.gate"]
    )


@pytest.mark.unit
def test_processor_dispatches_and_emits_canonical_weight_and_scale(monkeypatch):
    calls = []

    def fake_quantize(weight):
        calls.append(weight)
        return (
            torch.zeros(weight.shape, dtype=torch.float8_e4m3fn),
            torch.ones((*weight.shape[:-1], weight.shape[-1] // 32), dtype=torch.uint8),
        )

    monkeypatch.setattr(quantizer_mxfp8, "mxfp8_quantize", fake_quantize)
    converted = [
        ("model.layers.1.mlp.gate_proj.weight", torch.ones(4, 64, dtype=torch.bfloat16)),
        ("model.layers.1.mlp.up_proj.weight", torch.ones(4, 64, dtype=torch.bfloat16)),
    ]

    result = processors.quantize_params(
        types.SimpleNamespace(),
        "module.module.decoder.layers.1.mlp.linear_fc1.weight",
        converted,
        MXFP8_CONFIG,
    )

    assert [name for name, _ in result] == [
        "model.layers.1.mlp.gate_proj.weight",
        "model.layers.1.mlp.gate_proj.weight_scale_inv",
        "model.layers.1.mlp.up_proj.weight",
        "model.layers.1.mlp.up_proj.weight_scale_inv",
    ]
    assert len(calls) == 2
    assert result[0][1].dtype == torch.float8_e4m3fn
    assert result[1][1].dtype == torch.uint8


@pytest.mark.unit
@pytest.mark.parametrize(
    ("megatron_name", "hf_name"),
    [
        (
            "module.module.decoder.layers.1.self_attention.linear_qkv.weight",
            "model.layers.1.self_attn.q_proj.weight",
        ),
        (
            "module.module.decoder.layers.1.mlp.experts.linear_fc2.weight7",
            "model.layers.1.mlp.experts.7.down_proj.weight",
        ),
        (
            "module.module.decoder.layers.1.mlp.shared_experts.linear_fc2.weight",
            "model.layers.1.mlp.shared_expert.down_proj.weight",
        ),
        (
            "module.module.mtp.layers.0.transformer_layer.mlp.linear_fc2.weight",
            "mtp.layers.0.mlp.down_proj.weight",
        ),
    ],
)
def test_processor_quantizes_supported_megatron_weights(monkeypatch, megatron_name, hf_name):
    monkeypatch.setattr(
        quantizer_mxfp8,
        "mxfp8_quantize",
        lambda weight: (
            torch.zeros(weight.shape, dtype=torch.float8_e4m3fn),
            torch.ones((*weight.shape[:-1], weight.shape[-1] // 32), dtype=torch.uint8),
        ),
    )
    result = quantizer_mxfp8.quantize_params_mxfp8(
        types.SimpleNamespace(),
        megatron_name,
        [(hf_name, torch.ones(4, 64, dtype=torch.bfloat16))],
        MXFP8_CONFIG,
    )
    assert [name for name, _ in result] == [hf_name, hf_name.removesuffix(".weight") + ".weight_scale_inv"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("megatron_name", "hf_name"),
    [
        (
            "module.module.decoder.layers.1.mlp.experts.linear_fc1",
            "model.language_model.layers.1.mlp.experts.gate_up_proj",
        ),
        (
            "module.module.decoder.layers.1.mlp.experts.linear_fc2",
            "model.language_model.layers.1.mlp.experts.down_proj",
        ),
    ],
)
def test_processor_quantizes_grouped_experts(monkeypatch, megatron_name, hf_name):
    weight = torch.ones(3, 8, 64, dtype=torch.bfloat16)
    monkeypatch.setattr(
        quantizer_mxfp8,
        "mxfp8_quantize",
        lambda tensor: (
            torch.zeros(tensor.shape, dtype=torch.float8_e4m3fn),
            torch.ones((*tensor.shape[:-1], tensor.shape[-1] // 32), dtype=torch.uint8),
        ),
    )

    result = quantizer_mxfp8.quantize_params_mxfp8(
        types.SimpleNamespace(),
        megatron_name,
        [(hf_name, weight)],
        MXFP8_CONFIG,
    )

    assert [name for name, _ in result] == [hf_name, f"{hf_name}_scale_inv"]
    assert result[0][1].shape == (3, 8, 64)
    assert result[0][1].dtype == torch.float8_e4m3fn
    assert result[1][1].shape == (3, 8, 2)
    assert result[1][1].dtype == torch.uint8


@pytest.mark.unit
def test_processor_respects_checkpoint_exclusions_and_leaves_gdn_bf16(monkeypatch):
    monkeypatch.setattr(
        quantizer_mxfp8,
        "mxfp8_quantize",
        lambda _weight: pytest.fail("excluded weight was quantized"),
    )
    weight = torch.ones(4, 64, dtype=torch.bfloat16)
    excluded_config = {
        **MXFP8_CONFIG,
        "modules_to_not_convert": ["model.language_model.layers.1.mlp.gate_proj"],
    }
    dense = quantizer_mxfp8.quantize_params_mxfp8(
        types.SimpleNamespace(),
        "module.module.decoder.layers.1.mlp.linear_fc1.weight",
        [
            ("model.language_model.layers.1.mlp.gate_proj.weight", weight),
            ("model.language_model.layers.1.mlp.up_proj.weight", weight),
        ],
        excluded_config,
    )
    gdn = quantizer_mxfp8.quantize_params_mxfp8(
        types.SimpleNamespace(),
        "module.module.decoder.layers.1.self_attention.linear_attn.in_proj_qkv.weight",
        [("model.language_model.layers.1.linear_attn.in_proj_qkv.weight", weight)],
        MXFP8_CONFIG,
    )
    assert dense[0][1].dtype == torch.bfloat16
    assert dense[1][1].dtype == torch.bfloat16
    assert [name for name, _ in gdn] == ["model.language_model.layers.1.linear_attn.in_proj_qkv.weight"]
    assert gdn[0][1] is weight


@pytest.mark.unit
def test_converter_writes_qwen35_mxfp8_checkpoint(monkeypatch, tmp_path):
    converter = _load_converter_module()
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": {"num_hidden_layers": 4},
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")

    tensors_by_file = {
        "model-00001-of-00002.safetensors": {
            "model.language_model.layers.0.mlp.down_proj.weight": torch.ones(4, 64, dtype=torch.bfloat16),
            "model.language_model.layers.1.mlp.down_proj.weight": torch.ones(4, 64, dtype=torch.bfloat16),
            "model.language_model.layers.1.mlp.experts.gate_up_proj": torch.ones(3, 8, 64, dtype=torch.bfloat16),
            "model.language_model.layers.1.mlp.experts.down_proj": torch.ones(3, 8, 64, dtype=torch.bfloat16),
            "model.language_model.layers.1.linear_attn.in_proj_qkv.weight": torch.ones(4, 64, dtype=torch.bfloat16),
            "model.language_model.layers.1.linear_attn.in_proj_b.weight": torch.ones(4, 64, dtype=torch.bfloat16),
        },
        "model-00002-of-00002.safetensors": {
            "model.language_model.layers.1.linear_attn.in_proj_z.weight": torch.ones(4, 64, dtype=torch.bfloat16),
            "model.language_model.layers.1.linear_attn.in_proj_a.weight": torch.ones(4, 64, dtype=torch.bfloat16),
            "model.language_model.layers.1.linear_attn.out_proj.weight": torch.ones(4, 64, dtype=torch.bfloat16),
            "model.language_model.layers.1.mlp.shared_expert_gate.weight": torch.ones(1, 64, dtype=torch.bfloat16),
            "model.language_model.layers.3.mlp.down_proj.weight": torch.ones(4, 64, dtype=torch.bfloat16),
            "model.visual.blocks.0.mlp.weight": torch.ones(4, 64, dtype=torch.bfloat16),
        },
    }
    weight_map = {}
    for filename, tensors in tensors_by_file.items():
        save_file(tensors, source / filename)
        weight_map.update({name: filename for name in tensors})
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )

    monkeypatch.setattr(
        converter,
        "mxfp8_quantize",
        lambda weight: (
            torch.zeros(weight.shape, dtype=torch.float8_e4m3fn),
            torch.ones((*weight.shape[:-1], weight.shape[-1] // 32), dtype=torch.uint8),
        ),
    )
    converter.convert_mxfp8(
        source,
        output,
        "cpu",
        num_layers_at_start_in_bf16=1,
        num_layers_at_end_in_bf16=1,
    )

    output_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    quantization_config = output_config["quantization_config"]
    assert quantization_config["quant_method"] == "mxfp8"
    assert quantization_config["weight_block_size"] == [1, 32]
    exclusions = quantization_config["modules_to_not_convert"]
    for name in (
        "model.language_model.layers.0",
        "model.language_model.layers.3",
        "model.language_model.layers.1.linear_attn.in_proj_qkvz",
        "model.language_model.layers.1.linear_attn.in_proj_ba",
        "model.layers.1.linear_attn.in_proj_qkvz",
        "model.layers.1.linear_attn.in_proj_ba",
        "model.language_model.layers.1.mlp.shared_expert_gate",
        "model.visual",
    ):
        assert name in exclusions

    shard0 = load_file(output / "model-00001-of-00002.safetensors")
    assert shard0["model.language_model.layers.0.mlp.down_proj.weight"].dtype == torch.bfloat16
    assert shard0["model.language_model.layers.1.mlp.down_proj.weight"].dtype == torch.float8_e4m3fn
    assert shard0["model.language_model.layers.1.mlp.down_proj.weight_scale_inv"].dtype == torch.uint8
    assert shard0["model.language_model.layers.1.mlp.experts.gate_up_proj"].dtype == torch.float8_e4m3fn
    assert shard0["model.language_model.layers.1.mlp.experts.gate_up_proj_scale_inv"].shape == (3, 8, 2)
    assert shard0["model.language_model.layers.1.mlp.experts.gate_up_proj_scale_inv"].dtype == torch.uint8
    assert shard0["model.language_model.layers.1.mlp.experts.down_proj"].dtype == torch.float8_e4m3fn
    assert shard0["model.language_model.layers.1.mlp.experts.down_proj_scale_inv"].shape == (3, 8, 2)
    assert shard0["model.language_model.layers.1.mlp.experts.down_proj_scale_inv"].dtype == torch.uint8
    assert shard0["model.language_model.layers.1.linear_attn.in_proj_qkv.weight"].dtype == torch.bfloat16
    assert "model.language_model.layers.1.linear_attn.in_proj_qkv.weight_scale_inv" not in shard0
    shard1 = load_file(output / "model-00002-of-00002.safetensors")
    assert shard1["model.language_model.layers.1.mlp.shared_expert_gate.weight"].dtype == torch.bfloat16
    assert "model.language_model.layers.1.mlp.shared_expert_gate.weight_scale_inv" not in shard1

    output_index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert output_index["weight_map"]["model.language_model.layers.1.mlp.down_proj.weight_scale_inv"] == (
        "model-00001-of-00002.safetensors"
    )
    assert output_index["weight_map"]["model.language_model.layers.1.mlp.experts.gate_up_proj_scale_inv"] == (
        "model-00001-of-00002.safetensors"
    )
    total_size = sum(
        tensor.numel() * tensor.element_size()
        for filename in tensors_by_file
        for tensor in load_file(output / filename).values()
    )
    assert output_index["metadata"]["total_size"] == total_size
    assert (output / "tokenizer.json").read_text(encoding="utf-8") == "{}"
    assert "quantization_config" not in json.loads((source / "config.json").read_text(encoding="utf-8"))


@pytest.mark.unit
def test_converter_rejects_quantized_source_and_nonempty_output(tmp_path):
    converter = _load_converter_module()
    with pytest.raises(ValueError, match="Only unquantized"):
        converter._validate_source_config(
            {"model_type": "qwen3_5_moe", "quantization_config": {"quant_method": "fp8"}}
        )
    with pytest.raises(ValueError, match="only Qwen3.5"):
        converter._validate_source_config({"model_type": "llama"})

    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (output / "existing").write_text("occupied", encoding="utf-8")
    with pytest.raises(ValueError, match="must be empty"):
        converter.convert_mxfp8(source, output, "cpu")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
