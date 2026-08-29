import json
from types import SimpleNamespace

import pytest

from slime.utils.data import Dataset, _apply_prompt_template
from slime.utils.eval_config import EvalDatasetConfig, build_eval_dataset_configs


def test_apply_prompt_template_to_string():
    assert _apply_prompt_template("What is 1 + 1?", "Solve carefully.\n\n{prompt}\n\nAnswer in a box.") == (
        "Solve carefully.\n\nWhat is 1 + 1?\n\nAnswer in a box."
    )


def test_apply_prompt_template_to_last_user_message():
    prompt = [
        {"role": "system", "content": "You are a mathematician."},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
    ]

    transformed = _apply_prompt_template(prompt, "Instruction\n{prompt}\nFinal format")

    assert transformed[-1]["content"] == "Instruction\nSecond question\nFinal format"
    assert transformed[1]["content"] == "First question"


@pytest.mark.parametrize("template", ["no placeholder", "{prompt} and {prompt}"])
def test_eval_dataset_config_rejects_invalid_prompt_template(template):
    with pytest.raises(ValueError, match="exactly one"):
        EvalDatasetConfig(name="aime", path="aime.jsonl", prompt_template=template)


def test_prompt_template_can_be_set_in_eval_defaults():
    datasets = build_eval_dataset_configs(
        SimpleNamespace(),
        [{"name": "aime", "path": "aime.jsonl"}],
        {"prompt_template": "{prompt}\nUse boxed notation."},
    )

    assert datasets[0].prompt_template == "{prompt}\nUse boxed notation."


def test_dataset_applies_prompt_template_before_chat_template(tmp_path):
    path = tmp_path / "aime.jsonl"
    path.write_text(
        json.dumps({"prompt": [{"role": "user", "content": "What is 1 + 1?"}], "label": "2"}) + "\n",
        encoding="utf-8",
    )

    class Tokenizer:
        def apply_chat_template(self, prompt, **_kwargs):
            return prompt[-1]["content"]

        def __call__(self, prompts, **_kwargs):
            return {"input_ids": [[0] for _ in prompts]}

    dataset = Dataset(
        path=str(path),
        tokenizer=Tokenizer(),
        processor=None,
        max_length=None,
        prompt_key="prompt",
        label_key="label",
        apply_chat_template=True,
        prompt_template="Solve step by step.\n\n{prompt}\n\nAnswer: \\boxed{...}",
    )

    assert dataset[0].prompt == "Solve step by step.\n\nWhat is 1 + 1?\n\nAnswer: \\boxed{...}"
    assert dataset[0].label == "2"
