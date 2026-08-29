from slime.backends.sglang_utils.sglang_engine import SGLangEngine


def test_sglang_engine_forwards_weight_check_skip_list():
    engine = object.__new__(SGLangEngine)
    calls = []
    engine._make_request = lambda endpoint, payload: calls.append((endpoint, payload))

    engine.check_weights("snapshot", skip_tensor_list=["visual."])

    assert calls == [
        (
            "weights_checker",
            {"action": "snapshot", "skip_tensor_list": ["visual."]},
        )
    ]


def test_sglang_engine_omits_unset_weight_check_skip_list():
    engine = object.__new__(SGLangEngine)
    calls = []
    engine._make_request = lambda endpoint, payload: calls.append((endpoint, payload))

    engine.check_weights("compare")

    assert calls == [("weights_checker", {"action": "compare"})]
