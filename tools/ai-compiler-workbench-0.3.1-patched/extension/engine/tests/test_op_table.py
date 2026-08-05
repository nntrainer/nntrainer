from _util import load
op_table = load("op_table", "api/compatibility/op_table.py")
classify_op = op_table.classify_op


def test_linear_maps_to_fully_connected():
    e = classify_op("Linear")
    assert e["nntrainer_type"] == "fully_connected" and e["supported"] and e["emits_layer"]


def test_gpt2_conv1d_is_linear():
    e = classify_op("Conv1D")
    assert e["nntrainer_type"] == "fully_connected" and e["supported"]


def test_real_conv1d_is_not_matched_as_linear():
    # nn.Conv1d must NOT collide with GPT-2's Conv1D
    e = classify_op("Conv1d")
    assert e["nntrainer_type"] != "fully_connected"


def test_hf_gelu_activation():
    e = classify_op("GELUActivation")
    assert e["nntrainer_type"] == "activation"
    assert e["attributes"]["activation"] == "gelu" and e["supported"]


def test_hf_silu_activation():
    e = classify_op("SiLUActivation")
    assert e["attributes"]["activation"] == "swish"


def test_dropout_emits_no_layer():
    e = classify_op("Dropout")
    assert e["emits_layer"] is False and e["supported"]


def test_rmsnorm_family_recognized_but_manual():
    for name in ("RMSNorm", "LlamaRMSNorm", "Qwen2RMSNorm", "GemmaRMSNorm"):
        e = classify_op(name)
        assert e["emits_layer"] is True
        assert e["supported"] is False
        assert e["nntrainer_type"] is None       # never a guessed type string
        assert "RMS" in e["reason"]


def test_quick_gelu_not_silently_approximated():
    # deliberately unmapped -> stays unknown rather than pretending it's gelu
    e = classify_op("QuickGELUActivation")
    assert e["supported"] is False


def test_unknown_op():
    e = classify_op("SomeMadeUpOp")
    assert e["supported"] is False and e["emits_layer"] is True
    assert "unknown operator" in e["reason"]
