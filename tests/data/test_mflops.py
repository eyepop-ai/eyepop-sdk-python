import math

import flatbuffers
import numpy as np
import pytest

# the counter needs the admin extra
onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
tflite = pytest.importorskip("tflite")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from eyepop.data import mflops  # noqa: E402
from eyepop.data.mflops import (  # noqa: E402
    MFLOPS_PER_INVOCATION,
    UnresolvedShapeError,
    count_mflops,
    count_model_mflops,
    count_tflite_mflops,
    onnx_input_shapes,
    with_mflops,
)

ABILITY = {
    "format": "ONNX",
    "inputs": [{"tensor": {"name": "input", "type": "float32", "dimensions": [224, 224, 3, -1]}}],
}
# 2 x (64 x 224 x 224) x 3 x 3 x 3 conv, 64 x 224 x 224 relu and pool, 2 x 1000 x 64 matmul
CONV_MATMUL_MFLOPS = -(-(2 * 64 * 224 * 224 * 27 + 2 * 64 * 224 * 224 + 2 * 1000 * 64) // 1_000_000)


def _save(tmp_path, graph, name: str) -> str:
    path = str(tmp_path / name)
    # IR 10: the onnx package writes a newer IR version than onnxruntime may read
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10), path)
    return path


def _conv_matmul_model(tmp_path, batch: int | str = "batch") -> str:
    """A 3x224x224 image, a 3x3 conv to 64 channels, global pool, then a 64x1000 matmul."""
    weight = numpy_helper.from_array(np.zeros((64, 3, 3, 3), dtype=np.float32), "w")
    fc = numpy_helper.from_array(np.zeros((64, 1000), dtype=np.float32), "fc")
    nodes = [
        helper.make_node("Conv", ["input", "w"], ["conv"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv"], ["relu"]),
        helper.make_node("GlobalAveragePool", ["relu"], ["pool"]),
        helper.make_node("Flatten", ["pool"], ["flat"]),
        helper.make_node("MatMul", ["flat", "fc"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, 3, 224, 224])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [batch, 1000])],
        initializer=[weight, fc],
    )
    return _save(tmp_path, graph, "model.onnx")


def _value_dependent_model(tmp_path) -> str:
    """A MatMul over the indices NonZero finds in a 4x4x2 input: only a run knows how many."""
    nodes = [
        helper.make_node("Greater", ["input", "minus_one"], ["positive"]),
        helper.make_node("NonZero", ["positive"], ["indices"]),
        helper.make_node("Cast", ["indices"], ["float_indices"], to=TensorProto.FLOAT),
        helper.make_node("Transpose", ["float_indices"], ["rows"], perm=[1, 0]),
        helper.make_node("MatMul", ["rows", "fc"], ["out"]),
    ]
    graph = helper.make_graph(
        nodes,
        "dynamic",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 4, 2])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, None)],
        initializer=[
            numpy_helper.from_array(np.zeros((3, 16), dtype=np.float32), "fc"),
            numpy_helper.from_array(np.array(-1, dtype=np.float32), "minus_one"),
        ],
    )
    return _save(tmp_path, graph, "dynamic.onnx")


def test_counts_conv_matmul_and_elementwise_flops(tmp_path):
    assert count_mflops(_conv_matmul_model(tmp_path), {"input": [1, 3, 224, 224]}) == CONV_MATMUL_MFLOPS


def test_a_model_is_counted_from_its_bytes(tmp_path):
    with open(_conv_matmul_model(tmp_path), "rb") as f:
        model = f.read()

    assert count_mflops(model, {"input": [1, 3, 224, 224]}) == CONV_MATMUL_MFLOPS


def test_ability_dimensions_are_reversed_into_onnx_order_with_batch_one():
    assert onnx_input_shapes(ABILITY) == {"input": [1, 3, 224, 224]}


def test_a_dynamic_input_the_ability_does_not_declare_is_refused(tmp_path):
    with pytest.raises(UnresolvedShapeError):
        count_mflops(_conv_matmul_model(tmp_path), {})


def test_a_static_model_needs_no_declared_shape(tmp_path):
    assert count_mflops(_conv_matmul_model(tmp_path, batch=1), {}) == CONV_MATMUL_MFLOPS


def test_a_shape_that_depends_on_values_is_taken_from_a_run(tmp_path, monkeypatch):
    runs = []
    run_for_shapes = mflops._run_for_shapes
    monkeypatch.setattr(mflops, "_run_for_shapes", lambda *a: runs.append(a[1]) or run_for_shapes(*a))

    # 32 inputs, every one above -1: 32 rows x 3 inner x 16 outputs x 2 FLOPs
    assert count_mflops(_value_dependent_model(tmp_path), {}) == 1
    # one run: what NonZero found, and the boolean mask it found it in
    assert len(runs) == 1
    assert "indices" in runs[0]


def test_a_run_gives_the_values_it_is_asked_for_and_leaves_the_model(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    model = mflops._load_without_weights(_value_dependent_model(tmp_path), str(workdir))
    outputs = [o.name for o in model.graph.output]

    found = mflops._run_for_shapes(model, ["indices", "rows"], {}, str(workdir))

    assert found["indices"].shape == (3, 32)
    assert found["rows"].shape == (32, 3)
    assert [o.name for o in model.graph.output] == outputs


def test_shape_arithmetic_behind_an_if_is_frozen_to_its_value(tmp_path):
    """A Reshape whose target comes out of an If: inference cannot pick the branch."""
    then_graph = helper.make_graph(
        [helper.make_node("Constant", [], ["then_shape"], value=numpy_helper.from_array(np.array([4, 8], dtype=np.int64)))],
        "then",
        [],
        [helper.make_tensor_value_info("then_shape", TensorProto.INT64, [2])],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Constant", [], ["else_shape"], value=numpy_helper.from_array(np.array([8, 4], dtype=np.int64)))],
        "else",
        [],
        [helper.make_tensor_value_info("else_shape", TensorProto.INT64, [2])],
    )
    nodes = [
        helper.make_node("ReduceSum", ["input"], ["total"], keepdims=0),
        helper.make_node("Greater", ["total", "zero"], ["positive"]),
        helper.make_node("If", ["positive"], ["target"], then_branch=then_graph, else_branch=else_graph),
        helper.make_node("Reshape", ["input", "target"], ["reshaped"]),
        helper.make_node("MatMul", ["reshaped", "fc"], ["out"]),
    ]
    graph = helper.make_graph(
        nodes,
        "if",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [32])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, None)],
        initializer=[
            numpy_helper.from_array(np.zeros((8, 1000), dtype=np.float32), "fc"),
            numpy_helper.from_array(np.array(0, dtype=np.float32), "zero"),
        ],
    )
    path = _save(tmp_path, graph, "if.onnx")

    # random inputs are positive: [4, 8] x [8, 1000] is 2 x 4 x 1000 x 8 FLOPs, rounded up
    assert count_mflops(path, {}) == 1


def test_a_tensor_without_a_shape_is_unknown_not_a_scalar():
    graph = helper.make_graph(
        [],
        "g",
        [helper.make_tensor_value_info("unranked", TensorProto.FLOAT, None)],
        [helper.make_tensor_value_info("scalar", TensorProto.FLOAT, [])],
    )

    assert mflops._known_shapes(helper.make_model(graph)) == {"unranked": None, "scalar": []}


def _tflite_fully_connected(path, inputs: int, outputs: int) -> str:
    """A TensorFlow Lite model of one FULLY_CONNECTED op, [1, inputs] to [1, outputs]."""
    return _tflite_model(
        path,
        tflite.BuiltinOperator.FULLY_CONNECTED,
        [[1, inputs], [outputs, inputs], [1, outputs]],
    )


def _tflite_model(
    path,
    builtin: int,
    shapes: list[list[int] | None],
    adj_x: bool | None = None,
    has_rank: bool = False,
) -> str:
    """A TensorFlow Lite model of one operator: every tensor but the last is an input.

    A shape of None leaves the tensor's shape out. Written with raw flatbuffer slots, by the
    field numbers of TensorFlow Lite's schema: the tflite package reads models but cannot
    build them.
    """
    b = flatbuffers.Builder(1024)

    def int_vector(values):
        b.StartVector(4, len(values), 4)
        for value in reversed(values):
            b.PrependInt32(value)
        return b.EndVector()

    def offset_vector(offsets):
        b.StartVector(4, len(offsets), 4)
        for offset in reversed(offsets):
            b.PrependUOffsetTRelative(offset)
        return b.EndVector()

    def tensor(name, shape):
        name_offset = b.CreateString(name)
        shape_offset = None if shape is None else int_vector(shape)
        b.StartObject(9)  # shape, type, buffer, name, ..., has_rank
        if shape_offset is not None:
            b.PrependUOffsetTRelativeSlot(0, shape_offset, 0)
        b.PrependUOffsetTRelativeSlot(3, name_offset, 0)
        b.PrependBoolSlot(8, has_rank, False)
        return b.EndObject()

    tensors = offset_vector([tensor(f"t{i}", shape) for i, shape in enumerate(shapes)])
    operator_inputs = int_vector(list(range(len(shapes) - 1)))
    operator_outputs = int_vector([len(shapes) - 1])
    options = None
    if adj_x is not None:
        b.StartObject(3)  # adj_x, adj_y, asymmetric_quantize_inputs
        b.PrependBoolSlot(0, adj_x, False)
        options = b.EndObject()
    b.StartObject(5)  # opcode_index, inputs, outputs, builtin_options_type, builtin_options
    b.PrependUOffsetTRelativeSlot(1, operator_inputs, 0)
    b.PrependUOffsetTRelativeSlot(2, operator_outputs, 0)
    if options is not None:
        b.PrependUint8Slot(3, tflite.BuiltinOptions.BatchMatMulOptions, 0)
        b.PrependUOffsetTRelativeSlot(4, options, 0)
    operators = offset_vector([b.EndObject()])
    subgraph_inputs, subgraph_outputs = int_vector([0]), int_vector([len(shapes) - 1])
    b.StartObject(4)  # tensors, inputs, outputs, operators
    b.PrependUOffsetTRelativeSlot(0, tensors, 0)
    b.PrependUOffsetTRelativeSlot(1, subgraph_inputs, 0)
    b.PrependUOffsetTRelativeSlot(2, subgraph_outputs, 0)
    b.PrependUOffsetTRelativeSlot(3, operators, 0)
    subgraphs = offset_vector([b.EndObject()])
    b.StartObject(4)  # deprecated_builtin_code, custom_code, version, builtin_code
    b.PrependInt8Slot(0, min(builtin, 127), 0)
    b.PrependInt32Slot(3, builtin, 0)
    codes = offset_vector([b.EndObject()])
    b.StartObject(0)
    buffers = offset_vector([b.EndObject()])
    b.StartObject(5)  # version, operator_codes, subgraphs, description, buffers
    b.PrependUint32Slot(0, 3, 0)
    b.PrependUOffsetTRelativeSlot(1, codes, 0)
    b.PrependUOffsetTRelativeSlot(2, subgraphs, 0)
    b.PrependUOffsetTRelativeSlot(4, buffers, 0)
    b.Finish(b.EndObject(), file_identifier=b"TFL3")
    target = str(path / "model.tflite")
    with open(target, "wb") as f:
        f.write(b.Output())
    return target



def test_counts_a_tflite_models_flops(tmp_path):
    # 2 x 1000 outputs x 1000 inputs
    assert count_tflite_mflops(_tflite_fully_connected(tmp_path, 1000, 1000)) == 2


def test_a_tflite_model_is_counted_from_its_bytes(tmp_path):
    with open(_tflite_fully_connected(tmp_path, 1000, 1000), "rb") as f:
        assert count_tflite_mflops(f.read()) == 2


def test_a_tflite_batch_matmul_with_adj_x_sums_the_left_operands_rows(tmp_path):
    # adj_x: a [1, 4000, 10] is used as [1, 10, 4000], so every output element sums 4000 products
    path = _tflite_model(
        tmp_path, tflite.BuiltinOperator.BATCH_MATMUL, [[1, 4000, 10], [1, 4000, 500], [1, 10, 500]], adj_x=True
    )

    assert count_tflite_mflops(path) == 40  # 2 x 5000 outputs x 4000


def test_a_tflite_batch_matmul_without_adj_x_sums_the_left_operands_columns(tmp_path):
    path = _tflite_model(
        tmp_path, tflite.BuiltinOperator.BATCH_MATMUL, [[1, 10, 4000], [1, 4000, 500], [1, 10, 500]], adj_x=False
    )

    assert count_tflite_mflops(path) == 40


@pytest.mark.parametrize("has_rank", [True, False])
def test_a_tflite_reduction_to_a_scalar_without_a_shape_is_counted(tmp_path, has_rank):
    # a scalar output may leave its shape out; it is one element only where its rank is known
    path = _tflite_model(tmp_path, tflite.BuiltinOperator.MEAN, [[1, 10], [1], None], has_rank=has_rank)

    assert count_tflite_mflops(path) == (1 if has_rank else 0)


def test_a_model_is_told_apart_by_its_contents_not_its_root_files_format(tmp_path):
    # an ability of format ModelLess can carry an ONNX or a TensorFlow Lite model
    root = {**ABILITY, "format": "ModelLess"}
    tflite_path = _tflite_fully_connected(tmp_path, 1000, 1000)

    assert count_model_mflops(tflite_path, root) == 2
    assert count_model_mflops(_conv_matmul_model(tmp_path), root) == CONV_MATMUL_MFLOPS


def test_with_mflops_sets_the_counted_value(tmp_path):
    root = with_mflops(ABILITY, _conv_matmul_model(tmp_path))

    assert root == {**ABILITY, MFLOPS_PER_INVOCATION: CONV_MATMUL_MFLOPS}
    assert MFLOPS_PER_INVOCATION not in ABILITY


def test_with_mflops_keeps_a_declared_value(tmp_path):
    root = {**ABILITY, MFLOPS_PER_INVOCATION: 7}

    assert with_mflops(root, _conv_matmul_model(tmp_path)) == root


def test_with_mflops_leaves_a_model_it_cannot_count_as_it_is(tmp_path, caplog):
    # the root file declares no input, and the model's batch is dynamic
    root = {**ABILITY, "inputs": []}

    assert with_mflops(root, _conv_matmul_model(tmp_path)) == root
    assert "could not count" in caplog.text


def test_with_mflops_leaves_a_file_that_is_no_model_as_it_is(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"not a model")

    assert with_mflops(ABILITY, str(path)) == ABILITY


def test_a_variable_dimension_anywhere_is_counted_as_one():
    # SAM3's decoder takes language features [256, -1, 32]: a variable token count in the middle
    root = {"inputs": [{"tensor": {"name": "features", "dimensions": [256, -1, 32]}}]}

    assert onnx_input_shapes(root) == {"features": [32, 1, 256]}


def _one_node_model(tmp_path, node, inputs, initializers, name: str, out_type: int = TensorProto.FLOAT) -> str:
    graph = helper.make_graph(
        [node],
        "one",
        inputs,
        [helper.make_tensor_value_info(node.output[0], out_type, None)],
        initializer=initializers,
    )
    return _save(tmp_path, graph, name)


def test_a_quantized_matmul_counts_as_a_matmul(tmp_path):
    scale = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "scale")
    zero = numpy_helper.from_array(np.array(0, dtype=np.uint8), "zero")
    weight = numpy_helper.from_array(np.zeros((1000, 1000), dtype=np.uint8), "w")
    node = helper.make_node(
        "QLinearMatMul", ["a", "scale", "zero", "w", "scale", "zero", "scale", "zero"], ["out"]
    )
    path = _one_node_model(
        tmp_path, node, [helper.make_tensor_value_info("a", TensorProto.UINT8, [1, 1000])], [scale, zero, weight], "q.onnx", TensorProto.UINT8
    )

    # 2 x 1000 outputs x 1000 inner
    assert count_mflops(path, {}) == 2


def test_an_operator_the_count_cannot_price_keeps_the_default(tmp_path):
    weights = [
        numpy_helper.from_array(np.zeros((1, 8, 4), dtype=np.float32), "w"),
        numpy_helper.from_array(np.zeros((1, 8, 8), dtype=np.float32), "r"),
    ]
    node = helper.make_node("RNN", ["x", "w", "r"], ["out"], hidden_size=8)
    path = _one_node_model(
        tmp_path, node, [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4])], weights, "rnn.onnx"
    )

    with pytest.raises(UnresolvedShapeError, match="RNN"):
        count_mflops(path, {})
    assert with_mflops({"inputs": []}, path) == {"inputs": []}


def _einsum_flops(equation: str, *operand_shapes: list[int], output: list[int] | None = None) -> int:
    names = [f"x{i}" for i in range(len(operand_shapes))]
    node = helper.make_node("Einsum", names, ["out"], equation=equation)
    return mflops._node_flops(node, {**dict(zip(names, operand_shapes, strict=False)), "out": output})


def test_an_einsum_costs_one_multiply_accumulate_per_index_combination():
    # SAM3's mask predictor: queries x channels against channels x pixels
    assert _einsum_flops("bqc,bchw->bqhw", [1, 100, 256], [1, 256, 64, 64], output=[1, 100, 64, 64]) == (
        2 * 100 * 256 * 64 * 64
    )


def test_an_einsum_ellipsis_broadcasts_its_dimensions():
    flops = _einsum_flops("...ij,...jk->...ik", [2, 3, 10, 20], [3, 20, 30], output=[2, 3, 10, 30])

    assert flops == 2 * (2 * 3) * 10 * 20 * 30


def test_an_einsum_of_one_operand_costs_its_sums_and_nothing_to_transpose():
    assert _einsum_flops("ij->i", [10, 20], output=[10]) == 10 * 20
    assert _einsum_flops("ij->ji", [10, 20], output=[20, 10]) == 0
    # a trace: the implicit output keeps the indices that appear once, so it sums the diagonal
    assert _einsum_flops("ii", [10, 10]) == 10


def test_an_einsum_is_counted_in_a_model(tmp_path):
    node = helper.make_node("Einsum", ["a", "b"], ["out"], equation="ij,jk->ik")
    path = _one_node_model(
        tmp_path,
        node,
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [100, 200]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [200, 300]),
        ],
        [],
        "einsum.onnx",
    )

    assert count_mflops(path, {}) == math.ceil(2 * 100 * 200 * 300 / 1_000_000)


def test_a_gelu_onnx_runtime_fused_counts_as_elementwise(tmp_path):
    # PARSeq's: inference knows no com.microsoft schema, so the MatMul after it needs a run
    weight = numpy_helper.from_array(np.zeros((1000, 1000), dtype=np.float32), "w")
    graph = helper.make_graph(
        [
            helper.make_node("Gelu", ["a"], ["g"], domain="com.microsoft"),
            helper.make_node("MatMul", ["g", "w"], ["out"]),
        ],
        "gelu",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 1000])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, None)],
        initializer=[weight],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)], ir_version=10
    )
    path = str(tmp_path / "gelu.onnx")
    onnx.save(model, path)

    assert count_mflops(path, {}) == math.ceil((1000 + 2 * 1000 * 1000) / 1_000_000)


def test_a_gelu_without_an_inferred_output_is_priced_by_its_input():
    node = helper.make_node("Gelu", ["a"], ["g"], domain="com.microsoft")

    assert mflops._node_flops(node, {"a": [4, 1000], "g": None}) == 4000


def test_the_weights_are_kept_on_disk_while_counting(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()

    model = mflops._load_without_weights(_conv_matmul_model(tmp_path), str(workdir))

    assert all(len(i.raw_data) <= mflops._EXTERNAL_DATA_THRESHOLD for i in model.graph.initializer)
    assert (workdir / "weights").stat().st_size > 0


def test_a_model_with_external_weights_is_counted(tmp_path):
    source = onnx.load(_conv_matmul_model(tmp_path))
    external = tmp_path / "external"
    external.mkdir()
    onnx.save_model(source, str(external / "model.onnx"), save_as_external_data=True, location="model.data")

    assert count_mflops(str(external / "model.onnx"), {"input": [1, 3, 224, 224]}) == CONV_MATMUL_MFLOPS


def test_a_contrib_operator_keeps_the_default(tmp_path):
    node = helper.make_node("FusedGemm", ["a", "a"], ["out"], domain="com.microsoft")
    graph = helper.make_graph(
        [node],
        "contrib",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [8, 8])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [8, 8])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )
    path = str(tmp_path / "contrib.onnx")
    onnx.save(model, path)

    with pytest.raises(UnresolvedShapeError, match="com.microsoft"):
        count_mflops(path, {})


def test_a_matmul_inside_a_branch_keeps_the_default(tmp_path):
    weight = numpy_helper.from_array(np.zeros((8, 8), dtype=np.float32), "w")
    branch = helper.make_graph(
        [helper.make_node("MatMul", ["a", "w"], ["branch_out"])],
        "branch",
        [],
        [helper.make_tensor_value_info("branch_out", TensorProto.FLOAT, [8, 8])],
    )
    nodes = [
        helper.make_node("ReduceSum", ["a"], ["total"], keepdims=0),
        helper.make_node("Greater", ["total", "zero"], ["positive"]),
        helper.make_node("If", ["positive"], ["out"], then_branch=branch, else_branch=branch),
    ]
    graph = helper.make_graph(
        nodes,
        "branched",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [8, 8])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [8, 8])],
        initializer=[weight, numpy_helper.from_array(np.array(0, dtype=np.float32), "zero")],
    )
    path = _save(tmp_path, graph, "branched.onnx")

    with pytest.raises(UnresolvedShapeError, match="subgraph"):
        count_mflops(path, {})


def test_a_value_a_run_returned_is_not_asked_for_again(tmp_path, monkeypatch):
    asked: list[str] = []
    run_for_shapes = mflops._run_for_shapes
    monkeypatch.setattr(mflops, "_run_for_shapes", lambda *a: asked.extend(a[1]) or run_for_shapes(*a))

    count_mflops(_value_dependent_model(tmp_path), {})

    assert len(asked) == len(set(asked))
