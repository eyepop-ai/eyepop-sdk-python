"""How many MFLOPs one run of a model costs, for its ability's `mflops_per_invocation`.

Needs the `admin` extra: `pip install eyepop[admin]`.

The worker's pipeline adds an ability's `mflops_per_invocation` to its usage for every run of
the model, and the pipeline API turns that usage into milli compute units. Whoever publishes
a model's default export counts it with `with_mflops` before uploading the export's
`ability.json` or `model.json`; dataset-api copies the value into the model's other exports
(optimized, quantized, compiled).

The value is counted from the graph, not timed, so it does not depend on the hardware the
model runs on. One run is one batch of one input at the size the ability feeds it. A matrix
product or convolution costs two FLOPs per multiply-accumulate; an elementwise operation costs
one FLOP per output element. Operations that only move data (Reshape, Transpose, Gather, ...)
cost nothing.
"""

import logging
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from typing import Any

log = logging.getLogger(__name__)

MFLOPS_PER_INVOCATION = "mflops_per_invocation"

# a model is a file path or the model's bytes
Model = str | os.PathLike[str] | bytes

_TFLITE_IDENTIFIER = b"TFL3"

# where a counted operator's operands are: its input (or left operand) and its weight (or
# right operand); the quantized operators count as their float counterparts
_CONV_OPERANDS = {"Conv": (0, 1), "ConvTranspose": (0, 1), "ConvInteger": (0, 1), "QLinearConv": (0, 3)}
_MATMUL_OPERANDS = {"MatMul": (0, 1), "Gemm": (0, 1), "MatMulInteger": (0, 1), "QLinearMatMul": (0, 3)}
_PRODUCT_OPS = frozenset({*_CONV_OPERANDS, *_MATMUL_OPERANDS})

# operators whose every operand's shape the count needs: products, and Einsum
_COUNTED_OPS = frozenset({*_PRODUCT_OPS, "Einsum"})

# operators that compute but that the count cannot price: a model with one keeps the default
_UNPRICED_OPS = frozenset({"LSTM", "GRU", "RNN", "Attention"})

# com.microsoft operators that ONNX Runtime fuses from elementwise ones: one FLOP per output element
_CONTRIB_ELEMENTWISE_OPS = frozenset({"Gelu", "FastGelu", "BiasGelu", "QuickGelu"})

# initializers larger than this are kept on disk while counting, not in memory
_EXTERNAL_DATA_THRESHOLD = 1024

# runs of the model, at most, to resolve the shapes inference cannot
_MAX_RUNS = 8

# an integer tensor of at most this many elements is shape arithmetic, frozen to its value
_SMALL_INTEGER_ELEMENTS = 1024

_POOLING_OPS = frozenset(
    {"GlobalAveragePool", "AveragePool", "MaxPool", "ReduceMean", "ReduceSum", "ReduceMax"}
)

_ELEMENTWISE_OPS = frozenset(
    {
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Pow",
        "Sqrt",
        "Exp",
        "Log",
        "Erf",
        "Tanh",
        "Sigmoid",
        "Relu",
        "LeakyRelu",
        "PRelu",
        "HardSigmoid",
        "HardSwish",
        "Clip",
        "Max",
        "Min",
        "Neg",
        "Abs",
        "Reciprocal",
        "Softmax",
        "LogSoftmax",
        "BatchNormalization",
        "LayerNormalization",
        "InstanceNormalization",
        "Gelu",
        *_POOLING_OPS,
        "Resize",
    }
)

_NUMPY_TYPES = {
    "tensor(float)": "float32",
    "tensor(float16)": "float16",
    "tensor(double)": "float64",
    "tensor(int64)": "int64",
    "tensor(int32)": "int32",
    "tensor(uint8)": "uint8",
    "tensor(int8)": "int8",
    "tensor(bool)": "bool",
}


class UnresolvedShapeError(ValueError):
    """A shape the count needs is not known, and the count refuses to guess it."""


def with_mflops(root: Mapping[str, Any], model: Model) -> dict[str, Any]:
    """The export's root file (`ability.json` or `model.json`) with `mflops_per_invocation` set.

    A root file that already declares the value keeps it. A model that cannot be counted (an
    input dimension the root file does not declare, an operator the count cannot price, a file
    that is no ONNX or TensorFlow Lite model) is logged and leaves the root file as it is; the
    pipeline then falls back to its default.

    Raises:
        ImportError: the `admin` extra is not installed.
        OSError: the model's file cannot be read.

    Args:
        root: The export's root file, as it will be uploaded.
        model: The export's model, ONNX or TensorFlow Lite, as a path or its bytes.
    """
    result = dict(root)
    if result.get(MFLOPS_PER_INVOCATION):
        return result
    try:
        result[MFLOPS_PER_INVOCATION] = count_model_mflops(model, root)
    except (ImportError, OSError):
        # a setup error, not a model the count cannot price
        raise
    except Exception:  # noqa: BLE001 - an uncounted model falls back to the pipeline's default
        log.warning("could not count the model's MFLOPs; the pipeline will use its default", exc_info=True)
    return result


def count_model_mflops(model: Model, root: Mapping[str, Any]) -> int:
    """MFLOPs of one run of an ONNX or TensorFlow Lite model, told apart by its contents.

    Args:
        model: The model, as a path or its bytes.
        root: The export's root file, whose inputs give an ONNX model's input sizes.

    Raises:
        UnresolvedShapeError: a shape the count needs is not known.
    """
    if _is_tflite(model):
        return count_tflite_mflops(model)
    return count_mflops(model, onnx_input_shapes(root))


def onnx_input_shapes(root: Mapping[str, Any]) -> dict[str, list[int]]:
    """The ONNX input shapes a root file feeds its model, by tensor name.

    A root file declares each input tensor's dimensions innermost first (`[width, height,
    channels, batch]`), the reverse of the ONNX order. A dimension of `-1` or `0` is variable
    and counted as one: one run is one batch of one input, with one of each variable dimension
    (as one token of a text input).
    """
    shapes: dict[str, list[int]] = {}
    for model_input in root.get("inputs") or []:
        tensor = model_input.get("tensor") or {}
        name = tensor.get("name")
        dimensions = tensor.get("dimensions")
        if not name or not dimensions:
            continue
        shapes[name] = [1 if d in (-1, 0) else int(d) for d in reversed(dimensions)]
    return shapes


def count_mflops(model: Model, input_shapes: Mapping[str, list[int]]) -> int:
    """MFLOPs of one run of the ONNX model with the given input shapes, rounded up.

    Shapes come from ONNX shape inference. Where it cannot resolve the operands of a
    convolution or matrix product, as in a detection transformer whose shapes depend on
    values, the model runs on the CPU with random inputs. A run reports only the outputs of
    the nodes where inference stopped (the nodes whose inputs it knows) and the values of the
    graph's small integer tensors. Those values are mostly the shape arithmetic inference
    could not follow, such as a Reshape's target behind an If; they become constants, and
    inference carries the shapes on from there. One run usually resolves a model, and a run
    holds little more than an ordinary run of it.

    The weights are written to a temporary directory (under `TMPDIR`) and dropped from memory
    first: inference needs only their shapes, and a run reads them from there. So the count
    holds about one copy of the weights, in the run, and a model over protobuf's 2 GB limit
    counts too.

    Raises:
        UnresolvedShapeError: an input has a dynamic dimension `input_shapes` does not
            declare, or a shape stays unresolved even after the runs.
    """
    from onnx import shape_inference

    with tempfile.TemporaryDirectory(prefix="mflops-") as workdir:
        proto = _load_without_weights(model, workdir)
        _fix_inputs(proto, input_shapes)
        # the values a run has returned: not asked for again
        frozen: set[str] = set()
        for _ in range(_MAX_RUNS):
            inferred = shape_inference.infer_shapes(proto, strict_mode=False, data_prop=True)
            shapes = _known_shapes(inferred)
            if not _unresolved_operands(inferred, shapes):
                return _mflops(inferred, shapes)
            names = sorted(set(_frontier(inferred, shapes)) | _small_integers(proto, inferred, shapes, frozen))
            if not names:
                break
            del inferred
            found = _run_for_shapes(proto, names, input_shapes, workdir)
            _freeze(proto, found)
            frozen |= set(found)
        inferred = shape_inference.infer_shapes(proto, strict_mode=False, data_prop=True)
        # raises UnresolvedShapeError on the first operand still unresolved
        return _mflops(inferred, _known_shapes(inferred))


def _load_without_weights(model: Model, workdir: str) -> Any:
    """The model with its large initializers moved to a file in `workdir`; their shapes stay."""
    import onnx

    proto = onnx.load_from_string(model) if isinstance(model, bytes) else onnx.load(os.fspath(model))
    # writing the weights out clears them from the proto
    onnx.save_model(
        proto,
        os.path.join(workdir, "model.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights",
        size_threshold=_EXTERNAL_DATA_THRESHOLD,
    )
    return proto


def _mflops(model: Any, shapes: Mapping[str, list[int] | None]) -> int:
    return math.ceil(sum(_node_flops(node, shapes) for node in model.graph.node) / 1_000_000)


def count_tflite_mflops(model: Model) -> int:
    """MFLOPs of one run of the TensorFlow Lite model, rounded up.

    A TensorFlow Lite model stores every tensor's shape, so nothing needs inferring. A
    dynamic batch (-1) is stored as 1 in the shape, which is one run.

    Raises:
        UnresolvedShapeError: the model has a tensor with an unknown dimension a counted
            operation needs.
    """
    import tflite

    buffer = model if isinstance(model, bytes) else _read(model)
    parsed = tflite.Model.GetRootAsModel(buffer, 0)
    flops = 0
    for s in range(parsed.SubgraphsLength()):
        subgraph = parsed.Subgraphs(s)
        if subgraph is None:
            continue
        for o in range(subgraph.OperatorsLength()):
            operator = subgraph.Operators(o)
            if operator is None:
                continue
            code = parsed.OperatorCodes(operator.OpcodeIndex())
            if code is None:
                raise UnresolvedShapeError(f"operator {o} names no operator code")
            builtin = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
            flops += _tflite_operator_flops(builtin, operator, subgraph)
    return math.ceil(flops / 1_000_000)


def _read(path: str | os.PathLike[str]) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _is_tflite(model: Model) -> bool:
    # a TensorFlow Lite flatbuffer carries its file identifier at bytes 4 to 8
    if isinstance(model, bytes):
        return model[4:8] == _TFLITE_IDENTIFIER
    with open(model, "rb") as f:
        return f.read(8)[4:8] == _TFLITE_IDENTIFIER


def _fix_inputs(model: Any, input_shapes: Mapping[str, list[int]]) -> None:
    initializers = {i.name for i in model.graph.initializer}
    for graph_input in model.graph.input:
        if graph_input.name in initializers:
            continue
        dims = graph_input.type.tensor_type.shape.dim
        shape = input_shapes.get(graph_input.name)
        if shape is not None:
            if len(shape) != len(dims):
                raise UnresolvedShapeError(
                    f"input '{graph_input.name}' has {len(dims)} dimensions, "
                    f"the ability declares {len(shape)}"
                )
            for dim, value in zip(dims, shape, strict=True):
                dim.ClearField("dim_param")
                dim.dim_value = value
        elif any(not d.HasField("dim_value") or d.dim_value <= 0 for d in dims):
            raise UnresolvedShapeError(
                f"input '{graph_input.name}' has a dynamic dimension the ability does not declare"
            )


def _known_shapes(model: Any) -> dict[str, list[int] | None]:
    shapes: dict[str, list[int] | None] = {}
    for initializer in model.graph.initializer:
        shapes[initializer.name] = list(initializer.dims)
    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        if not value.type.HasField("tensor_type") or not value.type.tensor_type.HasField("shape"):
            # a tensor without a shape has an unknown rank, not no dimensions
            shapes.setdefault(value.name, None)
            continue
        dims = value.type.tensor_type.shape.dim
        if all(d.HasField("dim_value") and d.dim_value > 0 for d in dims):
            shapes[value.name] = [d.dim_value for d in dims]
        else:
            shapes.setdefault(value.name, None)
    return shapes


def _unresolved_operands(model: Any, shapes: Mapping[str, list[int] | None]) -> set[str]:
    missing: set[str] = set()
    for node in model.graph.node:
        if node.op_type == "Einsum":
            names: Iterable[str] = (*node.input, node.output[0])
        else:
            operands = _CONV_OPERANDS.get(node.op_type) or _MATMUL_OPERANDS.get(node.op_type)
            if operands is None:
                continue
            names = (node.input[operands[0]], node.input[operands[1]], node.output[0])
        missing.update(name for name in names if shapes.get(name) is None)
    return missing


def _frontier(model: Any, shapes: Mapping[str, list[int] | None]) -> list[str]:
    """The unknown outputs of the nodes whose inputs are all known: where inference stopped."""
    names: set[str] = set()
    for node in model.graph.node:
        if all(not i or shapes.get(i) is not None for i in node.input):
            names.update(o for o in node.output if o and shapes.get(o) is None)
    return sorted(names)


def _small_integers(
    model: Any, inferred: Any, shapes: Mapping[str, list[int] | None], frozen: set[str]
) -> set[str]:
    """The integer and boolean tensors computed in the graph that may be small: shape arithmetic."""
    import onnx

    integer_types = {onnx.TensorProto.INT64, onnx.TensorProto.INT32, onnx.TensorProto.BOOL}
    initializers = {i.name for i in model.graph.initializer}
    names: set[str] = set()
    for value in inferred.graph.value_info:
        if value.name in initializers or value.name in frozen:
            continue
        if value.type.tensor_type.elem_type not in integer_types:
            continue
        shape = shapes.get(value.name)
        if shape is None or math.prod(shape) <= _SMALL_INTEGER_ELEMENTS:
            names.add(value.name)
    return names


def _freeze(model: Any, found: Mapping[str, Any]) -> None:
    """Puts what a run found into the model.

    A small integer value becomes a constant, and a node whose outputs are all constants now
    is removed. Any other tensor's shape becomes a hint for inference.
    """
    import onnx
    from onnx import numpy_helper

    constants = {
        name: value for name, value in found.items() if value.dtype.kind in "iub" and value.size <= _SMALL_INTEGER_ELEMENTS
    }
    kept = [
        node for node in model.graph.node if not (any(node.output) and all(o in constants for o in node.output if o))
    ]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    produced = {o for node in model.graph.node for o in node.output}
    for name, value in found.items():
        if name in constants and name not in produced:
            model.graph.initializer.append(numpy_helper.from_array(value, name))
        else:
            # an output of a node that is kept, as a Split's beside a float output
            elem_type = onnx.helper.np_dtype_to_tensor_dtype(value.dtype)
            model.graph.value_info.append(onnx.helper.make_tensor_value_info(name, elem_type, list(value.shape)))


def _run_for_shapes(
    model: Any, names: list[str], input_shapes: Mapping[str, list[int]], workdir: str
) -> dict[str, Any]:
    """The values of `names` in one CPU run of the model with random inputs, as numpy arrays.

    The model's weights are in `workdir` (`_load_without_weights`). The model is left as it was.
    """
    import numpy as np
    import onnx
    import onnxruntime

    path = os.path.join(workdir, "run.onnx")
    outputs = len(model.graph.output)
    existing = {o.name for o in model.graph.output}
    model.graph.output.extend(onnx.ValueInfoProto(name=n) for n in names if n not in existing)
    try:
        onnx.save_model(model, path)
    finally:
        del model.graph.output[outputs:]
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    # the run is made once: an arena kept for later runs only adds to its peak
    options.enable_cpu_mem_arena = False
    session = onnxruntime.InferenceSession(path, options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    feeds = {
        graph_input.name: (rng.random(_feed_shape(graph_input, input_shapes)) * 255).astype(
            _numpy_type(graph_input.type)
        )
        for graph_input in session.get_inputs()
    }
    return {name: np.asarray(result) for name, result in zip(names, session.run(names, feeds), strict=True)}


def _feed_shape(graph_input: Any, input_shapes: Mapping[str, list[int]]) -> list[int]:
    shape = input_shapes.get(graph_input.name)
    if shape is not None:
        return shape
    # the inputs were fixed before the run, so an undeclared one is static
    return [int(d) for d in graph_input.shape]


def _numpy_type(onnx_type: str) -> str:
    try:
        return _NUMPY_TYPES[onnx_type]
    except KeyError:
        raise UnresolvedShapeError(f"an input of type {onnx_type} cannot be fed random values") from None


def _shape(name: str, shapes: Mapping[str, list[int] | None], node: Any) -> list[int]:
    shape = shapes.get(name)
    if shape is None:
        raise UnresolvedShapeError(f"{node.op_type} '{node.name}' has an unresolved shape for '{name}'")
    return shape


def _product(values: Iterable[int]) -> int:
    return math.prod(values)


def _attribute(node: Any, name: str, default: int) -> int:
    for attribute in node.attribute:
        if attribute.name == name:
            return int(attribute.i)
    return default


def _node_flops(node: Any, shapes: Mapping[str, list[int] | None]) -> int:
    op = node.op_type
    if node.domain == "com.microsoft" and op in _CONTRIB_ELEMENTWISE_OPS:
        # ONNX has no schema for them, so inference leaves the output unknown: it has the
        # first input's shape
        output = shapes.get(node.output[0]) or shapes.get(node.input[0])
        return _product(output) if output else 0
    if node.domain not in ("", "ai.onnx"):
        # a contrib operator, such as com.microsoft's fused attention, may compute anything
        raise UnresolvedShapeError(f"{node.domain}.{op} '{node.name}' is not an operator the count can price")
    if op in _UNPRICED_OPS:
        raise UnresolvedShapeError(f"{op} '{node.name}' is not an operator the count can price")
    if _subgraphs_compute(node):
        raise UnresolvedShapeError(f"{op} '{node.name}' computes in a subgraph the count does not follow")
    if op in _CONV_OPERANDS:
        source_index, weight_index = _CONV_OPERANDS[op]
        weight = _shape(node.input[weight_index], shapes, node)
        if op == "ConvTranspose":
            source = _shape(node.input[source_index], shapes, node)
            # every input element scatters into (out_channels / group) x kernel outputs
            macs = _product(source) * weight[1] * _product(weight[2:])
        else:
            output = _shape(node.output[0], shapes, node)
            # every output element sums (in_channels / group) x kernel products
            macs = _product(output) * weight[1] * _product(weight[2:])
        return 2 * macs
    if op == "Einsum":
        return _einsum_flops(node, shapes)
    if op in _MATMUL_OPERANDS:
        a = _shape(node.input[_MATMUL_OPERANDS[op][0]], shapes, node)
        output = _shape(node.output[0], shapes, node)
        transposed = op == "Gemm" and _attribute(node, "transA", 0) == 1
        inner = a[-2] if transposed else a[-1]
        return 2 * _product(output) * inner
    if op in _ELEMENTWISE_OPS and node.output:
        # an elementwise operation without a known shape is left out rather than guessed
        if op in _POOLING_OPS:
            source = shapes.get(node.input[0])
            return _product(source) if source else 0
        output = shapes.get(node.output[0])
        return _product(output) if output else 0
    return 0


def _einsum_flops(node: Any, shapes: Mapping[str, list[int] | None]) -> int:
    """One multiply-accumulate per combination of the equation's indices, two FLOPs each.

    A single operand that sums indices costs one FLOP per combination; one that only
    transposes or takes a diagonal costs nothing.
    """
    equation = ""
    for attribute in node.attribute:
        if attribute.name == "equation":
            equation = attribute.s.decode()
    operands_part, arrow, output_part = equation.replace(" ", "").partition("->")
    terms = operands_part.split(",")
    if len(terms) != len(node.input):
        raise UnresolvedShapeError(f"Einsum '{node.name}' has an equation for {len(terms)} operands")
    sizes: dict[str, int] = {}
    seen: list[str] = []
    for term, name in zip(terms, node.input, strict=True):
        shape = _shape(name, shapes, node)
        labels = _einsum_labels(term, len(shape), node)
        seen.extend(labels)
        for label, size in zip(labels, shape, strict=True):
            # an ellipsis dimension broadcasts: 1 beside n is n
            sizes[label] = max(sizes.get(label, 1), size)
    if arrow:
        output_rank = len(_shape(node.output[0], shapes, node))
        kept = set(_einsum_labels(output_part, output_rank, node))
    else:
        # the implicit output keeps the indices that appear once, and the ellipsis
        kept = {label for label in seen if seen.count(label) == 1 or label.startswith(".")}
    combinations = _product(sizes.values())
    if len(terms) > 1:
        return 2 * combinations
    return combinations if set(sizes) - kept else 0


def _einsum_labels(term: str, rank: int, node: Any) -> list[str]:
    """A term's index labels, its ellipsis as one label per dimension, numbered from the right."""
    head, ellipsis, tail = term.partition("...")
    if not ellipsis:
        if len(term) != rank:
            raise UnresolvedShapeError(f"Einsum '{node.name}' has term '{term}' for a rank {rank} operand")
        return list(term)
    covered = rank - len(head) - len(tail)
    if covered < 0:
        raise UnresolvedShapeError(f"Einsum '{node.name}' has term '{term}' for a rank {rank} operand")
    return [*head, *(f".{covered - i}" for i in range(covered)), *tail]


def _subgraphs_compute(node: Any) -> bool:
    """Whether a subgraph of the node (an If's branch, a Loop's or Scan's body) runs a counted operator."""
    import onnx

    for attribute in node.attribute:
        graphs = [attribute.g] if attribute.type == onnx.AttributeProto.GRAPH else list(attribute.graphs)
        for graph in graphs:
            for inner in graph.node:
                if inner.op_type in _COUNTED_OPS or inner.op_type in _UNPRICED_OPS or _subgraphs_compute(inner):
                    return True
    return False


def _tflite_shape(subgraph: Any, index: int) -> list[int]:
    if index < 0:
        raise UnresolvedShapeError("an operator names no tensor")
    tensor = subgraph.Tensors(index)
    # an absent shape reads as no dimensions; it is a scalar only where the rank is known
    shape = [int(tensor.Shape(i)) for i in range(tensor.ShapeLength())]
    if not shape and not tensor.HasRank():
        raise UnresolvedShapeError(f"tensor {index} has an unknown rank")
    if any(d <= 0 for d in shape):
        raise UnresolvedShapeError(f"tensor {index} has an unknown dimension: {shape}")
    return shape


def _tflite_operator_flops(builtin: int, operator: Any, subgraph: Any) -> int:
    import tflite

    ops = tflite.BuiltinOperator
    if builtin == ops.CONV_2D:
        # filter [out, kh, kw, in]: every output element sums kh x kw x in products
        weight = _tflite_shape(subgraph, operator.Inputs(1))
        return 2 * _product(_tflite_shape(subgraph, operator.Outputs(0))) * _product(weight[1:])
    if builtin == ops.DEPTHWISE_CONV_2D:
        # filter [1, kh, kw, out]: every output element sums kh x kw products
        weight = _tflite_shape(subgraph, operator.Inputs(1))
        return 2 * _product(_tflite_shape(subgraph, operator.Outputs(0))) * weight[1] * weight[2]
    if builtin == ops.TRANSPOSE_CONV:
        # inputs are output shape, filter [out, kh, kw, in], input: every input element scatters
        weight = _tflite_shape(subgraph, operator.Inputs(1))
        return 2 * _product(_tflite_shape(subgraph, operator.Inputs(2))) * weight[0] * weight[1] * weight[2]
    if builtin == ops.FULLY_CONNECTED:
        # weights [out, in]: every output element sums `in` products
        weight = _tflite_shape(subgraph, operator.Inputs(1))
        return 2 * _product(_tflite_shape(subgraph, operator.Outputs(0))) * weight[-1]
    if builtin == ops.BATCH_MATMUL:
        # adj_x transposes the left operand's last two dimensions: then it sums a[-2] products
        a = _tflite_shape(subgraph, operator.Inputs(0))
        inner = a[-2] if _tflite_adj_x(operator) else a[-1]
        return 2 * _product(_tflite_shape(subgraph, operator.Outputs(0))) * inner
    if builtin in _tflite_elementwise_ops():
        try:
            return _product(_tflite_shape(subgraph, operator.Outputs(0)))
        except UnresolvedShapeError:
            # an elementwise operation without a known shape is left out rather than guessed
            return 0
    return 0


def _tflite_adj_x(operator: Any) -> bool:
    import tflite

    table = operator.BuiltinOptions()
    if operator.BuiltinOptionsType() != tflite.BuiltinOptions.BatchMatMulOptions or table is None:
        return False
    options = tflite.BatchMatMulOptions()
    options.Init(table.Bytes, table.Pos)
    return bool(options.AdjX())


def _tflite_elementwise_ops() -> frozenset[int]:
    import tflite

    ops = tflite.BuiltinOperator
    return frozenset(
        {
            ops.ADD,
            ops.SUB,
            ops.MUL,
            ops.DIV,
            ops.POW,
            ops.SQRT,
            ops.RSQRT,
            ops.EXP,
            ops.LOG,
            ops.TANH,
            ops.LOGISTIC,
            ops.RELU,
            ops.RELU6,
            ops.RELU_N1_TO_1,
            ops.LEAKY_RELU,
            ops.PRELU,
            ops.HARD_SWISH,
            ops.MAXIMUM,
            ops.MINIMUM,
            ops.NEG,
            ops.ABS,
            ops.SOFTMAX,
            ops.LOG_SOFTMAX,
            ops.MEAN,
            ops.SUM,
            ops.REDUCE_MAX,
            ops.AVERAGE_POOL_2D,
            ops.MAX_POOL_2D,
            ops.RESIZE_BILINEAR,
            ops.RESIZE_NEAREST_NEIGHBOR,
            ops.GELU,
        }
    )
