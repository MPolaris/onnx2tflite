"""Decomposed operators — ONNX ops without direct TFLite builtin counterparts."""
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema
from tensorflow.lite.python.schema_py_generated import (
    BuiltinOperator as Op,
    AddOptionsT,
    MulOptionsT,
    DivOptionsT,
    SubOptionsT,
    MaximumMinimumOptionsT,
    FloorModOptionsT,
)
from onnx2tflite.tflite_backend.builder import Layout
from onnx2tflite.tflite_backend.op_mapping import _register


@_register("HardSigmoid")
def _hard_sigmoid(builder, node):
    """y = max(0, min(1, alpha*x + beta)). alpha=0.2, beta=0.5."""
    attrs = {}
    for attr in node.attribute:
        if attr.name == "alpha":
            attrs["alpha"] = attr.f
        elif attr.name == "beta":
            attrs["beta"] = attr.f
    alpha = attrs.get("alpha", 0.2)
    beta = attrs.get("beta", 0.5)

    x_idx = builder._tensor_map[node.input[0]]
    shape = builder._tensors[x_idx].shape
    layout = builder.get_layout(x_idx)

    # alpha*x
    alpha_data = np.full((1,), alpha, dtype=np.float32)
    alpha_idx = builder.register_weight(f"{node.output[0]}_alpha", alpha_data)
    mul_out = builder.register_tensor(f"{node.output[0]}_mul", shape)
    builder.add_op(Op.MUL, [x_idx, alpha_idx], [mul_out], MulOptionsT())

    # alpha*x + beta
    beta_data = np.full((1,), beta, dtype=np.float32)
    beta_idx = builder.register_weight(f"{node.output[0]}_beta", beta_data)
    add_out = builder.register_tensor(f"{node.output[0]}_add", shape)
    builder.add_op(Op.ADD, [mul_out, beta_idx], [add_out], AddOptionsT())

    # min(1, ...)
    one_data = np.full((1,), 1.0, dtype=np.float32)
    one_idx = builder.register_weight(f"{node.output[0]}_one", one_data)
    min_out = builder.register_tensor(f"{node.output[0]}_min", shape)
    builder.add_op(Op.MINIMUM, [add_out, one_idx], [min_out], MaximumMinimumOptionsT())

    # max(0, ...)
    zero_data = np.full((1,), 0.0, dtype=np.float32)
    zero_idx = builder.register_weight(f"{node.output[0]}_zero", zero_data)
    out = builder.register_tensor(node.output[0], shape)
    builder.add_op(Op.MAXIMUM, [min_out, zero_idx], [out], MaximumMinimumOptionsT())
    builder.set_layout(out, layout)
    return [out]


@_register("Celu")
def _celu(builder, node):
    """y = max(0, x) + min(0, alpha*(exp(x/alpha)-1))."""
    attrs = {}
    for attr in node.attribute:
        if attr.name == "alpha":
            attrs["alpha"] = attr.f
    alpha_val = attrs.get("alpha", 1.0)

    x_idx = builder._tensor_map[node.input[0]]
    shape = builder._tensors[x_idx].shape
    layout = builder.get_layout(x_idx)

    alpha_data = np.full((1,), alpha_val, dtype=np.float32)
    alpha_idx = builder.register_weight(f"{node.output[0]}_alpha", alpha_data)
    one_idx = builder.register_weight(f"{node.output[0]}_one", np.float32(1.0))
    zero_idx = builder.register_weight(f"{node.output[0]}_zero", np.float32(0.0))

    # max(0, x)
    max_out = builder.register_tensor(f"{node.output[0]}_max", shape)
    builder.add_op(Op.MAXIMUM, [x_idx, zero_idx], [max_out], MaximumMinimumOptionsT())

    # x/alpha
    div_out = builder.register_tensor(f"{node.output[0]}_div", shape)
    builder.add_op(Op.DIV, [x_idx, alpha_idx], [div_out], DivOptionsT())

    # exp(x/alpha)
    exp_out = builder.register_tensor(f"{node.output[0]}_exp", shape)
    builder.add_op(Op.EXP, [div_out], [exp_out])

    # exp(x/alpha) - 1
    sub_out = builder.register_tensor(f"{node.output[0]}_sub", shape)
    builder.add_op(Op.SUB, [exp_out, one_idx], [sub_out], SubOptionsT())

    # alpha * (...)
    mul_out = builder.register_tensor(f"{node.output[0]}_mul", shape)
    builder.add_op(Op.MUL, [sub_out, alpha_idx], [mul_out], MulOptionsT())

    # min(0, ...)
    min_out = builder.register_tensor(f"{node.output[0]}_min", shape)
    builder.add_op(Op.MINIMUM, [zero_idx, mul_out], [min_out], MaximumMinimumOptionsT())

    # max(0,x) + min(0,...)
    out = builder.register_tensor(node.output[0], shape)
    builder.add_op(Op.ADD, [max_out, min_out], [out], AddOptionsT())
    builder.set_layout(out, layout)
    return [out]


@_register("Mish")
def _mish(builder, node):
    """y = x * tanh(softplus(x)) = x * tanh(log(exp(x)+1))."""
    x_idx = builder._tensor_map[node.input[0]]
    shape = builder._tensors[x_idx].shape
    layout = builder.get_layout(x_idx)

    one_idx = builder.register_weight(f"{node.output[0]}_one", np.float32(1.0))

    # exp(x)
    exp_out = builder.register_tensor(f"{node.output[0]}_exp", shape)
    builder.add_op(Op.EXP, [x_idx], [exp_out])

    # exp(x) + 1
    add_out = builder.register_tensor(f"{node.output[0]}_add1", shape)
    builder.add_op(Op.ADD, [exp_out, one_idx], [add_out], AddOptionsT())

    # log(exp(x)+1) = softplus(x)
    sp_out = builder.register_tensor(f"{node.output[0]}_sp", shape)
    builder.add_op(Op.LOG, [add_out], [sp_out])

    # tanh(softplus(x))
    tanh_out = builder.register_tensor(f"{node.output[0]}_tanh", shape)
    builder.add_op(Op.TANH, [sp_out], [tanh_out])

    # x * tanh(...)
    out = builder.register_tensor(node.output[0], shape)
    builder.add_op(Op.MUL, [x_idx, tanh_out], [out], MulOptionsT())
    builder.set_layout(out, layout)
    return [out]


@_register("Softplus")
def _softplus(builder, node):
    """y = log(exp(x) + 1)."""
    x_idx = builder._tensor_map[node.input[0]]
    shape = builder._tensors[x_idx].shape
    layout = builder.get_layout(x_idx)

    one_idx = builder.register_weight(f"{node.output[0]}_one", np.float32(1.0))
    exp_out = builder.register_tensor(f"{node.output[0]}_exp", shape)
    builder.add_op(Op.EXP, [x_idx], [exp_out])
    add_out = builder.register_tensor(f"{node.output[0]}_add", shape)
    builder.add_op(Op.ADD, [exp_out, one_idx], [add_out], AddOptionsT())
    out = builder.register_tensor(node.output[0], shape)
    builder.add_op(Op.LOG, [add_out], [out])
    builder.set_layout(out, layout)
    return [out]


@_register("Softsign")
def _softsign(builder, node):
    """y = x / (1 + |x|)."""
    x_idx = builder._tensor_map[node.input[0]]
    shape = builder._tensors[x_idx].shape
    layout = builder.get_layout(x_idx)

    one_idx = builder.register_weight(f"{node.output[0]}_one", np.float32(1.0))

    # |x|
    abs_out = builder.register_tensor(f"{node.output[0]}_abs", shape)
    builder.add_op(Op.ABS, [x_idx], [abs_out])

    # 1 + |x|
    add_out = builder.register_tensor(f"{node.output[0]}_add", shape)
    builder.add_op(Op.ADD, [one_idx, abs_out], [add_out], AddOptionsT())

    # x / (1+|x|)
    out = builder.register_tensor(node.output[0], shape)
    builder.add_op(Op.DIV, [x_idx, add_out], [out], DivOptionsT())
    builder.set_layout(out, layout)
    return [out]


@_register("Selu")
def _selu(builder, node):
    """Keras decomposition (5 ops):
    SELU(x) = SELECT(x > 0, lambda * x, lambda * (exp(x) - alpha))
    where lambda=1.050700987, alpha=1.673263242 (ONNX defaults)"""
    attrs = {}
    for attr in node.attribute:
        if attr.name == "alpha": attrs["alpha"] = attr.f
        elif attr.name == "gamma": attrs["gamma"] = attr.f
    alpha_val = attrs.get("alpha", 1.673263242)
    gamma_val = attrs.get("gamma", 1.050700987)

    x_idx = builder._tensor_map[node.input[0]]
    shape = builder._tensors[x_idx].shape
    layout = builder.get_layout(x_idx)

    alpha = builder.register_weight(f"{node.output[0]}_alpha", np.float32(alpha_val))
    lmbda = builder.register_weight(f"{node.output[0]}_lambda", np.float32(gamma_val))
    zero = builder.register_weight(f"{node.output[0]}_zero", np.float32(0.0))

    # mask = x > 0
    mask = builder.register_tensor(f"{node.output[0]}_mask", shape, dtype=9)  # BOOL
    builder.add_op(Op.GREATER, [x_idx, zero], [mask])

    # positive branch: lambda * x
    pos = builder.register_tensor(f"{node.output[0]}_pos", shape)
    builder.add_op(Op.MUL, [x_idx, lmbda], [pos], MulOptionsT())

    # negative branch: lambda * (exp(x) - alpha)
    exp_out = builder.register_tensor(f"{node.output[0]}_exp", shape)
    builder.add_op(Op.EXP, [x_idx], [exp_out])
    sub_out = builder.register_tensor(f"{node.output[0]}_sub", shape)
    builder.add_op(Op.SUB, [exp_out, alpha], [sub_out], SubOptionsT())
    neg = builder.register_tensor(f"{node.output[0]}_neg", shape)
    builder.add_op(Op.MUL, [sub_out, lmbda], [neg], MulOptionsT())

    # SELECT(mask, pos, neg)
    out = builder.register_tensor(node.output[0], shape)
    builder.add_op(Op.SELECT, [mask, pos, neg], [out])
    builder.set_layout(out, layout)
    return [out]


@_register("Erf")
def _erf(builder, node):
    """Erf has no native TFLite op. Flex op ('FlexErf') requires SELECT_TF_OPS
    delegate which is not available in the default interpreter."""
    raise NotImplementedError("Erf not supported in TFLite — no builtin op available")


@_register("Clip")
def _clip(builder, node):
    """Clip: min(max(x, min_val), max_val)."""
    attrs = {}
    for attr in node.attribute:
        if attr.name == "min":
            attrs["min"] = attr.f
        elif attr.name == "max":
            attrs["max"] = attr.f
    # Also try reading min/max from inputs (opset 11+)
    if "min" not in attrs and len(node.input) > 1 and node.input[1] in builder.onnx_weights:
        attrs["min"] = float(builder.onnx_weights[node.input[1]])
    if "max" not in attrs and len(node.input) > 2 and node.input[2] in builder.onnx_weights:
        attrs["max"] = float(builder.onnx_weights[node.input[2]])
    min_val = attrs.get("min", None)
    max_val = attrs.get("max", None)

    x_idx = builder._tensor_map[node.input[0]]
    shape = builder._tensors[x_idx].shape
    layout = builder.get_layout(x_idx)

    current_out = x_idx
    if min_val is not None:
        min_data = np.full((1,), float(min_val), dtype=np.float32)
        min_idx = builder.register_weight(f"{node.output[0]}_min", min_data)
        tmp = builder.register_tensor(f"{node.output[0]}_maxout", shape)
        builder.add_op(Op.MAXIMUM, [current_out, min_idx], [tmp], MaximumMinimumOptionsT())
        current_out = tmp

    if max_val is not None:
        max_data = np.full((1,), float(max_val), dtype=np.float32)
        max_idx = builder.register_weight(f"{node.output[0]}_max", max_data)
        out = builder.register_tensor(node.output[0], shape)
        builder.add_op(Op.MINIMUM, [current_out, max_idx], [out], MaximumMinimumOptionsT())
        current_out = out

    builder.set_layout(current_out, layout)
    return [current_out]


@_register("Constant")
def _constant(builder, node):
    """Constant: store value in buffer and return tensor index."""
    for attr in node.attribute:
        if attr.name == "value":
            from onnx import numpy_helper
            data = numpy_helper.to_array(attr.t)
            idx = builder.register_weight(node.output[0], data)
            return [idx]
    raise ValueError("Constant node has no 'value' attribute")


@_register("Mod")
def _mod(builder, node):
    """Mod/fmod: floating point mod."""
    x_idx = builder._tensor_map[node.input[0]]
    shape = builder._tensors[x_idx].shape

    # Get divisor
    mod_val = None
    if len(node.input) > 1 and node.input[1] in builder.onnx_weights:
        mod_val = builder.onnx_weights[node.input[1]]
    for attr in node.attribute:
        if attr.name == "fmod":
            fmod = attr.i

    if mod_val is not None:
        mod_idx = builder.register_weight(f"{node.output[0]}_modval", mod_val.astype(np.float32))
    else:
        mod_idx = builder._tensor_map[node.input[1]]

    opt = FloorModOptionsT()
    out = builder.register_tensor(node.output[0], shape)
    builder.add_op(Op.FLOOR_MOD, [x_idx, mod_idx], [out], opt)
    builder.set_layout(out, builder.get_layout(x_idx))
    return [out]
