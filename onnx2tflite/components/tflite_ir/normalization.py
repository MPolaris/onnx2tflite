"""Normalization operators: BatchNorm (inference mode decomposition), InstanceNorm."""
import numpy as np
from tensorflow.lite.python.schema_py_generated import (
    BuiltinOperator as Op,
    MulOptionsT,
    AddOptionsT,
)
from onnx2tflite.components.tflite_ir.builder import Layout
from onnx2tflite.components.tflite_ir.op_mapping import _register


@_register("BatchNormalization")
def _batch_norm(builder, node):
    """Decompose inference-mode BatchNorm into Mul + Add."""
    x_idx = builder._tensor_map[node.input[0]]
    layout = builder.get_layout(x_idx)
    shape = builder._tensors[x_idx].shape

    scale = builder.onnx_weights[node.input[1]]   # gamma
    bias_val = builder.onnx_weights[node.input[2]]   # B
    mean = builder.onnx_weights[node.input[3]]
    var = builder.onnx_weights[node.input[4]]
    eps = 1e-5
    for attr in node.attribute:
        if attr.name == "epsilon":
            eps = attr.f

    # Precompute: new_scale = scale / sqrt(var + eps)
    #             new_bias = bias_val - mean * scale / sqrt(var + eps)
    inv_std = 1.0 / np.sqrt(var + eps)
    new_scale = scale * inv_std
    new_bias = bias_val - mean * new_scale

    # Reshape to broadcast: (C,) → (1, C, 1, 1) for NCHW or (1, 1, 1, C) for NHWC
    if layout == Layout.Channel_Last:
        rs = [1, 1, 1, -1]
    else:
        rs = [1, -1] + [1] * (len(shape) - 2)
    new_scale = new_scale.reshape(rs)
    new_bias = new_bias.reshape(rs)

    scale_idx = builder.register_weight(f"{node.output[0]}_scale", new_scale.astype(np.float32))
    bias_idx = builder.register_weight(f"{node.output[0]}_bias", new_bias.astype(np.float32))

    # x * scale
    mul_opt = MulOptionsT()
    mul_opt.fusedActivationFunction = 0
    mul_out = builder.register_tensor(f"{node.output[0]}_mul", shape)
    builder.add_op(Op.MUL, [x_idx, scale_idx], [mul_out], mul_opt)
    builder.set_layout(mul_out, layout)

    # + bias
    add_opt = AddOptionsT()
    add_opt.fusedActivationFunction = 0
    out = builder.register_tensor(node.output[0], shape)
    builder.add_op(Op.ADD, [mul_out, bias_idx], [out], add_opt)
    builder.set_layout(out, layout)
    return [out]


@_register("InstanceNormalization")
def _instance_norm(builder, node):
    """Decompose InstanceNorm into: (x - mean) / sqrt(var + eps) * scale + bias."""
    x_idx = builder._tensor_map[node.input[0]]
    layout = builder.get_layout(x_idx)
    shape = builder._tensors[x_idx].shape

    scale_data = builder.onnx_weights[node.input[1]]
    bias_data = builder.onnx_weights[node.input[2]]
    eps = 1e-5
    for attr in node.attribute:
        if attr.name == "epsilon":
            eps = attr.f

    # Reshape scale/bias for broadcasting
    if layout == Layout.Channel_Last:
        rs = [1, 1, 1, -1]
    else:
        rs = [1, -1] + [1] * (len(shape) - 2)

    scale_idx = builder.register_weight(f"{node.output[0]}_scale", scale_data.reshape(rs).astype(np.float32))
    bias_idx = builder.register_weight(f"{node.output[0]}_bias", bias_data.reshape(rs).astype(np.float32))
    eps_idx = builder.register_weight(f"{node.output[0]}_eps", np.array([eps], dtype=np.float32).reshape([1] * len(shape)))

    from tensorflow.lite.python.schema_py_generated import ReducerOptionsT, SubOptionsT, DivOptionsT, MulOptionsT
    # mean
    ropt = ReducerOptionsT()
    ropt.keepDims = True
    mean_out = builder.register_tensor(f"{node.output[0]}_mean", shape)
    builder.add_op(Op.MEAN, [x_idx], [mean_out], ropt)

    # x - mean
    sub_opt = SubOptionsT()
    sub_opt.fusedActivationFunction = 0
    diff_out = builder.register_tensor(f"{node.output[0]}_diff", shape)
    builder.add_op(Op.SUB, [x_idx, mean_out], [diff_out], sub_opt)

    # var = mean((x - mean)^2)
    sq_out = builder.register_tensor(f"{node.output[0]}_sq", shape)
    builder.add_op(Op.MUL, [diff_out, diff_out], [sq_out],
                   MulOptionsT())
    var_out = builder.register_tensor(f"{node.output[0]}_var", shape)
    builder.add_op(Op.MEAN, [sq_out], [var_out], ReducerOptionsT())

    # var + eps
    add1_out = builder.register_tensor(f"{node.output[0]}_vareps", shape)
    builder.add_op(Op.ADD, [var_out, eps_idx], [add1_out], AddOptionsT())

    # rsqrt(var + eps)
    rsqrt_out = builder.register_tensor(f"{node.output[0]}_rsqrt", shape)
    builder.add_op(Op.RSQRT, [add1_out], [rsqrt_out])

    # (x - mean) * rsqrt * scale
    norm_out = builder.register_tensor(f"{node.output[0]}_norm", shape)
    builder.add_op(Op.MUL, [diff_out, rsqrt_out], [norm_out], MulOptionsT())

    scaled_out = builder.register_tensor(f"{node.output[0]}_scaled", shape)
    builder.add_op(Op.MUL, [norm_out, scale_idx], [scaled_out], MulOptionsT())

    # + bias
    add_opt = AddOptionsT()
    add_opt.fusedActivationFunction = 0
    out = builder.register_tensor(node.output[0], shape)
    builder.add_op(Op.ADD, [scaled_out, bias_idx], [out], add_opt)
    builder.set_layout(out, layout)
    return [out]
