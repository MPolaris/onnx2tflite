import tensorflow as tf
from abc import ABC
from enum import Enum, unique

@unique
class Layout(Enum):
    Default = 0
    Channel_First = 1 << 0# for onnx format
    Channel_Last = 1 << 1 # for tensorflow format
    Channel_None = 1 << 2 # no channel

class Node_Layout:
    def __init__(self, name:str, pre:list=[], nxt:list=[]) -> None:
        self.name = name
        self.pre = pre
        self.nxt = nxt
        self.layout = Layout.Default

class BaseOP(ABC):
    def __init__(self, tensor_graph, const_weights, node_attributes, node_inputs, node_outputs, layout_dict) -> None:
        pass

onnx2tf_type = {
    1: tf.float32,   # ONNX_FLOAT
    2: tf.uint8,     # ONNX_UINT8
    3: tf.int8,      # ONNX_INT8
    4: tf.uint16,    # ONNX_UINT16
    5: tf.int16,     # ONNX_INT16
    6: tf.int32,     # ONNX_INT32
    7: tf.int64,     # ONNX_INT64
    8: tf.string,    # ONNX_STRING
    9: tf.bool,      # ONNX_BOOL
    10: tf.float16,  # ONNX_FLOAT16
    11: tf.float64,  # ONNX_DOUBLE
    12: tf.uint32,   # ONNX_UINT32
    13: tf.uint64,   # ONNX_UINT64
    14: tf.complex64,  # ONNX_COMPLEX64
    15: tf.complex128 # ONNX_COMPLEX128
}

np2tf_type = {
    "int32": tf.int32,
    "int64": tf.int64,
    "float32": tf.float32,
    "float64": tf.float64,
    "bool": tf.bool,
    "uint8": tf.uint8,
    "int8": tf.int8,
    "int16": tf.int16,
    "uint16": tf.uint16,
    "uint32": tf.uint32,
    "uint64": tf.uint64,
    "complex64": tf.complex64,
    "complex128": tf.complex128
}

FORCE_CHANNEL_LAST_OP = ["Conv", "ConvTranspose", "DepthToSpace", "Pad", "AveragePool", "MaxPool", "Upsample", "Resize", "Gemm"]
FORCE_CHANNEL_FIRST_OP = ["Reshape", "Transpose", "ScatterND", "MatMul"]

