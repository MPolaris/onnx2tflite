"""
TFLite FlatBuffer IR builder — constructs .tflite models directly from ONNX graphs.

Uses TF's built-in schema_py_generated module (FlatBuffers Object API) to build
TFLite models without going through Keras or the TF Lite converter.
"""

import flatbuffers
import numpy as np
from onnx import numpy_helper

from tensorflow.lite.python import schema_py_generated as schema


class Layout:
    """Tensor layout tracking."""
    Channel_First = 1   # NCHW / ONNX native
    Channel_Last = 2    # NHWC / TFLite native
    Channel_None = 3    # No spatial dims (1D/2D tensors)


# ONNX TensorProto element type → TFLite TensorType
_ONNX_TO_TFLITE_DTYPE = {
    1:  schema.TensorType.FLOAT32,
    6:  schema.TensorType.INT32,
    7:  schema.TensorType.INT64,
    9:  schema.TensorType.BOOL,
    10: schema.TensorType.FLOAT16,
    2:  schema.TensorType.UINT8,
    3:  schema.TensorType.INT8,
}


class TFLiteBuilder:
    """
    Builds a TFLite FlatBuffer model by walking an ONNX graph.

    Manages:
      - Tensor registry (name -> tensor_index in subgraph)
      - Buffer registry (weight data stored as flatbuffer buffers)
      - OperatorCode dedup (same builtin_code reused)
      - Layout tracking (which tensors are NCHW vs NHWC)
      - Auto-insertion of Transpose nodes for layout conversion
    """

    def __init__(self, model_proto):
        self.model_graph = model_proto.graph

        # Tensor registry: name -> tensor index in subgraph
        self._tensor_map: dict[str, int] = {}
        # Layout for each tensor index
        self._layout: dict[int, int] = {}

        # Buffers: buffer[0] is always empty (for intermediate tensors)
        self._buffers: list = [None]
        # Subgraph tensors
        self._tensors: list[schema.TensorT] = []
        # Subgraph operators
        self._operators: list[schema.OperatorT] = []

        # OperatorCode dedup: builtin_code -> opcode_index
        self._opcode_map: dict[int, int] = {}
        self._operator_codes: list[schema.OperatorCodeT] = []

        # Subgraph inputs/outputs (tensor indices)
        self._inputs: list[int] = []
        self._outputs: list[int] = []

        # Load ONNX initializers
        self.onnx_weights: dict[str, np.ndarray] = {}
        for init in model_proto.graph.initializer:
            self.onnx_weights[init.name] = numpy_helper.to_array(init)

        # Build graph inputs
        self._build_inputs()

    # ---- Public API ----

    def register_tensor(self, name: str, shape, dtype=0, layout=None) -> int:
        """Register a new tensor and return its index. Deduplicates by name.

        Args:
            dtype: TFLite TensorType code (0=FLOAT32, 2=INT32, 4=INT64)
        """
        if name in self._tensor_map:
            return self._tensor_map[name]

        tflite_dtype = dtype
        shape_list = [int(d) for d in shape] if shape else []

        quant = schema.QuantizationParametersT()
        tensor = schema.TensorT()
        tensor.name = name.encode() if isinstance(name, str) else name
        tensor.shape = shape_list
        tensor.type = tflite_dtype
        tensor.buffer = 0  # default empty buffer
        tensor.quantization = quant
        tensor.isVariable = False

        idx = len(self._tensors)
        self._tensors.append(tensor)
        self._tensor_map[name] = idx
        if layout is not None:
            self._layout[idx] = layout
        return idx

    def register_weight(self, name: str, data: np.ndarray) -> int:
        """Register a weight tensor (stored in flatbuffer buffer)."""
        # Determine TFLite dtype from numpy
        np_to_tfl = {
            np.dtype('float32'): schema.TensorType.FLOAT32,
            np.dtype('float64'): schema.TensorType.FLOAT32,  # downcast
            np.dtype('int32'): schema.TensorType.INT32,
            np.dtype('int64'): schema.TensorType.INT64,
            np.dtype('bool'): schema.TensorType.BOOL,
            np.dtype('float16'): schema.TensorType.FLOAT16,
        }
        tfl_dtype = np_to_tfl.get(data.dtype, schema.TensorType.FLOAT32)
        idx = self.register_tensor(name, data.shape, dtype=tfl_dtype)

        buf_idx = len(self._buffers)
        self._buffers.append(data.tobytes())
        self._tensors[idx].buffer = buf_idx

        # Weights start in ONNX NCHW format
        self._layout[idx] = Layout.Channel_First
        return idx

    def add_op(self, builtin_code: int, inputs: list[int], outputs: list[int],
               options=None, custom_code: str = None):
        """Add an operator to the subgraph. Returns output tensor indices."""
        op = schema.OperatorT()
        op.opcodeIndex = self._get_opcode_index(builtin_code, custom_code)
        op.inputs = [int(i) for i in inputs]
        op.outputs = [int(o) for o in outputs]
        if options is not None:
            op.builtinOptionsType = _OPTIONS_TYPE_MAP.get(type(options), 0)
            op.builtinOptions = options
        self._operators.append(op)
        return outputs

    def mark_output(self, tensor_idx: int):
        """Mark a tensor as graph output."""
        if tensor_idx not in self._outputs:
            self._outputs.append(tensor_idx)

    # ---- Layout Management ----

    def get_layout(self, tensor_idx: int) -> int:
        return self._layout.get(tensor_idx, Layout.Channel_First)

    def set_layout(self, tensor_idx: int, layout: int):
        self._layout[tensor_idx] = layout

    def ensure_nhwc(self, tensor_idx: int) -> int:
        """Insert Transpose NCHW→NHWC if needed. Returns (possibly new) tensor index."""
        if self._layout.get(tensor_idx) == Layout.Channel_Last:
            return tensor_idx

        shape = self._tensors[tensor_idx].shape
        if len(shape) <= 2:
            self._layout[tensor_idx] = Layout.Channel_None
            return tensor_idx

        # NCHW → NHWC: [0, 2, 3, ..., 1]
        perm = [0] + list(range(2, len(shape))) + [1]
        nhwc_shape = [shape[0]] + list(shape[2:]) + [shape[1]]

        # Transpose needs perm as second input (as int32 tensor)
        perm_name = f"{self._tensors[tensor_idx].name.decode()}_perm_nhwc"
        perm_idx = self.register_weight(perm_name, np.array(perm, dtype=np.int32))

        out_name = f"{self._tensors[tensor_idx].name.decode()}_nhwc"
        out_idx = self.register_tensor(out_name, nhwc_shape, layout=Layout.Channel_Last)

        self.add_op(schema.BuiltinOperator.TRANSPOSE, [tensor_idx, perm_idx], [out_idx])
        self._layout[out_idx] = Layout.Channel_Last
        return out_idx

    def ensure_nchw(self, tensor_idx: int) -> int:
        """Insert Transpose NHWC→NCHW if needed. Returns (possibly new) tensor index."""
        if self._layout.get(tensor_idx) == Layout.Channel_First:
            return tensor_idx

        shape = self._tensors[tensor_idx].shape
        if len(shape) <= 2:
            self._layout[tensor_idx] = Layout.Channel_None
            return tensor_idx

        # NHWC → NCHW: [0, -1, 1, 2, ..., -2]
        perm = [0, len(shape) - 1] + list(range(1, len(shape) - 1))
        nchw_shape = [shape[0], shape[-1]] + list(shape[1:-1])

        perm_name = f"{self._tensors[tensor_idx].name.decode()}_perm_nchw"
        perm_idx = self.register_weight(perm_name, np.array(perm, dtype=np.int32))

        out_name = f"{self._tensors[tensor_idx].name.decode()}_nchw"
        out_idx = self.register_tensor(out_name, nchw_shape, layout=Layout.Channel_First)

        self.add_op(schema.BuiltinOperator.TRANSPOSE, [tensor_idx, perm_idx], [out_idx])
        self._layout[out_idx] = Layout.Channel_First
        return out_idx

    # ---- Serialization ----

    def build(self) -> bytes:
        """Serialize the model to TFLite FlatBuffer bytes."""
        # buffer[0] is always empty (pre-allocated in __init__)
        buffer_objs = []
        for b in self._buffers:
            buf = schema.BufferT()
            buf.data = b
            buffer_objs.append(buf)

        subgraph = schema.SubGraphT()
        subgraph.name = b"main"
        subgraph.tensors = self._tensors
        subgraph.inputs = self._inputs
        subgraph.outputs = self._outputs
        subgraph.operators = self._operators

        model = schema.ModelT()
        model.version = 3
        model.subgraphs = [subgraph]
        model.operatorCodes = self._operator_codes
        model.description = b"onnx2tflite direct IR"
        model.buffers = buffer_objs

        builder = flatbuffers.Builder(4096)
        model_offset = model.Pack(builder)
        builder.Finish(model_offset, b"TFL3")
        return bytes(builder.Output())

    # ---- Internal ----

    def _get_opcode_index(self, builtin_code: int, custom_code: str = None) -> int:
        """Get or create an OperatorCode entry."""
        if builtin_code in self._opcode_map:
            return self._opcode_map[builtin_code]

        oc = schema.OperatorCodeT()
        if custom_code:
            oc.deprecatedBuiltinCode = schema.BuiltinOperator.CUSTOM
            oc.customCode = custom_code
        else:
            oc.builtinCode = builtin_code
            oc.deprecatedBuiltinCode = builtin_code
        oc.version = 1

        idx = len(self._operator_codes)
        self._operator_codes.append(oc)
        self._opcode_map[builtin_code] = idx
        return idx

    def _build_inputs(self):
        """Create TFLite input tensors from ONNX graph inputs."""
        for inp in self.model_graph.input:
            shape = [d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim]
            if not shape:
                continue

            # Convert ONNX elem_type to TFLite TensorType
            onnx_dtype = inp.type.tensor_type.elem_type
            tfl_dtype = _ONNX_TO_TFLITE_DTYPE.get(onnx_dtype, schema.TensorType.FLOAT32)

            if len(shape) >= 3:
                layout = Layout.Channel_First
            else:
                layout = Layout.Channel_None

            idx = self.register_tensor(inp.name, shape, dtype=tfl_dtype, layout=layout)
            self._inputs.append(idx)


# Map OptionsT type to its builtinOptionsType enum value
_OPTIONS_TYPE_MAP = {
    schema.Conv2DOptionsT: 1,
    schema.DepthwiseConv2DOptionsT: 2,
    schema.ConcatEmbeddingsOptionsT: 3,
    schema.LSHProjectionOptionsT: 4,
    schema.Pool2DOptionsT: 5,
    schema.SVDFOptionsT: 6,
    schema.RNNOptionsT: 7,
    schema.FullyConnectedOptionsT: 8,
    schema.SoftmaxOptionsT: 9,
    schema.ConcatenationOptionsT: 10,
    schema.AddOptionsT: 11,
    schema.L2NormOptionsT: 12,
    schema.LocalResponseNormalizationOptionsT: 13,
    schema.LSTMOptionsT: 14,
    schema.ResizeBilinearOptionsT: 15,
    schema.CallOptionsT: 16,
    schema.ReshapeOptionsT: 17,
    schema.SkipGramOptionsT: 18,
    schema.SpaceToDepthOptionsT: 19,
    schema.EmbeddingLookupSparseOptionsT: 20,
    schema.MulOptionsT: 21,
    schema.PadOptionsT: 22,
    schema.GatherOptionsT: 23,
    schema.BatchToSpaceNDOptionsT: 24,
    schema.SpaceToBatchNDOptionsT: 25,
    schema.TransposeOptionsT: 26,
    schema.ReducerOptionsT: 27,
    schema.SubOptionsT: 28,
    schema.DivOptionsT: 29,
    schema.SqueezeOptionsT: 30,
    schema.SequenceRNNOptionsT: 31,
    schema.StridedSliceOptionsT: 32,
    schema.ExpOptionsT: 33,
    schema.TopKV2OptionsT: 34,
    schema.SplitOptionsT: 35,
    schema.LogSoftmaxOptionsT: 36,
    schema.CastOptionsT: 37,
    schema.DequantizeOptionsT: 38,
    schema.MaximumMinimumOptionsT: 39,
    schema.ArgMaxOptionsT: 40,
    schema.LessOptionsT: 41,
    schema.NegOptionsT: 42,
    schema.PadV2OptionsT: 43,
    schema.GreaterOptionsT: 44,
    schema.GreaterEqualOptionsT: 45,
    schema.LessEqualOptionsT: 46,
    schema.SelectOptionsT: 47,
    schema.SliceOptionsT: 48,
    schema.TransposeConvOptionsT: 49,
    schema.SparseToDenseOptionsT: 50,
    schema.TileOptionsT: 51,
    schema.ExpandDimsOptionsT: 52,
    schema.EqualOptionsT: 53,
    schema.NotEqualOptionsT: 54,
    schema.ShapeOptionsT: 55,
    schema.PowOptionsT: 56,
    schema.ArgMinOptionsT: 57,
    schema.FakeQuantOptionsT: 58,
    schema.PackOptionsT: 59,
    schema.LogicalOrOptionsT: 60,
    schema.OneHotOptionsT: 61,
    schema.LogicalAndOptionsT: 62,
    schema.LogicalNotOptionsT: 63,
    schema.UnpackOptionsT: 64,
    schema.FloorDivOptionsT: 65,
    schema.SquareOptionsT: 66,
    schema.ZerosLikeOptionsT: 67,
    schema.FillOptionsT: 68,
    schema.BidirectionalSequenceLSTMOptionsT: 69,
    schema.BidirectionalSequenceRNNOptionsT: 70,
    schema.UnidirectionalSequenceLSTMOptionsT: 71,
    schema.FloorModOptionsT: 72,
    schema.RangeOptionsT: 73,
    schema.ResizeNearestNeighborOptionsT: 74,
    schema.LeakyReluOptionsT: 75,
    schema.SquaredDifferenceOptionsT: 76,
    schema.MirrorPadOptionsT: 77,
    schema.AbsOptionsT: 78,
    schema.SplitVOptionsT: 79,
    schema.UniqueOptionsT: 80,
    schema.ReverseV2OptionsT: 81,
    schema.AddNOptionsT: 82,
    schema.GatherNdOptionsT: 83,
    schema.CosOptionsT: 84,
    schema.WhereOptionsT: 85,
    schema.RankOptionsT: 86,
    schema.ReverseSequenceOptionsT: 88,
    schema.MatrixDiagOptionsT: 89,
    schema.QuantizeOptionsT: 90,
    schema.MatrixSetDiagOptionsT: 91,
    schema.HardSwishOptionsT: 92,
    schema.IfOptionsT: 93,
    schema.WhileOptionsT: 94,
    schema.DepthToSpaceOptionsT: 95,
    schema.NonMaxSuppressionV4OptionsT: 96,
    schema.NonMaxSuppressionV5OptionsT: 97,
    schema.ScatterNdOptionsT: 98,
    schema.SelectV2OptionsT: 99,
    schema.DensifyOptionsT: 100,
    schema.SegmentSumOptionsT: 101,
    schema.BatchMatMulOptionsT: 102,
    schema.CumsumOptionsT: 103,
    schema.CallOnceOptionsT: 104,
    schema.BroadcastToOptionsT: 105,
    schema.Rfft2dOptionsT: 106,
    schema.Conv3DOptionsT: 107,
    schema.DynamicUpdateSliceOptionsT: 108,
    schema.UnsortedSegmentProdOptionsT: 109,
    schema.UnsortedSegmentMaxOptionsT: 110,
    schema.UnsortedSegmentSumOptionsT: 111,
    schema.ATan2OptionsT: 112,
    schema.UnsortedSegmentMinOptionsT: 113,
    schema.SignOptionsT: 114,
    schema.HashtableOptionsT: 118,
    schema.HashtableFindOptionsT: 119,
    schema.HashtableImportOptionsT: 120,
    schema.HashtableSizeOptionsT: 121,
    schema.VarHandleOptionsT: 122,
    schema.ReadVariableOptionsT: 123,
    schema.AssignVariableOptionsT: 124,
    schema.RandomOptionsT: 125,
    schema.BucketizeOptionsT: 126,
    schema.GeluOptionsT: 127,
}
