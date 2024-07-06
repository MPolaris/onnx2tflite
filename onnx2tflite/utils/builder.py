import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from onnx import numpy_helper
from .op_registry import OPERATOR
from .dataloader import RandomLoader, ImageLoader

from onnx2tflite.layers import conv_layers

# copy from https://github.com/gmalivenko/onnx2keras
def decode_node_attribute(node)->dict:
    """
    Parse ONNX attributes to Python dictionary
    :param args: ONNX attributes object
    :return: Python dictionary
    """
    def onnx_attribute_to_dict(onnx_attr):
        """
        Parse ONNX attribute
        :param onnx_attr: ONNX attribute
        :return: Python data type
        """
        if onnx_attr.HasField('t'):
            return numpy_helper.to_array(getattr(onnx_attr, 't'))

        for attr_type in ['f', 'i']:
            if onnx_attr.HasField(attr_type):
                return getattr(onnx_attr, attr_type)

        # s need to be decode, bytes to string
        if onnx_attr.HasField('s'):
            return getattr(onnx_attr, 's').decode()

        for attr_type in ['floats', 'ints', 'strings']:
            if getattr(onnx_attr, attr_type):
                return list(getattr(onnx_attr, attr_type))
    return {arg.name: onnx_attribute_to_dict(arg) for arg in node.attribute}

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


def keras_builder(onnx_model, native_groupconv:bool=False):

    conv_layers.USE_NATIVE_GROUP_CONV = native_groupconv
    
    model_graph = onnx_model.graph

    '''
        init onnx model's build-in tensors
    '''
    onnx_weights = dict()
    for initializer in model_graph.initializer:
        onnx_weights[initializer.name] = numpy_helper.to_array(initializer)

    '''
        build input nodes
    '''
    tf_tensor, input_shape = {}, []
    for inp in model_graph.input:
        input_shape = [x.dim_value for x in inp.type.tensor_type.shape.dim]
        if input_shape == []:
            continue
        batch_size = 1 if input_shape[0] <= 0 else input_shape[0]
        input_shape = input_shape[2:] + input_shape[1:2]
        tf_tensor[inp.name] = keras.Input(shape=input_shape, batch_size=batch_size, dtype=onnx2tf_type.get(inp.type.tensor_type.elem_type))

    '''
        build model inline node by iterate onnx nodes.
    '''
    for node in model_graph.node:
        op_name, node_inputs, node_outputs = node.op_type, node.input, node.output
        op_attr = decode_node_attribute(node)
        
        tf_operator = OPERATOR.get(op_name)
        if tf_operator is None:
            raise KeyError(f"{op_name} not implemented yet")
        
        _inputs = None 
        if len(node_inputs) > 0:
            _inputs = tf_tensor[node_inputs[0]] if node_inputs[0] in tf_tensor else onnx_weights[node_inputs[0]]

        res = tf_operator(tf_tensor, onnx_weights, node_inputs, op_attr, outputs=node_outputs)(_inputs)
        if isinstance(res, list):
            for index in range(len(node_outputs)):
                tf_tensor[node_outputs[index]] = res[index]
        else:
            tf_tensor[node_outputs[0]] = res
    
    '''
        build keras model
    '''
    input_nodes = [tf_tensor[x.name] for x in model_graph.input]
    outputs_nodes = [tf_tensor[x.name] for x in model_graph.output]
    keras_model = keras.Model(inputs=input_nodes, outputs=outputs_nodes)
    keras_model.trainable = False
    # keras_model.summary()

    return keras_model

def tflite_builder(keras_model, weight_quant:bool=False, fp16_model=False, int8_model:bool=False, image_root:str=None,
                    int8_mean:list or float = [123.675, 116.28, 103.53], int8_std:list or float = [58.395, 57.12, 57.375]):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    if weight_quant or int8_model or fp16_model:
        converter.experimental_new_converter = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if fp16_model:
        converter.target_spec.supported_types = [tf.float16]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
    elif int8_model:
        assert len(keras_model.inputs) == 1, f"help want, only support single input model."
        shape = list(keras_model.inputs[0].shape)
        dataset = RandomLoader(shape) if image_root is None else ImageLoader(image_root, shape, int8_mean, int8_std)
        converter.representative_dataset = lambda: dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.target_spec.supported_types = []
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        converter.experimental_new_converter = True

    tflite_model = converter.convert()
    return tflite_model