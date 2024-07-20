import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from onnx import numpy_helper
from .dataloader import RandomLoader, ImageLoader

from onnx2tflite.utils import OPERATOR
from onnx2tflite.layers import conv_layers
from onnx2tflite.utils.definitions import *
from onnx2tflite.utils.graph_tools import build_tf_inputs, decode_node_attribute

def keras_builder(onnx_model, native_groupconv:bool=False):

    conv_layers.USE_NATIVE_GROUP_CONV = native_groupconv
    
    model_graph = onnx_model.graph
    layout_dict, tf_tensor = {}, {}

    '''
        init onnx model's build-in tensors
    '''
    onnx_weights = dict()
    for initializer in model_graph.initializer:
        onnx_weights[initializer.name] = numpy_helper.to_array(initializer)

    '''
        build input nodes
    '''
    input_nodes = build_tf_inputs(model_graph, layout_dict)
    tf_tensor.update(input_nodes)

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

        # init layout
        for index in range(len(node_outputs)):
            layout_dict[node_outputs[index]] = layout_dict.get(node_inputs[0], Layout.Default)
        
        res = tf_operator(tf_tensor, onnx_weights, node_inputs, op_attr, node_outputs, layout_dict)(_inputs)
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
    keras_model.summary()
    print(layout_dict)

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