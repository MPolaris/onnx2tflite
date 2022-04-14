import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from onnx import numpy_helper
from .op_registry import OPERATOR

def representative_dataset_gen(img_root, img_size):
    if img_root is None or (not os.path.exists(img_root)):
        for _ in range(20):
            input = np.random.rand(1, img_size[0], img_size[1], 3).astype(np.float32)
            yield [input]
    else:
        VALID_FORMAT = ['jpg', 'png']
        for i, fn in enumerate(os.listdir(img_root)):
            if fn.split(".")[-1] not in VALID_FORMAT:
                continue
            input = cv2.imread(os.path.join(img_root, fn))
            input = cv2.resize(input, img_size)[:, :, ::-1]
            input = np.expand_dims(input, axis=0).astype(np.float32)
            input /= 255
            yield [input]
            if i >= 40:
                break

def keras_builder(onnx_model):
    model_graph = onnx_model.graph
    onnx_weights = dict()
    for initializer in model_graph.initializer:
        onnx_weights[initializer.name] = numpy_helper.to_array(initializer)
    tf_tensor, input_shape = {}, []
    for inp in model_graph.input:
        input_shape = [x.dim_value for x in inp.type.tensor_type.shape.dim]
        tf_tensor[inp.name] = keras.Input(shape=(input_shape[2], input_shape[3], input_shape[1]), batch_size=input_shape[0])
    
    for node in model_graph.node:
        op_name, node_inputs, node_outputs, op_attr = node.op_type, node.input, node.output, dict()
        for x in node.attribute:
            if x.type == 1:
                op_attr[x.name] = x.f
            elif x.type == 2:
                op_attr[x.name] = x.i
            elif x.type == 3:
                op_attr[x.name] = x.s.decode()
            elif x.type == 7:
                op_attr[x.name] = x.ints

        tf_operator = OPERATOR.get(op_name)
        if tf_operator is None:
            raise KeyError(f"算子 {op_name} 还未实现")
        tf_tensor[node_outputs[0]] = tf_operator(tf_tensor, onnx_weights, node_inputs, op_attr)(tf_tensor[node_inputs[0]])

    keras_model = keras.Model(inputs=[tf_tensor[x.name] for x in model_graph.input], outputs=[tf_tensor[x.name] for x in model_graph.output])
    keras_model.trainable = False
    keras_model.summary()

    return keras_model

def tflite_builder(keras_model, weight_quant:bool=False, int8_model:bool=False, image_root:str=None):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if weight_quant or int8_model:
        converter.experimental_new_converter = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if int8_model:
        input_shape = (keras_model.inputs[0].shape[1], keras_model.inputs[0].shape[2])
        converter.representative_dataset = lambda: representative_dataset_gen(image_root, input_shape)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = []
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        converter.experimental_new_converter = True

    tflite_model = converter.convert()
    return tflite_model