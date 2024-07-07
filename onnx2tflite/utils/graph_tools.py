from onnx import numpy_helper
import tensorflow as tf
from tensorflow import keras
from .definitions import *

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

def build_tf_inputs(model_graph, node_dict:dict):
    inputs_name = []
    for inp in model_graph.input:
        input_shape = [x.dim_value for x in inp.type.tensor_type.shape.dim]
        if input_shape == []:
            continue
        inputs_name.append(inp.name)
        node_dict[inp.name] = Node_Layout(inp.name)
        if len(input_shape) < 3:
            node_dict[inp.name].layout = Layout.Channel_None

    _inputs_name = inputs_name.copy()
    for node in model_graph.node:
        op_name, node_inputs = node.op_type, node.input
        # output_layout = Layout.Default
        for ninp in node_inputs:
            if ninp in _inputs_name and op_name in FORCE_CHANNEL_LAST_OP and node_dict[ninp].layout == Layout.Default:
                node_dict[ninp].layout = Layout.Channel_Last
                _inputs_name.remove(ninp)
            if ninp in _inputs_name and op_name in FORCE_CHANNEL_FIRST_OP and node_dict[ninp].layout == Layout.Default:
                node_dict[ninp].layout = Layout.Channel_First
                _inputs_name.remove(ninp)
            # output_layout = output_layout | node_dict[ninp].layout
        
        if len(_inputs_name) == 0:
            break 
    
    input_nodes = {}
    for inp in model_graph.input:
        input_shape = [x.dim_value for x in inp.type.tensor_type.shape.dim]
        if input_shape == []:
            continue
        batch_size = 1 if input_shape[0] <= 0 else input_shape[0]
        input_shape = input_shape[1:]
        if node_dict[inp.name].layout == Layout.Channel_Last:
            input_shape = input_shape[1:] + input_shape[0:1]
        
        input_nodes[inp.name] = keras.Input(shape=input_shape, batch_size=batch_size, dtype=onnx2tf_type.get(inp.type.tensor_type.elem_type))

    return input_nodes
