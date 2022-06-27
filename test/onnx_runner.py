import os
import onnx
import onnxruntime

class ONNXModel():
    def __init__(self, onnx_path, interest_layers=[]):
        model = onnx.load(onnx_path)
        for il in interest_layers:
            model.graph.output.extend([onnx.ValueInfoProto(name=il)])
        self.onnx_session = onnxruntime.InferenceSession(model.SerializeToString())
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("-------------------{}------------------------".format(os.path.split(onnx_path)[-1]))
        print("load_model:{}".format(os.path.split(onnx_path)[-1]))
        print("input_name:{}, output_name:{}\n".format(self.input_name, self.output_name))
 
    def get_output_name(self, onnx_session):
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
 
    def get_input_name(self, onnx_session):
        input_name = []
        self.in_shape = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
            self.in_shape = node.shape
            if isinstance(self.in_shape[0], str):
                self.in_shape[0] = 1
        return input_name
        
    @property
    def input_shape(self):
        return self.in_shape

    def get_input_feed(self, input_name, image_tensor):
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed
 
    def forward(self, image_tensor):
        input_feed = self.get_input_feed(self.input_name, image_tensor)
        output = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return output
