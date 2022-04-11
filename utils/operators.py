# from . import OPERATOR
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras

# @OPERATOR.register_operator("BatchNormalization")
# class TFBatchNormalization(keras.layers.Layer):
#     def __init__(self, weight, bias, running_mean, running_var, epsilon=1e-5, momentum=0.9):
#         super().__init__()
#         epsilon = 1e-5 if epsilon is None else epsilon
#         momentum = 0.9 if epsilon is None else momentum
#         self.bn = keras.layers.BatchNormalization(
#             beta_initializer=keras.initializers.Constant(bias),
#             gamma_initializer=keras.initializers.Constant(weight),
#             moving_mean_initializer=keras.initializers.Constant(running_mean),
#             moving_variance_initializer=keras.initializers.Constant(running_var),
#             epsilon=epsilon,
#             momentum=momentum)

#     def call(self, inputs):
#         return self.bn(inputs)

# @OPERATOR.register_operator("Pad")
# class TFPad(keras.layers.Layer):
#     def __init__(self, pad, model="constant"):
#         super().__init__()
#         self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
#         self.model = model

#     @staticmethod
#     def autopad(k, p=None):  # kernel, padding
#         # Pad to 'same'
#         if p is None:
#             p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#         return p

#     def call(self, inputs):
#         return tf.pad(inputs, self.pad, mode=self.model)

# @OPERATOR.register_operator("Conv")
# class TFConv(keras.layers.Layer):
#     def __init__(self, c1, c2, k=1, s=1, dilations=1, p=None, g=1, w=None, b=None):
#         super().__init__()
#         assert g == 1, "TF v2.2 Conv2D does not support 'groups' argument"
#         if isinstance(dilations, int):
#             dilations = (dilations, dilations)
#         if isinstance(s, int):
#             s = (s, s)
#         if dilations[0] != 1 and s[0] != 1:
#             raise Exception("Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.")
#         conv_model_str = 'SAME' if s == 1 and k == 1 else "VALID"
#         conv = keras.layers.Conv2D(
#             c2, k, s, conv_model_str, use_bias=False if b is None else True,
#             kernel_initializer=keras.initializers.Constant(w),
#             bias_initializer='zeros' if b is None else keras.initializers.Constant(b),
#             dilation_rate=dilations)
#         if p is None:
#             self.conv = conv
#         else:
#             self.conv = keras.Sequential([TFPad(TFPad.autopad(k, p)), conv])

#     def call(self, inputs):
#         return self.conv(inputs)

# class TFGroupConv(keras.layers.Layer):
#     # Group Convolution 分组卷积
#     def __init__(self, cin, cout, k=1, s=1, dilations=1, p=None, groups=1, w=None, b=None):
#         super().__init__()
#         filters = w.shape[-2]
#         assert groups*filters == cout, "Input channels and filters must both be divisible by groups."
#         if isinstance(dilations, int):
#             dilations = (dilations, dilations)
#         if isinstance(s, int):
#             s = (s, s)
#         if dilations[0] != 1 and s[0] != 1:
#             raise Exception("Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.")
#         self.cin = cin
#         self.groups = groups
#         cout = int(cout//groups)
#         if p is None:
#             self.pad = None
#         else:
#             self.pad = TFPad(TFPad.autopad(k, p))
        
#         self.convs = []
#         for i in range(groups):
#             self.convs.append(keras.layers.Conv2D(
#                                 cout, k, s, 'VALID', use_bias=False if b is None else True,
#                                 dilation_rate=dilations,
#                                 kernel_initializer=keras.initializers.Constant(w[:, :, :, i*cout:(i+1)*cout]),
#                                 bias_initializer='zeros' if b is None else keras.initializers.Constant(b[i*cout:(i+1)*cout])))

#     def call(self, inputs):
#         if self.pad is not None:
#             inputs = self.pad(inputs)
#         outs = []
#         in_s = tf.split(inputs, num_or_size_splits=self.groups, axis=-1)
#         for i in range(self.groups):
#             outs.append(self.convs[i](in_s[i]))
#         outs = tf.concat(outs, axis=-1)
#         return outs

# class TFDepthwiseConv2D(keras.layers.Layer):
#     # 深度可分离卷积
#     def __init__(self, k=1, s=1, dilations=1, p=None, w=None, b=None) -> None:
#         super().__init__()
#         if p is None:
#             self.pad = None
#         else:
#             self.pad = TFPad(TFPad.autopad(k, p))

#         if isinstance(dilations, int):
#             dilations = (dilations, dilations)
#         if isinstance(s, int):
#             s = (s, s)
#         conv = keras.layers.DepthwiseConv2D(
#             k, s, "VALID", use_bias=False if b is None else True,
#             weights=[w] if b is None else [w, b],
#             dilation_rate=dilations,
#             activation=None,
#             kernel_initializer='zeros',
#             bias_initializer='zeros'
#         )
        
#         if p is None:
#             self.conv = conv
#         else:
#             self.conv = keras.Sequential([TFPad(TFConv.autopad(k, p)), conv])
            
#     def call(self, inputs):
#         return self.conv(inputs)