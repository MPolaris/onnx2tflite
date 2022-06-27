
def Torch2TFAxis(axis):
    if axis == 0:
        axis = 0
    elif axis == 1:
        axis = -1
    else:
        axis -= 1
    return axis

def TorchShape2TF(shape:list or tuple):
    if len(shape) <= 2:
        return tuple(shape)
    new_shape = [shape[0], *shape[2:], shape[1]]
    return tuple(new_shape)

def TorchWeights2TF(weights):
    if(len(weights.shape) > 2):
        shape = [i for i in range(len(weights.shape))]
        shape = TorchShape2TF(shape)
        weights = weights.transpose(*shape)
    return weights