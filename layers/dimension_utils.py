'''
    shape and axis transform utils func.
'''
def channel_to_last_dimension(axis):
    '''
        make channel first to channel last
    '''
    if axis == 0:
        axis = 0
    elif axis == 1:
        axis = -1
    else:
        axis -= 1
    return axis

def shape_NCD_to_NDC_format(shape:list or tuple):
    '''
        make shape format from channel first to channel last
    '''
    if len(shape) <= 2:
        return tuple(shape)
    new_shape = [shape[0], *shape[2:], shape[1]]
    return tuple(new_shape)

def tensor_NCD_to_NDC_format(tensor):
    '''
        make tensor format from channel first to channel last
    '''
    if(len(tensor.shape) > 2):
        shape = [i for i in range(len(tensor.shape))]
        shape = shape_NCD_to_NDC_format(shape)
        tensor = tensor.transpose(*shape)
    return tensor