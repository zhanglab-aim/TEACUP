from amber.architect import ModelSpace, Operation

def deserilizer(s):
    if s.startswith('Conv1D'):
        args = s.split('_')[1:]
        ms = conv1d_space(*args)
    elif s.startswith('Conv2D'):
        args = s.split('_')[1:]
        ms = conv2d_space(*args)
    elif s.startswith('bench201'):
        ms = fake_bench201_space()
    else:
        raise Exception("unknown string id: %s" %s)
    return ms


def conv1d_space(num_layers=9, num_pool=3, out_filters=64):
    model_space = ModelSpace()
    out_filters = int(out_filters)
    num_layers = int(num_layers)
    num_pool = int(num_pool)    
    expand_layers = [num_layers//num_pool-1 + i*(num_layers//num_pool) for i in range(num_pool-1)]
    for i in range(num_layers):
        model_space.add_layer(i, [
            Operation('conv1d', filters=out_filters, kernel_size=9, activation='relu'),
            Operation('conv1d', filters=out_filters, kernel_size=5, activation='relu'),
            Operation('conv1d', filters=out_filters, kernel_size=9, activation='relu', dilation=10),
            Operation('conv1d', filters=out_filters, kernel_size=5, activation='relu', dilation=10),
            # max/avg pool has underlying 1x1 conv
            Operation('maxpool1d', filters=out_filters, pool_size=5, strides=1),
            Operation('avgpool1d', filters=out_filters, pool_size=5, strides=1),
            Operation('identity', filters=out_filters),
      ])
        if i in expand_layers:
            out_filters *= 2
    return model_space


def conv2d_space(num_layers=9, num_pool=3, out_filters=64):
    model_space = ModelSpace()
    out_filters = int(out_filters)
    num_layers = int(num_layers)
    num_pool = int(num_pool)    
    expand_layers = [num_layers//num_pool-1 + i*(num_layers//num_pool) for i in range(num_pool-1)]
    for i in range(num_layers):
        model_space.add_layer(i, [
            Operation('conv2d', filters=out_filters, kernel_size=(5,5), activation='relu'),
            Operation('conv2d', filters=out_filters, kernel_size=(3,3), activation='relu'),
            #Operation('conv2d', filters=out_filters, kernel_size=(1,1), activation='relu'),
            Operation('conv2d', filters=out_filters, kernel_size=(5,5), activation='relu', dilation=4),
            Operation('conv2d', filters=out_filters, kernel_size=(3,3), activation='relu', dilation=4),
            Operation('maxpool2d', filters=out_filters, pool_size=(3,3), strides=1),
            Operation('avgpool2d', filters=out_filters, pool_size=(3,3), strides=1),
            Operation('identity', filters=out_filters),
      ])
        if i in expand_layers:
            out_filters *= 2
    return model_space


def fake_bench201_space(num_layers=6, out_filters=64):
    '''
    For now, only use this to generate categorical distribution for genetic algorithm
    '''
    model_space = ModelSpace()
    for i in range(num_layers):
        model_space.add_layer(i, [
            Operation('identity', filters=out_filters),
            Operation('identity', filters=out_filters),
            Operation('conv2d', filters=out_filters, kernel_size=(5,5), activation='relu', dilation=4),
            Operation('conv2d', filters=out_filters, kernel_size=(3,3), activation='relu', dilation=4),
            Operation('avgpool2d', filters=out_filters, pool_size=(3,3), strides=1),
      ])
    return model_space