# the channels correspond to number of feature_map layers. every other dimension will be the same


tf.reset_default_graph()

mm_channels = 7
screen_channels = 17

weights_screen = {  
    's_wc1': _weights_with_decay('s_wc1', [5, 5, screen_channels, 16], stddev=0.05), 
    's_wc2': _weights_with_decay('s_wc2', [3, 3, 16, 32], stddev=0.05),
    's_wfc1': _weights_with_decay('s_wfc1', [64 * 64 * 32, 32], stddev=0.04), 
    
}

weights_mm = {  
    'mm_wc1': _weights_with_decay('mm_wc1', [5, 5, mm_channels, 16], stddev=0.05),  
    'mm_wc2': _weights_with_decay('mm_wc2', [3, 3, 16, 32], stddev=0.05),
    'mm_wfc1': _weights_with_decay('mm_wfc1', [64 * 64 * 32, 32], stddev=0.04),  
    
}

biases = {
    'bc1': tf.get_variable('bc1', 16, initializer=tf.constant_initializer(0.0)),
    'bc2': tf.get_variable('bc2', 32, initializer=tf.constant_initializer(0.1)),
    'bfc1': tf.get_variable('bfc1', 256, initializer=tf.constant_initializer(0.1)),
}


def conv_layer(inputs, weights, biases, name, stride=1):
    
    conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME', name=name)
    conv = tf.nn.bias_add(conv, biases)
    
    return tf.nn.relu(conv)
    
    
def fully_connected(inputs, weights, biases, name):  # not sure how I am going to reconcile this, yet. Weights?
    fc1 = tf.add(tf.matmul(inputs, weights), biases, name=name)
    fc1 = tf.nn.tanh(fc1)
    return fc1


def spatial_conv(inputs, weights, biases, name, stride=1):
    spatial = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME', name=name)
    spatial = tf.nn.bias_add(spatial, biases)
    
    return tf.nn.tanh(spatial)
def create_network1(screen_channels, mm_channels, screen_weights, mm_weights, biases):  
    
    screen_input = tf.placeholder(tf.float32, [None, 64, 64, screen_channels], name='screen_input')
    mm_input = tf.placeholder(tf.float32, [None, 64, 64, mm_channels], name='mm_input')
    
    screen_conv1 = conv_layer(screen_input, screen_weights['s_wc1'], biases['bc1'], name='s_conv1')
    screen_conv2 = conv_layer(screen_conv1, screen_weights['s_wc2'], biases['bc2'], name='s_conv2')
    
    mm_conv1 = conv_layer(mm_input, mm_weights['mm_wc1'], biases['bc1'], name='mm_conv1')
    mm_conv2 = conv_layer(mm_conv1, mm_weights['mm_wc2'], biases['bc2'], name='mm_conv2')
    
    # broadcast vector (?). Where info is len(actions.FUNCTIONS)
#     info_fc = fully_connected(tf.layers.flatten(info), weights['wfc1'], biases['bfc1'], name='fc1')  
    
    # spatial actions -- i think should be 'state_rep'
    state_rep = tf.concat([mm_conv2, screen_conv2], axis=3)  #  [None, 64, 64, 64]
    spatial_bias = tf.get_variable('spatial_bias', [1], initializer=None)
    spatial_weights = _weights_with_decay('w_spat', [1, 1, 64, 1], stddev=0.05)
    
    spat_conv = spatial_conv(state_rep, spatial_weights, spatial_bias, name='spat_conv')  # working
#     print(spat_conv.get_shape().as_list())
#     print(tf.layers.flatten(mm_conv2).get_shape().as_list())
#     print(tf.layers.flatten(screen_conv2).get_shape().as_list())  # these work as well

#     spatial_action = tf.nn.softmax(tf.layers.flatten(spat_conv))  #  works until here

    # non-spatial actions and value
    non_spat_fc = tf.concat([tf.layers.flatten(mm_conv2), tf.layers.flatten(screen_conv2), info_fc], axis=1)
