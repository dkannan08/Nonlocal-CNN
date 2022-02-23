# Adapted from: https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
import tensorflow as tf

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    Args:
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    Returns:
        x (tensor): tensor as input to the next layer
    """    
    conv = tf.keras.layers.Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = tf.keras.layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = tf.keras.layers.Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=10, num_stages=3, spatial_pool=True, flop_mode=False):
    """ResNet Version 1 Model builder.

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each Unit/stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters for CIFAR10 is:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    Args:
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
        num_stages (int): number of Units or stages in the network
        spatial_pool (bool): If spatial subsampling should be performed betw. Units/stages and if last layer should be softmax or not. 
        (For segmentation, set it to True)
        flop_mode (bool): If the model should be built to calculate flops or for actual training.

    Returns:
        model (Model): tf.keras model instance
    """    
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44)')
    # Start model definition.
    num_filters = 16    #initial conv layer with so many number of filters 
    num_res_blocks = int((depth - 2) / 6)
    if flop_mode:  #if we want to count FLOPs, mini-batch size is set to 1, standard way to calculate FLOPs
        inputs = tf.keras.Input(shape=input_shape, batch_size=1)
    else:
        inputs = tf.keras.Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(num_stages):              #'stack' in the 'Unit' in the thesis
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                if spatial_pool:
                    strides = 2  # downsample
                else:
                    strides = 1
                    
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims at the start of each Unit
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y])           #skip connection addition
            x = tf.keras.layers.Activation('relu')(x)
        num_filters *= 2                   #after the end of each Unit, double the number of filters

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    if spatial_pool:    
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        y = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(num_classes,
                        activation=None,
                        kernel_initializer='he_normal')(y)
    else:     #forsegmentation tasks: no pooling
        outputs = tf.keras.layers.Conv2D(num_classes, 1, padding='same', kernel_initializer='he_normal')(x)

    # Instantiate model.
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

def resnet_v2(input_shape, depth, num_classes=10, num_stages=3, spatial_pool=True, flop_mode=False):
    """ResNet Version 2 Model builder.

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each Unit/stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    Args:
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
        num_stages (int): number of Units or stages in the network
        spatial_pool (bool): If spatial subsampling should be performed betw. 
        Units/stages and if last layer should be softmax or not. (For segmentation, set it to True)
        flop_mode (bool): If the model should be built to calculate flops or for actual training.
        
    Returns:
        model (Model): tf.keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110)')
    # Start model definition.
    num_filters_in = 16    #initial conv layer with so many number of filters 
    num_res_blocks = int((depth - 2) / 9)

    if flop_mode:  #if we want to count FLOPs, mini-batch size is set to 1, standard way to calculate FLOPs
        inputs = tf.keras.Input(shape=input_shape, batch_size=1)
    else:
        inputs = tf.keras.Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(num_stages):              #'stack' in the 'Unit' in the thesis
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2       #at the start of each Unit, double the number of filters
                if res_block == 0:  # first layer but not first stage
                    if spatial_pool:
                        strides = 2    # downsample
                    else:
                        strides = 1

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims at the start of each Unit
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y])      #skip connection addition

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if spatial_pool:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        y = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(num_classes,
                        activation=None,
                        kernel_initializer='he_normal')(y)
    else:     #forsegmentation tasks: no pooling
        outputs = tf.keras.layers.Conv2D(num_classes, 1, padding='same', kernel_initializer='he_normal')(x)
    # Instantiate model.
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model