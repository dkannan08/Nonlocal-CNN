import tensorflow as tf

class Combined_Tensor:
    """
    Stores the tensors of the last two iterations $Y_j$ and $Y_{j-1}$
    """
    def __init__(self, x, y):
        self.current = x
        self.previous = y
        
class Hamiltonian_Block(tf.keras.layers.Layer):
    """
    Forward propagation through the Hamiltonian Block
    """
    def __init__(self, no_channels, h, reg):
        """
        no_channels (int): number of channels/filters in the Block input
        h (float): discretization step size
        reg (float): regularization hyperparameter
        """
        super(Hamiltonian_Block, self).__init__()
        self.no_channels = no_channels
        self.h = h
        self.reg = reg
        
    def build(self, input_shape):
        #self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal',trainable=True)
        #self.b = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)
        self.conv1 = tf.keras.layers.Conv2D(int(self.no_channels/2), 3, padding = 'same', 
                                            kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(self.reg), 
                                            activation = None)
        self.conv2 = tf.keras.layers.Conv2D(int(self.no_channels/2), 3, padding = 'same', 
                                            kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(self.reg), 
                                            activation = None)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        #super(Block, self).build(input_shape)

    def call(self, inputs, training = False):     
        #Using the verlet discretization scheme
        self.Y, self.Z = tf.split(inputs, num_or_size_splits=2, axis=-1) #channel-wise splitting

        t = self.conv1(self.Z)
        t = tf.nn.relu(self.bn1(t, training = training))
        t = tf.nn.conv2d_transpose(t, self.conv1.weights[0], output_shape = (inputs.shape[0], inputs.shape[1], inputs.shape[2], int(inputs.shape[-1]/2)), strides = 1)
        t = tf.nn.bias_add(t,self.conv1.weights[1])
        self.Y = self.Y + self.h*t 
        
        t = self.conv2(self.Y)
        t = tf.nn.relu(self.bn2(t, training = training))
        t = tf.nn.conv2d_transpose(t, self.conv2.weights[0], output_shape = (inputs.shape[0], inputs.shape[1], inputs.shape[2], int(inputs.shape[-1]/2)), strides = 1)
        t = tf.nn.bias_add(t,self.conv2.weights[1])
        self.Z = self.Z - self.h*t
        
        return tf.concat([self.Y, self.Z], -1)   #channel-wise concatenation
        
class Midpoint_Block(tf.keras.layers.Layer):
    """
    Forward propagation through the Midpoint Block
    """
    def __init__(self, no_channels, h, reg):
        """
        no_channels (int): number of channels/filters in the Block input
        h (float): discretization step size
        reg (float): regularization hyperparameter
        """
        super(Midpoint_Block, self).__init__()
        self.no_channels = no_channels
        self.h = h
        self.reg = reg
        
    def build(self, input_shape):
        #self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal',trainable=True)
        #self.b = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)
        self.conv1 = tf.keras.layers.Conv2D(self.no_channels, 3, padding = 'same', 
                                            kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(self.reg), 
                                            activation = None)
        self.bn1 = tf.keras.layers.BatchNormalization()
        #super(Block, self).build(input_shape)
    
    def call(self, inputs, training = False):      #inputs is a "Combined_Tensor" object

        t = self.conv1(inputs.current)
        s = tf.nn.conv2d_transpose(inputs.current, self.conv1.weights[0], output_shape = (inputs.current.shape[0], inputs.current.shape[1], inputs.current.shape[2], inputs.current.shape[3]), strides = 1)
        #s = tf.nn.bias_add(s,self.conv1.weights[1])
        t = t - s
        t = tf.nn.relu(self.bn1(t, training = training))
        
        return Combined_Tensor(inputs.previous + 2*self.h*t, inputs.current)
'''
        s = inputs + h*t
        
        s = self.conv1(s)
        s = tf.nn.relu(self.bn3(s, training = training))
        s = -tf.nn.conv2d_transpose(s, self.conv1.weights[0], output_shape = (inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]), strides = 1)
        s = tf.nn.bias_add(s,self.conv1.weights[1])
        #s = tf.nn.relu(self.bn4(s, training = training))
        
        return inputs + (h/2)*t + (h/2)*s
'''

class Parabolic_Block(tf.keras.layers.Layer):
    """
    Forward propagation through the Parabolic Block
    """
    def __init__(self, no_channels, h, reg):
        """
        no_channels (int): number of channels/filters in the Block input
        h (float): discretization step size
        reg (float): regularization hyperparameter
        """
        super(Parabolic_Block, self).__init__()
        self.no_channels = no_channels
        self.h = h
        self.reg = reg
        
    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(self.no_channels, 3, padding = 'same', 
                                            kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(self.reg), 
                                            activation = None)
        self.bn1 = tf.keras.layers.BatchNormalization()
        #super(Block, self).build(input_shape)


    def call(self, inputs, training = False):
        
        t = self.conv1(inputs)
        t = tf.nn.relu(self.bn1(t, training = training))
        t = tf.nn.conv2d_transpose(t, self.conv1.weights[0], output_shape = (inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]), strides = 1)
        t = -tf.nn.bias_add(t,self.conv1.weights[1])
        return inputs + self.h*t
