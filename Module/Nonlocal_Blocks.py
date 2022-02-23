import tensorflow as tf
import numpy as np
#import warnings
        
class Nonlocal_diff(tf.keras.layers.Layer):
    """
    Forward propagation with nonlocal diffusion operator
    """
    def __init__(self, no_channels, version=2, affinity='dot1', iternum=2, lam=0.1, h=0.05, reg=0.0002, subsample=False, nl_pool_size=2):
        """
        no_channels (int): number of channels/filters in the Nonlocal Block
        version (int): version 2 computes $X_j - X_i$. version 1 is not used in the thesis.
                       version=1 is used to compute the same, but integrate is with $X_j$ and not with the difference term. 
                       Check: https://arxiv.org/pdf/1711.07971.pdf for details on version 1.
        affinity (str): For nonlocal diffusion operator, 'dot1' is scaled embedded dot product kernel,
        with normalizing const \mathcal{N} from thesis. 'dot2' is the one without this
        simplication of diving by \mathcal{N}. We divide by the \sum_j \omega(x_i, x_j)
        'embed_gaussian' will give us the scaled embedded gaussian kernel.
        
        iternum (int): number of stages in the Nonlocal Block
        lam (float): kernel scaling factor lambda from thesis
        h (float): discretization step size
        reg (float): regularization hyperparameter
        subsample (bool): to subsample in Nonlocal Block or not
        nl_pool_size (int): pooling size in the Nonlocal Block
        """        
        super(Nonlocal_diff, self).__init__()
        self.no_channels = no_channels
        self.version = version
        self.affinity = affinity
        self.iternum = iternum
        self.lam = lam
        self.h = h
        self.reg = reg
        self.subsample = subsample
        self.nl_pool_size = nl_pool_size
    
    def build(self, input_shape):
        #define the two embeddings
        self.conv_theta = tf.keras.layers.Conv2D(self.no_channels//2, 1, padding = 'same', 
                                            kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(self.reg), 
                                            activation = None)
        self.conv_phi = tf.keras.layers.Conv2D(self.no_channels//2, 1, padding = 'same', 
                                            kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(self.reg), 
                                            activation = None)
        if self.version == 1:   #see above what version 1 means
            self.conv_g = tf.keras.layers.Conv2D(self.no_channels//2, 1, padding = 'same', 
                                            kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(self.reg), 
                                            activation = None)
        if self.subsample:
            self.pool = tf.keras.layers.MaxPool2D(pool_size=self.nl_pool_size)
        self.W_layers = [tf.keras.layers.Conv2D(self.no_channels, 1, padding = 'same', 
                                            kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(self.reg), 
                                            activation = None) for i in range(self.iternum)]  #end 1x1 convolution layer K_1, K_2 from thesis
        self.bn_layers = [tf.keras.layers.BatchNormalization() for i in range(self.iternum)]
    
    def call(self, inputs, training = False):            
        theta_x = self.conv_theta(inputs)
        #theta_x = inputs
        #if self.subsample:    #sparsity of connections
        #    theta_x = self.pool(theta_x)
        theta_x = tf.reshape(theta_x,[theta_x.shape[0],-1, theta_x.shape[3]])
        
        phi_x = self.conv_phi(inputs)
        #phi_x = inputs
        if self.subsample:    #sparsity of connections
            phi_x = self.pool(phi_x)                      #subsampled embedding
        phi_x = tf.reshape(phi_x,[phi_x.shape[0],-1, phi_x.shape[3]])
        phi_x = tf.transpose(phi_x, perm=[0, 2, 1])
        
        f_x = tf.matmul(theta_x, phi_x)        #dot product of the two embeddings
        N = f_x.shape[-1]             #Normalizing factor \mathcal{N}
        f_x = self.lam*f_x           #scaling of the kernel
        lastval = inputs             #lastval just stores the original input to the Nonlocal Block
        if self.version==1:       #see above what version 1 means
            g_x = self.conv_g(inputs)
            #g_x = inputs
            if self.subsample:    #subsample the embedding
                g_x = self.pool(g_x)
            g_x = tf.reshape(g_x,[g_x.shape[0],-1, g_x.shape[3]])
            
            if self.affinity == 'dot1':
                f_x = f_x/N          #scaling
            if self.affinity == 'embed_gaussian':
                f_x = tf.nn.softmax(f_x, -1)             #check https://arxiv.org/pdf/1711.07971.pdf, why softmax is used
            if self.affinity =='dot2':
                f_sum = tf.reduce_sum(f_x, axis = 2, keepdims = True)        #stores the row sum of the matrix representing the kernel
                f_x  = tf.divide(f_x, f_sum)        #scaling
                
            for i in range(self.iternum):    #perform the nonlocal operation in stages, as discussed in thesis
                
                fg = tf.matmul(f_x, g_x)                                    
                fg = tf.reshape(fg, [inputs.shape[0], inputs.shape[1], inputs.shape[2], -1])
                output = self.W_layers[i](fg)         #1x1 conv at the end of the stage
                output = tf.nn.relu(self.bn_layers[i](output, training = training))
                output = lastval + self.h*output   #add to the input of the Nonlocal Block ('skip connections')
                
                #lastval = output
                g_x = output             #reshape the intermediate tensor and re-do the nonlocal operation in the next stage/iteration
                if self.subsample:
                    g_x = self.pool(g_x)
                g_x = tf.reshape(g_x,[g_x.shape[0],-1, g_x.shape[3]])
            
            return output
        
        if self.version==2:            
            #g_x = self.conv_g(inputs)
            g_x = inputs      #no embedding performed in version 2
            if self.subsample: #subsample the input to the Nonlocal Block
                sub_g_x = self.pool(g_x)
                sub_g_x = tf.reshape(sub_g_x,[sub_g_x.shape[0],-1, sub_g_x.shape[3]])
            g_x = tf.reshape(g_x,[g_x.shape[0],-1, g_x.shape[3]])

            if self.affinity == 'embed_gaussian':
                f_x = tf.math.exp(f_x)
            f_sum = tf.reduce_sum(f_x, axis = 2, keepdims = True)       #stores the row sum of the matrix representing the kernel

            for i in range(self.iternum):    #perform the nonlocal operation in stages, as discussed in thesis
                fg = tf.matmul(f_x, sub_g_x) if self.subsample else tf.matmul(f_x, g_x) 
                tmp = tf.multiply(g_x, f_sum)             #refer to equations 4.4 and 4.5 from thesis for details on this computation
                if self.affinity == 'dot1':     #scaling
                    fg = (fg-tmp)/N
                if self.affinity == 'dot2' or self.affinity == 'embed_gaussian':     #scaling
                    fg  = tf.divide(fg - tmp, f_sum)

                fg = tf.reshape(fg, [inputs.shape[0], inputs.shape[1], inputs.shape[2], -1])
                output = self.W_layers[i](fg)         #1x1 conv at the end of the stage
                output = tf.nn.relu(self.bn_layers[i](output, training = training))
                output = lastval + self.h*output   #add to the input of the Nonlocal Block ('skip connections')
                
                #lastval = output
                g_x = output             #reshape the intermediate tensor and re-do the nonlocal operation in the next stage/iteration
                if self.subsample:
                    sub_g_x = self.pool(g_x) 
                    sub_g_x = tf.reshape(sub_g_x,[sub_g_x.shape[0],-1, sub_g_x.shape[3]])                
                g_x = tf.reshape(g_x,[g_x.shape[0],-1, g_x.shape[3]])
            
            return output

class Nonlocal_pseudo(tf.keras.layers.Layer):
    """
    Forward propagation with pseudo-differential operators
    """    
    def __init__(self, no_channels, typ=2, n=2.0, s=0.5, norm='norm1', iternum=2, lam=0.1, h=0.05, reg=0.0002,  subsample=False, nl_pool_size=2):
        """
        no_channels (int): number of channels/filters in the Nonlocal Block
        typ (int): 1 for frac. Laplacian, 2 for for inv frac. laplacian, 3 for \Delta^{-1}
        n (int): dimension constant; kept at 2.
        s (float): power of fractional/inverse fractional Laplacian
        norm (str): For pseudo-differential operators, to normalize by \mathcal{N} from thesis ('norm1'),
        or by \sum_j \omega(x_i, x_j) ('norm2').
        
        iternum (int): number of stages in the Nonlocal Block
        lam (float): kernel scaling factor lambda from thesis
        h (float): discretization step size
        reg (float): regularization hyperparameter
        subsample (bool): to subsample in Nonlocal Block or not
        nl_pool_size (int): pooling size in the Nonlocal Block
        """ 
        super(Nonlocal_pseudo, self).__init__()
        self.no_channels = no_channels
        self.typ = typ
        self.n = n
        self.s = s
        self.norm = norm
        self.iternum = iternum
        self.lam = lam
        self.h = h
        self.reg = reg
        self.subsample = subsample
        self.nl_pool_size = nl_pool_size
        """
        version 2 computes $X_i - X_j$.
        version=1 is used to compute the same, but integrate is with $X_j$ and not with the difference term. 
        """  
        if self.typ == 2:                          #(-delta)^s
            self.power = self.n + (2*self.s)
            self.version = 2
        if self.typ == 3:                         #(-delta)^(-s)
            self.power = self.n - (2*self.s)
            self.version = 1
        if self.typ == 4:                          #(-delta)^(-1)
            self.version = 1
        #if self.norm!= 'dot1':
        #    warnings.warn('Passing <exponential> kernel for pseudo differential operator, nevertheless using <dot1> kernel')
        #    self.norm = 'dot1'
            
    def build(self, input_shape):
        #define the two embeddings
        self.conv_theta = tf.keras.layers.Conv2D(self.no_channels//2, 1, padding = 'same', 
                                            kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(self.reg), 
                                            activation = None)
        self.conv_phi = tf.keras.layers.Conv2D(self.no_channels//2, 1, padding = 'same', 
                                            kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(self.reg), 
                                            activation = None)
        #self.conv_g = tf.keras.layers.Conv2D(self.no_channels//2, 1, padding = 'same', 
        #                                    kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(self.reg), 
        #                                    activation = None)
        if self.subsample:
            self.pool = tf.keras.layers.MaxPool2D(pool_size=self.nl_pool_size)
        self.W_layers = [tf.keras.layers.Conv2D(self.no_channels, 1, padding = 'same', 
                                            kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(self.reg), 
                                            activation = None) for i in range(self.iternum)]   #end 1x1 convolution layer K_1, K_2 from thesis
        self.bn_layers = [tf.keras.layers.BatchNormalization() for i in range(self.iternum)]
     
    def call(self, inputs, training = False):                
        theta_x = self.conv_theta(inputs)
        #theta_x = inputs
        #if self.subsample:    #sparsity of connections
        #    theta_x = self.pool(theta_x)
        theta_x = tf.reshape(theta_x,[theta_x.shape[0],-1, theta_x.shape[3]])
        
        phi_x = self.conv_phi(inputs)
        #phi_x = inputs
        if self.subsample:    #sparsity of connections
            phi_x = self.pool(phi_x)                      #subsampled embedding
        phi_x = tf.reshape(phi_x,[phi_x.shape[0],-1, phi_x.shape[3]])
        '''
        diff = theta_x[:,:,tf.newaxis,:]-phi_x[:,tf.newaxis,:,:]
        f_x = tf.sqrt(tf.reduce_sum(diff**2,axis=-1)) 
        '''
        #pair-wise distance calculations based on the discussion on page 42 in the thesis
        row_norms_A = tf.reduce_sum(tf.square(theta_x), axis=-1)
        row_norms_A = tf.reshape(row_norms_A, [row_norms_A.shape[0], -1, 1])  
        row_norms_B = tf.reduce_sum(tf.square(phi_x), axis=-1)
        row_norms_B = tf.reshape(row_norms_B, [row_norms_B.shape[0], 1, -1])
        f_x = tf.sqrt(row_norms_A - 2 * tf.matmul(theta_x, tf.transpose(phi_x, [0,2,1])) + row_norms_B)
        
        #self.factor stores the c_{n,s} constant from thesis
        #f_x stores the kernel
        if self.typ == 2:                          #(-delta)^s
            self.factor = ((4**self.s)*tf.exp(tf.math.lgamma((self.n/2.0)+self.s)))/((np.pi**(self.n/2.0))*tf.abs(tf.exp(tf.math.lgamma(-self.s))))
            f_x = f_x**self.power
            f_x = tf.math.divide_no_nan(self.factor, f_x)
        if self.typ == 3:                         #(-delta)^(-s)
            self.factor = (tf.exp(tf.math.lgamma((self.n/2.0)-self.s)))/(4**self.s*(np.pi**(self.n/2.0))*tf.exp(tf.math.lgamma(self.s)))
            f_x = f_x**self.power
            f_x = tf.math.divide_no_nan(self.factor, f_x)
        if self.typ == 4:                          #(-delta)^(-1) assuming n=2
            euler_mascheroni_const = 0.5772156649
            self.factor = (1.0/(4.0*np.pi*tf.exp(tf.math.lgamma(1.0))))
            log_dist = tf.math.log(f_x)
            log_dist = tf.where(tf.math.is_inf(log_dist), -euler_mascheroni_const/2.0, log_dist)  #if entry is undefined, redefine the entry to have zero net contribution from that entry.
            f_x = self.factor*(-2.0*log_dist - euler_mascheroni_const)
            
        N = f_x.shape[-1]             #Normalizing factor \mathcal{N}
        f_x = self.lam*f_x           #scaling of the kernel
        lastval = inputs             #lastval just stores the original input to the Nonlocal Block
        if self.version==1: 
            #g_x = self.conv_g(inputs)
            g_x = inputs
            if self.subsample:    #subsample the input to the Nonlocal Block
                g_x = self.pool(g_x)
            g_x = tf.reshape(g_x,[g_x.shape[0],-1, g_x.shape[3]])
            
            if self.norm == 'norm1':
                f_x = f_x/N       #scaling
            #if self.affinity == 'embed_gaussian':
            #    f_x = tf.nn.softmax(f_x, -1)
            elif self.norm =='norm2':      #scaling
                f_sum = tf.reduce_sum(f_x, axis = 2, keepdims = True)       #stores the row sum of the matrix representing the kernel
                f_x  = tf.divide(f_x, f_sum) 
            else:
                raise NotImplementedError()
                
            for i in range(self.iternum):    #perform the nonlocal operation in stages, as discussed in thesis
                fg = tf.matmul(f_x, g_x)
                fg = tf.reshape(fg, [inputs.shape[0], inputs.shape[1], inputs.shape[2], -1])
                output = self.W_layers[i](fg)         #1x1 conv at the end of the stage
                output = tf.nn.relu(self.bn_layers[i](output, training = training))
                output = lastval + self.h*output   #add to the input of the Nonlocal Block ('skip connections')
                
                #lastval = output
                g_x = output             #reshape the intermediate tensor and re-do the nonlocal operation in the next stage/iteration
                if self.subsample:
                    g_x = self.pool(g_x)
                g_x = tf.reshape(g_x,[g_x.shape[0],-1, g_x.shape[3]])
            
            return output
        
        if self.version==2:  #nonlocal diffusion
            #g_x = self.conv_g(inputs)
            g_x = inputs
            if self.subsample:        #subsample the input to the Nonlocal Block
                sub_g_x = self.pool(g_x)
                sub_g_x = tf.reshape(sub_g_x,[sub_g_x.shape[0],-1, sub_g_x.shape[3]])
            g_x = tf.reshape(g_x,[g_x.shape[0],-1, g_x.shape[3]])
            
            #if self.affinity == 'exp_gaussian':
            #    f_x = tf.math.exp(f_x)
            f_sum = tf.reduce_sum(f_x, axis = 2, keepdims = True)       #stores the row sum of the matrix representing the kernel
            
            for i in range(self.iternum):    #perform the nonlocal operation in stages, as discussed in thesis
                fg = tf.matmul(f_x, sub_g_x) if self.subsample else tf.matmul(f_x, g_x) 
                tmp = tf.multiply(g_x, f_sum)             #refer to equations 4.4 and 4.5 from thesis for details on this computation
                if self.norm == 'norm1':       #scaling
                    fg = (tmp-fg)/N
                elif self.norm == 'norm2':     #scaling
                    fg  = tf.divide(tmp - fg, f_sum)
                else:
                    raise NotImplementedError()
                
                fg = tf.reshape(fg, [inputs.shape[0], inputs.shape[1], inputs.shape[2], -1])
                output = self.W_layers[i](fg)         #1x1 conv at the end of the stage
                output = tf.nn.relu(self.bn_layers[i](output, training = training))
                output = lastval + self.h*output   #add to the input of the Nonlocal Block ('skip connections')
                
                #lastval = output
                g_x = output             #reshape the intermediate tensor and re-do the nonlocal operation in the next stage/iteration
                if self.subsample:
                    sub_g_x = self.pool(g_x) 
                    sub_g_x = tf.reshape(sub_g_x,[sub_g_x.shape[0],-1, sub_g_x.shape[3]])
                g_x = tf.reshape(g_x,[g_x.shape[0],-1, g_x.shape[3]])                
            
            return output   