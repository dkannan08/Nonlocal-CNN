import tensorflow as tf
#import os
#os.chdir('/content/drive/My Drive/Colab Notebooks/')
from Module.ODE_Blocks import *
from Module.Nonlocal_Blocks import *
        
class Normal_Unit(tf.keras.layers.Layer):             #Unit for Hamiltonian, Parabolic
    """
    Defines a Hamiltonian/Parabolic Unit for the Nonlocal network. Each unit has 'num_block' Blocks.
    """
    def __init__(self, no_channels, block_typ="Hamiltonian", num_blocks=6, nonlocal_typ=0, s=0.5, affinity='dot1', norm='norm1', lam=0.1, h = 0.05, reg=0.0002, nl_pool_size=2, nl_subsample=False):
        """
        no_channels (int): number of channels/filters in each Unit
        block_typ (str): Only "Hamiltonian" is tested fully
        num_blocks (int): number of blocks in each Unit
        nonlocal_typ (int): number between 0 and 4. See readme.
        s (float): power of Laplacian and inverse frac. Laplacian
        affinity (str): For nonlocal diffusion operator, 'dot1' is scaled embedded dot product kernel,
        with normalizing const \mathcal{N} from thesis. 'dot2' is the one without this
        simplication of diving by \mathcal{N}. We divide by the \sum_j \omega(x_i, x_j)
        'embed_gaussian' will give us the scaled embedded gaussian kernel.
        
        norm (str): For pseudo-differential operators, to divide by \mathcal{N} from thesis ('norm1'),
        or by \sum_j \omega(x_i, x_j) ('norm2').
        lam (float): kernel scaling factor lambda from thesis
        h (float): discretization step size
        reg (float): regularization hyperparameter
        nl_pool_size (int): pooling size in the Nonlocal Block
        nl_subsample (bool): to subsample in Nonlocal Block or not
        """
        super(Normal_Unit, self).__init__()
        self.no_channels = no_channels
        self.block_typ = block_typ
        self.num_blocks = num_blocks
        self.nonlocal_typ = nonlocal_typ
        self.s = s
        self.affinity = affinity
        self.norm = norm
        self.lam = lam
        self.h = h
        self.reg = reg
        self.nl_subsample = nl_subsample
        self.nl_pool_size = nl_pool_size
        if self.s<0 or self.s>1:
            raise NotImplementedError()
            
        if self.block_typ == "Hamiltonian":
            Block = Hamiltonian_Block             
        elif self.block_typ == "Parabolic":
            Block = Parabolic_Block
        else:
            raise NotImplementedError()
        
        self.blocks = [Block(self.no_channels, self.h, self.reg) for i in range(self.num_blocks)]  #normal Blocks in the Units
        if self.nonlocal_typ == 0:  #add Nonlocal Block based on 'nonlocal_typ'
            pass
        elif self.nonlocal_typ==2 or self.nonlocal_typ == 3 or self.nonlocal_typ == 4:
            self.nl_block = Nonlocal_pseudo(self.no_channels, typ=self.nonlocal_typ, s=self.s, norm=self.norm, iternum=2, lam=self.lam, h=self.h, subsample=self.nl_subsample, nl_pool_size=self.nl_pool_size)
        elif self.nonlocal_typ == 1:
	        self.nl_block = Nonlocal_diff(self.no_channels, version=2, affinity=self.affinity, iternum=2, lam=self.lam, h=self.h, subsample=self.nl_subsample, nl_pool_size=self.nl_pool_size) 
        else:
            raise NotImplementedError()
        
    def call(self, inputs, training):
        if self.num_blocks<=6:
            x = self.blocks[0](inputs, training)   #opening two Blocks
            x = self.blocks[1](x, training)
            
            if self.nonlocal_typ>=1 and self.nonlocal_typ <=4:  #insert Nonlocal Block after 2nd normal Block in each Unit
                x = self.nl_block(x, training)
    
            for i in range(2, self.num_blocks):   #rest of the normal Blocks
                x = self.blocks[i](x, training)
        else:        #for deeper layers, add Nonlocal Block just before the last normal Block
            x = self.blocks[0](inputs, training) 
            
            for i in range(1, self.num_blocks-1):
                x = self.blocks[i](x, training)
                
            if self.nonlocal_typ>=1 and self.nonlocal_typ <=4:
                x = self.nl_block(x, training)
        
            x = self.blocks[self.num_blocks-1](x, training)   #last normal Block

        return x

class Double_Unit(tf.keras.layers.Layer):             #Unit for Midpoint, Leapfrog networks
    """
    Defines a Midpoint/Leapfrog Unit for the Nonlocal network. Each unit has 'num_block' Blocks.
    This Unit is still untested. It is supposed to be used for networks where the forward prop is governed by two previous step values Y_{j}, Y_{j-1}.
    """    
    def __init__(self, no_channels, block_typ="Hamiltonian", num_blocks=6, nonlocal_typ=0, s=0.5, affinity='dot1', norm='norm1', lam=0.1, h = 0.05, reg=0.0002, nl_pool_size=2, nl_subsample=False):
        """
        no_channels (int): number of channels/filters in each Unit
        block_typ (str): Only "Hamiltonian" is tested fully
        num_blocks (int): number of Blocks in each Unit
        nonlocal_typ (int): number between 0 and 4. See readme.
        s (float): power of Laplacian and inverse frac. Laplacian
        affinity (str): For nonlocal diffusion operator, 'dot1' is scaled embedded dot product kernel,
        with normalizing const \mathcal{N} from thesis. 'dot2' is the one without this
        simplication of diving by \mathcal{N}. We divide by the \sum_j \omega(x_i, x_j)
        'embed_gaussian' will give us the scaled embedded gaussian kernel.
        
        norm (str): For pseudo-differential operators, to divide by \mathcal{N} from thesis ('norm1'),
        or by \sum_j \omega(x_i, x_j) ('norm2').
        lam (float): kernel scaling factor lambda from thesis
        h (float): discretization step size
        reg (float): regularization hyperparameter
        nl_pool_size (int): pooling size in the Nonlocal Block
        nl_subsample (bool): to subsample in Nonlocal Block or not
        """
        super(Double_Unit, self).__init__()
        self.no_channels = no_channels
        self.block_typ = block_typ
        self.num_blocks = num_blocks
        self.nonlocal_typ = nonlocal_typ
        self.s = s
        self.affinity = affinity
        self.norm = norm
        self.lam = lam
        self.h = h
        self.reg = reg
        self.nl_subsample = nl_subsample
        self.nl_pool_size = nl_pool_size
        if self.s<0 or self.s>1:
            raise NotImplementedError()        
        
        if self.block_typ == "Midpoint":
            Block = Midpoint_Block       
        else:
            raise NotImplementedError()
            
        self.blocks = [Block(self.no_channels, self.h, self.reg) for i in range(self.num_blocks)]  #normal Blocks in the Units
        if self.nonlocal_typ == 0:  #add Nonlocal Block based on 'nonlocal_typ'
            pass
        elif self.nonlocal_typ==2 or self.nonlocal_typ == 3 or self.nonlocal_typ == 4:
            self.nl_block = Nonlocal_pseudo(self.no_channels, typ=self.nonlocal_typ, s=self.s, norm=self.norm, iternum=2, lam=self.lam, h=self.h, subsample=self.nl_subsample, nl_pool_size=self.nl_pool_size)
        elif self.nonlocal_typ == 1:
	        self.nl_block = Nonlocal_diff(self.no_channels, version=2, affinity=self.affinity, iternum=2, lam=self.lam, h=self.h, subsample=self.nl_subsample, nl_pool_size=self.nl_pool_size)
        else:
            raise NotImplementedError()
        
    def call(self, inputs, training):   #inputs is a "Combined_Tensor" object
        if self.num_blocks<=6:
            obj = self.blocks[0](inputs, training)  #stores the tensors $Y_j$ and $Y_{j-1}$ as a class object
            obj = self.blocks[1](obj, training)
            
            if self.nonlocal_typ>=1 and self.nonlocal_typ <=4:  #insert Nonlocal Block after 2nd normal Block in each Unit
                obj.current = self.nl_block(obj.current, training)
                
            for i in range(2, self.num_blocks):   #rest of the normal Blocks
                obj = self.blocks[i](obj, training)
        else:        #for deeper layers, add Nonlocal Block just before the last normal Block
            obj = self.blocks[0](inputs, training) 
            
            for i in range(1, self.num_blocks-1):
                obj = self.blocks[i](obj, training)
                
            if self.nonlocal_typ>=1 and self.nonlocal_typ <=4:
                obj.current = self.nl_block(obj.current, training)
        
            obj = self.blocks[self.num_blocks-1](obj, training)   #last normal Block
            
        return obj
    
class ODE_model(tf.keras.Model):
    """
    Builds the ODE-based models.
    """    
    def __init__(self, num_classes, CHANNELS, block_typ="Hamiltonian", num_blocks=6, nonlocal_typ=0, s=0.5, affinity='dot1', norm='norm1', lam=0.1, h=0.05, nl_subsample=False, dataset='stl10'):
        """
        num_classes (int): number of classes in the dataset; 10 for 'cifar10' and 'stl10', 100 for 'cifar100', 20 for 'bdd100k'
        CHANNELS (list): number of channels/filters in all the Units e.g. [64,128,256]
        block_typ (str): Only "Hamiltonian" is tested fully
        num_blocks (int): number of Blocks in each Unit
        nonlocal_typ (int): number between 0 and 4. See readme.
        s (float): power of Laplacian and inverse frac. Laplacian
        affinity (str): For nonlocal diffusion operator, 'dot1' is scaled embedded dot product kernel,
        with normalizing const \mathcal{N} from thesis. 'dot2' is the one without this
        simplication of diving by \mathcal{N}. We divide by the \sum_j \omega(x_i, x_j)
        'embed_gaussian' will give us the scaled embedded gaussian kernel.
        
        norm (str): For pseudo-differential operators, to normalize by \mathcal{N} from thesis ('norm1'),
        or by \sum_j \omega(x_i, x_j) ('norm2').
        lam (float): kernel scaling factor lambda from thesis
        h (float): discretization step size
        nl_subsample (bool): to subsample in Nonlocal Block or not
        dataset (str): dataset name: 'cifar10', 'stl10', 'cifar100', 'bdd100k'
        """
        
        super(ODE_model, self).__init__()
        
        self.num_classes = num_classes
        self.block_typ = block_typ
        self.num_blocks = num_blocks
        self.nonlocal_typ = nonlocal_typ
        self.s = s
        self.CHANNELS = CHANNELS
        self.affinity = affinity
        self.norm = norm
        self.lam = lam
        self.h = h
        self.nl_subsample = nl_subsample
        self.dataset = dataset
        if self.block_typ == "Hamiltonian" or self.block_typ == "Parabolic":
            Unit = Normal_Unit
        elif block_typ == "Midpoint":
            Unit = Double_Unit
        else:
            raise NotImplementedError()
        self.reg = 0.0005 if self.dataset == 'stl10' else 0.0002   #change regularization hyperparam based on dataset
        if self.dataset=='stl10':      #change the subsampling pool size in the Nonlocal Block, based on image resolution of the datasets.
            self.nl_pool_size = 4
        elif self.dataset=='bdd100k':
            self.nl_pool_size = 3
        else:
            self.nl_pool_size = 2
        self.units = [Unit(self.CHANNELS[i], self.block_typ, self.num_blocks, self.nonlocal_typ, s=self.s, affinity=self.affinity, norm=self.norm, 
                           lam=self.lam, h=self.h, reg=self.reg, nl_pool_size=self.nl_pool_size, nl_subsample=self.nl_subsample) for i in range(len(self.CHANNELS))]
        
        self.opening_conv = tf.keras.layers.Conv2D(CHANNELS[0], 3, padding= 'same', kernel_initializer = 'he_normal', 
                                   kernel_regularizer=tf.keras.regularizers.l2(self.reg),  
                                   activation = None)       #opening conv layer
        self.betw_conv = [tf.keras.layers.Conv2D(CHANNELS[i+1], 1, padding= 'same', kernel_initializer = 'he_normal', 
                                   kernel_regularizer=tf.keras.regularizers.l2(self.reg),  
                                   activation = 'relu') for i in range(len(self.CHANNELS)-1)]  #the conv layers between the Units
        
        self.bn = tf.keras.layers.BatchNormalization()
        
        if self.dataset=='stl10':   #due to bigger resolution, avg pooling with bigger pool size is performed
            self.pool2 = tf.keras.layers.AvgPool2D(pool_size=8, strides = 8)
        
        if self.dataset=='cifar10' or self.dataset=='cifar100' or self.dataset=='stl10':
            self.pool = tf.keras.layers.AvgPool2D(pool_size=2, strides = 2)
            self.flatten = tf.keras.layers.Flatten()
            self.classifier = tf.keras.layers.Dense(self.num_classes, kernel_initializer ='he_normal')
        elif self.dataset=='bdd100k':         #no fully-connected layer for segmentation task
            self.seg_final_layer = tf.keras.layers.Conv2D(num_classes, 1, padding='same', kernel_initializer='he_normal', 
                                   kernel_regularizer=tf.keras.regularizers.l2(self.reg))    #each pixel gets 'num_channels' number of predictions
        else:
            raise ValueError("Unknown dataset input")

    def call(self, inputs, training):
        if self.block_typ == "Hamiltonian" or self.block_typ == "Parabolic":
            x = self.opening_conv(inputs)
            x = tf.nn.relu(self.bn(x, training))
            
            for i in range(len(self.CHANNELS)):
                x = self.units[i](x, training)    #passing through each Unit
                if i!=(len(self.CHANNELS)-1):   #pooling between Units
                    if self.dataset=='cifar10' or self.dataset=='cifar100' or self.dataset=='stl10':
                        x = self.pool(x)
                    x = self.betw_conv[i](x)    #increase channels at the end of each Unit

            if self.dataset=='stl10':    #final pooling layer of the network
                x = self.pool2(x)
            elif self.dataset=='cifar10' or self.dataset=='cifar100':
                x = self.pool(x)
            elif self.dataset=='bdd100k':
                x = self.seg_final_layer(x)
                return x
                
        
        elif self.block_typ == "Midpoint":
            x = self.opening_conv(inputs)
            x = tf.nn.relu(self.bn(x, training))
            
            obj = Combined_Tensor(x, tf.zeros_like(x))  #stores the tensors $Y_j$ and $Y_{j-1}$ as a class object
            for i in range(len(self.CHANNELS)):
                obj = self.units[i](obj, training)     #passing through each Unit
                if i!=(len(self.CHANNELS)-1):   #pooling between Units
                    if self.dataset=='cifar10' or self.dataset=='cifar100' or self.dataset=='stl10':
                        obj.current = self.pool(obj.current)   #pool the output of the last two steps
                        obj.previous = self.pool(obj.previous)
                    obj.current = self.betw_conv[i](obj.current)    #increase channels at the end of each Unit
                    obj.previous = self.betw_conv[i](obj.previous)
            
            if self.dataset=='stl10':    #final pooling layer of the network
                x = self.pool2(obj.current)
            elif self.dataset=='cifar10' or self.dataset=='cifar100':
                x = self.pool(obj.current)                       
            elif self.dataset=='bdd100k':
                x = self.seg_final_layer(obj.current)
                return x        
        else:
            raise NotImplementedError()
        
        x = self.flatten(x)         #for classification task, flatten and fully-connected layer
        return self.classifier(x)
