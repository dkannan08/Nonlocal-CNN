import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from Module.Networks import *
#import numpy as np
#import os

#Modified from: https://github.com/tensorflow/tensorflow/issues/32809

def setup_arg_parsing():
    """
    Parse the commandline arguments
    """
    import argparse
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument('--num_blocks', dest='num_blocks', default=6, required=False, type=int, help='Number of Hamiltonian blocks in each Unit (default=%(default)s)')
    parser.add_argument('--nonlocal_typ', dest='nonlocal_typ', default=0, required=True, type=int, help='Type of nonlocal operator in Nonlocal Block (default=%(default)s)')
    parser.add_argument('--dataset', dest='dataset', default="cifar10", required=True, help='Bechmark dataset cifar10, cifar100, stl10 (default=%(default)s)')

    return parser.parse_args()

#this function computes the FLOPs by setting up a computational graph
def get_flops(num_blocks, nonlocal_typ, dataset, h, CHANNELS, block_typ, nl_subsample, num_classes, input_size):
    """
    Args:
        num_blocks (int): number of blocks in each Unit
        nonlocal_typ (int): number between 0 and 4. See readme.
        dataset (str): Name of dataset. See readme.
        h (float): discretization step size
        CHANNELS (list): number of channels in each unit
        block_typ (str): Leave this as "Hamiltonian". Other Blocks are not tested yet.
        nl_subsample (bool): to subsample in the Nonlocal Block or not
        num_classes (int): number of classes in the dataset
        input_size (tuple): size of the input for a mini-batch of size 1, e.g (1,32,32,3)
    Returns: (None). Prints FLOPs and number of trainable parameters
    """
    tf.compat.v1.reset_default_graph()
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            #model = tf.keras.applications.VGG19(input_tensor=tf.compat.v1.placeholder('float32', shape=(1, 224,224,3)))
            model=ODE_model(num_classes, CHANNELS, block_typ, num_blocks=num_blocks, nonlocal_typ=nonlocal_typ, lam=0.1, h=h, nl_subsample=nl_subsample, dataset=dataset)
            model.build(input_shape=input_size)       #build model with particular input shape
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            # Optional: save printed results to file
            # flops_log_path = os.path.join(tempfile.gettempdir(), 'tf_flops_log.txt')
            # opts['output'] = 'file:outfile={}'.format(flops_log_path)

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
            params = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)

            print("Model:", '%s_type_%d' % (block_typ, nonlocal_typ))
            print("Number of FLOPs (MACC):", (flops.total_float_ops/2)/1e6, "M")   #divide by 2 because tf calculates the double of the actual FLOPs always
            print("Number of trainable parameters:", params.total_parameters/1e6, "M")
            
def main():
    """
    Calculates FLOPs for Nonlocal Hamiltonian models
    """
    args = setup_arg_parsing()
    
    h = 0.06
    num_blocks = args.num_blocks
    dataset = args.dataset
    nonlocal_typ = args.nonlocal_typ
    CHANNELS = [32, 64, 112] if num_blocks<=6 else [32, 64, 128]      #number of channels/filters in each Unit of the network
    block_typ = "Hamiltonian"  
    nl_subsample = True         #to have subsampling in Nonlocal Block
    if dataset=='cifar10' or dataset=='cifar100':
        input_size = (1,32,32,3)
        if dataset=='cifar10':
            num_classes=10
        else:
            num_classes=100
    elif dataset=='stl10':
        input_size = (1,96,96,3)
        num_classes=10
    elif dataset=='bdd100k':
        input_size = (1,90,160,3)      #assuming that the bdd100k is resized by a factor of 2^3
        num_classes=20
    else:
        raise ValueError("Wrong dataset name")

    get_flops(num_blocks=num_blocks, nonlocal_typ=nonlocal_typ, dataset=dataset, h=h, CHANNELS=CHANNELS, block_typ=block_typ, nl_subsample=nl_subsample, num_classes=num_classes, input_size=input_size)
    
if __name__ == '__main__':
    main()