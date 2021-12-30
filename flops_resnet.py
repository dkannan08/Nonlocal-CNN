import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from Module.ResNet import *
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

    parser.add_argument('--n', dest='n', default=7, required=False, type=int, help='Number of Residual Blocks in each Unit. Choose n=7/18 for ResNet-44/110, n=2/6 for PreResNet-20/56 (default=%(default)s)')
    parser.add_argument('--version', dest='version', default=1, required=False, type=int, help='Version 1 for ResNet, Version 2 for PreResNet (default=%(default)s)')
    parser.add_argument('--dataset', dest='dataset', default="cifar10", required=True, help='Bechmark dataset cifar10, cifar100, stl10 (default=%(default)s)')

    return parser.parse_args()

#this function computes the FLOPs by setting up a computational graph
def get_flops(n, version, depth, input_size, num_classes, spatial_pool):
    """
    Args:
        n (int): number of residual blocks in each Unit
        version (int): ResNet (1) or PreResNet (2)
        num_classes (int): number of classes in the dataset
        input_size (tuple): size of the input, e.g (32,32,3)
        spatial_pool (bool): to have pooling between Units or not. 
        For segmentation tasks, set it to False.
    Returns: (None). Prints FLOPs and number of trainable parameters
    """
    tf.compat.v1.reset_default_graph()
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            #model = tf.keras.applications.VGG19(input_tensor=tf.compat.v1.placeholder('float32', shape=(1, 224,224,3)))
            if version ==1:
                model = resnet_v1(input_shape=input_size, depth=depth, num_classes=num_classes, flop_mode=True, spatial_pool=spatial_pool)
            else:
                model = resnet_v2(input_shape=input_size, depth=depth, num_classes=num_classes, flop_mode=True, spatial_pool=spatial_pool)
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            # Optional: save printed results to file
            # flops_log_path = os.path.join(tempfile.gettempdir(), 'tf_flops_log.txt')
            # opts['output'] = 'file:outfile={}'.format(flops_log_path)

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
            params = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)

            print("Model:", 'ResNet%dv%d' % (depth, version))
            print("Number of FLOPs (MACC):", (flops.total_float_ops/2)/1e6, "M")   #divide by 2 because tf calculates the double of the actual FLOPs always
            print("Number of trainable parameters:", params.total_parameters/1e6, "M")

def main():
    """
    Calculates FLOPs for ResNet models
    """
    args = setup_arg_parsing()
    
    dataset = args.dataset
    n = args.n
    version = args.version
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2
    if dataset=='cifar10' or dataset=='cifar100':
        input_size = (32,32,3)
        if dataset=='cifar10':
            num_classes=10
        else:
            num_classes=100
    elif dataset=='stl10':
        input_size = (96,96,3)
        num_classes=10
    elif dataset=='bdd100k':
        input_size = (90,160,3)     #assuming that the bdd100k is resized by a factor of 2^3
        num_classes=20
    else:
        raise ValueError("Wrong dataset name")
    spatial_pool=False if dataset=='bdd100k' else True   #don't pool between Units if it is a segmentation task
    
    get_flops(n=n, version=version, depth=depth, input_size=input_size, num_classes=num_classes, spatial_pool=spatial_pool)

if __name__ == '__main__':
    main()