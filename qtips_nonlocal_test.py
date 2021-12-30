import tensorflow as tf
from datetime import datetime
import numpy as np
tf.keras.backend.set_floatx('float32')
import pathlib
import os
from Module.Networks import *
from Module.utils import *
#tf.config.experimental_run_functions_eagerly(True)
#tf.keras.mixed_precision.experimental.set_policy('float32')

#Modified from: https://www.tensorflow.org/tutorials/quickstart/advanced
    
def setup_arg_parsing():
    """
    Parse the commandline arguments
    """
    import argparse
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument('--h', dest='h', default=0.05, required=False, type=float, help='Discretization step size between 0.03 and 0.06 (default=%(default)s)')
    parser.add_argument('--epochs', dest='epochs', default=30, required=False, type=int, help='Number of training epochs (default=%(default)s)')
    parser.add_argument('--block_typ', dest='block_typ', default="Hamiltonian", required=False, help='ODE Block type (default=%(default)s)')
    parser.add_argument('--nonlocal_typ', dest='nonlocal_typ', default=0, required=True, type=int, help='Type of nonlocal operator in Nonlocal Block (default=%(default)s)')
    parser.add_argument('--save_curve', dest='save_curve', default=False, required=False, type=bool, help='Save training and test accuracies for each epoch (default=%(default)s)')
    parser.add_argument('--save_model', dest='save_model', default=False, required=False, type=bool, help='Save model and training weights (default=%(default)s)')

    return parser.parse_args()

def main():
    """
    Trains and tests the Nonlocal Hamiltonian networks on Q-tips
    """
    args = setup_arg_parsing()
    
    h = args.h
    EPOCHS = args.epochs
    nonlocal_typ = args.nonlocal_typ  #See readme on how to choose nonlocal_typ
    block_typ = args.block_typ    #"Hamiltonian" is the only block type that is tested fully for this implementation
    #dataset = 'cifar10'
    s=0.5                   #power of fractional/inverse fractional laplacian
    
    BATCH_SIZE = 50
    INTERVAL = [80, 120,160, 180]    #intervals where learning rates change
    RATES = [0.01, 0.01, 0.001, 0.0001, 0.00001]    #learning rates
    CHANNELS = [32, 64, 112]     #number of channels/filters in each Unit of the network
    SMOOTH_REG = True    #weight smoothness decay is switched on
    
    THRESHOLD =  40.0     #model is saved if SAVE_MODEL is true and threshold test accuracy is crossed
    SAVE_MODEL = args.save_model
    SAVE_DATA = args.save_curve
    SAVE_PATH = './saved_models/'    #path where everything will be saved
    
    traindir = './data/train'  #directory where Q-tips is stored
    testdir = './data/val'
    
    data_dir1 = pathlib.Path(traindir)
    #list_ds1 = tf.data.Dataset.list_files(str(data_dir1/'*/*'))
    CLASS_NAMES = np.array([item.name for item in data_dir1.glob('*') if item.name != "filenames.txt"])
    num_classes = len(CLASS_NAMES)
    train_ds, test_ds, num_images_train = get_qtips_tf(traindir, testdir, BATCH_SIZE, noise=True)
    
    @tf.function                     #executes the function as a tensorflow graph
    def train_step(images, labels):  #function to train the model with each batch
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)    #training is a boolean that tells how the batch norm, dropout should behave. They behave differently for trains and test mode.
            loss = loss_fn(labels, predictions)
            loss += sum(model.losses)
            if SMOOTH_REG:
                loss += smooth_loss(model, h, l_2 = 1e-8)    #add the smooth weight decay penalty
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))   #SGD step based on gradients for the weights
    
        train_loss(loss)
        train_accuracy(labels, predictions)
        
    @tf.function
    def test_step(images, labels):   #function to test the model with each batch
        predictions = model(images, training=False)
        t_loss = loss_fn(labels, predictions)
        t_loss += sum(model.losses)
        if SMOOTH_REG:    
            t_loss += smooth_loss(model, h, l_2 = 1e-8)
        test_loss(t_loss)
        test_accuracy(labels, predictions)
    
    SAVE_PATH = SAVE_PATH + 'qtips' + "_" + str(block_typ) + "_nltype_" + str(nonlocal_typ) + "/"
    print("Model:", '%s_nltype_%d' % (block_typ, nonlocal_typ))
    start_time = datetime.now()
    print("Start time: ", start_time)
    start_time = start_time.strftime('%Y-%m-%d')+"_"+start_time.strftime('%H:%M:%S')
    #cifar10 is passed below to have same experimental setup such as regularization hyperparameters, etc. as for the cifar10 experiments.
    model = ODE_model(num_classes, CHANNELS, block_typ, num_blocks=6, nonlocal_typ=nonlocal_typ, s=s, affinity='dot1', lam=0.1, h=h, nl_subsample=True, dataset='cifar10')
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)        #not applying softmax but using logits for stability reasons, as suggested by tf
    
    lr_chng = [(num_images_train/BATCH_SIZE)*i for i in INTERVAL]
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_chng, RATES)
    optimizer = tf.keras.optimizers.SGD(learning_rate = lr_schedule, momentum=0.9)
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    
    nonlocal_train_acc_list = []    #lists to save the training and test accuracies
    nonlocal_test_acc_list = []
    for epoch in range(EPOCHS):
        train_loss.reset_states()    #reset metrics for the next epoch
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        
        for images, labels in train_ds:         #passing each batch of labels and images to train func
            train_step(images, labels)
        
        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)
    
        if SAVE_MODEL and test_accuracy.result()*100>=THRESHOLD:          #save model weights if threshold is exceeded
            filename = SAVE_PATH + '/model_'+start_time+'/EPOCH_' + str(epoch+1)+ '_acc_' +'%.2f' %(test_accuracy.result()*100)
            print('Saving checkpoint at:', filename)
            model.save_weights(filename, overwrite = True, save_format = 'tf')
    
        nonlocal_train_acc_list.append(train_accuracy.result().numpy())
        nonlocal_test_acc_list.append(test_accuracy.result().numpy())
        template = 'Epoch {:d}, Loss: {:.4f}, Accuracy: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}, Time: {:.19}'
        print(template.format(epoch+1, train_loss.result(), train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100, str(datetime.now())))   
    
    if SAVE_DATA:                    #save train and test accuracies in .npz file
        os.makedirs(SAVE_PATH, exist_ok=True)  
        ARR_FILE = SAVE_PATH+"traintestdata"+"_"+start_time+".npz"
        print("Training and test curves stored at: ", ARR_FILE)
        np.savez(ARR_FILE, train_acc=np.array(nonlocal_train_acc_list), test_acc=np.array(nonlocal_test_acc_list))

if __name__ == '__main__':
    main()