import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime
import numpy as np
tf.keras.backend.set_floatx('float32')
import os
from Module.ResNet import *
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

    parser.add_argument('--n', dest='n', default=7, required=False, type=int, help='Number of Residual Blocks in each Unit. Choose n=7/18 for ResNet-44/110, n=2/6 for PreResNet-20/56 (default=%(default)s)')
    parser.add_argument('--version', dest='version', default=1, required=False, type=int, help='Version 1 for ResNet, Version 2 for PreResNet (default=%(default)s)')
    parser.add_argument('--epochs', dest='epochs', default=200, required=False, type=int, help='Number of training epochs (default=%(default)s)')
    parser.add_argument('--save_model', dest='save_model', default=False, required=False, type=bool, help='Save model and training weights (default=%(default)s)')
    parser.add_argument('--save_curve', dest='save_curve', default=False, required=False, type=bool, help='Save training and test accuracies for each epoch (default=%(default)s)')
    parser.add_argument('--dataset', dest='dataset', default="cifar10", required=True, help='Bechmark dataset cifar10, cifar100, stl10 (default=%(default)s)')
    parser.add_argument('--gauss_noise', dest='gauss_noise', default=False, required=False, type=bool, help='add gaussian noise to test data (default=%(default)s)')
    parser.add_argument('--struct_noise', dest='struct_noise', default=False, required=False, type=bool, help='add structural noise to test data (default=%(default)s)')
    parser.add_argument('--frac', dest='frac', default=1, required=False, type=float, help='Fraction of the dataset that is used for training, >0 and <=1 (default=%(default)s)')    

    return parser.parse_args()

def main():
    """
    Trains and tests the ResNet models on benchmark datasets
    """
    args = setup_arg_parsing()
    
    n = args.n
    version = args.version
    EPOCHS = args.epochs
    dataset = args.dataset
    BATCH_SIZE = 50 if dataset=='bdd100k' else 128
    gauss_noise = args.gauss_noise
    struct_noise = args.struct_noise
    frac = args.frac
    if frac<=0 or frac >1:
        raise ValueError("Fraction value of the dataset out of bounds") 
        
    #threshold to save model depending on the dataset
    #Change the threshold if needed, based on the experiment you perform.
    if dataset=='stl10':
        THRESHOLD = 75.2
    elif dataset=='bdd100k':
        THRESHOLD = 83.0 if version==1 else 51.6
    elif dataset=='cifar10':
        THRESHOLD = 92.3
    else:
        THRESHOLD = 69.4   #cifar100        
    
    BATCH_SIZE = 64 if dataset=='stl10' and frac<1 else BATCH_SIZE    #due to small training data samples for stl10, compared to cifar10/100    
    INTERVAL = [1, 80, 120, 160, 180]    #intervals where learning rates change
    RATES = [0.01, 0.1, 0.01, 0.001, 0.0001, 0.00001]    #learning rates

    SAVE_MODEL = args.save_model
    SAVE_DATA = args.save_curve
    SAVE_PATH = './saved_models/'    #path where everything will be saved
    
    @tf.function                     #executes the function as a tensorflow graph
    def train_step(images, labels):  #function to train the model with each batch
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)    #training is a boolean that tells how the batch norm, dropout should behave. They behave differently for trains and test mode.
            loss = loss_fn(labels, predictions)
            loss += sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))   #SGD step based on gradients for the weights
    
        train_loss(loss)
        train_accuracy(labels, predictions)
        if dataset=='bdd100k':       #iou makes sense for only bdd100k dataset.
            train_mean_iou(labels, predictions)
        
    @tf.function
    def test_step(images, labels):   #function to test the model with each batch
        predictions = model(images, training=False)
        t_loss = loss_fn(labels, predictions)
        t_loss += sum(model.losses)
        test_loss(t_loss)
        test_accuracy(labels, predictions)
        if dataset=='bdd100k':       #iou makes sense for only bdd100k dataset.
            test_mean_iou(labels, predictions)
        
    if version == 1:   #network depth calculation based on resnet version
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2
    print("Model:", 'ResNet%dv%d' % (depth, version))
    start_time = datetime.now()
    print("Start time: ", start_time)
    start_time = start_time.strftime('%Y-%m-%d')+"_"+start_time.strftime('%H:%M:%S')
    SAVE_PATH = SAVE_PATH + dataset + "_ResNet_"+ str(depth) + "_version_" + str(version) + "/"
    train_ds, test_ds, num_images, num_classes = load_dataset(dataset=dataset, BATCH_SIZE=BATCH_SIZE, gauss_noise=gauss_noise, struct_noise=struct_noise, frac=frac)
    spatial_pool=False if dataset=='bdd100k' else True   #don't pool between Units if it is a segmentation task
    if version == 1:
        model = resnet_v1(input_shape=tuple(train_ds.element_spec[0].shape[1:]), depth=depth, num_classes=num_classes, spatial_pool=spatial_pool)
    else:
        model = resnet_v2(input_shape=tuple(train_ds.element_spec[0].shape[1:]), depth=depth, num_classes=num_classes, spatial_pool=spatial_pool)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)        #not applying softmax but using logits for stability reasons, as suggested by tf
    
    lr_chng = [int(num_images/BATCH_SIZE)*i for i in INTERVAL]
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_chng, RATES) #learning rate decay
    optimizer = tf.keras.optimizers.SGD(learning_rate = lr_schedule, momentum=0.9)
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    if dataset=='bdd100k':
        train_mean_iou = tfa.metrics.MeanMetricWrapper(iou_fn,'train_mean_IOU',dtype='float32')  #wraps the user-defined iou function as a tf metric
        test_mean_iou = tfa.metrics.MeanMetricWrapper(iou_fn,'test_mean_IOU',dtype='float32')
        train_iou_list = []
        test_iou_list = []
    
    train_acc_list = []    #lists to save the training and test accuracies
    test_acc_list = []
    for epoch in range(EPOCHS):
        train_loss.reset_states()    #reset metrics for the next epoch
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        if dataset == 'bdd100k':
            train_mean_iou.reset_states()
            test_mean_iou.reset_states()
        
        for images, labels in train_ds:         #passing each batch of labels and images to train func
            train_step(images, labels)
    
        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)
    
        if SAVE_MODEL and test_accuracy.result()*100>=THRESHOLD:          #save model weights if threshold is exceeded
            filename = SAVE_PATH + '/model_'+start_time+'/EPOCH_' + str(epoch+1)+ '_acc_' +'%.2f' %(test_accuracy.result()*100)
            print('Saving checkpoint at:', filename)
            model.save_weights(filename, overwrite = True, save_format = 'tf')
     
        train_acc_list.append(train_accuracy.result().numpy())
        test_acc_list.append(test_accuracy.result().numpy())    
        if dataset=='bdd100k':    #for segmentation tasks, iou metric also needs to be printed
            template = 'Epoch {:d}, Loss: {:.4f}, Accuracy: {:.4f}, IoU: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}, Test IoU: {:.4f}, Time: {:.19}'
            print(template.format(epoch+1, train_loss.result(), train_accuracy.result()*100, train_mean_iou.result()*100, test_loss.result(), test_accuracy.result()*100, test_mean_iou.result()*100, str(datetime.now())))
            train_iou_list.append(train_mean_iou.result().numpy())
            test_iou_list.append(test_mean_iou.result().numpy())
        else:
            template = 'Epoch {:d}, Loss: {:.4f}, Accuracy: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}, Time: {:.19}'
            print(template.format(epoch+1, train_loss.result(), train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100, str(datetime.now())))   
        
    if SAVE_DATA:                    #save train and test accuracies in .npz file
        os.makedirs(SAVE_PATH, exist_ok=True)  
        ARR_FILE = SAVE_PATH+"traintestdata"+"_"+start_time+".npz"
        print("Training and test curves stored at: ", ARR_FILE)
        if dataset=='bdd100k':    #save iou as well for segmentation tasks
            np.savez(ARR_FILE, train_acc=np.array(train_acc_list), test_acc=np.array(test_acc_list), train_iou=np.array(train_iou_list), test_iou=np.array(test_iou_list))
        else:
            np.savez(ARR_FILE, train_acc=np.array(train_acc_list), test_acc=np.array(test_acc_list))
        
if __name__ == '__main__':
    main()
