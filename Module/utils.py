import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import pathlib
import os
import PIL.Image as Image

canvasSize = None #to store image resolution of Q-tips dataset

def plot_dataset(ds):
    """
    Plot 5 times 5 images of a tf dataset
    Source: https://www.tensorflow.org/tutorials/load_data/images    
    Args:
        ds (tf.data.Dataset): input batched dataset with images and labels 
    Returns:
        None (Plots images with labels)
    """
    plt.figure(figsize=(10,10))
    for image_batch, label_batch in ds:
        for n in range(25):  
            ax = plt.subplot(5,5,n+1)
            plt.imshow(image_batch[n])
            plt.title(int(label_batch[n]), fontsize = 13)
            plt.axis('off')
        break
    
def augmentation(x, y):
    """
    Image augmentation for CIFAR-10/CIFAR-100    
    Args:
        x (numpy array or tf.Tensor): images
        y (numpy array or tf.Tensor): labels
    Returns:
        x (tf.Tensor): augmented image, y (tf.Tensor): labels
    """
    x = tf.image.resize_with_crop_or_pad(
        x, x.shape[0] + 8, x.shape[1] + 8)
    x = tf.image.random_crop(x, [x.shape[0]-8, x.shape[1]-8, x.shape[2]])
    x = tf.image.random_flip_left_right(x)
    return x, y

def augmentation_stl(x, y):
    """
    Image augmentation for STL-10
    Args:
        x (numpy array or tf.Tensor): images
        y (numpy array or tf.Tensor): labels
    Returns:
        x (tf.Tensor): augmented image, y (tf.Tensor): labels
    """
    x = tf.image.resize_with_crop_or_pad(
        x, x.shape[0] + 24, x.shape[1] + 24)
    x = tf.image.random_crop(x, [x.shape[0]-24, x.shape[1]-24, x.shape[2]])
    x = tf.image.random_flip_left_right(x)
    return x, y

def segment_augment(x, y):
    """
    Image augmentation for BDD100K   
    Args:
        x (numpy array or tf.Tensor): images
        y (numpy array or tf.Tensor): labels/masks
    Returns:
        x (tf.Tensor): augmented image, y (tf.Tensor): labels/masks
    """
    if tf.random.uniform([])<0.5:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)
    return x, y

def normalize(x, y):
    """
    per-image standardization   
    Args:
        x (numpy array or tf.Tensor): images
        y (numpy array or tf.Tensor): labels
    Returns:
        x (tf.Tensor): augmented image, y (tf.Tensor): labels
    """
    x = tf.image.per_image_standardization(x)
    return x, y
 
def smooth_loss(model, h, l_2=0.0002):
    """
    Weight smoothness decay  
    Args:
        model (Model): tf.keras model instance
        h (float): step size
        l_2 (float): regularization hyperparameter
    Returns:
        float: penalty for rapidly changing weights between the layers
    """
    loss_sum = 0
    for j in range(3):
        kernels = None
        for t in model.layers[j].weights:                    #For each of the three Unit
            if 'kernel' in t.name and 'block' in t.name and 'nonlocal' not in t.name: #only if the layers don't belong to Nonlocal Block
                if kernels is None:
                    kernels = tf.reshape(t, shape = [1, -1])
                else:
                    kernels = tf.concat([kernels, tf.reshape(t, shape = [1, -1])], axis = 0) #collect all the kernel weights
        K1 = kernels[::2]
        K2 = kernels[1::2]
        K1diff = K1[1:] - K1[:-1]           #difference in the kernel weights of adjacent layers
        K2diff = K2[1:] - K2[:-1]
        K1diff = tf.norm(K1diff/h, axis = 1)**2
        K2diff = tf.norm(K2diff/h, axis = 1)**2
        loss_sum += tf.reduce_sum(K1diff)
        loss_sum += tf.reduce_sum(K2diff)      
    
    return l_2*h*loss_sum

def add_gauss_noise(x, stv):
    """
    add Gaussian noise to each image    
    Args:
        x (numpy array or tf.Tensor): images
        stv (float): standard deviation of noise
    Returns:
        x (tf.Tensor): images with Gaussian noise
    """    
    noise_values = tf.random.normal(x.shape, mean = 0, stddev=stv, dtype=tf.float32)
    x = x + noise_values
    x = tf.clip_by_value(x, clip_value_min=0, clip_value_max=1)
    return x

def add_struct_noise(x, eps):
    """
    add structural noise to each image, i.e.
    randomly chosen test image is multiplied by 'eps' and added to all test images 'x'
    check: https://arxiv.org/pdf/1811.09885.pdf
    Not used in this thesis.
    Args:
        x (numpy array or tf.Tensor): images
        eps (float): fraction to multiply the randomly chosen image with
    Returns:
        x (tf.Tensor): images with structural noise
    """
    #random_index = np.random.randint(0,x.shape[0], size=1)[0]
    random_index = 100  #101st test image is used for structured noise
    noise_values = eps*x[random_index]
    x = x + noise_values
    x = tf.clip_by_value(x, clip_value_min=0, clip_value_max=1)
    return x

def load_dataset(dataset='cifar10', per_pixel_center=True, per_pixel_std=False,  BATCH_SIZE=100, gauss_noise=False, stv=0.02, struct_noise=False, eps=0.30, frac=1, dload_path=os.getcwd()):
    """
    load dataset with images and labels and return tf.data.Dataset instance
    Args:
        dataset (str): 'cifar10', 'cifar100', 'stl10' or 'bdd100k'
        per_pixel_mean (bool): per-pixel mean subtraction from train and test images
        per_pixel_std (bool): per-pixel standard deviation division for train and test images
        BATCH_SIZE (int): the mini-batch size of training and test data
        gauss_noise (bool): add Gaussian noise to test images
        stv (float): standard deviation of Gaussian noise
        struct_noise (bool): add structured noise to test images, check: https://arxiv.org/pdf/1811.09885.pdf
        eps (float): factor for the structured noise
        frac(float): decimal between >0 and <=1, fraction of the training data to be used
        dload_path (str): path to store STL10 data, so that it can be reused again.
    Returns:
        train_ds (tf.data.Dataset): batched and augmented training data
        test_ds (tf.data.Dataset): batched test data
        int: number of images in the training data
        int: number of classes in the dataset
    """
    if dataset=='cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train[:int(frac*x_train.shape[0])]        #data is already well shuffled, so we can take the first 5% or 10% of the training examples if needed.
        y_train = y_train[:int(frac*y_train.shape[0])]
        x_train, x_test = x_train.astype('float32')/255.0, x_test.astype('float32')/255.0
        if gauss_noise:
            x_test = add_gauss_noise(x_test, stv)
        if struct_noise:
            x_test = add_struct_noise(x_test, eps)
        m, st = np.mean(x_train, axis = (0)), np.std(x_train, axis = (0))
        if per_pixel_center:    
            x_train -= m; x_test -= m; 
        if per_pixel_std:
            x_train /= st; x_test /= st;
        
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(augmentation).shuffle(10000).batch(BATCH_SIZE)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(1000).batch(BATCH_SIZE)

    elif dataset=='cifar100':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        x_train = x_train[:int(frac*x_train.shape[0])]        #data is already well shuffled, so we can take the first 5% or 10% of the training examples if needed.
        y_train = y_train[:int(frac*y_train.shape[0])]        
        x_train, x_test = x_train.astype('float32')/255.0, x_test.astype('float32')/255.0
        if gauss_noise:
            x_test = add_gauss_noise(x_test, stv)
        if struct_noise:
            x_test = add_struct_noise(x_test, eps)
        m, st = np.mean(x_train, axis = (0)), np.std(x_train, axis = (0))
        if per_pixel_center:    
            x_train -= m; x_test -= m; 
        if per_pixel_std:
            x_train /= st; x_test /= st;
            
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(augmentation).shuffle(10000).batch(BATCH_SIZE)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(1000).batch(BATCH_SIZE)
        
    elif dataset=='stl10':
        train_ds, test_ds = tfds.load('stl10', split=['train', 'test'], data_dir=dload_path, batch_size = -1, shuffle_files=True, as_supervised=True)
        x_train = train_ds[0].numpy(); y_train = train_ds[1].numpy();
        x_test = test_ds[0].numpy(); y_test = test_ds[1].numpy();
        x_train = x_train[:int(frac*x_train.shape[0])]        #data is already well shuffled, so we can take the first 5% or 10% of the training examples if needed.
        y_train = y_train[:int(frac*y_train.shape[0])]  
        x_train, x_test = x_train.astype('float32')/255.0, x_test.astype('float32')/255.0
        if gauss_noise:
            x_test = add_gauss_noise(x_test, stv)
        if struct_noise:
            x_test = add_struct_noise(x_test, eps)
        m, st = np.mean(x_train, axis = (0)), np.std(x_train, axis = (0))
        if per_pixel_center:    
            x_train -= m; x_test -= m; 
        if per_pixel_std:
            x_train /= st; x_test /= st;
            
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(augmentation_stl).shuffle(1000).batch(BATCH_SIZE)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(1000).batch(BATCH_SIZE) 
        
    elif dataset=='bdd100k':
        train_data = np.load('./bddarray_resol_decr3/train_data.npz')
        test_data = np.load('./bddarray_resol_decr3/test_data.npz')
        x_train, y_train = train_data['images'], train_data['labels']
        x_test, y_test = test_data['images'], test_data['labels']
        
        y_train[y_train == 255] = 19  #Black parts are labeled as the 20th category (counting starts from 0)
        y_test[y_test == 255] = 19
        x_train = x_train[:int(frac*x_train.shape[0])]        #data is already well shuffled, so we can take the first 5% or 10% of the training examples if needed.
        y_train = y_train[:int(frac*y_train.shape[0])]
        x_train, x_test = x_train.astype('float32')/255.0, x_test.astype('float32')/255.0
        if gauss_noise:
            x_test = add_gauss_noise(x_test, stv)
        if struct_noise:
            x_test = add_struct_noise(x_test, eps)        
        m, st = np.mean(x_train, axis = (0)), np.std(x_train, axis = (0))
        if per_pixel_center:    
            x_train -= m; x_test -= m; 
        if per_pixel_std:
            x_train /= st; x_test /= st;
            
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(segment_augment).shuffle(100).batch(BATCH_SIZE)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(100).batch(BATCH_SIZE) 
        
    elif dataset=='oxfordpet':   #not fully tested oxfordpet
        print("Using untested dataset oxfordpet")
        ds, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
        train_ds = ds['train'].map(load_oxford).map(segment_augment).shuffle(100).batch(BATCH_SIZE)
        test_ds = ds['test'].map(load_oxford).shuffle(100).batch(BATCH_SIZE)
        return train_ds, test_ds, info.splits['train'].num_examples, 3
    
    else:
        raise ValueError("Wrong dataset name") 
    
    return train_ds, test_ds, len(x_train), np.max(y_train)+1

def get_label(file_path):
    """
    Get label of the Q-tips images from the directory names
    """
    parts = tf.strings.split(file_path, os.path.sep)  # convert the path to a list of path components
    return tf.expand_dims((int(parts[-2])-1),0)
    #return np.array([int(parts[-2])], dtype = np.uint8)
    #return int(parts[-2])
    #return parts[-2] == CLASS_NAMES # The second to last is the class-directory

def decode_img(img):
    """
    Load images of the Q-tips dataset
    """
    img = tf.image.decode_png(img, channels=3)  # convert the compressed string to a 3D uint8 tensor
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [canvasSize[0], canvasSize[1]])  # resize the image to the desired size.

def process_path(file_path):
    """
    Get label and image from Q-tips directory
    """
    label = get_label(file_path)
    img = tf.io.read_file(file_path)   # load the raw data from the file as a string
    img = decode_img(img)
    return img, label

def prepare_for_training(ds, BATCH_SIZE, cache=True, noise = True, shuffle_buffer_size=1000):
    """
    Batch the data and add some Gaussian noise to the Q-tips images
    Args:
        ds (tf.data.Dataset): tensorflow dataset
        BATCH_SIZE (int): mini-batch size
        Rest of the arguments are left unchanged.
    """
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    if noise:
        ds = ds.map(qtips_noise_tf)
    #ds = ds.repeat()  # Repeat forever
    ds = ds.batch(BATCH_SIZE)
    #ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

def qtips_noise_tf(x, y):
    """
    Add noise to Q-tips images
    """
    noise_values = tf.random.normal(x.shape, mean = 0, stddev=0.3, dtype=tf.float32)
    x = x + noise_values
    x = tf.clip_by_value(x, clip_value_min=0, clip_value_max=1)
    #np.putmask(image, image < 0, 0)
    #np.putmask(image, image > 1, 1)
    return x, y

def get_qtips_tf(traindir, testdir, BATCH_SIZE, noise = True):
    """
    Get Q-tips batched dataset.
    Source: https://www.tensorflow.org/tutorials/load_data/images
    Args:
        traindir (str): path to directory with training images
        testdir (str): path to directory with test images
        BATCH_SIZE (int): mini-batch size
        noise (bool): add Gaussian noise
    Returns:
        train_ds (tf.data.Dataset): training dataset
        test_ds (tf.data.Dataset): test dataset
        num_images (int): number of training images
    """
    pic = Image.open(str(traindir+'/1/'+os.listdir(str(traindir+'/1'))[0])) #open first image in folder "1" to check image size
    global canvasSize                                    #canvasSize stores the image resolution of Q-tips images
    canvasSize = np.array(pic.convert("RGB")).shape
    
    data_dir1 = pathlib.Path(traindir)
    list_ds1 = tf.data.Dataset.list_files(str(data_dir1/'*/*'))
    data_dir2 = pathlib.Path(testdir)
    list_ds2 = tf.data.Dataset.list_files(str(data_dir2/'*/*'))
        
    labeled_ds1 = list_ds1.map(process_path, num_parallel_calls= tf.data.experimental.AUTOTUNE)
    labeled_ds2 = list_ds2.map(process_path, num_parallel_calls= tf.data.experimental.AUTOTUNE)
    train_ds = prepare_for_training(labeled_ds1, BATCH_SIZE, noise = noise)
    test_ds = prepare_for_training(labeled_ds2, BATCH_SIZE, noise = noise)
    
    num_images = sum([len(files) for r, d, files in os.walk(traindir)])
    return train_ds, test_ds, num_images

def iou_fn(y_true, y_pred, smooth=1, num_classes=20):
    """
    User-defined IoU metric.
    The in-built IoU metric from tensorflow sets the fraction to zero, if the denominator is 0, i.e.
    if the class is not present in a batch of images, which leads to an underestimation of IoU values.
    There is some debate on whether the fraction should be 0 or 1: https://github.com/tensorflow/tensorflow/issues/12294
    We use the continuous approximation of IoU metric.
    See the formula and the implementation here:
        https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/DNN_IOU_SEGMENTATION.pdf
        https://ilmonteux.github.io/2019/05/10/segmentation-metrics.html   
    Args:
        y_true (numpy array or tf.Tensor): true labels for the pixels in sparse form i.e [0,2,4,1,...]
        y_pred (numpy array or tf.Tensor): predicted labels for the pixels. Each pixel has 'n' logits, where n is the number of classes.
                                           Logits are predictions made by the network for each pixel before the softmax classifier is applied.
        smooth (int): value to be added if the denominator is zero, i.e. the class doesn't show up in the batch
        num_classes (int): number of classes in the dataset
    Returns:
        iou (float): IoU metric for a batch of images and its predictions
    """
    y_true = tf.cast(tf.reshape(y_true, [y_true.shape[0], y_true.shape[1], y_true.shape[2]]), tf.int32)
    y_true = tf.one_hot(y_true, num_classes)
    y_pred = tf.nn.softmax(y_pred)

    intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1,2,3])
    union = tf.reduce_sum(y_true,[1,2,3]) + tf.reduce_sum(y_pred,[1,2,3]) - intersection
    iou = tf.reduce_mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def label_to_image(labels):
    '''
    Converts the labels of bdd100k dataset to an RGB-image.
    Color coding similar to Cityscapes dataset: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py    
    Args:
        labels (2D numpy array): label for each pixel in an image.  
        e.g.: [[0,10,4,3,2,...],[...],[...]]
    Returns:
        numpy array: 3-channel RGB-image
    '''
    # set of colours (similar to Cityscapes Dataset)
    colors = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
               (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
               (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
               (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
               (0, 0, 230), (119, 11, 32), (0, 0, 0)]
    color_img = np.tile(labels[:,:], 3)
    for i in range(len(colors)):
        color_img = np.where(color_img==i, colors[i], color_img)
    return color_img/255.   

def load_oxford(datapoint, image_size=64):
    '''
    Load the Oxford-IIIT Pet Dataset from tensorflow_datasets.
    Source: https://www.tensorflow.org/tutorials/images/segmentation#define_the_model
    Args:
        datapoint: tensorflow dataset datapoint
    image_size (int): resize images to dimension (image_size, image_size)
    Returns:
        input_image, input_mask: tensorflow dataset split into images and masks
    '''
    # resize images
    input_image = tf.image.resize(datapoint['image'], (image_size, image_size))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (image_size, image_size))                            
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

    
def qtips_noise_keras(image):
    """
    Add noise to Q-tips images, keras version of the function 'qtips_noise_tf'.
    Not used in out experiments.
    """
    noise_values = tf.random.normal(image.shape, mean = 0, stddev=15, dtype=tf.float32)
    image = image + noise_values
    image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=255)
    #np.putmask(image, image < 0, 0)
    #np.putmask(image, image > 1, 1)
    return image

def get_qtips_keras(traindir, testdir, BATCH_SIZE, noise = True):
    """
    Get Q-tips batched dataset, keras version of the function 'get_qtips_tf'. 
    This function is slower than the tensorflow version. Hence, it is not used for our experiments.
    """
    pic = Image.open(str(traindir+'/1/'+os.listdir(str(traindir+'/1'))[0])) #open first image in folder "1" to check image size
    canvasSize = np.array(pic.convert("RGB")).shape
    if noise:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip = False, rescale = 1./255, data_format = "channels_last", preprocessing_function = qtips_noise_keras)
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255, data_format = "channels_last", preprocessing_function = qtips_noise_keras)
    else:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip = False, rescale = 1./255, data_format = "channels_last")
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255, data_format = "channels_last")

    train_ds = train_datagen.flow_from_directory(traindir, color_mode="rgb", batch_size=BATCH_SIZE, class_mode="sparse", 
                                                    shuffle = True, target_size = (canvasSize[0], canvasSize[1]))
                                                    
    test_ds = test_datagen.flow_from_directory(testdir, color_mode="rgb", batch_size=BATCH_SIZE, class_mode = "sparse", 
                                                    shuffle = True, target_size = (canvasSize[0], canvasSize[1]))
    
    num_images = sum([len(files) for r, d, files in os.walk(traindir)])
    return train_ds, test_ds, num_images