# Deep Neural Networks and PIDE discretizations - Nonlocal CNNs inspired by PIDEs
-Bastian Bohn, Michael Griebel, Dinesh Kannan

-ReadMe for the code.

Convolutional neural networks with nonlocal interactions. Based on Hamiltonian networks by [Chang, Meng, Haber et al.](https://arxiv.org/abs/1709.03698).
Nonlocal Blocks inserted in each Unit of the network. 

## Requirements
- Python 3.5+
- pip 19.0 or later
- Numpy 1.19.1+
- Matplotlib (for plotting)
- Tensorflow 2.3 (ideally). 
The tests for BDD100K might not work properly for earlier versions (Tensorflow 2.1 etc.) because tensorflow addons are used here, which depend a bit on tf versions.
- tensorflow-datasets-nightly: `pip install tfds-nightly`
- tensorflow addons: `pip install tensorflow-addons`
- Tensorflow GPU requirements such as cuDNN/CUDA etc. (https://www.tensorflow.org/install/gpu). Check correct CUDA/cuDNN version corresponding to the Tensorflow version [here](https://www.tensorflow.org/install/source#gpu_support_3). Tested on Ubuntu 18.04, but for other operating systems, check [here](https://www.tensorflow.org/install/pip).
- OpenCV: `pip install opencv-python`
- pathlib (for Q-tips): `pip install pathlib` 
- PIL pillow: `pip install Pillow`
- torchvision (for generating Q-tips dataset) (torch is optional): `pip install torch torchvision` 

Quick installation:

For Google Colab users: `pip install -r requirements.txt`

For non-Colab users: Remove all the irrelevant packages from requirements.txt file and run `pip install -r requirements.txt`

## Usage

### 1. Q-tips curves
The Q-tips dataset can be generated by running `python qtips_gen.py`. This will create a 'data' folder with the training and test images in the 'Nonlocal_CNN' folder. An example dataset is provided in the CD because the dataset doesn't take that much storage space. If you now run the python file, then the 'data' file will be overwritten.
#### 1.1 Using ResNets
Run `python qtips_resnet_test.py` to train ResNet-44 on Q-tips dataset, i.e. `n`=7, `version`=1.

With optional arguments:<br/>

`python qtips_resnet_test.py --n 7 --version 1 --epochs 30 --save_curve True --save_model True` \

- `n` is the number of Residual Blocks in a Residual Unit.  \
- `version` is 1 for ResNets and 2 for PreResNets. \
 Use n=7 and version=1 for ResNet-44, n=2 and version=2 for PreResNet-20. Use n=18 and version=1 for ResNet-110, n=6 and version=2 for PreResNet-56, etc. \
- `epochs` training epochs can be set to a value other than 30. \
- `save_curve` can be passed as True to store the training and test accuracies for each epoch. A .npz file will be stored in the 'saved_models' folder with a name similar to 'testtraindata_YYYY-MM-DD_HH:MM:SS.npz'\
To load the file, use: \
`data=np.load('saved_models/stl10_Hamiltonian_nltype_4/traintestdata_2020-09-12_07:49:38')`\
Now one can use, `data['test_acc']`, `data['train_acc']` to access the numpy arrays.
Change filename appropriately. e.g. 'save_models/qtips_ResNet_20_version_2/...'

- `save_model` can be passed as True and the model weights are saved if a certain threshold of test accuracy is crossed. Change the THRESHOLD variable in file 'qtips_resnet_test.py' if needed. A file with model weights in the 'saved_models' folder will be stored with a name similar to 'EPOCH_EEE_acc_71.24'\
To see how to save and load model weights, check: https://www.tensorflow.org/guide/keras/save_and_serialize <br/>
Example to load weights:<br/>
First train the same model for an epoch or so. Or just build it using model.build. Then use \
`model.load_weights('saved_models/stl10_Hamiltonian_nltype_4/model_2020-09-12_07:49:38/EPOCH_162_acc_80.60')` <br/> to load the model weights that were saved after the 162nd training epoch. Change filename appropriately. e.g. 'save_models/qtips_ResNet_44_version_1/.../...'

#### 1.2 Using Nonlocal Hamiltonian networks
Run `python qtips_nonlocal_test.py --nonlocal_typ 1` to train Nonlocal Hamiltonian networks on Q-tips dataset.

`nonlocal_typ` can be between 0 and 4. 0 stands for the original Hamiltonian Network **without** any Nonlocal Block in each Unit. 1 is for the nonlocal diffusion operator in the Nonlocal Block. 2 is for $\Delta^s$, 3 is for $\Delta^{-s}$, 4 is for $\Delta^{-1}$. 

The value of $s$ is kept at 0.5 for Q-tips experiments. \

With optional arguments:<br/>

`python qtips_nonlocal_test.py --nonlocal_typ 2 --h 0.05 --block_typ Hamiltonian --epochs 50 --save_curve True --save_model True` \

- `h` Discretization hyperparameter and step size. Default 0.05.
- `block_typ` 'Hamiltonian', 'Midpoint', 'Parabolic' normal Blocks. Only **Hamiltonian** is fully tested so far and is the default value.
- `epochs` training epochs can be set to a value other than 30.
- `save_curve` can be passed as True to store the training and test accuracies for each epoch. See above regarding how to load the saved train and test accuracy arrays (same as ResNets). Change filename appropriately.
- `save_model` can be passed as True and the model weights are saved if a certain threshold of test accuracy is crossed. Change the THRESHOLD variable in file 'qtips_nonlocal_test.py' if needed. See above regarding how to load the saved model weights (same as ResNets). Change filename appropriately.

### 2. Benchmark datasets experiments
#### 2.1 Preparation for BDD100K
For the segmentation task on BDD100K, we need to prepare the data. For other datasets, skip this step.

1. Download the data from [here](https://bdd-data.berkeley.edu/). Download only the 'Segmentation' part of the dataset which is around 1.2GB in size. Unfortunately, the file is too big to be put on the CD that I submit.
2. Store this 'bdd100k' folder in the folder 'Nonlocal_CNN', i.e. the path should look like this: 'Nonlocal_CNN/bdd100k/seg/...'
3. Now run the file 'bdd100k_to_array.py': <br/>
`python bdd100k_to_array.py` \
This will store two .npz files in the directory 'Nonlocal_CNN/bddarray_resol_decr3/...' <br/>
The 'decr3' suggests that the 1280x720 resolution is reduced by a factor of 2<sup>3</sup>, i.e. to a size of 160x90. These two .npz files will be used from now on for all the BDD100K experiments, which is faster than going through the entire data of high-resolution images, every time we run an experiment with BDD100K.

#### 2.2 Using ResNets for benchmark datasets
Run `python resnet_test.py --dataset stl10` to train ResNet-44 on a particular dataset, i.e. `n`=7, `version`=1. You can pass the following dataset arguments: 'cifar10', 'cifar100', 'stl10', 'bdd100k' <br/>

With optional arguments:<br/>

`python resnet_test.py --dataset cifar10 --n 7 --version 1 --epochs 30 --save_curve True --save_model True --gauss_noise True --frac 0.5` \

- `n` is the number of Residual Blocks in a Residual Unit.  \
- `version` is 1 for ResNets and 2 for PreResNets. \
 Use n=7 and version=1 for ResNet-44, n=2 and version=2 for PreResNet-20. Use n=18 and version=1 for ResNet-110, n=6 and version=2 for PreResNet-56, etc. \
- `epochs` training epochs can be set to a value other than 200. \
- `save_curve` can be passed as True to store the training and test accuracies for each epoch. A .npz file will be stored in the 'saved_models' folder with a name similar to 'testtraindata_YYYY-MM-DD_HH:MM:SS.npz'\
To load the file, use: \
`data=np.load('saved_models/stl10_Hamiltonian_nltype_4/traintestdata_2020-09-12_07:49:38')`\
Now one can use, `data['test_acc']`, `data['train_acc']`, `data['train_iou']`, `data['test_iou']` to access the numpy arrays with the metrics for each epoch of training.
Change filename appropriately. e.g. 'save_models/cifar100_ResNet_20_version_2/...'

- `save_model` can be passed as True and the model weights are saved if a certain threshold of test accuracy is crossed. Change the THRESHOLD variable in file 'resnet_test.py' if needed. A file with model weights in the 'saved_models' folder will be stored with a name similar to 'EPOCH_EEE_acc_71.24'\
To see how to save and load model weights, check: https://www.tensorflow.org/guide/keras/save_and_serialize <br/>
Example to load weights: <br/>
First train the same model for an epoch or so. Or just build it using model.build. Then use \
`model.load_weights('saved_models/stl10_Hamiltonian_nltype_4/model_2020-09-12_07:49:38/EPOCH_162_acc_80.60')` <br/> to load the model weights that were saved after the 162nd training epoch. Change filename appropriately. e.g. 'save_models/cifar100_ResNet_44_version_1/.../...'

- `gauss_noise`: Whether to add Gaussian noise to test images while checking for robustness to noise. Default is False.
- `frac`: fraction of the training data to be used while checking for robustness to training data subsampling. Number >0 and <=1. Default is 100%, i.e. 1.

#### 2.2 Using Nonlocal Hamiltonian networks for benchmark datasets
Run `python nonlocal_test.py --dataset cifar100 --nonlocal_typ 1` to train Nonlocal Hamiltonian networks on a particular dataset. You can pass the following dataset arguments: 'cifar10', 'cifar100', 'stl10', 'bdd100k' <br/>

`nonlocal_typ` can be between 0 and 4. 0 stands for the original Hamiltonian Network **without** any Nonlocal Block in each Unit. 1 is for the nonlocal diffusion operator in the Nonlocal Block. 2 is for $\Delta^s$, 3 is for $\Delta^{-s}$, 4 is for $\Delta^{-1}$. \

With optional arguments:<br/>

`python nonlocal_test.py --dataset cifar100 --nonlocal_typ 2 --h 0.05 --epochs 50 --block_typ Hamiltonian --num_blocks 18 --s 0.25 --save_curve True --save_model True --gauss_noise True --frac 0.25` \

- `h` Discretization hyperparameter and step size. Default 0.06.
- `block_typ` 'Hamiltonian', 'Midpoint', 'Parabolic' normal Blocks. Only **Hamiltonian** is fully tested so far as is the default value.
- `epochs` training epochs can be set to a value other than 200.
- `save_curve` can be passed as True to store the training and test accuracies for each epoch. See above regarding how to load the saved train and test accuracy arrays (same as ResNets). Change filename appropriately.
- `save_model` can be passed as True and the model weights are saved if a certain threshold of test accuracy is crossed. Change the THRESHOLD variable in file 'nonlocal_test.py' if needed. See above regarding how to load the saved model weights (same as ResNets). Change filename appropriately.
- `num_blocks`: Number of normal (Hamiltonian) Blocks blocks in each Unit. Default is 6, i.e. 6-6-6 architecture from the thesis. Set it to 18, etc. for deeper networks.
- `s`: power of the fractional or inverse fractional laplacian. Default is 0.5. Set it to a value >0 and &lt;1
- `gauss_noise`: Whether to add Gaussian noise to test images while checking for robustness to noise. Default is False.
- `frac`: fraction of the training data to be used while checking for robustness to training data subsampling. Number >0 and <=1. Default is 100%, i.e. 1.

### 3. Outputs for the network training
The outputs of the training should look similar to:
```
...
Epoch 29, Loss: 1.0556, Accuracy: 94.3359, Test Loss: 2.6210, Test Accuracy: 36.7188, Time: 2020-09-18 07:39:36
Epoch 30, Loss: 1.0734, Accuracy: 92.3828, Test Loss: 2.4197, Test Accuracy: 40.6250, Time: 2020-09-18 07:39:42
...
```
The IoU is automatically also printed (not in the example above) if the dataset is chosen as 'bdd100k'.

### 4. FLOPs calculation
To calculate the FLOPs of the models, run:
#### 4.1 For ResNets:
`python flops_resnet.py --dataset cifar100`

This calculates the FLOPs for ResNet-44, i.e. `n`=7, `version`=1. <br/>
`dataset` can also be set as 'cifar10', 'cifar100', 'stl10' and 'bdd100k'.

With optional arguments:

`python flops_resnet.py --dataset cifar100 --n 7 --version 1`

- `n` and `version`: Use n=7 and version=1 for ResNet-44, n=2 and version=2 for PreResNet-20. Use n=18 and version=1 for ResNet-110, n=6 and version=2 for PreResNet-56, etc.

#### 4.2 For Nonlocal Hamiltonian networks:

`python flops_nonlocal.py --nonlocal_typ 3 --dataset cifar10`

`nonlocal_typ` can be set between 0 and 4 as mentioned above.
`dataset` can also be set as 'cifar10', 'cifar100', 'stl10' and 'bdd100k'.

With optional arguments:

`python flops_nonlocal.py --nonlocal_typ 3 --dataset cifar10 --num_blocks 18`

- `num_blocks`: Default is 6. Set to any other value for calculating FLOPs of different network depths.

The FLOPs can be reduced in any model by increasing the pool size of the subsampling in the Nonlocal Block. To do this, change the value of `self.nl_pool_size` in 'Networks.py' in the Module folder. If the FLOPs are too high for a given CPU/GPU, increase the pool size. The performance is relatively worse, but nevertheless better than the original Hamiltonian network.

The FLOPs of the bdd100k network are relatively high because the implementation doesn't involve pooling layers etc., which is necessary while dealing with high-resolution images. The experiments for bdd100k were just to demonstrate the potential in image segmentation. More efficient implementations with skip connections (U-Net, etc.) might mitigate the issue partially at least.

The outputs after running these files look like the following:
```
...
Model: Hamiltonian_type_3
Number of FLOPs (MACC): 2011.01579 M
Number of trainable parameters: 0.555562 M
...
```

