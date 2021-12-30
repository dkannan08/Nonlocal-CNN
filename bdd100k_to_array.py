import numpy as np
import os
import cv2

def save_bdd(train_or_val='train', resol_decr=3):
    '''
    Converts the bdd100k data (for image segmentation) to numpy arrays and stores it in .npz file.  Resizes the images if resol_decr>0.
    This function might take some time to execute. Please be patient. Run it on Nvidia Tesla V100, etc. for faster execution.

    Args:
        train_or_val: 'train' or 'val' to fetch training or validation data

    resol_decr: int
    if resol_decr = 0, original resolution is maintained, if equal to 1, every second row and column taken
    if = 2, this process is repeated twice, i.e. every second row and column is taken and the same is done again to the resulting image
    if = 3,4.. same iterative proceduce follows.
    This way, pixel-wise labels stay the same.
    Also the reason why interpolation is not a good idea here.
    Returns:
        numpy arrays: containing resized images and corresponding labels
    '''

    x, y = [], []
    if train_or_val == 'train':
        image_dir = './bdd100k/seg/images/train/'
        label_dir = './bdd100k/seg/labels/train/'
    else:
        image_dir = './bdd100k/seg/images/val/'
        label_dir = './bdd100k/seg/labels/val/'

    for f in os.listdir(image_dir):
        img = cv2.imread(image_dir+f)
        lab = cv2.imread(label_dir + f[:-4] +'_train_id.png') #read the corresponding labels of the image
        #decrease the resolution
        img = img[2**resol_decr-1::2**resol_decr, 2**resol_decr-1::2**resol_decr,:]
        lab = lab[2**resol_decr-1::2**resol_decr, 2**resol_decr-1::2**resol_decr,:]
        x.append(img)
        y.append(lab[:,:,0, np.newaxis])
        del img, lab
    return np.array(x), np.array(y)

if __name__ == '__main__':
    resol_decr = 3
    x_train, y_train = save_bdd(train_or_val='train', resol_decr=resol_decr)
    x_test, y_test = save_bdd(train_or_val='val', resol_decr=resol_decr)

    save_path = './bddarray_resol_decr'+str(resol_decr)+'/'
    os.makedirs(save_path, exist_ok=True) 
    np.savez(save_path+'train_data.npz', images=x_train, labels=y_train)
    np.savez(save_path+'test_data.npz', images=x_test, labels=y_test)
