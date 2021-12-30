#generate qtips data
#Modified from: https://github.com/HaberGroup/SemiImplicitDNNs
import PIL.Image as Image
import numpy as np
import torchvision.transforms as transforms
import os
import shutil
    
def main():
    """
    Generates the Q-tips train and test images in a folder named 'data'
    """
    canvasSize = np.array([64,64,3]) #image resolution
    minLength = 32; maxLength = 60   #length of stick
    minWidth = 4; maxWidth = 6       #width of stick
    bgVal = 112; bgClass = 0; objVal = 30  #background value and stick midsection value
    nClasses = 15
        
    def image_gen():  #returns the generated image and the corresponding label
        data_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(180),
                transforms.ToTensor()
            ])          #random rotation
    
        image = np.zeros(canvasSize, dtype=np.uint8)
        image.fill(bgVal)
    
        length = np.random.randint(minLength, high = maxLength)
        width = np.random.randint(minWidth, high = maxWidth)
        target = np.random.randint(1, high=nClasses + 1)
    
        # Find the top left corner
        x = canvasSize[0]//2 - length//2
        y = canvasSize[1]//2 - width//2
    
        # Place the object
        image[y:(y+width), x:(x+length), :] = objVal
        #label[y:(y+width), x:(x+length)] = target
    
        if target==1:
            # red-red category
            image[y:(y+width), x:(x+width),0] = 255 # Left marker
            image[y:(y+width), (x+length-width):(x+length), 0] = 255 # Right marker
    
        elif target==2:
            # red-green category
            image[y:(y+width), x:(x+width),0] = 255 # Left marker
            image[y:(y+width), (x+length-width):(x+length), 1] = 255 # Right marker
    
        elif target==3:
            # red-blue category
            image[y:(y+width), x:(x+width),0] = 255 # Left marker
            image[y:(y+width), (x+length-width):(x+length), 2] = 255 # Right marker
    
        elif target==4:
            # red-yellow category
            image[y:(y+width), x:(x+width),0] = 255 # Left marker
            image[y:(y+width), (x+length-width):(x+length), 0] = 255 # Right marker
            image[y:(y+width), (x+length-width):(x+length), 1] = 255 # Right marker
            
        elif target==5:
            # red-pink category
            image[y:(y+width), x:(x+width),0] = 255 # Left marker
            image[y:(y+width), (x+length-width):(x+length), 0] = 255 # Right marker
            image[y:(y+width), (x+length-width):(x+length), 2] = 255 # Right marker
    
        elif target==6:
            # green-green category
            image[y:(y+width), x:(x+width),1] = 255 # Left marker
            image[y:(y+width), (x+length-width):(x+length), 1] = 255 # Right marker
    
        elif target==7:
            # green-blue category
            image[y:(y+width), x:(x+width),1] = 255 # Left marker
            image[y:(y+width), (x+length-width):(x+length), 2] = 255 # Right marker
    
        elif target==8:
            # green-yellow category
            image[y:(y+width), x:(x+width),1] = 255 # Left marker
            image[y:(y+width), (x+length-width):(x+length), 0] = 255 # Right marker
            image[y:(y+width), (x+length-width):(x+length), 1] = 255 # Right marker
            
        elif target==9:
            # green-pink category
            image[y:(y+width), x:(x+width),1] = 255 # Left marker
            image[y:(y+width), (x+length-width):(x+length), 0] = 255 # Right marker
            image[y:(y+width), (x+length-width):(x+length), 2] = 255 # Right marker
    
        elif target==10:
            # blue-blue category
            image[y:(y+width), x:(x+width),2] = 255 # Left marker
            image[y:(y+width), (x+length-width):(x+length), 2] = 255 # Right marker
    
        elif target==11:
            # blue-yellow category
            image[y:(y+width), x:(x+width),2] = 255 # Left marker
            image[y:(y+width), (x+length-width):(x+length), 0] = 255 # Right marker
            image[y:(y+width), (x+length-width):(x+length), 1] = 255 # Right marker
            
        elif target==12:
            # blue-pink category
            image[y:(y+width), x:(x+width),2] = 255 # Left marker
            image[y:(y+width), (x+length-width):(x+length), 0] = 255 # Right marker
            image[y:(y+width), (x+length-width):(x+length), 2] = 255 # Right marker        
    
        elif target==13:
            # yellow-yellow category
            image[y:(y+width), x:(x+width),0] = 255 # Left marker
            image[y:(y+width), x:(x+width),1] = 255 # Left marker
            image[y:(y+width), (x+length-width):(x+length), 0] = 255 # Right marker
            image[y:(y+width), (x+length-width):(x+length), 1] = 255 # Right marker
            
        elif target==14:
            # yellow-pink category
            image[y:(y+width), x:(x+width),0] = 255 # Left marker
            image[y:(y+width), x:(x+width),1] = 255 # Left marker
            image[y:(y+width), (x+length-width):(x+length), 0] = 255 # Right marker
            image[y:(y+width), (x+length-width):(x+length), 2] = 255 # Right marker     
            
        elif target==15:
            # pink-pink category
            image[y:(y+width), x:(x+width),0] = 255 # Left marker
            image[y:(y+width), x:(x+width),2] = 255 # Left marker
            image[y:(y+width), (x+length-width):(x+length), 0] = 255 # Right marker
            image[y:(y+width), (x+length-width):(x+length), 2] = 255 # Right marker
    
        else:
            raise Exception('Target value out of range')
    
        image = data_transforms(image).permute(1,2,0)
        image = (image*255).numpy().astype(np.uint8)
        image[np.logical_and(image[:,:,0]==0, image[:,:,1]==0, image[:,:,2]==0)]=bgVal
        
        return image, target
    
    def generator(N, datadir, overwrite = True, updateRate = 100):  #generates several image,label pairs and save it in 'datadir' directory
        if os.path.isdir(datadir):               # Check if dir exists, if not make it
            if overwrite:
                print('Overwriting %s' % datadir)
                shutil.rmtree(datadir)
            else:
                raise Exception('Datadir already exists and overwiting is turned off: %s' % datadir)
        else:
            print('Creating Output Directory: %s' % datadir)
            
        Path = os.path.join(datadir)
        os.makedirs(Path)
        
        for i in range(nClasses):
            subdirectory = os.path.join(datadir, str(i+1))    #generate subdirectories for each class
            os.makedirs(subdirectory)
    
        # Generate images
        print('Generating %d examples...' % N)
        filenames = []
    
        for i in range(N):
    
            if ((i+1)%updateRate == 0) and not (i==0):  #print after creation of certain number of images
                print('%6d' % (i+1))
    
            image, label = image_gen()
    
            # Save image and label
            img = Image.fromarray(image)
            lab = label
    
            filename = '%06d.png' % i
            img.save(os.path.join(Path, str(lab), filename))
            filenames.append(filename)
    
        # Save list of filenames for dataloading later
        fileListPath = os.path.join(datadir, 'filenames.txt')
        print('Saving Filelist to %s' % fileListPath)
        with open(fileListPath, 'w') as f:
            for item in filenames:
                f.write("%s\n" % item)
                
    generator(512, 'data/train/')
    generator(512, 'data/val/')

if __name__ == '__main__':
    main()