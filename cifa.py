from torchvision.datasets import CIFAR10
import os
from torch.utils.data import Dataset
import pickle
import numpy as np
import cv2
import torch.nn as nn
from torchvision.transforms import ToTensor
train_dataset = CIFAR10( './cifar',train = True)
"""
image, label = train_dataset.__getitem__(2000)
print(label)
image.show()
"""
# print(train_dataset.class_to_idx)
class MyDataset(Dataset):
    def __init__(self, root ='cifar/cifar-10-batches-py', train = True, transform = None):
        self.root = root
        if train == True:
            data_files = [os.path.join(root,"data_batch_{}".format(i)) for i in range(1,6)]
            # join all the files in the root file
        else:
            data_files = [os.path.join(root,"test_batch")]
        # print(data_files)
        self.images = []
        self.labels = []
        for data_file in data_files:
            with open(data_file, "rb") as fo:
                data = pickle.load(fo, encoding= "bytes")
                self.images.extend(data[b'data'])
                self.labels.extend(data[b'labels'])
                #flatten the files, if we use append it will return 5 because the picture is an element in the files
        #print(len(self.labels))
        #print(len(self.images))
    def __len__(self):
            return len(self.labels)
        
    def __getitem__(self,item):
            image = self.images[item].reshape((3,32,32)).astype(np.float32)
            label = self.labels[item]
            return image/255., label

    """
            print(len(data[b'data']))
            print(len(data[b'labels']))
            print('---------------')
            #print(data.keys())
    """
if __name__ == '__main__':
    """
    dataset = MyDataset(root= "cifar/cifar-10-batches-py", train = True)
    image, label = dataset.__getitem__(234)
    #image = np.reshape(image,(32,32,3))
    image = np.reshape(image,(3,32,32))
    image = np.transpose(image,(1,2,0))
    print(image.shape)
    print(label)
    cv2.imshow("image",cv2.resize(image,(320,320)))
    cv2.waitKey(0)
    """
    