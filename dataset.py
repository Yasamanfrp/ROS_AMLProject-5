import torch
import torch.utils.data as data
from PIL import Image
from random import random
import random
import torchvision.transforms.functional as TF
import numpy as np



def _dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


class Dataset(data.Dataset):
    def __init__(self, names, labels, path_dataset,img_transformer=None):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer

    def __getitem__(self, index):

        "here we access the images and process them"
        "first of all we want to have the path of the images so we just read the .txt file"
        #open the path file and convert
        img = Image.open(self.data_path+self.names[index])
        img = img.convert('RGB')
        #Applies preprocessing when accessing the image
        img = self._image_transformer(img)

        "here we produce the corresponded labels of roatation angels"
        "0°, 90°, 180°, 270° that correspond to labels 0, 1, 2, 3"
    
        
        #label_rot is the number of rotations==>
        #90° roation means we rotate the image one time
        
        label_rot = (random.randint(0,3))
        index_rot = label_rot
        img_rot = torch.rot90(img,k=label_rot,dims=[1,2])
        return img, int(self.labels[index]), img_rot, index_rot
       

    def __len__(self):
        return len(self.names)


"we need to return test dataset and process it the same as Dtaset"
class TestDataset(data.Dataset):
    def __init__(self, names, labels, path_dataset,img_transformer=None):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer

    def __getitem__(self, index):

        img = Image.open(self.data_path+self.names[index])
        img = img.convert('RGB')
        img = self._image_transformer(img)
        label_rot = (random.randint(0,3))
        index_rot = label_rot
        img_rot = torch.rot90(img,k=label_rot,dims=[1,2])
        return img, int(self.labels[index]), img_rot, index_rot
       

    def __len__(self):
        return len(self.names)

