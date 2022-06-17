import numpy as np;
import torch

import torch.utils.data as data
import binvox_rw;
import os
import random
import cv2


'''
    Description of File:
    ---------------------
        This file is all code relevant to interpreting the ShapeNet dataset for PyTorch to understand while training.
        
        The training data is a set of multiple files (in this case, I decided to set a constant amount because I ran into issues trying
            to train with variable images for training datapoint.
        
        The dataset class requires __len__ and __getitem__ as overloaded operators for this to work with PyTorch. 
            __len__ is to understand training data amount.
            __getitem__ is to understand how to take data from ShapeNet
            
'''

class R2N2_Data(data.Dataset):
    def __init__(self, path_to_data, path_to_labels, k=5):
        '''
            path_to_data: 'str' as the name entails. this leads right to the ShapeNet data directory.
                the file hierarchy should be path_to_data/hash/rendering/
                
            path_to_labels: 'str' as the name entails. this leads to the 3d reconstruction of each datapoint.
                the file hierarchy should be path_to_labels/hash/model.binvox
            
            k: number of images to train from. default if 5 because that's my project. hehe
                constraint: 0 < k < 24
        '''
        self.root_data = path_to_data
        self.root_labels = path_to_labels
        self.dataPointsFileNames = os.listdir(path_to_data)
        self.n = len(self.dataPointsFileNames)
        self.k = k # desired images to take
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        '''
            NOTE:
            ------
            whenever getitem is called, this is what happens behind the scenes.
            I did not want to use too much memory, so everything that is used is just strings and python "os" libarry
            doing some work.
            
            compared to loading thousands of images, this is much much cheaper in memory.
        '''
        directory = self.root_data + self.dataPointsFileNames[idx]
        os.listdir(directory)
        all_png = [directory+'/rendering/'+i for i in os.listdir(directory + '/rendering/') if i.endswith('.png')]
        choices = random.sample(all_png,k=self.k)
        images = []
        for path in choices:
            im = cv2.imread(path)
            im = cv2.resize(im, (127,127), interpolation=cv2.INTER_AREA)
            im.transpose(2,0,1)
            im = torch.tensor(np.rollaxis(im,2,start=0), device='cuda').float()
            images.append(im)
        with open(self.root_labels + self.dataPointsFileNames[idx] + '/model.binvox','rb') as f:
            m1 = binvox_rw.read_as_3d_array(f)
        occ_grid = m1.data
        occ_grid = torch.tensor(occ_grid.astype(np.uint8)).long()
        occ_grid = occ_grid.cuda()
        
        data_dict = {'data': images, 'label': occ_grid}
        return data_dict
