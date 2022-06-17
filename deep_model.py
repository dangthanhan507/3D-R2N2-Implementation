import torch
from torch import nn
import numpy as np

'''
    This is the deeper residual implementation of the 
'''

class DeepEncodeNetwork(nn.Module):
    def __init__(self):
        super(DeepEncodeNetwork, self).__init__()
        # NETWORK DEFINITION OF THE ENCODER PART OF NETWORK
        self.conv2d1_1 = nn.Conv2d(3, 96, (7,7),padding='same')
        self.conv2d1_2 = nn.Conv2d(96, 96, (3,3),padding='same')
        
        self.conv2d2_1 = nn.Conv2d(96, 128, (3,3),padding='same')
        self.conv2d2_2 = nn.Conv2d(128, 128, (3,3),padding='same')
        #extra resid
        self.conv2d2_3 = nn.Conv2d(96, 128, (1,1), padding='same')
        
        self.conv2d3_1 = nn.Conv2d(128, 256, (3,3), padding='same')
        self.conv2d3_2 = nn.Conv2d(256, 256, (3,3), padding='same')
        #extra resid
        self.conv2d3_3 = nn.Conv2d(128, 256, (1,1), padding='same')
        
        
        self.conv2d4_1 = nn.Conv2d(256, 256, (3,3),padding='same')
        self.conv2d4_2 = nn.Conv2d(256, 256, (3,3),padding='same')
        
        self.conv2d5_1 = nn.Conv2d(256, 256, (3,3), padding='same')
        self.conv2d5_2 = nn.Conv2d(256, 256, (3,3), padding='same')
        #extra resid
        self.conv2d5_3 = nn.Conv2d(256, 256, (1,1), padding='same')
        
        self.conv2d6_1 = nn.Conv2d(256, 256, (3,3), padding='same')
        self.conv2d6_2 = nn.Conv2d(256, 256, (3,3), padding='same')
        
        self.pool = nn.MaxPool2d( (2,2) )
        self.relu = nn.LeakyReLU()
        
        self.flat = nn.Flatten() # how have i never seen this before???????
        self.pool = nn.MaxPool2d( (2,2) )
        self.relu = nn.LeakyReLU()
        self.hidden = nn.Linear(256, 1024)
        
    def forward(self, x):
        x = self.conv2d1_1(x)
        x = self.relu(x)
        
        #First Layer
        x = self.conv2d1_2(x)
        x = self.relu(x)
        pool_x = self.pool(x)
        
        #Second
        x = self.conv2d2_1(pool_x)
        x = self.relu(x)
        x = self.conv2d2_2(x)
        x = self.relu(x)
        res_x = self.conv2d2_3(pool_x) # residual convolution
        x = x + res_x # add together current sequence with residual
        pool_x = self.pool(x)
        
        #Third
        x = self.conv2d3_1(pool_x)
        x = self.relu(x)
        x = self.conv2d3_2(x)
        x = self.relu(x)
        res_x = self.conv2d3_3(pool_x)
        x = x + res_x
        pool_x = self.pool(x)
        
        #Fourth
        x = self.conv2d4_1(pool_x)
        x = self.relu(x)
        x = self.conv2d4_2(x)
        x = self.relu(x)
        pool_x = self.pool(x)
        
        #Fifth
        x = self.conv2d5_1(pool_x)
        x = self.relu(x)
        x = self.conv2d5_2(x)
        x = self.relu(x)
        res_x = self.conv2d5_3(x)
        x = x + res_x
        pool_x = self.pool(x)
        
        #Sixth
        x = self.conv2d6_1(pool_x)
        x = self.relu(x)
        x = self.conv2d6_2(x)
        x = self.relu(x)
        x = x + pool_x
        x = self.pool(x)
        
        #Flatten and make ready to pass through
        x = self.flat(x)
        x = self.hidden(x)
        x = self.relu(x)
        return x
        

class GRU_Network(nn.Module):
    def __init__(self, N=4, N_h=128):
        # NETWORK DEFINITION OF THE RECURRENT PART OF NETWORK
        super(GRU_Network, self).__init__()
        self.conv3d1 = nn.Conv3d(N_h, N_h, (3,3,3),padding='same', bias=True)
        self.conv3d2 = nn.Conv3d(N_h, N_h, (3,3,3),padding='same', bias=True)
        self.conv3d3 = nn.Conv3d(N_h, N_h, (3,3,3),padding='same', bias=True)        
        
        self.hidden1 = nn.Linear(1024, N_h*N*N*N)
        self.hidden2 = nn.Linear(1024, N_h*N*N*N)
        self.hidden3 = nn.Linear(1024, N_h*N*N*N)
        self.N = N
        self.N_h = N_h
    def forward(self, x, h_prev):
        x_1 = self.hidden1(x)
        x_2 = self.hidden2(x)
        x_3 = self.hidden3(x)
        
        h_1 = self.conv3d1(h_prev)
        h_2 = self.conv3d2(h_prev)
        #h_3 = self.conv3d3(h_prev)
        
        if len(x_1.shape) == 2:
            rt = torch.sigmoid( x_1.reshape(x_1.shape[0],self.N_h,self.N,self.N,self.N) + h_1 )
            ut = torch.sigmoid( x_2.reshape(x_2.shape[0],self.N_h,self.N,self.N,self.N) + h_2 )
            gt = torch.tanh(x_3.reshape(x_3.shape[0],self.N_h,self.N,self.N,self.N) + self.conv3d3(rt * h_prev) )
        else:
            rt = torch.sigmoid( x_1.reshape(self.N_h,self.N,self.N,self.N) + h_1 )
            ut = torch.sigmoid( x_2.reshape(self.N_h,self.N,self.N,self.N) + h_2 )
            gt = torch.tanh(x_3.reshape(self.N_h,self.N,self.N,self.N) + self.conv3d3(rt * h_prev) )            
        final = ((1-ut) * h_prev) + (ut * gt)
        return final
class DeepDecodeNetwork(nn.Module):
    def __init__(self, N_h=128):
        # NETWORK DEFINITION OF THE DECODER PART OF NETWORK
        super(DeepDecodeNetwork, self).__init__() 
        self.relu = nn.LeakyReLU()
        self.softmax = torch.nn.LogSoftmax(dim=1)
        
        self.conv3d1_1 = torch.nn.Conv3d(128,128, (3,3,3), padding='same')
        self.conv3d1_2 = torch.nn.Conv3d(128,128, (3,3,3), padding='same')
        
        self.conv3d2_1 = torch.nn.Conv3d(128,128, (3,3,3), padding='same')
        self.conv3d2_2 = torch.nn.Conv3d(128,128, (3,3,3), padding='same')
        
        self.conv3d3_1 = torch.nn.Conv3d(128,64, (3,3,3), padding='same')
        self.conv3d3_2 = torch.nn.Conv3d(64,64, (3,3,3), padding='same')
        self.conv3d3_3 = torch.nn.Conv3d(128,64, (1,1,1), padding='same')
        
        self.conv3d4_1 = torch.nn.Conv3d(64,32, (3,3,3), padding='same')
        self.conv3d4_2 = torch.nn.Conv3d(32,32, (3,3,3), padding='same')
        self.conv3d4_3 = torch.nn.Conv3d(32,32, (3,3,3), padding='same')
        
        self.conv3d5_1 = torch.nn.Conv3d(32,2, (3,3,3), padding='same')
    def forward(self, x):
        pool_y = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        y = self.relu(pool_y)
        y = self.conv3d1_1(y)
        y = self.relu(y)
        y = self.conv3d1_2(y)
        y = self.relu(y)
        y = y + pool_y
        
        pool_y = nn.functional.interpolate(y, scale_factor=2, mode='nearest')
        y = self.conv3d2_1(pool_y)
        y = self.relu(y)
        y = self.conv3d2_2(y)
        y = self.relu(y)
        y = y + pool_y
        
        pool_y = nn.functional.interpolate(y, scale_factor=2, mode='nearest')
        y = self.conv3d3_1(pool_y)
        y = self.relu(y)
        y = self.conv3d3_2(y)
        y = self.relu(y)
        #residual connection
        res_y = self.conv3d3_3(pool_y)
        y = y + res_y
        
        y = self.conv3d4_1(y)
        prev_y = self.relu(y)
        y = self.conv3d4_2(prev_y)
        y = self.relu(y)
        
        prev_y = self.conv3d4_3(prev_y)
        y = y + prev_y
        
        y = self.conv3d5_1(y)
        final = self.softmax(y)
        return final

