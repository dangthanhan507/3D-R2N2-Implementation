import torch
from torch import nn
import numpy as np

class EncodeNetwork(nn.Module):
    def __init__(self):
        super(EncodeNetwork, self).__init__()
        self.conv2d1 = nn.Conv2d(3, 96, (7,7),padding='same')
        self.conv2d2 = nn.Conv2d(96, 128, (3,3),padding='same')
        self.conv2d3 = nn.Conv2d(128, 256, (3,3),padding='same')
        self.conv2d4 = nn.Conv2d(256, 256, (3,3),padding='same')
        self.conv2d5 = nn.Conv2d(256, 256, (3,3),padding='same')
        self.conv2d6 = nn.Conv2d(256, 256, (3,3),padding='same')

        self.pool = nn.MaxPool2d( (2,2) )
        self.relu = nn.LeakyReLU()
        self.hidden = nn.Linear(256, 1024)
        self.sequence = nn.Sequential( self.conv2d1, self.pool, self.relu,
                                        self.conv2d2, self.pool, self.relu,
                                        self.conv2d3, self.pool, self.relu,
                                        self.conv2d4, self.pool, self.relu,
                                        self.conv2d5, self.pool, self.relu,
                                        self.conv2d6, self.pool, self.relu)
    def forward(self, x):
        y = self.sequence(x)
        
        if len(y.shape) == 4:
            y = y.reshape(y.shape[0], -1)
        else:
            y = torch.flatten(y)
        y = self.hidden(y)
        y = self.relu(y)
        return y

class LSTM_Network(nn.Module):
    def __init__(self, N=4, N_h=128):
        super(LSTM_Network, self).__init__()
        #N_h is the number of NxNxN vectors
        #N is the grid space for spatial resolution. Higher -> More resolution
        self.conv3d1 = nn.Conv3d(N_h, N_h, (3,3,3),padding='same', bias=True)
        self.conv3d2 = nn.Conv3d(N_h, N_h, (3,3,3),padding='same', bias=True)
        self.conv3d3 = nn.Conv3d(N_h, N_h, (3,3,3),padding='same', bias=True)
        
        self.hidden1 = nn.Linear(1024, N_h*N*N*N)
        self.hidden2 = nn.Linear(1024, N_h*N*N*N)
        self.hidden3 = nn.Linear(1024, N_h*N*N*N)
        
        self.N_h = N_h
        self.N = N
    def forward(self, x, s_prev, h_prev):
        # if this is first iteration, s_prev and h_prev should be zero'd tensors
        x_1 = self.hidden1(x)
        x_2 = self.hidden2(x)
        x_3 = self.hidden3(x)
        
        h_1 = self.conv3d1(h_prev)
        h_2 = self.conv3d2(h_prev)
        h_3 = self.conv3d3(h_prev)
        
        if len(x_1.shape) == 2:
            
            ft = torch.sigmoid(x_1.reshape(x_1.shape[0],self.N_h,self.N,self.N,self.N) + h_1)
            it = torch.sigmoid(x_2.reshape(x_2.shape[0],self.N_h,self.N,self.N,self.N) + h_2)
            gt = torch.tanh( x_3.reshape(x_3.shape[0],self.N_h,self.N,self.N,self.N) + h_3 )
        else:
            ft = torch.sigmoid(x_1.reshape(self.N_h,self.N,self.N,self.N) + h_1)
            it = torch.sigmoid(x_2.reshape(self.N_h,self.N,self.N,self.N) + h_2)
            gt = torch.tanh( x_3.reshape(self.N_h,self.N,self.N,self.N) + h_3 )
        
        st = ft * s_prev + it * gt
        ht = torch.tanh(st)
        
        return st,ht # use these again and then return ht for the decoder network once done
        
class DecodeNetwork(nn.Module):
    def __init__(self, N_h=128):
        super(DecodeNetwork, self).__init__()
        self.conv3d_dec1 = torch.nn.Conv3d(128, 128, (3,3,3),padding='same', bias=True)
        self.conv3d_dec2 = torch.nn.Conv3d(128, 128, (3,3,3),padding='same', bias=True)
        self.conv3d_dec3 = torch.nn.Conv3d(128, 64, (3,3,3),padding='same', bias=True)
        self.conv3d_dec4 = torch.nn.Conv3d(64, 32, (3,3,3),padding='same', bias=True)
        self.conv3d_dec5 = torch.nn.Conv3d(32, 2, (3,3,3),padding='same', bias=True)
        
        self.relu = nn.LeakyReLU()
        self.softmax = torch.nn.LogSoftmax(dim=1)
    def forward(self, x):
        # cannot plug directly from LSTM_Network
        y = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        y = self.conv3d_dec1(y)
        y = self.relu(y)
        y = nn.functional.interpolate(y, scale_factor=2, mode='nearest')
        y = self.conv3d_dec2(y)
        y = self.relu(y)
        y = nn.functional.interpolate(y, scale_factor=2, mode='nearest')
        y = self.conv3d_dec3(y)
        y = self.relu(y)
        y = self.conv3d_dec4(y)
        y = self.relu(y)
        y = self.conv3d_dec5(y)

        final = self.softmax(y)
        return final

