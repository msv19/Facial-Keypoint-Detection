## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # Input size 224 x 224 x 1   # (W-F)/S + 1 
        # This network architecture/model is based on NaimishNet with certain modifications 
        
        # 1st Convolutional Layer 
        self.conv1 = nn.Conv2d(1, 32, 5)     # Output tensor shape - 220 x 220 x 32
        I.xavier_uniform( self.conv1.weight,gain = 1)
        self.pool1 = nn.MaxPool2d(2)         # output tensor shape - 110 x 110 x 32
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout(0.1)
        
        # 2nd Convolutional layer
        self.conv2 = nn.Conv2d(32, 48, 3)    # output tensor shape - 108 x 108 x 48
        I.xavier_uniform( self.conv2.weight,gain = 1)    
        self.pool2 = nn.MaxPool2d(2)         # output tensor shape - 54 x 54 x 48
        self.bn2 = nn.BatchNorm2d(48)
        self.drop2 = nn.Dropout(0.2)
        
        # 3rd Convolutional layer
        self.conv3 = nn.Conv2d(48, 64, 3)    # output tensor shape - 52 x 52 x 64
        I.xavier_uniform( self.conv3.weight,gain = 1)
        self.pool3 = nn.MaxPool2d(2)         # outputtensor shape - 26 x 26 x 64
        self.bn3 = nn.BatchNorm2d(64)
        self.drop3 = nn.Dropout(0.3)
        
        # 4th Convolutional layer
        self.conv4 = nn.Conv2d(64,128,3)    # output tensor shape - 24 x 24 x 128
        I.xavier_uniform( self.conv4.weight,gain = 1)
        self.pool4 = nn.MaxPool2d(2)        # output tensor shape - 12 x 12 x 128
        self.bn4 = nn.BatchNorm2d(128)
        self.drop4 = nn.Dropout(0.4)
        
        # Fully connected layers
        self.fc1 = nn.Linear(12*12*128, 1024) 
        self.drop5 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(1024,512)
        self.drop6 = nn.Dropout(0.6)
        
        self.fc3 = nn.Linear(512,136)
    
   
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x =  self.pool1(F.elu(self.bn1(self.conv1(x))))
        #x =  self.pool1(F.elu(self.conv1(x)))
        x =  self.drop1(x)
        
        x =  self.pool2(F.elu(self.bn2(self.conv2(x))))
        #x =  self.pool2(F.elu(self.conv2(x)))
        x =  self.drop2(x)
        
        x =  self.pool3(F.elu(self.bn3(self.conv3(x))))
        #x =  self.pool3(F.elu(self.conv3(x)))
        x =  self.drop3(x)
        
        x =  self.pool4(F.elu(self.bn4(self.conv4(x))))
        #x =  self.pool4(F.elu(self.conv4(x)))
        x =  self.drop4(x)
         
        # Flatten the output for FC layers
        x = x.view(x.size(0),-1)
        
        #Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.drop5(x)
        
        x = F.relu(self.fc2(x))
        x = self.drop6(x)
        
        x = self.fc3(x)
       
        # a modified x, having gone through all the layers of your model, should be returned
        return x
