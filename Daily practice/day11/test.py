#!/usr/bin/env python

#https://discuss.pytorch.org/t/pytorch-trained-model-on-webcam/23928/5

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np  
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
from torch.autograd import Variable
from torchvision import transforms
import PIL 
import cv2
import os
import matplotlib.pyplot as plt
import time

#This is the Label
Labels = { 0 : '0',
           1 : '1',
           2 : '2',
           3 : '3',
           4 : '4',
           5 : '5',
        }

# Let's preprocess the inputted frame

single_transforms = transforms.Compose(
        [
        #transforms.Resize((64,64)),
        transforms.CenterCrop((256,256)),
        transforms.Resize((64,64)),
        #transforms.ColorJitter(brightness=0.4, contrast=0, saturation=0, hue=0),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ]
)


##########################################################
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()
        self.num_channels = 32.
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)
        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(8*8*self.num_channels*4, self.num_channels*4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)
        self.fc2 = nn.Linear(self.num_channels*4, 6)       
        self.dropout_rate = 0.8

    def forward(self, s):

        #                                                  -> batch_size x 3 x 64 x 64
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = self.bn1(self.conv1(s))                         # batch_size x num_channels x 64 x 64
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels x 32 x 32
        s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 32 x 32
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 16 x 16
        s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 16 x 16
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*4 x 8 x 8

        # flatten the output for each image
        s = s.view(-1, 8*8*self.num_channels*4)             # batch_size x 8*8*num_channels*4

        # apply 2 fully connected layers with dropout
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), 
            p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*4
        s = self.fc2(s)                                     # batch_size x 6

        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(s, dim=1)
##########################################################


def restore_net(path):
    global model
    #path = path + '/day11/cnn_stanford2.pkl'
    print(path)
    model  = torch.load('cnn_stanford.pkl').cpu()
    model  = model.cpu()
    model.eval()                #set the device to eval() mode for testing


def preprocess(image):
    image = PIL.Image.fromarray(image) #Webcam frames are numpy array format
                                       #Therefore transform back to PIL image
    #print(image)                             
    image = single_transforms(image)
    #image = image.float()
    image2 = image
    #image = Variable(image, requires_autograd=True)
    #image = image.cuda()
    image = image.unsqueeze(0) #I don't know for sure but Resnet-50 model seems to only
                               #accpets 4-D Vector Tensor so we need to squeeze another
    return image2, image                            #dimension out of our 3-D vector Tensor



if __name__ == '__main__':
    print("Restore net from :")
    path = os.getcwd()+'/day11/cnn_stanford2.pkl'
    print(path)
    try:
        restore_net(path)
        print("Restore net done!!!")
    except:
        print("Restore error!!  Check path.")
        os.system('shutdown -s')
        time.sleep(100)
    time.sleep(2)

    print("Reading from webcam...")

    count = 0
    cap = cv2.VideoCapture(0)

    while 1:
        _ , frame = cap.read()
        frame = cv2.resize(frame, (640, 360))
        #det_frame = cv2.resize(frame, (64, 64))
        det_frame = frame
        frame_sized, det_frame = preprocess(det_frame)
        #print(det_frame.shape)
        
        frame_sized = frame_sized.numpy().transpose((1, 2, 0))
        #rint(det_frame.shape)

        if count % 5 == 0:
            prediction = model(det_frame)
            prediction = prediction.detach().numpy()
            value = np.argmax(prediction, axis=1)
            #value = torch.max(prediction, 1)[1].data.numpy().squeeze()
            #print("dd")
            count = 0
        count = count+1

        cv2.putText(frame, format(value), (560, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("SIGN DETECTER", frame)
        cv2.imshow("SIGN ", frame_sized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow("SIGN DETECTER")


