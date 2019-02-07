#!/usr/bin/env python

#https://discuss.pytorch.org/t/pytorch-trained-model-on-webcam/23928/5
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

#This is the Label
Labels = { 0 : '0',
           1 : '1',
           2 : '2',
           3 : '3',
           4 : '4',
           5 : '5',
        }

# Let's preprocess the inputted frame

data_transforms = transforms.Compose(
        [
        transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ]
)

test_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
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


def restore_net():
    global model
    model  = torch.load('cnn_stanford.pkl')
    model  = model.cpu()
    model.eval()                #set the device to eval() mode for testing



def argmax(prediction):
    prediction = prediction.cpu()
    p = torch.max(prediction, 1)[1].data.numpy().squeeze()
    print(p)
    prediction = prediction.detach().numpy()
    print(prediction)
    top_1 = np.argmax(prediction, axis=1)
    score = np.amax(prediction)
    score = '{:6f}'.format(score)
    prediction = top_1[0]
    result = Labels[prediction]

    return result,score



def preprocess(image):
    image = PIL.Image.fromarray(image) #Webcam frames are numpy array format
                                       #Therefore transform back to PIL image
    print(image)                             
    image = data_transforms(image)
    image = image.float()
    #image = Variable(image, requires_autograd=True)
    #image = image.cuda()
    image = image.unsqueeze(0) #I don't know for sure but Resnet-50 model seems to only
                               #accpets 4-D Vector Tensor so we need to squeeze another
    return image                            #dimension out of our 3-D vector Tensor


img = cv2.imread('image1.jpg')
 
show_score = 0
show_res = "Nothing"
sequence = 0



if __name__ == '__main__':
    print("Restore net...")
    restore_net()
    print("Restore net done!!!")
    
    imgd        = img
    #imgd = cv2.resize(image, (64, 64))
    print(imgd.shape)
    image_data   = preprocess(imgd)
    print(image_data.shape)
    
    prediction   = model(image_data)
    print("Prediction: %s" %format(prediction))
    result,score = argmax(prediction)
    print("Result: %s" %format(result))
    print("Score: %s" %format(score))
    if float(score) >= 0.5:
        show_res  = result
        show_score= score
    else:
        show_res   = "Nothing"
        show_score = score
        
        
    #cv2.putText(image, '%s' %(show_res),(950,250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
    #cv2.putText(image, '(score = %.5f)' %(float(show_score)), (950,300), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    #cv2.rectangle(image,(400,150),(900,550), (250,0,0), 2)
    while True:
        image = cv2.resize(imgd, (480, 600))
        cv2.imshow("SIGN DETECTER", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow("SIGN DETECTER")

