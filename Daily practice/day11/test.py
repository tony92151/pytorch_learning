#!/usr/bin/env python

#https://discuss.pytorch.org/t/pytorch-trained-model-on-webcam/23928/5
import numpy as np  
import torch
import torch.nn
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
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
) 
print(os.getcwd())
print("Load model")

def restore_net():
    model  = torch.load('net.pkl')
    model  = model.cuda()
    model.eval()                #set the device to eval() mode for testing



def argmax(prediction):
    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()
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
    image = image.cuda()
    image = image.unsqueeze(0) #I don't know for sure but Resnet-50 model seems to only
                               #accpets 4-D Vector Tensor so we need to squeeze another
    return image                            #dimension out of our 3-D vector Tensor


img = cv2.imread('image.jpg')

show_score = 0
show_res = "Nothing"
sequence = 0



if __name__ == '__main__':
    print("Restore net...")
    restore_net()
    print("Restore net done!!!")
    while True:
        image        = img[100:450,150:570]
        image_data   = preprocess(image)
        print(image_data)
        prediction   = model(image_data)
        result,score = argmax(prediction)
        if result >= 0.5:
            show_res  = result
            show_score= score
        else:
            show_res   = "Nothing"
            show_score = score
            
        cv2.putText(image, '%s' %(show_res),(950,250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        cv2.putText(image, '(score = %.5f)' %(show_score), (950,300), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
        cv2.rectangle(image,(400,150),(900,550), (250,0,0), 2)
        cv2.imshow("SIGN DETECTER", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyWindow("SIGN DETECTER")

