#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#https://discuss.pytorch.org/t/pytorch-trained-model-on-webcam/23928/5
import numpy as np  
import torch
import torch.nn
import torchvision 
from torch.autograd import Variable
from torchvision import transforms
import PIL 
#import cv2

#This is the Label
Labels = { 0 : '0',
           1 : '1',
           2 : '2',
           3 : '3',
           4 : '4',
           5 : '5',
        }


# In[ ]:


# Let's preprocess the inputted frame

data_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0,0.225])
    ]
) 

model  = torch.load("cnn_stanford.pkl") #Load model to CPU
model  = model.cuda()
model.eval()                #set the device to eval() mode for testing


# In[ ]:


#Set the Webcam 
def Webcam_720p():
    cap.set(3,1280)
    cap.set(4,720)


# In[ ]:


def argmax(prediction):
    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()
    top_1 = np.argmax(prediction, axis=1)
    score = np.amax(prediction)
    score = '{:6f}'.format(score)
    prediction = top_1[0]
    result = Labels[prediction]

    return result,score


# In[ ]:


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


# In[ ]:


img = cv2.imread('image.jpg')

fps = 0
show_score = 0
show_res = 'Nothing'
sequence = 0


# In[ ]:


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

