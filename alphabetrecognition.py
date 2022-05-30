import cv2 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os
import ssl
import time
x = np.load("image.npz")["arr_0"]
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","v","W","X","Y","Z"]
nclasses = len(classes)
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state = 9,train_size = 7500 , test_size = 2500)
xtrainscaled = xtrain/255
xtestscaled = xtest/255
clf = LogisticRegression(solver = "saga",multi_class = "multinomial").fit(xtrainscaled,ytrain)
yprediction = clf.predict(xtestscaled)
accuracy = accuracy_score(ytest,yprediction)
print(accuracy)
cap = cv2.VideoCapture(0)
while True:
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.color_BGR2GRAY)
        height,width = gray.shape
        upperleft = (int(width/2 - 56) , int(height/2 - 56))
        bottomright = (int(width/2 + 56) ,int(height/2 + 56))
        cv2.rectangle(gray,upperleft,bottomright,(0,255,0),2)
        roi = gray[upperleft[1]:bottomright[1],upperleft[0]:bottomright[0]]
        image_PIL = Image.fromarray(roi)
        image_bw = image_PIL.convert("L")
        image_bw_resize = image_bw.resize((28,28),Image.ANTIALIAS)
        image_bw_resize_inverted = PIL.ImageOps.invert(image_bw_resize)
        pixelfilter = 20
        minpixel = np.percentile(image_bw_resize_inverted,pixelfilter)
        image_bw_resize_inverted_scaled = np.clip(image_bw_resize_inverted - minpixel , 0 , 255)
        max_pixel = np.max(image_bw_resize_inverted)
        image_bw_resize_inverted_scaled = np.asarray(image_bw_resize_inverted_scaled)/max_pixel
        testsample = np.asarray(image_bw_resize_inverted_scaled).reshape(1,784)
        testprediction = clf.predict(testsample)
        print(testprediction)
        cv2.imshow("frame",gray)
    except Exception as e:
        pass
cap.relase()
cv2.destroyAllWindows()
