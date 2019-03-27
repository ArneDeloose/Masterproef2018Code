#Module which calculates the features from a spectrogram

#Load packages
from __future__ import division #changes / to 'true division'
import numpy as np
import math
import cv2
from skimage.measure import compare_ssim as ssim

#load modules
import AD3_Features as AD3

#calculates features from nested dictionaries
def calc_features(rectangles, regions, templates, num_reg, list_bats, num_total):
    #!regions and rectangles can have different index because regions can be deleted if they overlap with the previous spectrogram
    features=np.zeros((len(templates)+7, num_reg))
    features_freq=np.zeros((7, num_reg)) #unscaled freq info
    count=0
    k=0 #index of the rectangle
    features_key={}
    for i,d in regions.items():
        k=0 #reset index every timestep
        for j,d in regions[i].items():
            features_key[count]=(i,j)
            features[0, count]=rectangles[i][3,k] #freq range
            features[1, count]=rectangles[i][1,k] #min freq
            features[2, count]=rectangles[i][1,k]+rectangles[i][3,k] #max freq
            features[3, count]=rectangles[i][1,k]+rectangles[i][3,k]/2 #av freq
            features[4, count]=rectangles[i][2,k] #duration
            index=np.argmax(regions[i][j]) #position peak frequency
            l=len(regions[i][j][0,:]) #number of timesteps
            a=index%l #timestep at peak freq
            b=math.floor(index/l) #frequency at peak freq
            features[5, count]=a/l #peak frequency T
            features[6, count]=b+rectangles[i][1,k] #peak frequency F
            for l in range(len(templates)):
                features[l+7, count]=AD3.compare_img2(regions[i][j], templates[l])
            features_freq[:, count]=features[:7, count]
            count+=1
            k+=1
    #Feature scaling
    for m in range(7):
        features[m,:]=(features[m, :]-features[m, :].min())/(features[m, :].max()-features[m, :].min())
    return(features, features_key, features_freq)

#variant function for single dictionaries (non-nested), used in DML code
def calc_features2(rectangles, regions, templates, list_bats, num_total):
    #regions and rectangles can have different index because regions can be deleted if they overlap with the previous spectrogram
    num_reg=len(regions)
    features=np.zeros((len(templates)+7, num_reg))
    for i in range(len(regions)):
        features[0, i]=rectangles[i][3] #freq range
        features[1, i]=rectangles[i][1] #min freq
        features[2, i]=rectangles[i][1]+rectangles[i][3] #max freq
        features[3, i]=rectangles[i][1]+rectangles[i][3]/2 #av freq
        features[4, i]=rectangles[i][2] #duration
        index=np.argmax(regions[i]) #position peak frequency
        l=len(regions[i][0, :]) #number of timesteps
        a=index%l #timestep at peak freq
        b=math.floor(index/l) #frequency at peak freq
        features[5, i]=a/l #peak frequency T
        features[6, i]=b+rectangles[i][1] #peak frequency F
        for l in range(len(templates)):
            features[l+7, i]=AD3.compare_img2(regions[i], templates[l])
    return(features)

#calculates the SSIM between two images
#Always scales the images according to the largest image
def compare_img2(img1, img2):
    a=len(img1[0,:])
    b=len(img1)
    c=len(img2[0,:])
    d=len(img2)
    if a*b==c*d: #same size
        if a>c: #img2->img1
            si=(a,b)
            img2_new=cv2.resize(img2, si)
            score=ssim(img1, img2_new, multichannel=True)
        else: #img1->img2
            si=(c,d)
            img1_new=cv2.resize(img1, si)
            score=ssim(img1_new, img2, multichannel=True)
    else: #different size
        if a*b>c*d: #1st image is bigger
            si=(a,b)
            img2_new=cv2.resize(img2, si)
            score=ssim(img1, img2_new, multichannel=True)
        else:
            si=(c,d)
            img1_new=cv2.resize(img1, si)
            score=ssim(img1_new, img2, multichannel=True)
    return(score)

def calc_num_regions(regions):
    num_reg=0
    for i,d in regions.items():
        for j,d in regions[i].items():
            num_reg+=1
    return(num_reg)
