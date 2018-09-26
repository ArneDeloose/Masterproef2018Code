#Example of an ROI using an image of text ('shine on you crazy diamond')
import cv2
import numpy as np

#import image
image = cv2.imread('C:\\Users\\arne\\Documents\\School\\Thesis\\test_ROI.png')

#grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray', gray)
#cv2.waitKey(0)

#binary
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
#cv2.imshow('second', thresh)
#cv2.waitKey(0)

#dilation, normal
kernel = np.ones((1,1), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
#cv2.imshow('dilated', img_dilation)
#cv2.waitKey(0)

#Bigger kernel (to select larger areas)
#kernel = np.ones((10,15), np.uint8)
#img_dilation = cv2.dilate(thresh, kernel, iterations=1)
#cv2.imshow('dilated', img_dilation)
#cv2.waitKey(0)


#find contours
im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = image[y:y+h, x:x+w]

    # show ROI
    #cv2.imshow('segment no:'+str(i),roi)
    cv2.rectangle(image,(x,y),( x + w, y + h ),(0,255,0),2)
    #cv2.waitKey(0)

    #Write out sections to seperate files
    if w > 15 and h > 15:
        cv2.imwrite('C:\\Users\\arne\\Documents\\School\\Thesis\\{}.png'.format(i), roi)
        
    
cv2.imshow('marked areas',image)
cv2.waitKey(0)
