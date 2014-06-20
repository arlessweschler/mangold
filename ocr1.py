#!/usr/bin/env python

import sys
import numpy as np
import cv2

im = cv2.imread(sys.argv[1])
im3 = im.copy()

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

#################      Now finding Contours         ###################

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

samples =  np.empty((0,100))
responses = []
#keys = [i for i in range(48,58)]
chars = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz"
keys = set([ord(c) for c in chars])

for cnt in contours:
    #if cv2.contourArea(cnt)>50:
    if cv2.contourArea(cnt)>30:
        [x,y,w,h] = cv2.boundingRect(cnt)

        #if h>28:
        if h>14:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            #roismall = cv2.resize(roi,(8,8))
            cv2.imshow('norm',im)
            key = cv2.waitKey(0)

            print key, chr(key)

            if key == 27:  # (escape to quit)
                #sys.exit()
                break
            elif key in keys:
                responses.append(key)
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print "training complete"

print responses

np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)

