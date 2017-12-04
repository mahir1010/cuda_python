from MedianFilter import *
import numpy as np
import cv2
import time


sp_noise='lena_sp_noise.png'

img = cv2.imread(sp_noise,cv2.IMREAD_GRAYSCALE)
x,y=img.shape
if img is None:
	print('No Image found')
	exit()

d_img=cuda.to_device(img)
TPB=(kernel_width,kernel_width)
BPG=(x,y)

start =time.time()
median_filter[BPG,TPB](d_img,x-1,y-1)
cuda.synchronize()
end1=time.time()-start
output=d_img.copy_to_host()

start=time.time()
blur = cv2.medianBlur(img,3)
end2=time.time()-start

cv2.imshow('Original',img)
cv2.imshow('Cuda',output)
cv2.imshow('Opencv',blur)
cv2.waitKey(0)

print('Cuda Implementation:',end1*1000,' ms')
print('Opencv Implementation:',end2*1000,' ms')
cv2.destroyAllWindows()


