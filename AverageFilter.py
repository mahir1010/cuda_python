from numba import cuda,uint8
import numpy as np
import cv2
import time


kernel_width=3

@cuda.jit('void(uint8[:,:],int32,int32)')
def avg_filter(pixels,m,n):
	sub_img=cuda.shared.array(shape=(kernel_width,kernel_width),dtype=uint8)
	block_id_x=cuda.blockIdx.x
	block_id_y=cuda.blockIdx.y
	thread_id_x=cuda.threadIdx.x
	thread_id_y=cuda.threadIdx.y
	Memory_position_x=block_id_x+thread_id_x
	Memory_position_y=block_id_y+thread_id_y	
	isVisible= False if Memory_position_x < 0 or Memory_position_x > m or Memory_position_y < 0 or Memory_position_y > n else True
	if not isVisible:
		sub_img[thread_id_x,thread_id_y]=0
	else:
		sub_img[thread_id_x,thread_id_y]=pixels[Memory_position_x,Memory_position_y]
	cuda.syncthreads()
	sum=0
	for i in range(0,kernel_width):
		for j in range(0,kernel_width):
			sum+=sub_img[i,j]
	sum//=kernel_width*kernel_width
	if isVisible:
		pixels[Memory_position_x,Memory_position_y]=sum

sp_noise='lena_sp_noise.png'

img = cv2.imread('/home/mahir/SciPy/'+sp_noise,cv2.IMREAD_GRAYSCALE)
x,y=img.shape
if img is None:
	print('No Image found')
	exit()

d_img=cuda.to_device(img)
TPB=(kernel_width,kernel_width)
BPG=(x,y)

start =time.time()
avg_filter[BPG,TPB](d_img,x-1,y-1)
cuda.synchronize()
end1=time.time()-start
output=d_img.copy_to_host()

start=time.time()
blur = cv2.blur(img,(3,3))
end2=time.time()-start

cv2.imshow('Original',img)
cv2.imshow('Cuda',output)
cv2.imshow('Opencv',blur)
cv2.waitKey(0)

print('Cuda Implementation:',end1*1000,' ms')
print('Opencv Implementation:',end2*1000,' ms')


