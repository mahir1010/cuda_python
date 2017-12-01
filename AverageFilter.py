from numba import cuda,uint8
import numpy as np
import cv2
import time


kernel_width=3

@cuda.jit('void(uint8[:,:],int32,int32)')
def avg_filter(pixels,m,n):
	"""	
	avg_filter : A function which uses a window/kernel to traverse over every pixels of the image .
		It computes average of every elements in a window/kernel and and sets every element to that value.
		This function runs parallely by computing window/kernel at every pixel simultaneously.
		Number of thread blocks = Number of Pixels
		Number of threads per block = kernel_width * kernel_width
		Number of threads = Number of Thread Blocks * Number of threads per block
	Parameters:
		pixels: An array of size mxn pixels where m and n are the dimensions of an image and the elements of the array are the intensity 
		values at every pixel.
		m:it is the width of the image
		n:it is the hight of the image
	Example:
		[ [0,0,32,123,255,134,....]
		  [0,0,22,23,35,1,........]
		  [6,0,102,13,25,34,......]
		  [0,10,72,1,55,13,.......]
		  [10,6,22,12,202,14,.....]
		  ...
		  ...
		  [0,0,32,123,255,134,....]]
	
		Pixel Coordinate = [1,1]
		kernel=[[0,0,32]
				[0,0,22]
				[6,0,102]]
		average= (0+0+32+0+0+22+6+0+102) // 9 =18
		output in image = [[18,18,18]
						   [18,18,18]
						   [18,18,18]]

	Note:
		For border-pixels the adjacent pixels in the kernel which are non existant in the image are set to zero intensity
		These pixel are detected by computing whether their cordinates are either negative or greater than the dimension 
		of the image
		For first block the center of the kernel would be the first pixel of the image therefore the pixels at position:
		[0,0],[0,1],[0,2],[1,0] and [2,0] would have global coordinates having value less than 0 hence they are actually
		non existant.Therefore their intensity value is set to zero.
		The global coordinates in the memory are computed by following formulae:
		Memory_position_x=block_id_x+thread_id_x-1
		Memory_position_y=block_id_y+thread_id_y-1 
	"""

	sub_img=cuda.shared.array(shape=(kernel_width,kernel_width),dtype=uint8) #A shared array to store the elements of the kernel
	block_id_x=cuda.blockIdx.x
	block_id_y=cuda.blockIdx.y
	thread_id_x=cuda.threadIdx.x
	thread_id_y=cuda.threadIdx.y
	Memory_position_x=block_id_x+thread_id_x-1
	Memory_position_y=block_id_y+thread_id_y-1
	"""
		isVisible computes whether the pixel actually exists or not
		is false for every pixels outside the image boundary hence the intensity level must be set to 0
	"""	
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
		
