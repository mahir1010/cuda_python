from numba import cuda,uint8

kernel_width=3
kernel_size=(kernel_width*kernel_width)

@cuda.jit('void (uint8[:,:],int32,int32)')
def median_filter(pixels,m,n):
	sub_img=cuda.shared.array(shape=kernel_size,dtype=uint8) #A shared array to store the elements of the kernel
	block_id_x=cuda.blockIdx.x
	block_id_y=cuda.blockIdx.y
	thread_id_x=cuda.threadIdx.x
	thread_id_y=cuda.threadIdx.y
	Memory_position_x=block_id_x+thread_id_x-1
	Memory_position_y=block_id_y+thread_id_y-1
	linear_index=thread_id_x*kernel_width+thread_id_y
	isVisible= False if Memory_position_x < 0 or Memory_position_x > m or Memory_position_y < 0 or Memory_position_y > n else True
	if not isVisible:
		sub_img[linear_index]=0
	else:
		sub_img[linear_index]=pixels[Memory_position_x,Memory_position_y]
	
	if thread_id_x==1 and thread_id_y==1:
		for x in range(len(sub_img)-1,0,-1):
			for i in range(x):
				if sub_img[i]>sub_img[i+1]:
					temp=sub_img[i]
					sub_img[i]=sub_img[i+1]
					sub_img[i+1]=temp
		median=sub_img[int((kernel_size+1)/2)-1] if (kernel_size)%2==1 else uint8((sub_img[int((kernel_size)/2)-1]+sub_img[int((kernel_size)/2)])/2)
		pixels[Memory_position_x,Memory_position_y]=median