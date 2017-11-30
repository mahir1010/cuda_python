from numba import cuda,vectorize,float32
import numpy as np
import time
import sys


"""Cuda Implementation of Numpy ufunction """
@vectorize(['float32(float32, float32)'], target='cuda')
def add_ufunc(x, y):
    return x + y


""" Naive Implementation without Parallelization """
def naive_sum(a,b):
	for i in range(0,n):
		a[i]=a[i]+b[i]
	return a

""" Cuda kernel """
@cuda.jit('void(float32[:],float32[:],float32[:])')
def sum(a,b,out):
	id=cuda.grid(1)
	out[id]=a[id]+b[id]

""" Cude Kernel with Shared Memory"""
@cuda.jit('void(float32[:],float32[:])')
def sum_shared_memory(a,b):
	sh_a=cuda.shared.array(shape=256,dtype=float32)
	sh_b=cuda.shared.array(shape=256,dtype=float32)
	thread_block_id=cuda.threadIdx.x
	id=cuda.grid(1)
	if id < a.size:
		sh_a[thread_block_id]=a[id]
		sh_b[thread_block_id]=b[id]
		sh_a[thread_block_id]+=sh_b[thread_block_id]
		a[id]=sh_a[thread_block_id]
	

n = 1000000
a=np.random.rand(n).astype(np.float32)
b=np.random.rand(n).astype(np.float32)

d_a=cuda.to_device(a)
d_out=cuda.device_array_like(d_a)
d_b=cuda.to_device(b)

	

"""Naive Function Call and Profiling """
naive_a=np.copy(a)
naive_b=np.copy(b)
naive_start=time.time()
naive_a=naive_sum(naive_a,naive_b)
naive_eta=time.time()-naive_start

"""Cuda Kernel Call"""
threadperblock=32
blockspergrid=(n+threadperblock-1)//threadperblock
start=time.time()

sum[blockspergrid,threadperblock](d_a,d_b,d_out) 
cuda.synchronize()

eta=time.time()-start

h_out=d_out.copy_to_host()

""" Numpy UFunction Call"""
start=time.time()
out=np.add(a,b)
end=time.time()-start

""" Cuda Ufunction Call"""
start=time.time()
output=add_ufunc(a,b)
end_cuda_ufunc=time.time()-start


""" Cuda Add Function with Shared Memory"""
threadperblock=256
blockspergrid=(n+threadperblock-1)//threadperblock
start=time.time()
sum_shared_memory[blockspergrid,threadperblock](d_a,d_b)
cuda.synchronize()
end_shared=time.time()-start

var=True if len(sys.argv)>1 else False

print('\vAddition of two ',n,' sized array','\n\n\tTime to Compute:\n')
print('\tNaive Implementation:\t\t%10f'%(naive_eta*1000),' ms')
if var and sys.argv[1]=='-no':
	exit()
print('\tCuda  uFunction     :\t\t%10f'%(end_cuda_ufunc*1000),' ms')
if var and sys.argv[1]=='-cu':
	exit()
print('\tNumpy Implementation:\t\t%10f'%(end*1000), ' ms')
if var and sys.argv[1]=='-ni':
	exit()
print('\tCUDA  Implementation:\t\t%10f'%(eta*1000),' ms')
if var and sys.argv[1]=='-ci':
	exit()
print('\tCuda  Shared Memory :\t\t%10f'%(end_shared*1000),' ms')
