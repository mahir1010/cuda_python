"""
Execution syntax:
python sum.py [-no,-cu,-ni,-ci]

args:
	-no: prints only upto naive call
	-cu: prints only upto cuda ufunction call
	-ni: prints only upto Numpy call
	-ci: prints only upto cuda kernel call
	<empty> : prints everything including cuda kernel with shared memory.

"""
from numba import cuda,vectorize,float32
import numpy as np
import time
import sys


var=True if len(sys.argv)>1 else False

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

print('\vAddition of two ',n,' sized array','\n\n\tTime to Compute:\n')	

"""Naive Function Call and Profiling """
naive_a=np.copy(a)
naive_b=np.copy(b)
start=time.time()
naive_a=naive_sum(naive_a,naive_b)
end=time.time()-start
print('\tNaive Implementation:\t\t%10f'%(end*1000),' ms')
if var and sys.argv[1]=='-no':
	exit()

""" Cuda Ufunction Call"""
start=time.time()
output=add_ufunc(a,b)
end=time.time()-start

print('\tCuda  uFunction     :\t\t%10f'%(end*1000),' ms')
if var and sys.argv[1]=='-cu':
	exit()

""" Numpy UFunction Call"""
start=time.time()
out=np.add(a,b)
end=time.time()-start
print('\tNumpy Implementation:\t\t%10f'%(end*1000), ' ms')
if var and sys.argv[1]=='-ni':
	exit()

"""Cuda Kernel Call"""
threadperblock=32
blockspergrid=(n+threadperblock-1)//threadperblock

d_a=cuda.to_device(a)
d_out=cuda.device_array_like(d_a)
d_b=cuda.to_device(b)

start=time.time()

sum[blockspergrid,threadperblock](d_a,d_b,d_out) 
cuda.synchronize()

end=time.time()-start

print('\tCUDA  Implementation:\t\t%10f'%(end*1000),' ms')
if var and sys.argv[1]=='-ci':
	exit()

""" Cuda Add Function with Shared Memory"""
threadperblock=32
blockspergrid=(n+threadperblock-1)//threadperblock
start=time.time()
sum_shared_memory[blockspergrid,threadperblock](d_a,d_b)
cuda.synchronize()
end=time.time()-start

print('\tCuda  Shared Memory :\t\t%10f'%(end*1000),' ms')
