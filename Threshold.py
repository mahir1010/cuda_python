
from numba import cuda
import matplotlib.pyplot as plt
import numpy as np
import time

n = 100000

noise = np.random.normal(size=n) * 3
pulses = np.maximum(np.sin(np.arange(n) / (n / 23)) - 0.3, 0.0)
waveform = ((pulses * 300) + noise).astype(np.int16)
plt.figure(1)
plt.plot(waveform)

def naive_zero_suppress(signal,threshold):
	for i in range(0,signal.size):
		if signal[i] < threshold:
			signal[i]=0

@cuda.jit('void(int16[:],int16)')
def zero_suppress(device_array,threshold):
	id=cuda.grid(1)
	if(id<device_array.size):
		if device_array[id]<threshold:
			device_array[id]=0

"""Calling Cuda Function"""
d_waveform=cuda.to_device(waveform)
threadsperBlock=32
blockspergrid=(d_waveform.size+ (threadsperBlock-1)) // threadsperBlock
start=time.time()
zero_suppress[blockspergrid,threadsperBlock](d_waveform,15)
cuda.synchronize()
cuda_end=time.time()-start

start=time.time()
naive_zero_suppress(waveform,15)
end=time.time()-start


waveform=d_waveform.copy_to_host()
plt.figure(2)
plt.plot(waveform)
plt.show()


print('\v\v\t\t Cuda= ',(cuda_end)*1000,' ms')
print('\v\v\t\t Naive= ',(end)*1000,' ms')
