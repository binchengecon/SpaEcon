
import time
import numpy as np 
import cupy as cp 
import argparse


parser = argparse.ArgumentParser(description="xi_r values")
parser.add_argument("--len", type=int,default=4)
args = parser.parse_args()

len = args.len


tpt1 = time.time()
x = cp.arange(10**(2*len)).reshape(10**(len), 10**(len)).astype('f')
x.sum(axis=1)
tpt2 = time.time()

tpt3 = time.time()
x = np.arange(10**(2*len)).reshape(10**(len), 10**(len)).astype('f')
x.sum(axis=1)
tpt4 = time.time()


print("dim =(10^{},10^{}),cptime={},nptime={}".format((len),(len),tpt2-tpt1,tpt4-tpt3))


len = 6
tpt1 = time.time()
x = cp.arange(5**(2*len)).reshape(5**(len), 5**(len)).astype('f')
x.sum(axis=1)
tpt2 = time.time()

tpt3 = time.time()
x = np.arange(5**(2*len)).reshape(5**(len), 5**(len)).astype('f')
x.sum(axis=1)
tpt4 = time.time()


print("dim =(10^{},10^{}),cptime={},nptime={}".format((len),(len),tpt2-tpt1,tpt4-tpt3))

