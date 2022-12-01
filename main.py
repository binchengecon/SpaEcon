import cupy as cp 
x = cp.arange(6).reshape(2, 3).astype('f')
x
x.sum(axis=1)
print("scc")