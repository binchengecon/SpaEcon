import numpy as np

def integral_3D(fun_matrix,dx1,dx2,dx3,dim):
    """
    Goal:
    Intergrate a three-dimensional function on desiganated space of dimensions from 1 to 2.

    Input: 
    1. Function meshmatrix as fun
    2. Two uniform meshgrids as x1, x2, x3

    Output:
    1. Integral of fun as Int_fun
    Comment:
    1. There are spaces for c-parallelism
    """



    if dim == 2:
        Int_fun = np.sum(4*fun_matrix[1:-1,1:-1,:] , axis=(0,1))
        Int_fun += fun_matrix[0,0,:]
        Int_fun += fun_matrix[0,-1,:]
        Int_fun += fun_matrix[-1,0,:] 
        Int_fun += fun_matrix[-1,-1,:] 
        Int_fun += np.sum(2*fun_matrix[0,1:-1,:] , axis=(0,1))
        Int_fun += np.sum(2*fun_matrix[-1,1:-1,:] , axis=(0,1))
        Int_fun += np.sum(2*fun_matrix[1:-1,0,:] , axis=(0,1))
        Int_fun += np.sum(2*fun_matrix[1:-1,-1,:] , axis=(0,1))
        Int_fun = dx1*dx2*Int_fun/4.0

    if dim == 1:
        Int_fun = np.sum(2*fun_matrix[1:-1,:,:],axis=0)
        Int_fun += fun_matrix[0,:,:]
        Int_fun += fun_matrix[-1,:,:]
        Int_fun = dx1*Int_fun/2.0
        

    return Int_fun

def integral_2D(fun_matrix,dx1,dx2,dim):
    """
    Goal:
    Intergrate a three-dimensional function on desiganated space of dimensions from 1 to 2.

    Input: 
    1. Function meshmatrix as fun
    2. Two uniform meshgrids as x1, x2, x3

    Output:
    1. Integral of fun as Int_fun
    Comment:
    1. There are spaces for c-parallelism
    """

    if dim == 2:
        Int_fun = np.sum(4*fun_matrix[1:-1,1:-1] , axis=(0,1))
        Int_fun += fun_matrix[0,0]
        Int_fun += fun_matrix[0,-1]
        Int_fun += fun_matrix[-1,0] 
        Int_fun += fun_matrix[-1,-1] 
        Int_fun += np.sum(2*fun_matrix[0,1:-1] , axis=(0,1))
        Int_fun += np.sum(2*fun_matrix[-1,1:-1] , axis=(0,1))
        Int_fun += np.sum(2*fun_matrix[1:-1,0] , axis=(0,1))
        Int_fun += np.sum(2*fun_matrix[1:-1,-1] , axis=(0,1))
        Int_fun = dx1*dx2*Int_fun/4.0

    if dim == 1:
        Int_fun = np.sum(2*fun_matrix[1:-1,:],axis=0)
        Int_fun += fun_matrix[0,:]
        Int_fun += fun_matrix[-1,:]
        Int_fun = dx1*Int_fun/2.0
        

    return Int_fun


def derivative_2D(fun_matrix,dx1,dx2,direction):
    
    dfun_matrix = np.zeros(fun_matrix.shape)

    if direction =='F':

        dfun_matrix[0:-1,:] = (fun_matrix[1:,:]-fun_matrix[0:-1,:])/dx1

    if direction =='B':

        dfun_matrix[1:,:] = (fun_matrix[1:,:]-fun_matrix[0:-1,:])/dx1

    return dfun_matrix


def derivative_3D(fun_matrix,dx1,dx2,dx3,direction):
    
    dfun_matrix = np.zeros(fun_matrix.shape)

    if direction =='F':

        dfun_matrix[:,0:-1,:] = (fun_matrix[:,1:,:]-fun_matrix[:,0:-1,:])/dx2

    if direction =='B':

        dfun_matrix[:,1:,:] = (fun_matrix[:,1:,:]-fun_matrix[:,0:-1,:])/dx2

    return dfun_matrix