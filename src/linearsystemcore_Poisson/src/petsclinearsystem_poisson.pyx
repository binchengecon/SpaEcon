from petsc4py.PETSc cimport Mat, PetscMat

from petsc4py.PETSc import Error
cimport numpy as np # this allows access to the data member

cdef extern from "petsclinearsystem_poisson.h":
    ctypedef struct Params:
        double lambda_
    int FormLinearSystem_C_HJB(double *A, double *BF, double *BB, double *C, int n1, int n2, int n, double dt, double dX, PetscMat petsc_mat)
    int FormLinearSystem_C_KFE(double *A, double *BF, double *BB, double *C, int n1, int n2, int n, double dt, double dX, PetscMat petsc_mat)
 
def formLinearSystem_HJB(np.ndarray A, np.ndarray BF, np.ndarray BB, np.ndarray C, int n1, int n2, int n,double dt, double dX, Mat pymat):
    cdef int ierr = 0

    ierr = FormLinearSystem_C_HJB(<double*>A.data,<double*>BF.data, <double*>BB.data, <double*>C.data, n1, n2, n, dt, dX, pymat.mat)
    if ierr != 0: raise Error(ierr)
    return pymat

def formLinearSystem_KFE(np.ndarray A, np.ndarray BF, np.ndarray BB, np.ndarray C, int n1, int n2, int n,double dt, double dX, Mat pymat):
    cdef int ierr = 0

    ierr = FormLinearSystem_C_KFE(<double*>A.data,<double*>BF.data, <double*>BB.data, <double*>C.data, n1, n2, n, dt, dX, pymat.mat)
    if ierr != 0: raise Error(ierr)
    return pymat