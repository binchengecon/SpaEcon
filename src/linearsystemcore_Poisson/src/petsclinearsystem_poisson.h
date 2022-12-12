#ifndef PETSCLINEARSYSTEM_H
#define PETSCLINEARSYSTEM_H

#include <petsc.h>

typedef struct Params {
  PetscScalar lambda_;
} Params;

// PetscErrorCode FormLinearSystem_C(PetscScalar *R, PetscScalar *F, PetscScalar *K, PetscScalar *A, PetscScalar *B_r, PetscScalar *B_f, PetscScalar *B_k, PetscScalar *C_rr, PetscScalar *C_ff, PetscScalar *C_kk, PetscScalar dt, PetscScalar *lowerLims, PetscScalar *upperLims, PetscScalar *dVec, PetscInt *incVec, PetscInt n, Mat petsc_mat);
PetscErrorCode FormLinearSystem_C_HJB(PetscScalar *A, PetscScalar *BF, PetscScalar *BB, PetscScalar *C, PetscInt n1, PetscInt n2, PetscInt n, PetscScalar dt, PetscScalar dX, Mat petsc_mat);
PetscErrorCode FormLinearSystem_C_KFE(PetscScalar *A, PetscScalar *BF, PetscScalar *BB, PetscScalar *C, PetscInt n1, PetscInt n2, PetscInt n, PetscScalar dt, PetscScalar dX, Mat petsc_mat);

#endif /* !PETSCLINEARSYSTEM */