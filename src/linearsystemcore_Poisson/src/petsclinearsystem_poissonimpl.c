#include "petsclinearsystem_poisson.h"


static inline void fill_mat_values_Pi_HJB(PetscInt index, PetscInt Pi_center, PetscInt n1, PetscScalar *BF, PetscScalar * BB, PetscScalar dX, PetscInt * Pi_cols, PetscScalar *Pi_vals)
{

  if (index % n1==0) //LB
  {
    Pi_cols[Pi_center] = index;
    Pi_cols[Pi_center+1] = index+1;

    Pi_vals[Pi_center] = (BF[index]*(BF[index]>0)-BB[index]*(BB[index]<0))/dX;
    Pi_vals[Pi_center+1] = -(BF[index]*(BF[index]>0))/dX;

  }
  else if (index % n1 ==(n1-1))
  {
    Pi_cols[Pi_center-1] = index-1;
    Pi_cols[Pi_center]   = index;

    Pi_vals[Pi_center] = (BF[index]*(BF[index]>0)-BB[index]*(BB[index]<0))/dX;
    Pi_vals[Pi_center-1] = (BB[index]*(BB[index]<0))/dX;

  }
  else 
  {
    Pi_cols[Pi_center-1] = index-1;
    Pi_cols[Pi_center] = index;
    Pi_cols[Pi_center+1] = index+1;

    Pi_vals[Pi_center-1] = (BB[index]*(BB[index]<0))/dX;
    Pi_vals[Pi_center] = (BF[index]*(BF[index]>0)-BB[index]*(BB[index]<0))/dX;
    Pi_vals[Pi_center+1] = -(BF[index]*(BF[index]>0))/dX;

  }
}

static inline void fill_mat_values_Xi_HJB(PetscInt index,  PetscInt n1, PetscInt n2, PetscScalar *C , PetscInt * Xi_cols, PetscScalar *Xi_vals)
{
  PetscInt k;

  for (k=0; k<n2; k++)
  {
    Xi_cols[k] = index % n1 + k*n1;
    Xi_vals[k] = -C[index+k*n1*n2];

  }
}



PetscErrorCode FormLinearSystem_C_HJB(PetscScalar *A, PetscScalar *BF, PetscScalar *BB, PetscScalar *C, PetscInt n1, PetscInt n2, PetscInt n, PetscScalar dt, PetscScalar dX, Mat petsc_mat)
{

  PetscErrorCode ierr;

  PetscInt       index, Pi_center;
  
  
  
  PetscInt       Pi_cols[3], Xi_cols[n2];
  PetscScalar    Pi_vals[3], Xi_vals[n2];

  Pi_center = 1;

  PetscFunctionBegin;
  for (index  = 0; index < n; index++) {


    // Pi_center = 1;
    memset(Pi_vals,0,3*sizeof(PetscScalar));
    memset(Pi_cols,-1,3*sizeof(PetscInt));

    memset(Xi_vals,0,n2*sizeof(PetscScalar));
    memset(Xi_cols,-1,n2*sizeof(PetscInt));
    
    Pi_cols[Pi_center] = index;
    Pi_vals[Pi_center] = 1.0/dt - A[index];


    fill_mat_values_Pi_HJB(index, Pi_center, n1, BF, BB, dX, Pi_cols,Pi_vals);
    fill_mat_values_Xi_HJB(index, n1, n2, C, Xi_cols,Xi_vals);
    ierr = MatSetValues(petsc_mat,1,&index,3,Pi_cols,Pi_vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(petsc_mat,1,&index,n2,Xi_cols,Xi_vals,INSERT_VALUES);CHKERRQ(ierr);


  }
  ierr = MatAssemblyBegin(petsc_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(petsc_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



static inline void fill_mat_values_Pi_KFE(PetscInt index, PetscInt Pi_center, PetscInt n1, PetscScalar *BF, PetscScalar * BB, PetscScalar dX, PetscInt * Pi_cols, PetscScalar *Pi_vals)
{

  if (index % n1==0) //LB
  {
    Pi_cols[Pi_center] = index;
    Pi_cols[Pi_center+1] = index+1;

    Pi_vals[Pi_center] = (BF[index]*(BF[index]>0)-BB[index]*(BB[index]<0))/dX;
    Pi_vals[Pi_center+1] = (BB[index+1]*(BB[index+1]<0))/dX;

  }
  else if (index % n1 ==(n1-1))
  {
    Pi_cols[Pi_center-1] = index-1;
    Pi_cols[Pi_center]   = index;

    Pi_vals[Pi_center] = (BF[index]*(BF[index]>0)-BB[index]*(BB[index]<0))/dX;
    Pi_vals[Pi_center-1] = -(BF[index-1]*(BF[index-1]>0))/dX;

  }
  else 
  {
    Pi_cols[Pi_center-1] = index-1;
    Pi_cols[Pi_center] = index;
    Pi_cols[Pi_center+1] = index+1;

    Pi_vals[Pi_center-1] = -(BF[index-1]*(BF[index-1]>0))/dX;
    Pi_vals[Pi_center] = (BF[index]*(BF[index]>0)-BB[index]*(BB[index]<0))/dX;
    Pi_vals[Pi_center+1] = (BB[index+1]*(BB[index+1]<0))/dX;

  }
}

static inline void fill_mat_values_Xi_KFE(PetscInt index,  PetscInt n1, PetscInt n2, PetscScalar *C , PetscInt * Xi_cols, PetscScalar *Xi_vals)
{
  PetscInt k;

  for (k=0; k<n2; k++)
  {
    Xi_cols[k] = index % n1 + k*n1;
    Xi_vals[k] = -C[index+k*n1*n2];

  }
}



PetscErrorCode FormLinearSystem_C_KFE(PetscScalar *A, PetscScalar *BF, PetscScalar *BB, PetscScalar *C, PetscInt n1, PetscInt n2, PetscInt n, PetscScalar dt, PetscScalar dX, Mat petsc_mat)
{

  PetscErrorCode ierr;

  PetscInt       index, Pi_center;
  
  
  
  PetscInt       Pi_cols[3], Xi_cols[n2];
  PetscScalar    Pi_vals[3], Xi_vals[n2];

  Pi_center = 1;

  PetscFunctionBegin;
  for (index  = 0; index < n; index++) {


    // Pi_center = 1;
    memset(Pi_vals,0,3*sizeof(PetscScalar));
    memset(Pi_cols,-1,3*sizeof(PetscInt));

    memset(Xi_vals,0,n2*sizeof(PetscScalar));
    memset(Xi_cols,-1,n2*sizeof(PetscInt));
    
    Pi_cols[Pi_center] = index;
    Pi_vals[Pi_center] = 1.0/dt - A[index];


    fill_mat_values_Pi_KFE(index, Pi_center, n1, BF, BB, dX, Pi_cols,Pi_vals);
    fill_mat_values_Xi_KFE(index, n1, n2, C, Xi_cols,Xi_vals);
    ierr = MatSetValues(petsc_mat,1,&index,3,Pi_cols,Pi_vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(petsc_mat,1,&index,n2,Xi_cols,Xi_vals,INSERT_VALUES);CHKERRQ(ierr);


  }
  ierr = MatAssemblyBegin(petsc_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(petsc_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
