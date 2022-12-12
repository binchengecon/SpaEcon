#include "petsclinearsystem_poisson.h"


static inline void fill_mat_values_Pi(PetscInt index, PetscInt Pi_center, PetscInt n1, PetscScalar *BF, PetscScalar * BB, PetscScalar dX, PetscScalar * Pi_cols, PetscScalar *Pi_vals)
{

  if (index % n1==0) //LB
  {
    Pi_cols[Pi_center] = index;
    Pi_cols[Pi_center+1] = index+1;

    Pi_vals[Pi_center] = (BF[index]-BB[index])/dX;
    Pi_vals[Pi_center+1] = -(BF[index])/dX;

  }
  else if (index % n1 ==(n1-1))
  {
    Pi_cols[Pi_center-1] = index-1;
    Pi_cols[Pi_center]   = index;

    Pi_vals[Pi_center] = (BF[index]-BB[index])/dX;
    Pi_vals[Pi_center-1] = (BB[index])/dX;

  }
  else 
  {
    Pi_cols[Pi_center-1] = index-1;
    Pi_cols[Pi_center] = index;
    Pi_cols[Pi_center+1] = index+1;

    Pi_vals[Pi_center-1] = (BB[index])/dX;
    Pi_vals[Pi_center] = (BF[index]-BB[index])/dX;
    Pi_vals[Pi_center+1] = -(BF[index])/dX;

  }
}

static inline void fill_mat_values_Xi(PetscInt index,  PetscInt n1, PetscInt n2, PetscScalar *C , PetscScalar * Xi_cols, PetscScalar *Xi_vals)
{
  PetscInt k;

  for (k=0; k<n2; k++)
  {
    Xi_cols[k] = index % n1 + k*n1;
    Xi_vals[k] = C[index+k*n1*n2];

  }
}



PetscErrorCode FormLinearSystem_Poisson_C(PetscScalar *A, PetscScalar *BF, PetscScalar *BB, PetscScalar *C, PetscInt n1, PetscInt n2, PetscInt n, PetscScalar dt, PetscScalar dX, Mat petsc_mat);
{

  PetscErrorCode ierr;

  PetscInt       index, Pi_center;
  
  
  
  PetscInt       Pi_cols[3], Xi_cols[n2];
  PetscScalar    Pi_vals[3], Xi_vals[n2];

  Pi_center = 1;

  PetscFunctionBegin;
  for (index  = 0; index < n; index++) {


    center = 1;
    memset(Pi_vals,0,3*sizeof(PetscScalar));
    memset(Pi_cols,-1,3*sizeof(PetscInt));

    memset(Xi_vals,0,n2*sizeof(PetscScalar));
    memset(Xi_cols,-1,n2*sizeof(PetscInt));
    
    Pi_cols[Pi_center] = index;
    Pi_vals[Pi_center] = 1.0/dt - A[index];


    fill_mat_values_Pi(index, n1, BF, BB, dX, Pi_cols,Pi_vals);
    fill_mat_values_Xi(index, n1, n2, C, Xi_cols,Xi_vals);
    ierr = MatSetValues(petsc_mat,1,&index,3,Pi_cols,Pi_vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(petsc_mat,1,&index,n2,Xi_cols,Xi_vals,INSERT_VALUES);CHKERRQ(ierr);


  }
  ierr = MatAssemblyBegin(petsc_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(petsc_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
