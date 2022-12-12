import numpy as np
from scipy.sparse import spdiags, diags
from src.support import integral_3D, integral_2D, derivative_3D, derivative_2D

import petsc4py
from petsc4py import PETSc
import petsclinearsystempoi
# Initilization

A = 100
Rho = 0.2
Lambda = 0.3
Epsilon = 0.1
Alpha = 0.2
Zeta = 0.1
Beta = 0.3
Gamma = 0.2 

Kappa_bar = 0.2
Eta = 0.3
L = 1

N_city = 378
B_city = np.arange(1,N_city+1,1)
T_city = np.arange(1,N_city+1,1)
p_city = np.arange(1,N_city+1,1)
theta_city = np.arange(1,N_city+1,1)
X_city = np.arange(1,N_city+1,1)
L_city = np.arange(1,N_city+1,1)


diag = np.transpose(-np.tile(np.arange(-1,0,1/N_city),(N_city,1)))
tau_city = spdiags(diag,np.arange(0,N_city,1),N_city,N_city)
tau_city += spdiags(diag,-np.arange(0,N_city,1),N_city,N_city)
tau_city -= spdiags(np.tile(1,N_city),0,N_city,N_city)

Eps_Terminal = 0.02
Kap_Terminal = 0.1

# Grid Definition

# X1,X2,X3 = Age, Wealth, City
#             a      x     n
#             j      i     n
dX1 = 1
X1_min = 0
X1_max = A
X1 = np.arange(X1_min,X1_max+dX1,dX1)
nX1 = len(X1)

dX2 = 0.05
X2_min = 0
X2_max = 500
X2 = np.arange(X2_min,X2_max+dX2,dX2)
nX2 = len(X2)

dX3 = 1
X3_min = 1
X3_max = N_city
X3 = np.arange(X3_min,X3_max+dX3,dX3)
nX3 = len(X3)

# Broadcasting for computation

X2_mesh, X3_mesh = np.meshgrid(X2, X3, indexing='ij')



B_city_mesh = np.tile(B_city,(nX2,1))
T_city_mesh = np.tile(T_city,(nX2,1))
p_city_mesh = np.tile(p_city,(nX2,1))
theta_city_mesh = np.tile(theta_city,(nX2,1))
X_city_mesh = np.tile(X_city,(nX2,1))
L_city_mesh = np.tile(L_city,(nX2,1))

V_Terminal = Eps_Terminal*(Kap_Terminal+X2_mesh)**(1-Gamma)/(1-Gamma)


V= np.tile(V_Terminal,(nX1,1,1))
tol = 1e-8

Tolerance = 1e-7
fraction = 0.1

while Dist > Tolerance:

    for count in range(len(X1)-1,0,-1):
        V_temp = V[count,:,:]

        dVdX2_F = derivative_2D(V_temp,X2,X3,'F')
        # dVdX2_F[-1,:] =  
        Sigma_F = (   B_city_mesh * T_city_mesh * L_city_mesh**Alpha/(Kappa_bar* dVdX2_F)  * (X2_mesh/X_city_mesh)**(1-Beta))**(1/(Eta-1))
        drift_F = Kappa_bar/Eta*Sigma_F**Eta*  X2_mesh**Beta* X_city_mesh**(1-Beta)-Gamma*X2_mesh
        I_drift_F =  drift_F > 0


        dVdX2_B = derivative_2D(V_temp,X2,X3,'B')
        # dVdX2_B[0,:] = 
        Sigma_B = (   B_city_mesh * T_city_mesh * L_city_mesh**Alpha/(Kappa_bar* dVdX2_B)  * (X2_mesh/X_city_mesh)**(1-Beta))**(1/(Eta-1))
        drift_B = Kappa_bar/Eta*Sigma_B**Eta*  X2_mesh**Beta* X_city_mesh**(1-Beta)-Gamma*X2_mesh
        I_drift_B =  drift_B < 0

        I_drift_SS = 1-I_drift_F-I_drift_B
        Sigma_SS = (Gamma * Eta/Kappa_bar * (X2_mesh/X_city_mesh)**(1-Beta)    )**(1/Eta)
        dVdX2_SS = B_city_mesh * T_city_mesh * L_city_mesh**Alpha/ (Kappa_bar* Sigma_SS**(Eta-1)  ) * (X2_mesh/X_city_mesh)**(1-Beta)
        # drift_SS = np,zeros(X2_mesh.shape)


        dVdX2_Upwind = I_drift_F*dVdX2_F + I_drift_B*I_drift_B + I_drift_SS*dVdX2_SS
        Sigma = (   B_city_mesh * T_city_mesh * L_city_mesh**Alpha/(Kappa_bar* dVdX2_Upwind)  * (X2_mesh/X_city_mesh)**(1-Beta))**(1/(Eta-1))
        # refinement
        Sigma[Sigma>=1] = 1
        Sigma[Sigma<=0] = 0



        Utility = B_city_mesh* (T_city_mesh*L_city_mesh**theta_city_mesh)*X2_mesh


        tau_city_mesh = np.tile(tau_city,(nX2,1,1))
        V_temp_mesh   = np.tile(V_temp,(1,nX2,1))
        Mu = tau_city_mesh**(-Epsilon)*V_temp_mesh**(Epsilon)/np.sum(tau_city_mesh**(-Epsilon)*V_temp_mesh**(Epsilon),axis=2)
        Ksi = 1/N_city * Mu**(-1/Epsilon-1)/tau_city_mesh


        # Assisgnment

        A   =   -(Rho-Gamma+Lambda)*np.ones(X2_mesh.shape)
        B_F =   drift_F
        B_B =   drift_B
        C   =   Lambda*Mu*Ksi # Only this term is non-X2_mesh.shape
        D   =   Utility

        # SizeVec = np.array([nX2, nX3],dtype=np.int32)

        petsc_mat = PETSc.Mat().create()
        petsc_mat.setType('aij')
        petsc_mat.setSizes([nX2 * nX3, nX2 * nX3])
        petsc_mat.setPreallocationNNZ(13)
        petsc_mat.setUp()
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_WORLD)
        ksp.setType('bcgs')
        ksp.getPC().setType('ilu')
        ksp.setFromOptions()


        A_1d = A.ravel(order = 'F')
        B_F_1d = B_F.ravel(order = 'F')
        B_B_1d = B_B.ravel(order = 'F')
        C_1d = C.ravel(order = 'F')

        petsclinearsystempoi.formLinearSystem_HJB(A_1d,B_F_1d,B_B_1d,C_1d, nX2, nX3, nX2*nX3, dX1,dX2, petsc_mat)

        D_1d = D.ravel(order = 'F')
        V_temp_1d = V_temp.ravel(order = 'F')
        b = V_temp_1d/dX1 + D_1d
        
        petsc_rhs = PETSc.Vec().createWithArray(b)

        x = petsc_mat.createVecRight()
        ksp.setOperators(petsc_mat)
        ksp.setTolerances(rtol=tol)
        ksp.solve(petsc_rhs, x)
        petsc_rhs.destroy()
        x.destroy()
        V_temp_new = np.array(ksp.getSolution()).reshape(A.shape,order = "F")

        # num_iter = ksp.getIterationNumber()

        V[count-1,:,:] = V_temp_new

    Phi = np.zeros(V.shape)

    Phi_Initial = np.zeros(X2_mesh.shape)

    Phi_Initial[0,:] = 1/nX3 # ???????? why dX2 in paper
    Phi = np.tile(Phi_Initial,(nX1,1,1))

    for count in range(0,len(X1)-1,1):
        Phi_temp = Phi[count,:,:] 

        V_temp = V[count,:,:]

        dVdX2_F = derivative_2D(V_temp,X2,X3,'F')
        # dVdX2_F[-1,:] =  
        Sigma_F = (   B_city_mesh * T_city_mesh * L_city_mesh**Alpha/(Kappa_bar* dVdX2_F)  * (X2_mesh/X_city_mesh)**(1-Beta))**(1/(Eta-1))
        drift_F = Kappa_bar/Eta*Sigma_F**Eta*  X2_mesh**Beta* X_city_mesh**(1-Beta)-Gamma*X2_mesh
        I_drift_F =  drift_F > 0


        dVdX2_B = derivative_2D(V_temp,X2,X3,'B')
        # dVdX2_B[0,:] = 
        Sigma_B = (   B_city_mesh * T_city_mesh * L_city_mesh**Alpha/(Kappa_bar* dVdX2_B)  * (X2_mesh/X_city_mesh)**(1-Beta))**(1/(Eta-1))
        drift_B = Kappa_bar/Eta*Sigma_B**Eta*  X2_mesh**Beta* X_city_mesh**(1-Beta)-Gamma*X2_mesh
        I_drift_B =  drift_B < 0

        I_drift_SS = 1-I_drift_F-I_drift_B
        Sigma_SS = (Gamma * Eta/Kappa_bar * (X2_mesh/X_city_mesh)**(1-Beta)    )**(1/Eta)
        dVdX2_SS = B_city_mesh * T_city_mesh * L_city_mesh**Alpha/ (Kappa_bar* Sigma_SS**(Eta-1)  ) * (X2_mesh/X_city_mesh)**(1-Beta)
        # drift_SS = np,zeros(X2_mesh.shape)


        dVdX2_Upwind = I_drift_F*dVdX2_F + I_drift_B*I_drift_B + I_drift_SS*dVdX2_SS
        Sigma = (   B_city_mesh * T_city_mesh * L_city_mesh**Alpha/(Kappa_bar* dVdX2_Upwind)  * (X2_mesh/X_city_mesh)**(1-Beta))**(1/(Eta-1))
        # refinement
        Sigma[Sigma>=1] = 1
        Sigma[Sigma<=0] = 0



        Utility = B_city_mesh* (T_city_mesh*L_city_mesh**theta_city_mesh)*X2_mesh


        tau_city_mesh = np.tile(tau_city,(nX2,1,1))
        V_temp_mesh   = np.tile(V_temp,(1,nX2,1))
        Mu = tau_city_mesh**(-Epsilon)*V_temp_mesh**(Epsilon)/np.sum(tau_city_mesh**(-Epsilon)*V_temp_mesh**(Epsilon),axis=2)
        # Ksi = 1/N_city * Mu**(-1/Epsilon-1)/tau_city_mesh


        # Assisgnment

        A   =   -Lambda*np.ones(X2_mesh.shape)
        B_F =   drift_F
        B_B =   drift_B
        C   =   Lambda*Mu# Only this term is non-X2_mesh.shape
        D   =   np.zeros(X2_mesh.shape)


        petsc_mat = PETSc.Mat().create()
        petsc_mat.setType('aij')
        petsc_mat.setSizes([nX2 * nX3, nX2 * nX3])
        petsc_mat.setPreallocationNNZ(13)
        petsc_mat.setUp()
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_WORLD)
        ksp.setType('bcgs')
        ksp.getPC().setType('ilu')
        ksp.setFromOptions()


        A_1d = A.ravel(order = 'F')
        B_F_1d = B_F.ravel(order = 'F')
        B_B_1d = B_B.ravel(order = 'F')
        C_1d = C.ravel(order = 'F')

        petsclinearsystempoi.formLinearSystem_KFE(A_1d,B_F_1d,B_B_1d,C_1d, nX2, nX3, nX2*nX3, dX1,dX2, petsc_mat)

        D_1d = D.ravel(order = 'F')
        Phi_temp_1d = Phi_temp.ravel(order = 'F')
        b = Phi_temp_1d/dX1 + D_1d
        
        petsc_rhs = PETSc.Vec().createWithArray(b)

        x = petsc_mat.createVecRight()
        ksp.setOperators(petsc_mat)
        ksp.setTolerances(rtol=tol)
        ksp.solve(petsc_rhs, x)
        petsc_rhs.destroy()
        x.destroy()
        Phi_temp_new = np.array(ksp.getSolution()).reshape(A.shape,order = "F")

        Phi[count+1,:,:] = Phi_temp_new

    L_city_new = L * integral_3D(Phi, dX1, dX2, dX3, 2)
    X_city_new = (L * integral_3D(Phi * X2_mesh**Zeta, X1, X2, 2 ))**(1/Zeta)

    dVdX2_F = derivative_3D(V, dX1, dX2, dX3, 'F')
    # dVdX2_F[-1,:] =  


    dVdX2_B = derivative_3D(V, dX1, dX2, dX3, 'B')
    # dVdX2_B[0,:] = 

    Sigma_F = (   B_city_mesh * T_city_mesh * L_city_mesh**Alpha/(Kappa_bar* dVdX2_F)  * (X2_mesh/X_city_mesh)**(1-Beta))**(1/(Eta-1))
    drift_F = Kappa_bar/Eta*Sigma_F**Eta*  X2_mesh**Beta* X_city_mesh**(1-Beta)-Gamma*X2_mesh
    I_drift_F =  drift_F > 0


    Sigma_B = (   B_city_mesh * T_city_mesh * L_city_mesh**Alpha/(Kappa_bar* dVdX2_B)  * (X2_mesh/X_city_mesh)**(1-Beta))**(1/(Eta-1))
    drift_B = Kappa_bar/Eta*Sigma_B**Eta*  X2_mesh**Beta* X_city_mesh**(1-Beta)-Gamma*X2_mesh
    I_drift_B =  drift_B < 0

    I_drift_SS = 1-I_drift_F-I_drift_B
    Sigma_SS = (Gamma * Eta/Kappa_bar * (X2_mesh/X_city_mesh)**(1-Beta)    )**(1/Eta)
    dVdX2_SS = B_city_mesh * T_city_mesh * L_city_mesh**Alpha/ (Kappa_bar* Sigma_SS**(Eta-1)  ) * (X2_mesh/X_city_mesh)**(1-Beta)
    # drift_SS = np,zeros(X2_mesh.shape)


    dVdX2_Upwind = I_drift_F*dVdX2_F + I_drift_B*I_drift_B + I_drift_SS*dVdX2_SS
    Sigma = (   B_city_mesh * T_city_mesh * L_city_mesh**Alpha/(Kappa_bar* dVdX2_Upwind)  * (X2_mesh/X_city_mesh)**(1-Beta))**(1/(Eta-1))
    # refinement
    Sigma[Sigma>=1] = 1
    Sigma[Sigma<=0] = 0

    h = Kappa_bar/Eta*Sigma_B**Eta*  X2_mesh**Beta* X_city_mesh**(1-Beta)

    Gamma_Vec = np.sum(integral_3D(h*Phi,dX1,dX2, dX3,1) , axis=1)/ np.sum(integral_3D(X2_mesh*Phi,dX1, dX2, dX3,1), axis = 1)

    Gamma_new = np.mean(Gamma_Vec)

    Dist_L = np.abs(L_city_new-L_city)
    Dist_X = np.abs(X_city_new-X_city)
    Dist_Gamma = np.abs(Gamma_new-Gamma)

    Dist = max(Dist_L, Dist_X, Dist_Gamma)

    L_city = fraction *L_city_new + (1-fraction) * L_city
    X_city = fraction * X_city_new + (1-fraction) * X_city
    Gamma  = fraction * Gamma_new + (1-fraction) * Gamma