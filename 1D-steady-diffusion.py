# This is an example of a python code that solve the 1D steady diffusion equation with linear finite elements.
# There are as many comments as possible  
#
import numpy as np
import matplotlib.pyplot as plt 

# Define physical domain = [0,L]
L=10.

O=0.

# number of elements
NEL=100

# Define diffusivity
diffusivity=np.zeros(NEL) * 1

# choose several type of diffusivity profiles (here constant per element): 1=constant, 2= linear increase, 3= non-linear decrease, 4= 2 values)
diff_profile=1

for iel in range(0,NEL):  
# constant 
    if (diff_profile==1):
        diffusivity[iel]=1.
        dlabel='constant diffusivity'
# linearly increasing
    if (diff_profile==2):
        diffusivity[iel]=float(iel)             
        dlabel='increasing diffusivity'
# non-linearly decreasing
    if (diff_profile==3):
        diffusivity[iel]=1./float(iel+.001)     
        dlabel='decreasing diffusivity'
# 2 values of diffusivity
    if (diff_profile==4):
        dlabel='2 values diffusivity'
        if (iel <= NEL/2):
            diffusivity[iel]=1.
        else:
            diffusivity[iel]=5.
# number of corners
NC=NEL+1
# global nodes (here we use linear elements, there are no interior points in the elements, so NG=NC)
NG=NC
global_nodes=np.linspace(O,L,NC) # warning: remember later that in python the indices start at 0.
# local nodes (we use linear elements so NL=NEL*2. In general NL=NEL*(POLYNOMIAL-ORDER+1))
NL=NEL*2

# Define the permanent source
S=np.zeros(NG)
# Dirac source close to the center of the domain
#S[np.int(np.floor(NG/2))]=1.
# gaussian like source
for i in range(NG):
    pos=global_nodes[i]
    S[i]=np.exp(-20*((pos-L/3)/L)**2)


# Initialize arrays of corners
left_corner=np.zeros(NEL)
right_corner=np.zeros(NEL)

# Initialize jacobian
jacobian=np.zeros(NEL)

# Define grid points and elements
for iel in range(0,NEL):  
    a_e=global_nodes[iel]
    b_e=global_nodes[iel+1]
    left_corner[iel]=a_e
    right_corner[iel]=b_e
    jacobian[iel]=(b_e-a_e)/2
#    print('iel, a_e, b_e, alpha_e :',iel,left_corner[iel],right_corner[iel],jacobian[iel])
    
# Connectivity matrix: rectangular matrix containing integers
Q=np.zeros((NL,NG),dtype=int)
# define an integer array which will give the global number for each corner of each element: iglob(iel,icorner).
iglob=np.empty((NEL,2),dtype=int)

# initialize iglob for the first element
iglob[0,0]=1
iglob[0,1]=2
# current count of global nodes is 2
ig=2
# Loop on the elements and assign the next global node to the right corner
for iel in range(1,NEL):
    ig=ig+1
    iglob[iel,1]=ig
# Loop again on the elements and assign the global number of the left corner
for iel in range(1,NEL):
        iglob[iel,0]=iglob[iel-1,1]

# Fill the connectivity matrix: the number of the line is the local number of the point. 
# We put a 1 in the column corresponding to the global number of this point.

# we start a counter at -1 because indices start at 0 in python
iline=-1
for iel in range(1,NEL+1): # loop on the elements
    for icorner in range(2): # loop on the corners of the elements
        iline=iline+1 # increase the line counter
#        print('iel,icorner,iline =',iel,icorner,iline)
        icol=iglob[iel-1,icorner]
        Q[iline,icol-1]=1
        
# Define the transpose of the connectivity matrix
        Qt=Q.transpose()

# Define elementary mass and stiffness matrices
mass_matrix_elem=np.zeros((NEL,2,2))
stif_matrix_elem=np.zeros((NEL,2,2))

for iel in range(NEL):
    mass_matrix_elem[iel,0,0]=jacobian[iel]*(2/3)
    mass_matrix_elem[iel,0,1]=jacobian[iel]*(1/3)
    mass_matrix_elem[iel,1,0]=jacobian[iel]*(1/3)
    mass_matrix_elem[iel,1,1]=jacobian[iel]*(2/3)
    #
    stif_matrix_elem[iel,0,0]=diffusivity[iel]*(1/2)/jacobian[iel]
    stif_matrix_elem[iel,0,1]=diffusivity[iel]*(-1/2)/jacobian[iel]
    stif_matrix_elem[iel,1,0]=diffusivity[iel]*(-1/2)/jacobian[iel]
    stif_matrix_elem[iel,1,1]=diffusivity[iel]*(1/2)/jacobian[iel]
    
# Define the local matrices
mass_matrix_local=np.zeros((NEL*2,NEL*2))
stif_matrix_local=np.zeros((NEL*2,NEL*2))

for iel in range(NEL):
    for icorner in range(2):
        ibeg=(iel*2)
        mass_matrix_local[ibeg,ibeg]=mass_matrix_elem[iel,0,0]
        mass_matrix_local[ibeg,ibeg+1]=mass_matrix_elem[iel,0,1]
        mass_matrix_local[ibeg+1,ibeg]=mass_matrix_elem[iel,1,0]
        mass_matrix_local[ibeg+1,ibeg+1]=mass_matrix_elem[iel,1,1]
        #
        stif_matrix_local[ibeg,ibeg]=stif_matrix_elem[iel,0,0]
        stif_matrix_local[ibeg,ibeg+1]=stif_matrix_elem[iel,0,1]
        stif_matrix_local[ibeg+1,ibeg]=stif_matrix_elem[iel,1,0]
        stif_matrix_local[ibeg+1,ibeg+1]=stif_matrix_elem[iel,1,1]

# Define the global matrices
mass_matrix_global=np.zeros((NG,NG))
stif_matrix_global=np.zeros((NG,NG))

mass_matrix_global=np.matmul(Qt,np.matmul(mass_matrix_local,Q))
stif_matrix_global=np.matmul(Qt,np.matmul(stif_matrix_local,Q))

# We may check that the mass matrix is ok by summing all its elements and check that we obtain the lenght of the domain
print('Sum of the global mass matrix coefficients:',mass_matrix_global.sum())
print('To be compared with the length of the domain:',L)
# compute the source term
F=np.matmul(mass_matrix_global,S)

# Now, we need to remove the first and last lines and columns because of the boundary conditions
MG=mass_matrix_global[1:NG-1,1:NG-1]
KG=stif_matrix_global[1:NG-1,1:NG-1]
FG=F[1:NG-1]
UG=np.linalg.solve(KG,FG)
XG=global_nodes[1:NG-1]

# Plot result
plt.figure()
plt.plot(XG,UG)
plt.xlabel('X')
plt.ylabel('U')
plt.title('1D steady diffusion equation, '+dlabel)
plt.show()
