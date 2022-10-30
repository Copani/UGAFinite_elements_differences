import numpy as np
from sympy import *
import scipy as sc
import matplotlib.pyplot as plt
from time import time

class Finel:
  '''Implements the finite Elements method for a 1D second order differential equation of the form
  -d/dx[k(x) * du(d)/dx)] = s(x), where
  k(x) is a known system property,
  s(x) is a source term
  u(x) is the spacially distributed quantity we want to solve for.

  It needs two static (diriclet) boundary condicions (BC) left, right
  and max. one derivative (van neuman) BC. If a van neumann BC is set,
  the diriclet BC is ignored.

  INPUTS (correct order)
  - x: 1D array of grid points
  - k: function of one variable
  - s: function of one variable
  OPT INPUTS
  - order: Order of integration scheme, default: 1
  - left: left static (Diriclet) boundary condicion
  - right: right static (Diriclet) boundary condicion

  IMPORTANT PROPERTIES
  - .Diric_left: (input) left diriclet boundary cond.: u(a)
  - .Diric_right: (input) right diriclet boundary cond. u(b)
  - .Neum_left: (input) value of left van Neumann cond. du(x)/dx|x=a
  - .Neum_right: (input) value of right van Neumann cond. du(x)/dx|x=b
  - .x: (output) 1D array of grid points
  - .u (output) 1D array of solution points

  IMPORTANT METHODS
  - .solve(): solves the equation
  - .plot(): plots result
  '''
  def __init__(self, x, k, s, order = 1, left=0, right=0):
    self.order = order
    self.x_nointerp = x # x without interpolation
    self.N = len(self.x_nointerp) # number of non-interpolated grid points
    self.x = self.get_x() # input x if order 1, otherwise interpolated x
    self.n = len(self.x) # number of local grid points (including interpolated points)
    self.k = k(self.x) # elasticity coefficient
    self.s = s(self.x) # source func.
    self.u = np.zeros(self.n) # make space for the solution later
    self.phi = self.get_elem_func() # list of elementary functions on [-1,1]
    self.J = (x[1:] - x[:-1])/2 # Jacobian for every cell (in between grid points)
    self.Q = self.get_Q() # connectivity matrix
    self.M = self.get_M()  # wheight Mtx
    self.K = self.get_K() # stiffness Mtx
    self.Diric_left = left # left diriclet boundary cond.: u(a)
    self.Diric_right = right # right diriclet boundary cond. u(b)
    self.Neum_left = None # left van Neumann cond. du(x)/dx|x=a
    self.Neum_right = None # right van Neumann cond. du(x)/dx|x=b


  def solve(self):
    '''Solve Ku = Ms, respecting the boundary condicions
    OUTPUT
    - self.u: 1D array of solution, same shape as self.x'''
    if self.Neum_left == None and self.Neum_right == None: # Then we have only Diriclet boundary condicions.
      # take the general formulation where we cut off first and last dimension to get u(a) = u(b) = 0 and
      # a linear source term is added to ensure the set boundary condicions.
      K_Diric = self.K[1:self.n-1,1:self.n-1] # just truncate (cut)
      a = self.x[0]
      b = self.x[-1]
      u_a = self.Diric_left
      u_b = self.Diric_right
      u0 = ((b - self.x) / (b - a) * u_a + (self.x - a) / (b - a) * u_b)[1:self.n-1] # linear function from (a,u_a) to (b,u_b)
      dx = self.x[1:] - self.x[:-1]
      dkdx = (self.k[2:] - self.k[:-2]) / (dx[:-1] + dx[1:]) #  symmetric O(1) scheme, 2J = dx
      s0 =  dkdx * (u_b - u_a) / (b - a) # extra source term d/dx*(k*du/dx)
      s  = self.s[1:self.n-1] # trncated source term
      M_Diric = self.M[1:self.n-1,1:self.n-1]
      Ms_Diric = (M_Diric @ (s + s0))
      u_star = np.linalg.solve(K_Diric, Ms_Diric)
      self.u[1:-1] = u_star + u0
      self.u[0] = u_a
      self.u[-1] = u_b

    elif self.Neum_left != None:
      if self.Neum_right != None:
        print('If there are only Diriclet boundary conditions, the y axis shift of the solution is undefined.')
        return
      K_Neum = self.K[:self.n - 1, :self.n - 1]  # just truncate (cut) left side
      phi_a = self.Neum_left
      u_b = self.Diric_right
      s = self.s[:self.n - 1]  # truncated source term
      M_Neum = self.M[:self.n - 1,:self.n - 1] # truncated M mtx
      Ms_Neum = (M_Neum @ s) - phi_a
      u_star = np.linalg.solve(K_Neum, Ms_Neum)
      self.u[:-1] = u_star + u_b
      self.u[-1] = u_b

    else:
      K_Neum = self.K[1:self.n, 1:self.n]  # just truncate (cut) left side
      phi_b = self.Neum_right
      u_a = self.Diric_left
      s = self.s[1:self.n]  # truncated source term
      M_Neum = self.M[1:self.n, 1:self.n] # truncated M mtx
      Ms_Neum = (M_Neum @ s) - phi_b # right hand side of matrix eq. "M s"
      u_star = np.linalg.solve(K_Neum, Ms_Neum)
      self.u[1:] = u_star + u_a
      self.u[0] = u_a


### unimportant tehnical stuff below
  def get_Q(self):
    '''generate connectivity matrix based on number of elements
    OUTPUT
    (2(n-1) x n) array, where n is number of 1D grid points'''
    if self.order == 1:
      Q = np.zeros([(self.n - 1)*2, self.n])
      for i in range(0,self.n-2):
        Q[2*i+1:2*i+3,i+1] = np.ones(2)
      Q[0,0] = 1
      Q[-1,-1] = 1
    if self.order == 2:
      # note that n is bigger here due to 2nd order interpolation
      Q = np.zeros([(self.n - 1)*3//2, self.n])
      for i in range((self.n - 3)//2):
        Q[3*i+1:3*i+4,2*i+1:2*i+3] = np.array([[1,0],[0,1],[0,1]])
      Q[0,0] = 1
      Q[-2:,-2:] = np.eye(2)
    return Q


  def get_x(self):
    '''give input x if order == 1, otherwise return interpolated x:
    1D array'''
    if self.order == 1:
      return self.x_nointerp
    if self.order == 2:
      Interp = np.zeros([self.N + (self.N-1),self.N]) # interpolation matrix
      for j in range(self.N-1):
        Interp[2*j,j] = 1
        Interp[2*j+1,j:j+2] = np.ones(2) / 2
      Interp[-1,-1] = 1
      return Interp @ self.x_nointerp


  def get_elem_func(self):
    '''define elementary cell functions on e = [-1,1]
    General note: use evalf() to evaluate expressions in sympy
    OUTPUT
    List of sympy expressions'''
    if self.order == 1:
      self.e = Symbol('e')
      phi1 = (self.e + 1)/2
      phi2 = (1 - self.e)/2
      return [phi1, phi2]

    if self.order == 2:
      self.e = Symbol('e')
      phi1 = self.e * (self.e - 1) / 2
      phi2 = -(self.e + 1) * (self.e - 1)
      phi3 = self.e * (self.e + 1) / 2
      return [phi1, phi2, phi3]


  def get_M(self):
    '''generate weight matrix M from elementary cells and jacobian
    OUTPUT
    (n x n) array'''

    Me = np.zeros([self.order + 1,self.order + 1]) # calc. elementary mass mtx
    for i in range(self.order+1):
      for j in range(self.order + 1):
        Me[i,j] = integrate(self.phi[i] * self.phi[j],(self.e,-1,1))

    Ml = Me * self.J[0] # calc. local mass mtx
    for k in range(1, self.N - 1):
      Ml = sc.linalg.block_diag(Ml,Me * self.J[k])

    Mg = self.Q.T @ Ml @ self.Q # calc. global mass mtx.
    return Mg


  def get_K(self):
    '''generate stiffness matrix K from elementary cells and jacobian
    OUTPUT
    (n x n) array'''

    Ke = np.zeros([self.order + 1,self.order + 1]) # calc. elementary stiffness mtx
    for i in range(self.order+1):
      for j in range(self.order+1):
        Ke[i,j] = integrate(diff(self.phi[i],self.e) * diff(self.phi[j],self.e),(self.e,-1,1))

    k0 = (self.k[0] + self.k[1]) / 2
    Kl = k0 * Ke / self.J[0] # calc. local mass mtx
    for m in range(1, self.N - 1):
      k_m = (self.k[m] + self.k[m + 1]) / 2
      Kl = sc.linalg.block_diag(Kl,k_m * Ke / self.J[m]) # iteratively adds a diagonal block
      # Here, we had to devide by jacobian instead of multiplying!

    Kg = self.Q.T @ Kl @ self.Q # calc. global mass mtx.
    return Kg


  def plot(self):
    if self.Neum_left == None and self.Neum_right == None:
      BC = 'Diriclet'
    else:
      BC = 'Mixed'
    plt.figure()
    plt.title('Steady diffuction for {} boundary condicions'.format(BC))
    plt.plot(self.x, self.u)
    plt.xlabel('$x$')
    plt.ylabel('$u(x)$')
    plt.show()
