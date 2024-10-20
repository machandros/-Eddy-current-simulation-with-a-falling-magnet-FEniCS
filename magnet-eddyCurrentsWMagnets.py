import numpy as np
import matplotlib.pyplot as plt

from fenics import *

import cupy
import cupyx
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import cupyx.scipy.sparse
import cupyx.scipy.sparse.linalg

import datetime

parameters['linear_algebra_backend'] = 'Eigen'

def tran2SparseMatrix(A):
    row, col, val = as_backend_type(A).data()
    return sps.csr_matrix((val, col, row))

mempool = cupy.get_default_memory_pool()

with cupy.cuda.Device(0):
    mempool.set_limit(size=3.5*1024**3)

set_log_level(20)

n = 65
mesh = UnitCubeMesh(n,n,n)

# define solution files
# # Create XDMF files for visualization output
xdmffile_A = XDMFFile('solutions/magScalarPotential.xdmf')
xdmffile_V = XDMFFile('solutions/ElecPotential.xdmf')
xdmffile_B = XDMFFile('solutions/magField.xdmf')
xdmffile_E = XDMFFile('solutions/elecCurrent.xdmf')

# Create time series (for use in reaction_system.py)
timeseries_B = TimeSeries('solutions/magScalarPotential')
timeseries_V = TimeSeries('solutions/ElecPotential')
timeseries_BF = TimeSeries('solutions/magField')
timeseries_EF = TimeSeries('solutions/elecCurrent')

# define dx
dx = Measure('dx', domain=mesh)

# Function spaces
CG_V = VectorElement('CG',mesh.ufl_cell(),1)
CG_F = FiniteElement('CG',mesh.ufl_cell(),1)
CG_VF = MixedElement([CG_V,CG_F])
V_VF = FunctionSpace(mesh,CG_VF)

W = VectorFunctionSpace(mesh, 'CG', 1)
P = V_VF.sub(0).collapse()
Z = V_VF.sub(1).collapse()

A, V = TrialFunctions(V_VF)
v, q = TestFunctions(V_VF)
A_out = Function(V_VF)

A0 = interpolate(Constant((0, 0, 0)),P)
A_diff = interpolate(Constant((0, 0, 0)),P)

B = Function(W)
E = Function(W)

assigner = FunctionAssigner([P,Z],V_VF)

# define constants and user expressions

# permiability of air = 1.25e-6
# permiability of neodymium = 1.3e-6
# permiability of iron = 6.3e-3
# permiability of copper/aluminium = 1.26e-6
# permiability of vacuum = 4*pi*1e-7
# Define magnetic permeability
class Permeability(UserExpression):
    def __init__(self, coords, **kwargs):
        super().__init__(**kwargs) # This part is new!
        self.coords = coords
    def eval_cell(self, values, x, cell):
        if x[0] > self.coords[0] and x[0] < self.coords[1] and x[1] > self.coords[4] and x[1] < self.coords[7] and x[2] > 0.4 and x[2] < 0.6:
            values[0] = 1.3e-6
        elif x[0] > 0.25 and x[0] < 0.75 and x[1] > 0.2 and x[1] < 0.3 and x[2] > 0.25 and x[2] < 0.75:
            values[0] = 1.26e-6
        else:
            values[0] = 1.25e-6

# Define conductivity
class Conductivity(UserExpression):
    def __init__(self, coords, **kwargs):
        super().__init__(**kwargs) # This part is new!
        self.coords = coords
    def eval_cell(self, values, x, cell):
        if x[0] > self.coords[0] and x[0] < self.coords[1] and x[1] > self.coords[4] and x[1] < self.coords[7] and x[2] > 0.4 and x[2] < 0.6: # magnet
            values[0] =  6.6e5
        elif x[0] > 0.25 and x[0] < 0.75 and x[1] > 0.2 and x[1] < 0.3 and x[2] > 0.25 and x[2] < 0.75:
            values[0] = 5.96e7
        else:
            values[0] = 0

class Magnet(UserExpression):
    def __init__(self, coords,ex,t, **kwargs):
        super().__init__(**kwargs) # This part is new!
        self.coords = coords
        self.ex = ex
        self.t = t

    def eval_cell(self, values, x, cell):
        if (x[0] > self.coords[0] and x[0] < self.coords[1] and x[1] > self.coords[4]+self.t*velocity and x[1] < self.coords[7]+self.t*velocity
            and x[2] > 0.4 and x[2] < 0.6):
            values[:] =  self.ex(x)

    def value_shape(self): return (3, )

dy = 0

def identifyMagnetCoords(t,velocity):
    global dy 
    dy = t*velocity
    x0 = 0.4
    y0 = 0.4 + dy
    x1 = 0.6
    y1 = 0.4 + dy
    x2 = 0.6
    y2 = 0.7 + dy
    x3 = 0.4
    y3 = 0.7+ dy
    return np.array([x0, x1, x2, x3, y0, y1, y2, y3]) 

velocity = -1.4286 #m/s
t = 0
dt = 0.01

# ex = Expression(('0','-1/(4*3.14*1e-7)','0'),degree = 1)
ex = Expression(('0','-1','0'),degree = 1)
coords = np.copy(identifyMagnetCoords(t,velocity))
M = Magnet(coords,ex,t,degree=1)

# define governing equations

# Define boundary condition
bcs = []
bcs.append(DirichletBC(V_VF.sub(0),Constant((0,0,0)),'on_boundary'))
bcs.append(DirichletBC(V_VF.sub(1),Constant((0)),'on_boundary'))

count = 0

for t in np.arange(0,0.07,dt):

    print("\n ################### t = ", t, " ##################################### \n")
    coords = np.copy(identifyMagnetCoords(t,velocity))
    M = Magnet(coords,ex,t,degree=1)
    
    mu = Permeability(coords, degree=1)
    sigma = Conductivity(coords,degree=1)
    nu = 1/mu
    
    f_A = (inner(nu*grad(A),grad(v))*dx
    + inner(Constant((1/(1.3e-6)))*Constant((4*pi*1e-7))*(-M),curl(v))*dx
    + inner(sigma*((A-A0)/dt),v)*dx
    + inner(sigma*grad(V),v)*dx
    )

    f_V = (inner(sigma*grad(V),grad(q))*dx
        - inner(sigma*((A-A0)/dt),grad(q))*dx
        )

    a1 = lhs(f_A+f_V)
    l1 = rhs(f_A+f_V)

    print("done 0")
    A1 = assemble(a1)
    [bc.apply(A1) for bc in bcs]

    print("done 1")
    L1 = assemble(l1)
    [bc.apply(L1) for bc in bcs]

    print("done 2")
    A1001 = tran2SparseMatrix(A1)
    b = L1[:]

    print("done 3")
    As = cupyx.scipy.sparse.csr_matrix(A1001)
    bs = cupy.array(b)
    print("done 4, starting solver at ",datetime.datetime.now())
    start = datetime.datetime.now()
    # A.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.lsqr(As, bs)[:1][0]) #slow
    A_out.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.gmres(As, bs,maxiter=50000)[:1][0]) #slow
    #A.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.cg(As, bs)[:1][0]) #not working
    #A.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.cgs(As, bs)[:1][0]) #not working
    # A.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.minres(As, bs)[:1][0]) #fast
    print("Finished solving at ",datetime.datetime.now(),"\n That took ", datetime.datetime.now()-start )

    if True:#count%5 == 0:

        print("####### post processing step ##########")
        # B = project(as_vector(curl(A.split()[0])), W)
        # B.rename("B","")

        B_trial = TrialFunction(W)
        B_Test = TestFunction(W)
        eqn_LHS = inner(B_trial,B_Test)*dx
        eqn_RHS = inner(curl(A_out.split()[0]),B_Test)*dx

        eqn_BC = DirichletBC(W,Constant((0,0,0)),'on_boundary')

        eqn_LHSA = assemble(eqn_LHS)
        eqn_BC.apply(eqn_LHSA)

        eqn_RHSA = assemble(eqn_RHS)
        eqn_BC.apply(eqn_RHSA)

        B_LHS = tran2SparseMatrix(eqn_LHSA)
        B_RHS = eqn_RHSA[:]

        As_B = cupyx.scipy.sparse.csr_matrix(B_LHS)
        bs_B = cupy.array(B_RHS)

        # u.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.lsqr(As, bs)[:1][0])
        B.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.gmres(As_B, bs_B,maxiter=30000)[:1][0])
        # B.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.minres(As_B, bs_B)[:1][0])
        B.rename("B","")
        # # Save solution to file (XDMF/HDF5)

        E_trial = TrialFunction(W)
        E_test = TestFunction(W)
        eqn_LHS = inner(E_trial,E_test)*dx
        eqn_RHS = inner(sigma*(((A_out.split()[0]-A0)/dt)+grad(A_out.split()[1])),E_test)*dx

        eqn_BC = DirichletBC(W,Constant((0,0,0)),'on_boundary')

        eqn_LHSA = assemble(eqn_LHS)
        eqn_BC.apply(eqn_LHSA)

        eqn_RHSA = assemble(eqn_RHS)
        eqn_BC.apply(eqn_RHSA)

        E_LHS = tran2SparseMatrix(eqn_LHSA)
        E_RHS = eqn_RHSA[:]

        As_E = cupyx.scipy.sparse.csr_matrix(E_LHS)
        bs_E = cupy.array(E_RHS)

        # u.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.lsqr(As, bs)[:1][0])
        E.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.gmres(As_E, bs_E,maxiter=30000)[:1][0])
        # B.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.minres(As_B, bs_B)[:1][0])
        E.rename("J","")

        xdmffile_A.write(A_out.split()[0], t)
        xdmffile_V.write(A_out.split()[1],t)
        xdmffile_B.write(B,t)
        xdmffile_E.write(E,t)

        # # # Save nodal values to file
        timeseries_B.store(A_out.split()[0].vector(), t)
        timeseries_V.store(A_out.split()[1].vector(),t)
        timeseries_BF.store(B.vector(),t)
        timeseries_EF.store(E.vector(),t)

    A_diff = A_out.split()[0] - A0
    assigner.assign([A0, Function(Z)],A_out)
    count+=1
