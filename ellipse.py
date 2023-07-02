'''
Optimization of the shape of an ellipse so that it contains a set of 3d points.


decide:
 - w in so3: ellipse orientation
 - r in r3: ellipse main dimensions
minimizing:
  r1*r2*r3 the volum of the ellipse
so that:
  r>=0
  for all points pk in a list,     pk in ellipse:     (pk-c)@A@pk-c)<=1

with A,c the matrix representation of the ellipsoid A=exp(w)@diag(1/r**2)@exp(w).T

'''

import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import numpy as np
import example_robot_data as robex
import matplotlib.pyplot as plt; plt.ion()
from utils.meshcat_viewer_wrapper import MeshcatVisualizer


# --- Load robot model
robot = robex.load('ur10')
viz = MeshcatVisualizer(robot)
viz.display(robot.q0)

g = robot.collision_model.geometryObjects[5]
vert = g.geometry.vertices()

NS = 1 # Subsample rate
for i in  np.arange(0,vert.shape[0]):
    pass
    #gv.addSphere(f'world/point_{i}',5e-3,[1,0,0,0.8] if i % NS == 0 else [.3,.3,1,.1])
    #gv.applyConfiguration(f'world/point_{i}',vert[i].tolist()+[1,0,0,0])
    viz.addSphere(f'world/point_{i}',5e-3,[1,0,0,0.8] if i % NS == 0 else [.3,.3,1,.1])
    viz.applyConfiguration(f'world/point_{i}',vert[i].tolist()+[1,0,0,0])

### CASADI
cw = casadi.SX.sym('w',3)
exp = casadi.Function('exp3',[cw], [cpin.exp3(cw)])

###
'''
decide 
 - w in so3: ellipse orientation
 - r in r3: ellipse main dimensions
minimizing:
  r1*r2*r3 the volum of the ellipse
so that:
  r>=0
  for all points pk in a list,     pk in ellipse

'''
opti = casadi.Opti()
var_w = opti.variable(3)
var_r = opti.variable(3)
var_c = opti.variable(3)

# The ellipsoid matrix is represented by w=log3(R),diag(P) with R,P=eig(A)
R = exp(var_w)
A = R@casadi.diag(1/var_r**2)@R.T


totalcost = var_r[0]*var_r[1]*var_r[2]
opti.subject_to( var_r >= 0)

for i in  np.arange(0,vert.shape[0],NS):
    # Convert point from geometry frame to joint frame
    p = g.placement.act(vert[i])
    # Constraint the ellipsoid to be including the point
    opti.subject_to( (p-var_c).T@A@(p-var_c)  <= 1  )

### SOLVE
opti.minimize(totalcost)
opti.solver("ipopt") # set numerical backend
opti.set_initial(var_r,.1)

sol = opti.solve_limited()

sol_r = opti.value(var_r)
sol_A = opti.value(A)
sol_c = opti.value(var_c)
sol_R = opti.value(exp(var_w))

from utils.plot_ellipse import plotEllipse,plotVertices
fig,ax = plt.subplots(1,subplot_kw={'projection':'3d'})
plotEllipse(ax,sol_A,sol_c)
plotVertices(ax,np.vstack([g.placement.act(p) for p in vert]),NS)

e,P = np.linalg.eig(sol_A)
sol_r = 1/e**.5
sol_R = P
M = pin.SE3(sol_R,sol_c)
viz.addEllipsoid('el',sol_r,[.3,.9,.3,.3])
viz.applyConfiguration('el',M)

print(f'SimpleNamespace(name="{robot.model.names[g.parentJoint]}",\n'+
      f'                A=np.{repr(sol_A)},\n'+
      f'                center=np.{repr(sol_c)})')


M0 = pin.SE3.Random()
for i in  np.arange(0,vert.shape[0]):
    viz.applyConfiguration(f'world/point_{i}',M0.act(vert[i]).tolist()+[1,0,0,0])
viz.applyConfiguration(viz.getViewerNodeName(g,pin.VISUAL),M0)
viz.applyConfiguration('el',M0*M)
