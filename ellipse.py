'''
Optimization of the shape of an ellipse so that it contains a set of 3d points.


decide:
 - w in so3: ellipse orientation
 - r in r3: ellipse main dimensions
minimizing:
  r1*r2*r3 the volum of the ellipse
so that:
  r>=0
  for all points pk in a list,     pk in ellipse


'''

import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import numpy as np
import example_robot_data as robex
import matplotlib.pyplot as plt; plt.ion()
from pinocchio.visualize import GepettoVisualizer

### HYPER PARAMETERS
# Hyperparameters defining the optimal control problem.
T = 20
q0 = np.array([0, -2.5, 2, -1.2, -1.7, 0])
costWeightsRunning = np.array([])  # sin, 1-cos, y, ydot, thdot, f
costWeightsTerminal = np.array([])

### LOAD AND DISPLAY PENDULUM
# Load the robot model from example robot data and display it if possible in Gepetto-viewer
robot = robex.load('ur10')
viz = GepettoVisualizer(robot.model,robot.collision_model,robot.visual_model)
viz.initViewer()
viz.loadViewerModel()

# The pinocchio model is what we are really interested by.
model = robot.model
data = model.createData()
idTool = model.getFrameId('tool0')
x0 = np.concatenate([q0,np.zeros(model.nv)])
target = np.array([  .4,.0,.4 ])
target = np.array([  -.4,.0,.2 ])
model.armature = np.array([.5]*6)
gv = viz.viewer.gui
gv.addSphere("world/target", 0.05, [1.0, 0, 0, 1])
gv.applyConfiguration("world/target", target.tolist() + [1, 0, 0, 0])
gv.refresh()



for g in robot.collision_model.geometryObjects:
    viz.viewer.gui.setVisibility(viz.getViewerNodeName(g,pin.VISUAL), 'OFF')

g = robot.collision_model.geometryObjects[2]
gv.setVisibility(viz.getViewerNodeName(g,pin.COLLISION), 'ON')
vert = g.geometry.vertices()

NS = 30
for i in  np.arange(0,vert.shape[0]):
    gv.addSphere(f'world/point_{i}',5e-3,[1,0,0,0.8] if i % NS == 0 else [.3,.3,1,.1])
    gv.applyConfiguration(f'world/point_{i}',vert[i].tolist()+[1,0,0,0])

gv.applyConfiguration(viz.getViewerNodeName(g,pin.COLLISION),[0,.1,0,1,0,0,0])
gv.refresh()
    

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
A = R.T@casadi.diag(1/var_r**2)@R


totalcost = var_r[0]*var_r[1]*var_r[2]
opti.subject_to( var_r >= 0.1)

NS = 30 # Subsample rate
for i in  np.arange(0,vert.shape[0],NS):
    p = vert[i]
    opti.subject_to( (p-var_c).T@A@(p-var_c)  <= 1  )

### SOLVE
opti.minimize(totalcost)
opti.solver("ipopt") # set numerical backend
opti.set_initial(var_r,.1)

sol = opti.solve_limited()

sol_r = opti.value(var_r)
opti_A = opti.value(A)
opti_c = opti.value(var_c)


from utils.plot_ellipse import plotEllipse,plotVertices
fig,ax = plt.subplots(1,subplot_kw={'projection':'3d'})
plotEllipse(ax,opti_A,opti_c)
plotVertices(ax,vert,NS)


