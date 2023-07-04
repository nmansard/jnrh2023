'''
Implement and solve the following nonlinear program:
decide q \in R^NQ
minimizing   sum_t || q - robot.q0 ||**2
so that 
      h(q) = target
      forall obstacles o,    (e_p - e_c)' e_A (e_p-e_c) >= 1
with h(q) the forward geometry (position of end effector) to be at target position,
e_A,e_c the ellipse matrix and center in the attached joint frame e_, and e_p = oMe^-1 o_p
the position of the obstacle point p in frame e_. 

The following tools are used:
- the ur10 model (loaded by example-robot-data)
- pinocchio.casadi for writing the problem and computing its derivatives
- the IpOpt solver wrapped in casadi
- the meshcat viewer

It assumes that the ellipses parameters are already computed, see ellipses.py for that.
'''

import time
import unittest
import numpy as np
import pinocchio as pin
import casadi
from pinocchio import casadi as cpin
import example_robot_data as robex
from numpy.linalg import norm
from types import SimpleNamespace

from utils.meshcat_viewer_wrapper import MeshcatVisualizer

# Change numerical print
pin.SE3.__repr__=pin.SE3.__str__
np.set_printoptions(precision=2, linewidth=300, suppress=True,threshold=1e6)

### HYPER PARAMETERS
Mtarget = pin.SE3(pin.utils.rotate('y', 3), np.array([-0.8, -0.1, 0.2]))  # x,y,z
q0 = np.array([ 0,5,3,0,2,0 ])
endEffectorFrameName = 'tool0'

# %jupyter_snippet 
# These values are computed using encapsulating_ellipse.py
ellipses = [
    SimpleNamespace(name="shoulder_lift_joint",
                    A=np.array([[75.09157846,  0.34008563, -0.08817025],
                                [ 0.34008563, 60.94969446, -0.55672959],
                                [-0.08817025, -0.55672959,  3.54456814]]),
                    center=np.array([-1.05980885e-04, -5.23471160e-02,  2.26280651e-01])),
    SimpleNamespace(name="elbow_joint",
                    A=np.array([[ 1.30344372e+02, -5.60880392e-02, -1.87555288e-02],
                                [-5.60880392e-02,  9.06119040e+01,  1.65531606e-01],
                                [-1.87555288e-02,  1.65531606e-01,  4.08568387e+00]]),
                    center=np.array([-2.01944435e-05,  7.22262249e-03,  2.38805264e-01])),
    SimpleNamespace(name="wrist_1_joint",
                    A=np.array([[ 2.31625634e+02,  5.29558437e-01, -1.62729657e-01],
                                [ 5.29558437e-01,  2.18145143e+02, -1.42425434e+01],
                                [-1.62729657e-01, -1.42425434e+01,  1.73855962e+02]]),
                    center=np.array([-9.78431524e-05,  1.10181763e-01,  6.67932259e-03])),
    SimpleNamespace(name="wrist_2_joint",
                    A=np.array([[ 2.32274519e+02,  1.10812959e-01, -1.12998357e-02],
                                [ 1.10812959e-01,  1.72324444e+02, -1.40077876e+01],
                                [-1.12998357e-02, -1.40077876e+01,  2.19132854e+02]]),
                    center=np.array([-2.64650554e-06,  6.27960760e-03,  1.11112087e-01])),
]
# %jupyter_snippet 

# %jupyter_snippet 
# Obstacle positions are arbitrary. Their radius is meaningless, just for visualization.
obstacles = [
    SimpleNamespace(radius=.01, pos=np.array([-.4,0.2+s,0.5]),name=f"obs_{i_s}")
    for i_s,s in enumerate(np.arange(-.5,.5,.1))
]
# %jupyter_snippet 

# --- Load robot model
robot = robex.load('ur10')
robot.q0 = q0
# Open the viewer
viz = MeshcatVisualizer(robot)
viz.display(robot.q0)
# %jupyter_snippet 
for e in ellipses:
    e.id = robot.model.getJointId(e.name)
    l,P = np.linalg.eig(e.A)
    e.radius = 1/l**.5
    e.rotation = P
    e.placement = pin.SE3(P,e.center)
# %jupyter_snippet 

# The pinocchio model is what we are really interested by.
model = robot.model
data = model.createData()
endEffector_ID = model.getFrameId(endEffectorFrameName)

# --- Add box to represent target
# Add a vizualization for the target
boxID = "world/box"
viz.addBox(boxID, [.05, .1, .2], [1., .2, .2, .5])
# Add a vizualisation for the tip of the arm.
tipID = "world/blue"
viz.addBox(tipID, [.08] * 3, [.2, .2, 1., .5])
# %jupyter_snippet 
for e in ellipses:
    viz.addEllipsoid(f'el_{e.name}',e.radius,[.3,.9,.3,.3])
for io,o in enumerate(obstacles):
    viz.addSphere(f'obs_{io}',o.radius,[.8,.3,.3,.9])
# %jupyter_snippet 

# %jupyter_snippet 
def displayScene(q,dt=1e-1):
    '''
    Given the robot configuration, display:
    - the robot
    - a box representing endEffector_ID
    - a box representing Mtarget
    '''
    pin.framesForwardKinematics(model,data,q)
    M = data.oMf[endEffector_ID]
    viz.applyConfiguration(boxID, Mtarget)
    viz.applyConfiguration(tipID, M)
    for e in ellipses:
        M = data.oMi[e.id]
        viz.applyConfiguration(f'el_{e.name}',M*e.placement)
    for io,o in enumerate(obstacles):
        viz.applyConfiguration(f'obs_{io}',pin.SE3(np.eye(3),o.pos))
    viz.display(q)
    time.sleep(dt)
displayScene(robot.q0)
# %jupyter_snippet 

q = robot.q0
                    
# --- Casadi helpers
cmodel = cpin.Model(model)
cdata = cmodel.createData()

cq = casadi.SX.sym("q",model.nq,1)
cpin.framesForwardKinematics(cmodel,cdata,cq)
error6_tool = casadi.Function('etool', [cq],
                             [ cpin.log6(cdata.oMf[endEffector_ID].inverse() * cpin.SE3(Mtarget)).vector ])
error3_tool = casadi.Function('etool', [cq],
                             [ cdata.oMf[endEffector_ID].translation - Mtarget.translation ])
error_tool = error3_tool

# %jupyter_snippet 
cpos = casadi.SX.sym('p',3)
for e in ellipses:
    e.e_pos = [  casadi.Function(f"e{e.id}_pos_{o.name}",[cq],
                                 [ cdata.oMi[e.id].inverse().act(casadi.SX(o.pos)) ])
                 for o in obstacles ]
    e.e_pos2 =  casadi.Function(f"e{e.id}",[cq,cpos],
                                 [ cdata.oMi[e.id].inverse().act(casadi.SX(cpos)) ])
                
# %jupyter_snippet 

opti = casadi.Opti()
var_q = opti.variable(model.nq)
totalcost = casadi.sumsqr(error_tool(var_q))

for e in ellipses:
    for (io,o) in enumerate(obstacles):
        # obstacle position in ellipsoid (joint) frame
        e_pos = e.e_pos[io](var_q)
        e_pos = e.e_pos2(var_q,o.pos)
        opti.subject_to( (e_pos-e.center).T@e.A@(e_pos-e.center) >=1 )

### SOLVE
opti.minimize(totalcost)
p_opts = dict(print_time=False, verbose=False)
s_opts = dict(print_level=0)
opti.solver("ipopt") # set numerical backend
opti.set_initial(var_q,q)

# Caution: in case the solver does not converge, we are picking the candidate values
# at the last iteration in opti.debug, and they are NO guarantee of what they mean.
try:
    sol = opti.solve_limited()
    sol_q = opti.value(var_q)

    q = opti.value(var_q)
    displayScene(q)
    time.sleep(1)

    dist = [ opti.value((e.e_pos[io](var_q)-e.center).T@e.A@(e.e_pos[io](var_q)-e.center))
             for io,o in enumerate(obstacles) for e in ellipses ]  
    dist = np.array(dist)
    print(dist)
    assert(np.all(dist>=1-1e-6))

except:
    print( 'ERROR IN CONVERGENCE')


# Test
ref_q = np.array([ 0.24658,  4.14076,  4.71167, -2.14954,  2.36311,  0.     ])
assert( np.allclose(sol_q,ref_q,1e-5) )
assert( opti.stats()['iter_count'] == 55)
