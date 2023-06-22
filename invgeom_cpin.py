'''
Stand-alone inverse geometry for a manipulator robot with a 6d objective.

Implement and solve the following nonlinear program:
decide q \in R^NQ
minimizing   || log( M(q)^-1 M^* ||^2 
with M(q) \in SE(3) the placement of the robot end-effector, and M^* the target.

The following tools are used:
- the ur10 model (loaded by example-robot-data)
- pinocchio.casadi for writing the problem and computing its derivatives
- the IpOpt solver wrapped in casadi
- the meshcat viewer
'''

import time
import unittest
import numpy as np
import pinocchio as pin
import casadi
from pinocchio import casadi as cpin
import example_robot_data as robex
from scipy.optimize import fmin_bfgs
from numpy.linalg import norm

from utils.meshcat_viewer_wrapper import MeshcatVisualizer

# Change numerical print
pin.SE3.__repr__=pin.SE3.__str__
np.set_printoptions(precision=2, linewidth=300, suppress=True,threshold=1e6)

### HYPER PARAMETERS
Mtarget = pin.SE3(pin.utils.rotate('x', 3.14 / 4), np.array([-0.5, 0.1, 0.2]))  # x,y,z
Mtarget = pin.SE3(pin.utils.rotate('x', 3), np.array([-0.5, 0.1, 0.2]))  # x,y,z
q0 = np.array([0, -3.14 / 2, 0, 0, 0, 0])

# --- Load robot model
robot = robex.load('ur5')
robot.q0 = q0
# Open the viewer
viz = MeshcatVisualizer(robot)
viz.display(robot.q0)
time.sleep(.3)
print("Let's go to pdes ... with casadi")

# The pinocchio model is what we are really interested by.
model = robot.model
data = model.createData()
idTool = model.getFrameId('tool0')
idElbow = model.getFrameId('elbow_joint')

# --- Add box to represent target
# Add a vizualization for the target
boxID = "world/box"
viz.addBox(boxID, [.05, .1, .2], [1., .2, .2, .5])
# %jupyter_snippet 1
# Add a vizualisation for the tip of the arm.
tipID = "world/blue"
viz.addBox(tipID, [.08] * 3, [.2, .2, 1., .5])

cmodel = cpin.Model(model)
cdata = cmodel.createData()

# Casadi helper functions
# S is a skew-symmetric matrix
# s is the 3-vector extracted from S
def wedge(S):
    s = casadi.vertcat(S[2,1], S[0,2], S[1,0])
    return s
    
# R is a rotation matrix not far from the identity
# w is the approximated rotation vector equivalent to R
def log3_approx(R):
    w = wedge(R - R.T) / 2
    return w



cq = casadi.SX.sym("x",model.nq,1)
cpin.framesForwardKinematics(cmodel,cdata,cq)
pos_tool = casadi.Function('ptool', [cq], [ cdata.oMf[idTool].translation ])
c_dR = cdata.oMf[idTool].rotation.T @ Mtarget.rotation
dR_tool = casadi.Function('Rtool', [cq], [ c_dR ])
Rerr_tool = casadi.Function('Rtool', [cq], [ log3_approx(c_dR) ])
error_tool = casadi.Function('etool', [cq], [ cpin.log6(cdata.oMf[idTool].inverse() * cpin.SE3(Mtarget)).vector ])

### PROBLEM
opti = casadi.Opti()
var_q = opti.variable(model.nq)

totalcost = casadi.sumsqr( pos_tool(var_q) - Mtarget.translation )
totalcost += casadi.sumsqr( Rerr_tool(var_q) )
#totalcost = casadi.sumsqr( error_tool(var_q) )

### SOLVE
opti.minimize(totalcost)
opti.solver("ipopt") # set numerical backend

def logger_function(niter):
    q = opti.debug.value(var_q)
    pin.framesForwardKinematics(model,data,q)
    M = data.oMf[idTool]
    viz.applyConfiguration(boxID, Mtarget)
    viz.applyConfiguration(tipID, M)
    viz.display(q)
    time.sleep(1e-1)
opti.callback(logger_function)

# Caution: in case the solver does not converge, we are picking the candidate values
# at the last iteration in opti.debug, and they are NO guarantee of what they mean.
try:
    sol = opti.solve_limited()
    sol_q = opti.value(var_q)
except:
    print('ERROR in convergence, plotting debug info.')
    sol_q = opti.debug.value(var_q)

