"""
Stand-alone inverse geometry for a manipulator robot with a 6d objective.

Implement and solve the following nonlinear program:
decide q \in R^NQ
minimizing   || log( M(q)^-1 M^* ||^2 
with M(q) \in SE(3) the placement of the robot end-effector, and M^* the target.

The following tools are used:
- the ur10 model (loaded by example-robot-data)
- pinocchio.casadi for writing the problem and computing its derivatives
- the IpOpt solver wrapped in casadi

The test leads to Nan at first iteration of IpOpt, likely due to an improper derivation
of the log function. This error is sensitive: change a little bit the target and everything
comes back to work.
Two targets are provided: MsaneTarget works, MfailureTarget don't. 
Change anything in MfailureTarget (angle or axis), and it is back to normal functionning.
"""

import time
import unittest
import numpy as np
import pinocchio as pin
import casadi
from pinocchio import casadi as cpin
import example_robot_data as robex
from scipy.optimize import fmin_bfgs
from numpy.linalg import norm


# Change numerical print
pin.SE3.__repr__ = pin.SE3.__str__
np.set_printoptions(precision=2, linewidth=300, suppress=True, threshold=1e6)

### HYPER PARAMETERS

# This target leads to proper convergence
MsaneTarget = pin.SE3(pin.utils.rotate("x", 3), np.array([-0.5, 0.1, 0.2]))  # x,y,z
# This target leads to Nan
MfailureTarget = pin.SE3(
    pin.utils.rotate("x", 3.14 / 4), np.array([-0.5, 0.1, 0.2])
)  # x,y,z
q0 = np.array([0, -3.14 / 2, 0, 0, 0, 0])

# --- Load robot model
robot = robex.load("ur5")
robot.q0 = q0

# The pinocchio model is what we are really interested by.
model = robot.model
data = model.createData()
idTool = model.getFrameId("tool0")

# Casadi helpers
cmodel = cpin.Model(model)
cdata = cmodel.createData()
cq = casadi.SX.sym("x", model.nq, 1)
cpin.framesForwardKinematics(cmodel, cdata, cq)
error_tool = casadi.Function(
    "etool",
    [cq],
    [cpin.log6(cdata.oMf[idTool].inverse() * cpin.SE3(MfailureTarget)).vector],
)

### PROBLEM
opti = casadi.Opti()
var_q = opti.variable(model.nq)
totalcost = casadi.sumsqr(error_tool(var_q))

### SOLVE
opti.minimize(totalcost)
opti.solver("ipopt")  # set numerical backend

# Caution: in case the solver does not converge, we are picking the candidate values
# at the last iteration in opti.debug, and they are NO guarantee of what they mean.
try:
    sol = opti.solve_limited()
    sol_q = opti.value(var_q)
except:
    print("ERROR in convergence, plotting debug info.")
    sol_q = opti.debug.value(var_q)

assert opti.return_status() == "Invalid_Number_Detected"
