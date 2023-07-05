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
MsaneTarget = pin.SE3(pin.utils.rotate("x", 3), np.array([-0.5, 0.1, 0.2]))  # x,y,z
q0 = np.array([0, -3.14 / 2, 0, 0, 0, 0])

# --- Load robot model
robot = robex.load("ur5")
robot.q0 = q0

# The pinocchio model is what we are really interested by.
model = robot.model
data = model.createData()
idTool = model.getFrameId("tool0")
idElbow = model.getFrameId("elbow_joint")


# Casadi helpers
cmodel = cpin.Model(model)
cdata = cmodel.createData()


# Casadi helper functions
# S is a skew-symmetric matrix
# s is the 3-vector extracted from S
def wedge(S):
    s = casadi.vertcat(S[2, 1], S[0, 2], S[1, 0])
    return s


# R is a rotation matrix not far from the identity
# w is the approximated rotation vector equivalent to R
def log3_approx(R):
    w = wedge(R - R.T) / 2
    return w


cq = casadi.SX.sym("x", model.nq, 1)
cpin.framesForwardKinematics(cmodel, cdata, cq)
error_tool = casadi.Function(
    "etool",
    [cq],
    [cpin.log6(cdata.oMf[idTool].inverse() * cpin.SE3(MsaneTarget)).vector],
)
pos_tool = casadi.Function(
    "ptool", [cq], [cdata.oMf[idTool].translation - MsaneTarget.translation]
)
rot_tool = casadi.Function(
    "rtool", [cq], [log3_approx(cdata.oMf[idTool].rotation.T @ MsaneTarget.rotation)]
)
dR = casadi.Function("dR", [cq], [cdata.oMf[idTool].rotation.T @ MsaneTarget.rotation])


### PROBLEM
opti = casadi.Opti()
var_q = opti.variable(model.nq)

# totalcost = casadi.sumsqr( error_tool(var_q) )
totalcost = casadi.sumsqr(pos_tool(var_q))
totalcost += casadi.sumsqr(rot_tool(var_q))

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
