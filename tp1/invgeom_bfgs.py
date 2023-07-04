"""
Stand-alone inverse geometry for a manipulator robot with a 6d objective.

Implement and solve the following nonlinear program:
decide q \in R^NQ
minimizing   || log( M(q)^-1 M^* ||^2
with M(q) \in SE(3) the placement of the robot end-effector, and M^* the target.

The following tools are used:
- the ur10 model (loaded by example-robot-data)
- the fmin_bfgs solver of scipy (with finite differences automatically implemented)
- the meshcat viewer
"""

# %jupyter_snippet imports
import time
import unittest

import example_robot_data as robex
import numpy as np
import pinocchio as pin
from numpy.linalg import norm
from scipy.optimize import fmin_bfgs

from utils.meshcat_viewer_wrapper import MeshcatVisualizer

# %end_jupyter_snippet


# %jupyter_snippet params
Mtarget = pin.SE3(
    pin.utils.rotate("x", 3.14 / 4), np.array([-0.5, 0.1, 0.2])
)  # x,y,z
q0 = np.array([0, -3.14 / 2, 0, 0, 0, 0])
# %end_jupyter_snippet

# --- Load robot model
robot = robex.load("ur5")
robot.q0 = q0

# Open the viewer
viz = MeshcatVisualizer(robot)
viz.display(robot.q0)
time.sleep(0.3)
print("Let's go to pdes.")

# The pinocchio model is what we are really interested by.
model = robot.model
data = model.createData()
idTool = model.getFrameId("tool0")
idElbow = model.getFrameId("elbow_joint")

# --- Add box to represent target
# Add a vizualization for the target
boxID = "world/box"
viz.addBox(boxID, [0.05, 0.1, 0.2], [1.0, 0.2, 0.2, 0.5])
# %jupyter_snippet 1
# Add a vizualisation for the tip of the arm.
tipID = "world/blue"
viz.addBox(tipID, [0.08] * 3, [0.2, 0.2, 1.0, 0.5])

#
# OPTIM 6D #########################################################
#


def cost(q):
    """Compute score from a configuration"""
    pin.framesForwardKinematics(model, data, q)
    M = data.oMf[idTool]
    return norm(pin.log(M.inverse() * Mtarget).vector)


def callback(q):
    pin.framesForwardKinematics(model, data, q)
    M = data.oMf[idTool]
    viz.applyConfiguration(boxID, Mtarget)
    viz.applyConfiguration(tipID, M)
    viz.display(q)
    time.sleep(1e-1)


qguess = robot.q0
qguess = np.array([0.12, -2.2, -1.45, 1.82, -0.95, 0.17])
qopt = fmin_bfgs(cost, qguess, callback=callback)

print(
    "The robot finally reached effector placement at\n",
    robot.placement(qopt, 6),
)
# %end_jupyter_snippet

### TEST ZONE ############################################################
### Some asserts below to check the behavior of this script in stand-alone
class InvGeom6DTest(unittest.TestCase):
    def test_qopt_6d(self):
        pin.framesForwardKinematics(model, data, qopt)
        Mopt = data.oMf[idTool]
        self.assertTrue(
            (np.abs(Mtarget.translation - Mopt.translation) < 1e-7).all()
        )
        self.assertTrue(
            np.allclose(pin.log(Mtarget.inverse() * Mopt).vector, 0, atol=1e-6)
        )


InvGeom6DTest().test_qopt_6d()
