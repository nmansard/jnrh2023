r"""
Stand-alone inverse geometry for a manipulator robot with a 6d objective.

Implement and solve the following nonlinear program:
decide q \in R^NQ
minimizing   || log( M(q)^-1 M^* ||^2
with M(q) \in SE(3) the placement of the robot end-effector, and M^* the
target.

The following tools are used:
- the ur10 model (loaded by example-robot-data)
- the fmin_bfgs solver of scipy (with finite differences automatically
  implemented)
- the meshcat viewer
"""

# %lupyter_snippet imports
import time
import unittest

import example_robot_data as robex
import meshcat
import numpy as np
import pinocchio as pin
from numpy.linalg import norm
from scipy.optimize import fmin_bfgs

# %end_lupyter_snippet

# %lupyter_snippet robot
robot = robex.load("ur5")
robot.q0 = np.array([0, -np.pi / 2, 0, 0, 0, 0])
model = robot.model
data = robot.data
# %end_lupyter_snippet

# %lupyter_snippet visualizer
viz = pin.visualize.MeshcatVisualizer(
    robot.model, robot.collision_model, robot.visual_model
)
robot.setVisualizer(viz, init=False)
robot.setVisualizer(viz, init=False)
viz.initViewer(open=False)
viz.loadViewerModel()
viz.display(robot.q0)
# %end_lupyter_snippet

# %lupyter_snippet params
tool_id = model.getFrameId("tool0")
transform_target_to_world = pin.SE3(
    pin.utils.rotate("x", np.pi / 4),
    np.array([-0.5, 0.1, 0.2]),
)
# %end_lupyter_snippet

# The pinocchio model is what we are really interested by.

#
# OPTIM 6D #########################################################
#


def cost(q: np.ndarray) -> float:
    """Compute score from a configuration"""
    pin.framesForwardKinematics(model, data, q)
    transform_tool_to_world = data.oMf[tool_id]
    return norm(
        pin.log(transform_tool_to_world.inverse() * transform_target_to_world).vector
    )


def meshcat_frame(
    handle: meshcat.Visualizer,
    axis_length: float = 0.1,
    axis_thickness: float = 0.005,
    opacity: float = 1.0,
    origin_color: int = 0x000000,
    origin_radius: float = 0.01,
) -> None:
    """
    Set MeshCat handle to a frame, represented by an origin and three axes.

    Args:
        handle: MeshCat handle to attach the frame to.
        axis_length: Length of axis unit vectors, in [m].
        axis_thickness: Thickness of axis unit vectors, in [m].
        opacity: Opacity of all three unit vectors.
        origin_color: Color of the origin sphere.
        origin_radius: Radius of the frame origin sphere, in [m].

    Note:
        As per the de-facto standard (Blender, OpenRAVE, RViz, ...), the
        x-axis is red, the y-axis is green and the z-axis is blue.
    """
    material = meshcat.geometry.MeshLambertMaterial(color=origin_color, opacity=opacity)
    sphere = meshcat.geometry.Sphere(origin_radius)
    handle.set_object(sphere, material)
    direction_names = ["x", "y", "z"]
    colors = [0xFF0000, 0x00FF00, 0x0000FF]
    rotation_axes = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    position_cylinder_in_body = 0.5 * axis_length * np.eye(3)
    for i in range(3):
        dir_name = direction_names[i]
        material = meshcat.geometry.MeshLambertMaterial(
            color=colors[i], opacity=opacity
        )
        transform_cylinder_to_body = meshcat.transformations.rotation_matrix(
            np.pi / 2, rotation_axes[i]
        )
        transform_cylinder_to_body[0:3, 3] = position_cylinder_in_body[i]
        cylinder = meshcat.geometry.Cylinder(axis_length, axis_thickness)
        handle[dir_name].set_object(cylinder, material)
        handle[dir_name].set_transform(transform_cylinder_to_body)


# --- Callback for visualization
viewer = viz.viewer
meshcat_frame(viewer["target"], opacity=1.0)
meshcat_frame(viewer["current"], opacity=0.5)


def callback(q: np.ndarray):
    pin.framesForwardKinematics(model, data, q)
    transform_frame_to_world = data.oMf[tool_id]
    viewer["target"].set_transform(transform_target_to_world.np)
    viewer["current"].set_transform(transform_frame_to_world.np)
    viz.display(q)
    time.sleep(1e-1)


qguess = robot.q0
qguess = np.array([0.12, -2.2, -1.45, 1.82, -0.95, 0.17])
qopt = fmin_bfgs(cost, qguess, callback=callback)

print(
    "The robot finally reached effector placement at\n",
    robot.placement(qopt, 6),
)
# %end_lupyter_snippet

# TEST ZONE ############################################################
# Some asserts below to check the behavior of this script in stand-alone


class InvGeom6DTest(unittest.TestCase):
    def test_qopt_6d(self):
        pin.framesForwardKinematics(model, data, qopt)
        Mopt = data.oMf[tool_id]
        self.assertTrue(
            (
                np.abs(transform_target_to_world.translation - Mopt.translation) < 1e-7
            ).all()
        )
        self.assertTrue(
            np.allclose(
                pin.log(transform_target_to_world.inverse() * Mopt).vector,
                0,
                atol=1e-6,
            )
        )


InvGeom6DTest().test_qopt_6d()
