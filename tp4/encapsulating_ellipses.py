"""
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

"""

import casadi
import example_robot_data as robex
import numpy as np
# %jupyter_snippet import
import pinocchio as pin
from pinocchio import casadi as cpin

from utils.meshcat_viewer_wrapper import MeshcatVisualizer

# %end_jupyter_snippet

# %jupyter_snippet load
# --- Load robot model
robot = robex.load("ur10")
viz = MeshcatVisualizer(robot)
viz.display(robot.q0)
# %end_jupyter_snippet

# %jupyter_snippet vertices
geom = robot.collision_model.geometryObjects[5]
vertices = geom.geometry.vertices()

for i in np.arange(0, vertices.shape[0]):
    viz.addSphere(f"world/point_{i}", 5e-3, [1, 0, 0, 0.8])
    viz.applyConfiguration(f"world/point_{i}", vertices[i].tolist() + [1, 0, 0, 0])
# %end_jupyter_snippet

### CASADI
# %jupyter_snippet helper
cw = casadi.SX.sym("w", 3)
exp = casadi.Function("exp3", [cw], [cpin.exp3(cw)])
# %end_jupyter_snippet

###
"""
decide 
 - w in so3: ellipse orientation
 - r in r3: ellipse main dimensions
minimizing:
  r1*r2*r3 the volum of the ellipse
so that:
  r>=0
  for all points pk in a list,     pk in ellipse

"""
# %jupyter_snippet vars
opti = casadi.Opti()
var_w = opti.variable(3)
var_r = opti.variable(3)
var_c = opti.variable(3)
# %end_jupyter_snippet

# %jupyter_snippet RA
# The ellipsoid matrix is represented by w=log3(R),diag(P) with R,P=eig(A)
R = exp(var_w)
A = R @ casadi.diag(1 / var_r**2) @ R.T
# %end_jupyter_snippet

# %jupyter_snippet cost
totalcost = var_r[0] * var_r[1] * var_r[2]
# %end_jupyter_snippet

# %jupyter_snippet rplus
opti.subject_to(var_r >= 0)
# %end_jupyter_snippet

# %jupyter_snippet points
for g_v in vertices:
    # g_v is the vertex v expressed in the geometry frame.
    # Convert point from geometry frame to joint frame
    j_v = geom.placement.act(g_v)
    # Constraint the ellipsoid to be including the point
    opti.subject_to((j_v - var_c).T @ A @ (j_v - var_c) <= 1)
# %end_jupyter_snippet

### SOLVE
# %jupyter_snippet solve
opti.minimize(totalcost)
opti.solver("ipopt")  # set numerical backend
opti.set_initial(var_r, 10)

sol = opti.solve_limited()

sol_r = opti.value(var_r)
sol_A = opti.value(A)
sol_c = opti.value(var_c)
sol_R = opti.value(exp(var_w))
# %end_jupyter_snippet

# Recover r,R from A (for fun)
e, P = np.linalg.eig(sol_A)
recons_r = 1 / e**0.5
recons_R = P

# %jupyter_snippet meshcat
# Build the ellipsoid 3d shape
# Ellipsoid in meshcat
viz.addEllipsoid("el", sol_r, [0.3, 0.9, 0.3, 0.3])
# jMel is the placement of the ellipsoid in the joint frame
jMel = pin.SE3(sol_R, sol_c)
# %end_jupyter_snippet

# %jupyter_snippet vizplace
# Place the body, the vertices and the ellispod at a random configuration oMj_rand
oMj_rand = pin.SE3.Random()
viz.applyConfiguration(viz.getViewerNodeName(geom, pin.VISUAL), oMj_rand)
for i in np.arange(0, vertices.shape[0]):
    viz.applyConfiguration(
        f"world/point_{i}", oMj_rand.act(vertices[i]).tolist() + [1, 0, 0, 0]
    )
viz.applyConfiguration("el", oMj_rand * jMel)
# %end_jupyter_snippet

# For future use ...
print(
    f'SimpleNamespace(name="{robot.model.names[geom.parentJoint]}",\n'
    + f"                A=np.{repr(sol_A)},\n"
    + f"                center=np.{repr(sol_c)})"
)

# Matplotlib (for fun)
import matplotlib.pyplot as plt

plt.ion()
from utils.plot_ellipse import plotEllipse, plotVertices

fig, ax = plt.subplots(1, subplot_kw={"projection": "3d"})
plotEllipse(ax, sol_A, sol_c)
plotVertices(ax, np.vstack([geom.placement.act(p) for p in vertices]), 1)
