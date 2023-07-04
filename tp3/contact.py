'''
Implement and solve the following nonlinear program:
decide 
  x_0 ... x_T \in R^NXxT+1
  a_0 ... a_T-1 \in R^NVxT
  tau_0 ... tau_T-1 \in R^NUxT
minimizing   sum_t || v - v^* ||^2 
so that:
   q_0 = robot.q0
   forall t=0..T, x_t+1 = x_t + integral_0^DT a_t
                  alpha_c(x_t,a_t) = baumgart(x,K)
                  a_t = cdyn(q_t,v_t,tau_t ; contacts )

with:
   x = [q,v] the robot state composed of configuration and velocity
   a the robot acceleration
   q^* an arbitrary reference configuration
   integral a numerical integration step (Euler implicit) for the acceleration
   alpha(x,a) the acceleration of the tip of the robot (3d / 6d cartesian quantity)
   baumgart(x,K) is a corrector for the contact dynamics, with K its stifness parameters.
   and cdyn(q,v,tau ; contacts) is the forward constrained dynamics, given the contacts.

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
from types import SimpleNamespace

from utils.meshcat_viewer_wrapper import MeshcatVisualizer

# Change numerical print
pin.SE3.__repr__=pin.SE3.__str__
np.set_printoptions(precision=2, linewidth=300, suppress=True,threshold=1e6)

### HYPER PARAMETERS
# %jupyter_snippet frames
Mtarget = pin.SE3(pin.utils.rotate('y', 3), np.array([-0.1, 0.2, 0.45094]))  # x,y,z
contacts = [ SimpleNamespace(name='left_sole_link', type=pin.ContactType.CONTACT_6D) ]
endEffectorFrameName = 'right_sole_link'
# %end_jupyter_snippet
# %jupyter_snippet hyper
T = 50
DT = .002
w_vel = .1
w_conf = 5
# %end_jupyter_snippet
# %jupyter_snippet contact_solver
# Baumgart correction
Kv = 20; Kp = 0#Kv**2/4
Kp = 0; Kv = 2*np.sqrt(Kp)
# Tuning of the proximal solver (minimal version)
prox_settings = pin.ProximalSettings(0,1e-6,1)
# %end_jupyter_snippet

# --- Load robot model
# %jupyter_snippet talos
robot = robex.load('talos_legs')
# Open the viewer
viz = MeshcatVisualizer(robot)
viz.display(robot.q0)

# The pinocchio model is what we are really interested by.
model = robot.model
data = model.createData()
# %end_jupyter_snippet

# %jupyter_snippet framesId
endEffector_ID = model.getFrameId(endEffectorFrameName)
for c in contacts:
    c.id = model.getFrameId(c.name)
    assert(c.id<len(model.frames))
    c.jid = model.frames[c.id].parentJoint
    c.placement = model.frames[c.id].placement
    c.model = pin.RigidConstraintModel(c.type,model,c.jid,c.placement)
contact_models = [ c.model for c in contacts ] 
# %end_jupyter_snippet

# %jupyter_snippet contact_solver
contact_datas = [ c.createData() for c in contact_models ]
for c in contact_models:
    c.corrector.Kd=Kv
    c.corrector.Kp=Kp
# %end_jupyter_snippet


# %jupyter_snippet viz
# --- Add box to represent target
# Add a vizualization for the target
boxID = "world/box"
viz.addBox(boxID, [.05, .1, .2], [1., .2, .2, .5])
# Add a vizualisation for the tip of the arm.
tipID = "world/blue"
viz.addBox(tipID, [.08] * 3, [.2, .2, 1., .5])
for c in contacts:
    c.viz =  f"world/contact_{c.name}"
    viz.addSphere(c.viz, [.07], [.8, .8, .2, .5])

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
    for c in contacts:
        viz.applyConfiguration(c.viz, data.oMf[c.id])
    viz.display(q)
    time.sleep(dt)
def displayTraj(qs,dt=1e-2):
    for q in qs[1:]:
        displayScene(q,dt=dt)
displayScene(robot.q0)
# %end_jupyter_snippet

# %jupyter_snippet helpers
# --- Casadi helpers
cmodel = cpin.Model(model)
cdata = cmodel.createData()
ccontact_models = [ cpin.RigidConstraintModel(c) for c in contact_models ]
ccontact_datas = [ c.createData() for c in ccontact_models ]
cprox_settings = cpin.ProximalSettings(prox_settings.absolute_accuracy,
                                       prox_settings.mu,prox_settings.max_iter)
cpin.initConstraintDynamics(cmodel,cdata, ccontact_models)

nq = model.nq
nv = model.nv
nx = nq+nv
ndx = 2*nv
cx = casadi.SX.sym("x",nx,1)
cdx = casadi.SX.sym("dx",nv*2,1)
cq = cx[:nq]
cv = cx[nq:]
caq = casadi.SX.sym("a",nv,1)
ctauq = casadi.SX.sym("tau",nv,1)

# Compute kinematics casadi graphs
cpin.constraintDynamics(cmodel,cdata,cq,cv,ctauq,ccontact_models,ccontact_datas)
cpin.forwardKinematics(cmodel,cdata,cq,cv,caq)
cpin.updateFramePlacements(cmodel,cdata)
# %end_jupyter_snippet

# %jupyter_snippet contact_placement
# Get initial contact position (for Baumgart correction)
pin.framesForwardKinematics(model,data,robot.q0)
# %end_jupyter_snippet

# %jupyter_snippet integrate
# Sym graph for the integration operation x,dx -> x(+)dx = [model.integrate(q,dq),v+dv]
cintegrate = casadi.Function('integrate',[cx,cdx],
                             [ casadi.vertcat(cpin.integrate(cmodel,cx[:nq],cdx[:nv]),
                                              cx[nq:]+cdx[nv:]) ])
# %end_jupyter_snippet

# %jupyter_snippet cnext
# Sym graph for the integration operation x' = [ q+vDT+aDT**2, v+aDT ]
cnext = casadi.Function('next', [cx,caq],
                        [ casadi.vertcat( cpin.integrate(cmodel,cx[:nq],cx[nq:]*DT + caq*DT**2),
                                          cx[nq:] + caq*DT ) ])
# %end_jupyter_snippet

# %jupyter_snippet aba
# Sym graph for the aba operation
caba = casadi.Function('fdyn', [cx,ctauq],[ cdata.ddq ])
# %end_jupyter_snippet

# %jupyter_snippet error
# Sym graph for the operational error
error_tool = casadi.Function('etool3', [cx],
                             [ cdata.oMf[endEffector_ID].translation - Mtarget.translation ])
# %end_jupyter_snippet

### PROBLEM

# %jupyter_snippet ocp1
opti = casadi.Opti()
var_dxs = [ opti.variable(ndx) for t in range(T+1) ]
var_as = [ opti.variable(nv) for t in range(T) ]
var_us = [ opti.variable(nv-6) for t in range(T) ]
var_xs = [ cintegrate( np.concatenate([robot.q0,np.zeros(nv)]),var_dx) for var_dx in var_dxs ]
# %end_jupyter_snippet

# %jupyter_snippet ocp2
totalcost = 0
# Define the running cost
for t in range(T):
    totalcost += 1e-3 * DT * casadi.sumsqr( var_xs[t][nq:] )
    totalcost += 1e-4 * DT * casadi.sumsqr( var_as[t] )
totalcost += 1e1 * casadi.sumsqr( error_tool(var_xs[T]) )
# %end_jupyter_snippet

# %jupyter_snippet ocp3
opti.subject_to(var_xs[0][:nq] == robot.q0)
opti.subject_to(var_xs[0][nq:] == 0) # zero initial velocity
opti.subject_to(var_xs[T][nq:] == 0) # zero terminal velocity
# %end_jupyter_snippet

# Define the integration constraints
# %jupyter_snippet integration
for t in range(T):
    tau = casadi.vertcat(np.zeros(6),var_us[t])
    opti.subject_to( caba(var_xs[t],tau) == var_as[t] )
    opti.subject_to( cnext(var_xs[t],var_as[t]) == var_xs[t+1] )
# %end_jupyter_snippet


for x in var_dxs: opti.set_initial(x,np.zeros(ndx))
for a in var_as: opti.set_initial(a,np.zeros(nv))

# u0 = 
# for u in us: opti.set_initial(u,u0)


# %jupyter_snippet ocp5
### SOLVE
opti.minimize(totalcost)
opti.solver("ipopt") # set numerical backend
opti.callback(lambda i: displayScene(opti.debug.value(var_xs[-1][:nq])))

# Caution: in case the solver does not converge, we are picking the candidate values
# at the last iteration in opti.debug, and they are NO guarantee of what they mean.
try:
    sol = opti.solve_limited()
    sol_xs = [ opti.value(var_x) for var_x in var_xs ]
    sol_as = [ opti.value(var_a) for var_a in var_as ]
    sol_us = [ opti.value(var_u) for var_u in var_us ]
except:
    print('ERROR in convergence, plotting debug info.')
    sol_xs = [ opti.debug.value(var_x) for var_x in var_xs ]
    sol_as = [ opti.debug.value(var_a) for var_a in var_as ]
    sol_us = [ opti.debug.value(var_u) for var_u in var_us ]
# %end_jupyter_snippet

# %jupyter_snippet ocp6
print("***** Display the resulting trajectory ...")
displayScene(robot.q0,1)
displayTraj([ x[:nq] for x in sol_xs],DT)
# %end_jupyter_snippet
