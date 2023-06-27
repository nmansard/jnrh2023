'''
Implement and solve the following nonlinear program:
decide 
  x_0 ... x_T \in R^NXxT+1
  a_0 ... a_T-1 \in R^NUxT
minimizing   sum_t || q - q^* ||^2 
so that:
   q_0 = robot.q0
   forall t=0..T, x_t+1 = x_t + integral_0^DT a_t
                  acc(x_t,a_t) = PD(x,K)

with:
   x = [q,v] the robot state composed of configuration and velocity
   a the robot acceleration
   q^* an arbitrary reference configuration
   integral a numerical integration step (Euler implicit) for the acceleration
   acc(x,a) the acceleration of the tip of the robot (3d cartesian quantity)
   and PD(x,K) is a corrector for the contact dynamics, with K its stifness parameters.

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
Mtarget = pin.SE3(pin.utils.rotate('y', 3), np.array([-0.1, 0.2, 0.45094]))  # x,y,z
T = 50
DT = .002
w_vel = .1
w_conf = 5

# --- Load robot model
robot = robex.load('talos_legs')
# Open the viewer
viz = MeshcatVisualizer(robot)
viz.display(robot.q0)
time.sleep(.3)
print("Let's go to pdes ... with casadi")

# The pinocchio model is what we are really interested by.
model = robot.model
data = model.createData()
idTool = model.getFrameId('left_sole_link')
idElb = model.getFrameId('right_sole_link')

# --- Add box to represent target
# Add a vizualization for the target
boxID = "world/box"
viz.addBox(boxID, [.05, .1, .2], [1., .2, .2, .5])
# %jupyter_snippet 1
# Add a vizualisation for the tip of the arm.
tipID = "world/blue"
viz.addBox(tipID, [.08] * 3, [.2, .2, 1., .5])
elbowID = "world/yellow"
viz.addSphere(elbowID, [.07], [.8, .8, .2, .5])
def displayScene(q,dt=1e-1):
    '''
    Given the robot configuration, display:
    - the robot
    - a box representing idTool
    - a box representing Mtarget
    '''
    pin.framesForwardKinematics(model,data,q)
    M = data.oMf[idTool]
    viz.applyConfiguration(boxID, Mtarget)
    viz.applyConfiguration(tipID, M)
    viz.applyConfiguration(elbowID, data.oMf[idElb])
    viz.display(q)
    time.sleep(dt)
def displayTraj(qs,dt=1e-2):
    for q in qs[1:]:
        displayScene(q,dt=dt)
displayScene(robot.q0)

# --- Casadi helpers
cmodel = cpin.Model(model)
cdata = cmodel.createData()

nq = model.nq
nv = model.nv
nx = nq+nv
ndx = 2*nv
cx = casadi.SX.sym("x",nx,1)
cdx = casadi.SX.sym("dx",nv*2,1)
cq = cx[:nq]
cv = cx[nq:]
caq = casadi.SX.sym("a",nv,1)

cpin.forwardKinematics(cmodel,cdata,cq,cv,caq)
cpin.updateFramePlacements(cmodel,cdata)
# Get initial contact position
pin.framesForwardKinematics(model,data,robot.q0)
p0 = data.oMf[idTool].translation.copy()

cintegrate = casadi.Function('integrate',[cx,cdx],
                             [ casadi.vertcat(cpin.integrate(cmodel,cx[:nq],cdx[:nv]),
                                              cx[nq:]+cdx[nv:]) ])
cnext = casadi.Function('next', [cx,caq],
                        [ casadi.vertcat( cpin.integrate(cmodel,cx[:nq],cx[nq:]*DT + caq*DT**2),
                                          cx[nq:] + caq*DT ) ])

error_tool = casadi.Function('etool3', [cx],
                             [ cdata.oMf[idElb].translation - Mtarget.translation ])

dpcontact = casadi.Function('dpacontact', [cx],
                           [ -(cdata.oMf[idTool].inverse().act(casadi.SX(p0))) ])
vcontact = casadi.Function('vcontact', [cx],
                           [cpin.getFrameVelocity( cmodel,cdata,idTool,
                                                   pin.LOCAL ).linear])
acontact = casadi.Function('acontact', [cx,caq],
                           [cpin.getFrameClassicalAcceleration( cmodel,cdata,idTool,
                                                                pin.LOCAL ).linear])


### PROBLEM
opti = casadi.Opti()
var_dxs = [ opti.variable(ndx) for t in range(T+1) ]
var_as = [ opti.variable(nv) for t in range(T) ]
var_xs = [ cintegrate( np.concatenate([robot.q0,np.zeros(nv)]),var_dx) for var_dx in var_dxs ]

totalcost = 0
opti.subject_to(var_xs[0][:nq] == robot.q0)
opti.subject_to(var_xs[0][nq:] == 0)

for t in range(T):
    #totalcost += 1e2 * DT * casadi.sumsqr( var_xs[t][:nq] - qtarget )
    totalcost += 1e-3 * DT * casadi.sumsqr( var_xs[t][nq:] )
    totalcost += 1e-4 * DT * casadi.sumsqr( var_as[t] )
    
    opti.subject_to( cnext(var_xs[t],var_as[t]) == var_xs[t+1] )
    Kp = 100
    Kv = 2*np.sqrt(Kp)
    opti.subject_to( acontact(var_xs[t],var_as[t]) + Kv* vcontact(var_xs[t]) + Kp * dpcontact(var_xs[t]) == 0)

totalcost += 1e4 * casadi.sumsqr( error_tool(var_xs[T]) )

#opti.subject_to(var_xs[T][nq:] == 0)
#opti.subject_to( error_tool(var_xs[T]) == 0)

### SOLVE
opti.minimize(totalcost)
opti.solver("ipopt") # set numerical backend
opti.callback(lambda i: displayScene(opti.debug.value(var_xs[-1][:nq])))

tau0 = pin.rnea(model,data,robot.q0,np.zeros(model.nv),np.zeros(model.nv))

# Caution: in case the solver does not converge, we are picking the candidate values
# at the last iteration in opti.debug, and they are NO guarantee of what they mean.
try:
    sol = opti.solve_limited()
    sol_xs = [ opti.value(var_x) for var_x in var_xs ]
    sol_as = [ opti.value(var_a) for var_a in var_as ]
except:
    print('ERROR in convergence, plotting debug info.')
    sol_xs = [ opti.debug.value(var_x) for var_x in var_xs ]
    sol_as = [ opti.debug.value(var_a) for var_a in var_as ]

print("***** Display the resulting trajectory ...")
displayScene(robot.q0,1)
displayTraj([ x[:nq] for x in sol_xs],DT)


pcontacts = []
vcontacts = []
acontacts = []
for t in range(T):
    x=sol_xs[t]; q=x[:nq]; v=x[nq:]; a=sol_as[t]
    pin.forwardKinematics(model,data,q,v,a)
    pin.updateFramePlacements(model,data)
    pcontacts.append( data.oMf[idTool].translation.copy() )
    vcontacts.append( pin.getFrameVelocity(model,data,idTool,pin.LOCAL).linear )
    acontacts.append( pin.getFrameClassicalAcceleration(model,data,idTool,pin.LOCAL).linear )
    
import matplotlib.pylab as plt; plt.ion()
fig,ax=plt.subplots(2,1,figsize=(6,6))
ax[0].plot([p-pcontacts[0] for p in pcontacts])
ax[0].set_title('delta position')
ax[0].axis((-2.45, 51.45, -.5e-3, .5e-3))
ax[1].plot(vcontacts)
ax[1].set_title('velocity')
ax[1].axis((-2.45, 51.45, -0.006627568040194312, 0.007463128239663308))
