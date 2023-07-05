import random

import meshcat
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer as PMV

from . import colors


def materialFromColor(color):
    if isinstance(color, meshcat.geometry.MeshPhongMaterial):
        return color
    elif isinstance(color, str):
        material = colors.colormap[color]
    elif isinstance(color, list):
        material = meshcat.geometry.MeshPhongMaterial()
        material.color = colors.rgb2int(*[int(c * 255) for c in color[:3]])
        if len(color) == 3:
            material.transparent = False
        else:
            material.transparent = color[3] < 1
            material.opacity = float(color[3])
    elif color is None:
        material = random.sample(list(colors.colormap), 1)[0]
    else:
        material = colors.black
    return material


class MeshcatVisualizer(PMV):
    def __init__(
        self,
        robot=None,
        model=None,
        collision_model=None,
        visual_model=None,
        url=None,
        autoclean=False,
    ):
        if robot is not None:
            super().__init__(robot.model, robot.collision_model, robot.visual_model)
        elif model is not None:
            super().__init__(model, collision_model, visual_model)

        if url is not None:
            if url == "classical":
                url = "tcp://127.0.0.1:6000"
            print("Wrapper tries to connect to server <%s>" % url)
            server = meshcat.Visualizer(zmq_url=url)
        else:
            server = None

        if robot is not None or model is not None:
            self.initViewer(loadModel=True, viewer=server)
        else:
            self.viewer = server if server is not None else meshcat.Visualizer()

        if autoclean:
            self.clean()

    def addSphere(self, name, radius, color):
        material = materialFromColor(color)
        self.viewer[name].set_object(meshcat.geometry.Sphere(radius), material)

    def addCylinder(self, name, length, radius, color=None):
        material = materialFromColor(color)
        self.viewer[name].set_object(
            meshcat.geometry.Cylinder(length, radius), material
        )

    def addBox(self, name, dims, color):
        material = materialFromColor(color)
        self.viewer[name].set_object(meshcat.geometry.Box(dims), material)

    def addEllipsoid(self, name, dims, color):
        material = materialFromColor(color)
        self.viewer[name].set_object(meshcat.geometry.Ellipsoid(dims), material)

    def applyConfiguration(self, name, placement):
        if isinstance(placement, list) or isinstance(placement, tuple):
            placement = np.array(placement)
        if isinstance(placement, pin.SE3):
            R, p = placement.rotation, placement.translation
            T = np.r_[np.c_[R, p], [[0, 0, 0, 1]]]
        elif isinstance(placement, np.ndarray):
            if placement.shape == (7,):  # XYZ-quat
                R = pin.Quaternion(np.reshape(placement[3:], [4, 1])).matrix()
                p = placement[:3]
                T = np.r_[np.c_[R, p], [[0, 0, 0, 1]]]
            else:
                print("Error, np.shape of placement is not accepted")
                return False
        else:
            print("Error format of placement is not accepted")
            return False
        self.viewer[name].set_transform(T)

    def delete(self, name):
        self.viewer[name].delete()

    def __getitem__(self, name):
        return self.viewer[name]
