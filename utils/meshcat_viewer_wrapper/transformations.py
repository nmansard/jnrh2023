'''
Collection of super simple transformations to ease the use of the viewer.
'''

import numpy as np

def planar(x, y, theta):
    '''Convert a 3d vector (x,y,theta) into a transformation in the Y,Z plane.'''
    s,c=np.sin(theta/2),np.cos(theta / 2)
    return [0, x, y, s,0,0,c]  # Rotation around X

def translation2d(x,y):
    ''' Convert a 2d vector (x,y) into a 3d transformation translating the Y,Z plane. '''
    return [0,x,y,1,0,0,0]

