import numpy as np
from mpl_toolkits.mplot3d import Axes3D



def plotEllipse(ax,opti_A,opti_c):
    # find the rotation matrix and radii of the axes
    U, s, rotation = np.linalg.svd(opti_A)
    radii = 1.0/np.sqrt(s)

    # now carry on with EOL's answer
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + opti_c

    # plot
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)

def plotVertices(ax,vertices,nsubsample):
    '''
    plot a 3xN array of vertices in 3d. If nsubsample is not 0, plot the main once in 
    large red, the others in small green.
    '''
    vert = vertices
    NS = nsubsample

    # Plot the vertices
    ax.plot3D(vert[::,0],vert[::,1],vert[::,2],'g.',markersize=1)
    ax.plot3D(vert[::NS,0],vert[::NS,1],vert[::NS,2],'r*')

    # Change the scalling for a regular one centered on the vertices.
    m,M = np.min(vert,0),np.max(vert,0)
    plot_center = (m+M)/2
    plot_length = max(M-m)/2
    ax.axes.set_xlim3d(left=plot_center[0]-plot_length, right=plot_center[0]+plot_length) 
    ax.axes.set_ylim3d(bottom=plot_center[1]-plot_length, top=plot_center[1]+plot_length) 
    ax.axes.set_zlim3d(bottom=plot_center[2]-plot_length, top=plot_center[2]+plot_length) 
