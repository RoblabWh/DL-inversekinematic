# %matplotlib inline
# from ipywidgets import interactive
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, Latex

def robot_pose(theta1, l1, theta2):
    l2 = 0.5
    ux = 0.4
    uy = 0.4
    tpx = -3.0
    tpy = 3.0
    tphi =  np.pi / 4.0
    phi = theta1 + theta2
    theta1 = theta1 * np.pi / 180.0
    theta2 = theta2 * np.pi / 180.0
    px1 = l1 * np.cos(theta1) + ux
    py1 = l1 * np.sin(theta1) + uy
    px2 = l1*np.cos(theta1) + l2*np.cos(theta1 + theta2) + ux
    py2 = l1*np.sin(theta1) + l2*np.sin(theta1 + theta2) + uy

    plt.figure(1)
    plt.plot([-5.0, 5.0], [0.0, 0.0], 'k--', linewidth=1)
    plt.plot([0.0, 0.0], [-5.0, 5.0], 'k--', linewidth=1)
    plt.plot([-3.0, 3.0], [3.0, 3.0], 'b--', linewidth=1)
    x = [ux, ux, px1, px2]
    y = [0.0, uy, py1, py2]
    plt.plot(x,y, 'ro-', label='line 1', linewidth=2)
    pose = str("Pose: x = " + "{:2.2f}".format(px2) + ", y = " + "{:2.2f}".format(py2) + ", phi = " + "{:3.0f}".format(phi))
    print(pose)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.xlim(-5, 5)
    plt.ylim(-4, 4)
    if np.linalg.norm([px2-tpx, py2-tpy, (phi*np.pi/180)-tphi]) < 0.1:
        print('Touched')
        touched = True
        plt.plot(tpx,tpy,'go',markersize=10)
    plt.show()

def robot_pose2(q, camera):
    l2 = 0.5
    ux = 0.4
    uy = 0.4
    theta1 = float((q[0] -np.pi/2))
    l1 = float(q[1])
    theta2 = float(q[2] + np.pi/2)
    phi = theta1 + theta2
    px1 = l1 * np.cos(theta1) + ux
    py1 = l1 * np.sin(theta1) + uy
    px2 = l1*np.cos(theta1) + l2*np.cos(theta1 + theta2) + ux
    py2 = l1*np.sin(theta1) + l2*np.sin(theta1 + theta2) + uy
    plt.plot([-5.0, 5.0], [0.0, 0.0], 'k--', linewidth=1)
    plt.plot([0.0, 0.0], [-5.0, 5.0], 'k--', linewidth=1)
    plt.plot([-3.0, 3.0], [3.0, 3.0], 'b--', linewidth=1)
    x = [ux, ux, px1, px2]
    y = [0.0, uy, py1, py2]
    plt.plot(x,y, 'ro-', label='line 1', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-5, 5)
    plt.ylim(-4, 4)
#    plt.show()
    camera.snap()
