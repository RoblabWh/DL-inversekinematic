import numpy as np
import matplotlib.pyplot as plt
import time
import math
import base64
import matplotlib.animation as animation
import copy

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from pytransform3d.rotations import *
from mpl_toolkits.mplot3d import proj3d
import os
import sys

class Robot():
    def __init__(self, dh_values=None, joint_limits=None, robot=None):
        #[dh_theta_values, dh_d_values, dh_a_values, dh_alpha_values]
        self.name = "Robot"
        if robot:
            if robot=="youbot":
                balon = b"AAAAAAAAAAAYLURU+yH5vwAAAAAAAAAAGC1EVPsh+T8YLURU+yH5PzMzMzMzM7M/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgZVDi2zny \
                        z9MN4lBYOWgP9ejcD0K18M/SOF6FK5HwT8AAAAAAAAAAAAAAAAAAAAAGC1EVPsh+b8AAAAAAAAAAAAAAAAAAAAAGC1EVPsh+T8AAAAAAAAAAA=="
                berong = base64.decodebytes(balon)
                mokuba = np.frombuffer(berong, dtype=np.float64).reshape((4,5))

                a1 = [169, -169]
                a2 = [90, -65]
                a3 = [146, -151]
                a4 = [102.5, -102.5]
                a5 = [167.5, -167.5]

                joint_limits = [a1, a2, a3, a4, a5]
            elif robot == "baxter":
                balon = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABI4XoUrkfRPwAAAAAAAAAAsp3vp8ZL1z8A \
                AAAAAAAAAFYOLbKd79c/AAAAAAAAAADsUbgehevRP0SLbOf7qbE/AAAAAAAAAABEi2zn+6mxPwAAAAAAAAAAexSuR+F6hD8AAAAAAAAAAAAAAAAAAAAAGC \
                1EVPsh+b8YLURU+yH5PxgtRFT7Ifm/GC1EVPsh+T8YLURU+yH5vxgtRFT7Ifk/AAAAAAAAAAA="
                berong = base64.decodebytes(balon)
                mokuba = np.frombuffer(berong, dtype=np.float64).reshape((4,7))

                joint_limits = [[180.0, -180.0] for i in range(7)]
                self.name = "Baxter"
            else:
                print("\'youbot\' and \'baxter\' are supported.")
                return
            self.dh_values = mokuba
            self.joint_limits = joint_limits
        else:
            if dh_values and joint_limits:
                self.joint_limits = joint_limits
                self.dh_values = dh_values
            else:
                print("Either \'dh_values\' or \'joint_limits\' or both are not defined.")
                return

        self.coordinates = [np.empty(len(self.joint_limits) + 1, dtype=float), np.empty(len(self.joint_limits) + 1, dtype=float), np.empty(len(self.joint_limits) + 1, dtype=float)]
        self.rotation = np.empty((len(self.joint_limits) + 1, 3, 3), dtype=float)


    def dhIthFrame(self, theta, d, a, alpha):
        rot_theta = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                            [np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        trans_d = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]])
        trans_a = np.array([[1, 0, 0, a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        rot_alpha = np.array([[1, 0, 0, 0],
                            [0, np.cos(alpha), -np.sin(alpha), 0],
                            [0, np.sin(alpha), np.cos(alpha), 0], [0, 0, 0, 1]])

        dh_ith_frame = rot_theta.dot(trans_d).dot(trans_a).dot(rot_alpha)

        return dh_ith_frame

    def getdhIthFrame(self, start, idx):
        return self.dhIthFrame(start + self.dh_values[0][idx], self.dh_values[1][idx], self.dh_values[2][idx], self.dh_values[3][idx])


    def buildDhTcpFrame(self, q_array):
        dh_frame = np.identity(4)

        for i in range(len(self.joint_limits)):
            tmp_dh_ith = self.dhIthFrame(q_array[i] + self.dh_values[0][i],
                                    self.dh_values[1][i],
                                    self.dh_values[2][i],
                                    self.dh_values[3][i])
            dh_frame = dh_frame.dot(tmp_dh_ith)

        return dh_frame

    def calc_x_y_z(self, q_array):  # calc x_y_z coordinates
        _coordinates = [np.empty(len(self.joint_limits) + 1, dtype=float), np.empty(len(self.joint_limits) + 1, dtype=float), np.empty(len(self.joint_limits) + 1, dtype=float)]
        _rotation = np.empty((len(self.joint_limits) + 1, 3, 3), dtype=float)
        dh = np.identity(4)

        xyz = dh[:-1, -1]
        _coordinates[0][0] = xyz[0]
        _coordinates[1][0] = xyz[1]
        _coordinates[2][0] = xyz[2]
        _rotation[0] = dh[:3,:-1]

        for i in range(len(self.joint_limits)):
            tmp_dh = self.dhIthFrame(q_array[i] + self.dh_values[0][i],
                                    self.dh_values[1][i],
                                    self.dh_values[2][i],
                                    self.dh_values[3][i])

            dh = np.matmul(dh, tmp_dh)

            xyz = dh[:-1, -1]
            _coordinates[0][i + 1] = xyz[0]
            _coordinates[1][i + 1] = xyz[1]
            _coordinates[2][i + 1] = xyz[2]
            _rotation[i + 1] = dh[:3,:-1]
        
        self.coordinates = copy.deepcopy(_coordinates)
        self.rotation = copy.deepcopy(_rotation)

        return _coordinates, _rotation

class DhAnimation(object):

    def __init__(self, robotanitmation):
        self.robotanimation = robotanitmation
        self.fig = self.robotanimation.fig
        self.ax = self.robotanimation.ax
        self.robot = self.robotanimation.robot
        self.coordinates = copy.deepcopy(self.robot.coordinates)
        self.rotation = np.array([np.identity(3) for i in range(len(self.robot.joint_limits))])
        self.solved = [False for i in range(len(self.robot.joint_limits))]
        self.solved_dh_params = np.zeros((len(self.robot.joint_limits) + 1, 4))
        self.mutex_transform = False

        # Stores angle and basis plots
        self.current_angle = None
        self.current_basis = None

        self.angle_slider = widgets.IntSlider(value=1, min=1, max=len(self.robot.joint_limits), description="Gelenk")
        self.angle_slider.observe(self.angle_slider_changes, names='value')
        display(self.angle_slider)
        self.rot_z = widgets.BoundedFloatText(value=0, min=-180, max=180, step=1, description="Rot \u03B8 [°]")
        self.rot_z.observe(self.rotate_theta, names='value')
        display(self.rot_z)
        max_trans = np.array(self.robot.dh_values[2:]).max()
        self.trans_z = widgets.BoundedFloatText(value=0, min=0, max=max_trans, step=0.01, description="Trans d [m]")
        self.trans_z.observe(self.translate_z, names='value')
        display(self.trans_z)
        self.trans_x = widgets.BoundedFloatText(value=0, min=0, max=max_trans, step=0.01, description="Trans a [m]")
        self.trans_x.observe(self.translate_x, names='value')
        display(self.trans_x)
        self.rot_x = widgets.BoundedFloatText(value=0, min=-180, max=180, step=1, description="Rot \u03B1 [°]")
        self.rot_x.observe(self.rotate_alpha, names='value')
        display(self.rot_x)

        self.angle_slider_changes({"new" : 1, "old" : 1})


    def check_gt(self):
        i = self.angle_slider.value - 1
        a = self.robot.rotation[i + 1]
        b = self.rotation[i]
        xyz_gt = np.array([self.robot.coordinates[0][i + 1], self.robot.coordinates[1][i + 1], self.robot.coordinates[2][i + 1]])
        xyz = np.array([self.coordinates[0][i], self.coordinates[1][i], self.coordinates[2][i]])
        if np.allclose(a, b, rtol=1e-10, atol=1e-03) and np.allclose(xyz, xyz_gt, rtol=1e-10, atol=1e-03):
            if all(x == True for x in self.solved[:i]):
                self.solved[i] = True
                self.robotanimation.text.set_text("Dieser Winkel ist korrekt.\nWeiter zum Nächsten!")
                self.solved_dh_params[i] = [self.rot_z.value, self.trans_z.value, self.trans_x.value, self.rot_x.value]
        else:
            self.solved[i] = False
            self.solved_dh_params[i] = [0,0,0,0]
            self.robotanimation.text.set_text("")


    def translate_z(self, change):
        if not self.mutex_transform:
            self.clear_plot()
            translation_z_d = self.trans_z.value - change['old']
            angle = self.angle_slider.value - 1
            trans_d = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, translation_z_d], [0, 0, 0, 1]])
            robot_rotation = self.rotation[angle].tolist()
            for i in range(3):
                robot_rotation[i].append(self.coordinates[i][angle])
            robot_rotation.append([0,0,0,1])
            robot_rotation = np.array(robot_rotation)
            dh = robot_rotation.dot(trans_d)
            xyz = dh[:-1, -1]
            self.coordinates[0][angle] = xyz[0]
            self.coordinates[1][angle] = xyz[1]
            self.coordinates[2][angle] = xyz[2]
            self.plot_coords(angle)

    def translate_x(self, change):
        if not self.mutex_transform:
            self.clear_plot()
            translation_x_a = self.trans_x.value - change['old']
            angle = self.angle_slider.value - 1
            trans_a = np.array([[1, 0, 0, translation_x_a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            robot_rotation = self.rotation[angle].tolist()
            for i in range(3):
                robot_rotation[i].append(self.coordinates[i][angle])
            robot_rotation.append([0,0,0,1])
            robot_rotation = np.array(robot_rotation)
            dh = robot_rotation.dot(trans_a)
            xyz = dh[:-1, -1]
            self.coordinates[0][angle] = xyz[0]
            self.coordinates[1][angle] = xyz[1]
            self.coordinates[2][angle] = xyz[2]
            self.plot_coords(angle)

    def rotate_theta(self, change):
        if not self.mutex_transform:
            self.clear_plot()
            rotation_angle_theta = self.rot_z.value - change['old']
            rotation_angle_theta = np.deg2rad(rotation_angle_theta)
            rotation_theta = np.array([[np.cos(rotation_angle_theta), -np.sin(rotation_angle_theta), 0],
                                    [np.sin(rotation_angle_theta), np.cos(rotation_angle_theta), 0], 
                                    [0, 0, 1]])
            angle = self.angle_slider.value - 1
            robot_rotation = np.asarray(self.rotation[angle])
            self.rotation[angle] = robot_rotation.dot(rotation_theta)
            self.plot_coords(angle)

    def rotate_alpha(self, change):
        if not self.mutex_transform:
            self.clear_plot()
            rotation_angle_alpha = self.rot_x.value - change['old']
            rotation_angle_alpha = np.deg2rad(rotation_angle_alpha)
            rotation_alpha = np.array([[1, 0, 0],
                                    [0, np.cos(rotation_angle_alpha), -np.sin(rotation_angle_alpha)],
                                    [0, np.sin(rotation_angle_alpha), np.cos(rotation_angle_alpha)]])
            angle = self.angle_slider.value - 1
            robot_rotation = np.asarray(self.rotation[angle])
            self.rotation[angle] = robot_rotation.dot(rotation_alpha)
            self.plot_coords(angle)

    def angle_slider_changes(self, change):
        angle = change['new'] - 1
        old_angle = change['old'] - 1
        self.mutex_transform = True
        params = self.solved_dh_params[angle]
        self.rot_z.value = params[0]
        self.trans_z.value = params[1]
        self.trans_x.value = params[2]
        self.rot_x.value = params[3]
        self.mutex_transform = False
        if not self.solved[old_angle]:
            self.rotation[old_angle] = np.identity(3)
            self.coordinates[0][old_angle] = copy.deepcopy(self.robot.coordinates[0][old_angle])
            self.coordinates[1][old_angle] = copy.deepcopy(self.robot.coordinates[1][old_angle])
            self.coordinates[2][old_angle] = copy.deepcopy(self.robot.coordinates[2][old_angle])
        if angle != 0 and self.solved[angle - 1] and not self.solved[angle]:
            self.rotation[angle] = copy.deepcopy(self.robot.rotation[angle])
            self.coordinates[0][angle] = copy.deepcopy(self.robot.coordinates[0][angle])
            self.coordinates[1][angle] = copy.deepcopy(self.robot.coordinates[1][angle])
            self.coordinates[2][angle] = copy.deepcopy(self.robot.coordinates[2][angle])
        self.clear_plot()
        self.plot_coords(angle)

    def plot_coords(self, angle):
        x = (self.coordinates[0][angle] * self.robotanimation.robotscale).tolist()
        y = (self.coordinates[1][angle] * self.robotanimation.robotscale).tolist()
        z = (self.coordinates[2][angle] * self.robotanimation.robotscale).tolist()
        self.current_basis = plot_basis(self.ax, R=self.rotation[angle], p=[x, y, z], s=self.robotanimation.coordscale)
        self.current_angle = self.ax.plot(x, y, z, 'o-', markersize=3,
                                markerfacecolor="orange", linewidth=3, color="orange")
        self.check_gt()
        #msg="hello world"
        #sys.stdout.write(msg + '\n')
        #os.write(1, msg.encode())

    def clear_plot(self):
        if self.current_angle:
            self.current_angle[0].remove()
        if self.current_basis:
            for i, line in enumerate(self.current_basis.get_lines()):
                if i != 0:
                    line.remove()
    
class RobotAnimation(object):

    def __init__(self, robot, basis=True):
        self.robot = robot
        self.scale = np.max([self.robot.dh_values[1], self.robot.dh_values[2]])
        #self.axscale = self.scale * 918 * 2 
        self.robotscale = self.scale * 4587
        self.axscale = np.array([self.robot.dh_values[1], self.robot.dh_values[2]]).sum() * self.robotscale
        self.coordscale = self.axscale / 9

        self.goal = None
        self.plot_basis = basis

        self.wframe = None
        self.xw, self.yw, self.zw = [], [], []
        self.robot_states = []
        self.robot_frame = None

        self.fig = plt.figure(f"{self.robot.name} Simulator")
        self.ax = plt.axes([-0.05, -0.05, 1.20, 1.20], projection='3d', autoscale_on=False)
        self.text= self.ax.text(0,0,self.axscale + self.axscale/10, s="", va="bottom", ha="left")

        #self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        self.DhAnimation = None
        self.angle_slider = None
        self.current_angle = None
        self.current_basis = None
        self.slider = None

    def format_coord(self, xd, yd):
        # nearest edge
        p0, p1 = min(self.ax.tunit_edges(),
                     key=lambda edge: proj3d._line2d_seg_dist(
                         edge[0], edge[1], (xd, yd)))
        self.debug = p0
        # scale the z value to match
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        d0 = np.hypot(x0-xd, y0-yd)
        d1 = np.hypot(x1-xd, y1-yd)
        dt = d0+d1
        z = d1/dt * z0 + d0/dt * z1

        x, y, z = proj3d.inv_transform(xd, yd, z, self.ax.M)

        xs = self.ax.format_xdata(x)
        ys = self.ax.format_ydata(y)
        zs = self.ax.format_zdata(z)
        return xs, ys, zs

    def manual_control(self, goal=None):
        if goal is not None:
            dhFrame = self.robot.buildDhTcpFrame(goal)
            self.goal = dhFrame[:-1, -1] * self.robotscale
        self.slider = []
        for i in range(len(self.robot.joint_limits)):
            min = self.robot.joint_limits[i][1]
            max = self.robot.joint_limits[i][0]
            self.slider.append(widgets.IntSlider(value= min+max, min=min, max=max, description=f'q{i + 1} [°]'))
            self.slider[i].observe(self.on_value_change, names='value')
            display(self.slider[i])
        self.on_value_change(None)

    def onclick(self, event):
        #tx = 'button=%d, x=%d, y=%d, z=%d, xdata=%f, ydata=%f, zdata=%f' % (event.button, event.x, event.y, event.z, event.xdata, event.ydata, event.zdata)
        xs, ys, zs = self.format_coord(event.xdata, event.ydata)
        tx = 'x=%s, y=%s, z=%s' % (xs, ys, zs)
        #tx = f"{event.x}, {event.y}"
        self.text.set_text(tx)

    def on_value_change(self, change):
        #new_a = change['new']
        q_array = [slider.value for slider in self.slider]
        q_array = np.around(np.radians(q_array), decimals=4)
        self.ax.cla()
        if self.goal is not None:
            self.ax.plot(self.goal[0], self.goal[2], self.goal[2], '-o', markersize=5, markerfacecolor="red")
        self.draw_robot(q_array)

    def dh_animation(self):
        position = np.array([0 for i in range(len(self.robot.joint_limits))])
        self.plot_basis = False
        self.draw_robot(position)
        self.DhAnimation = DhAnimation(self)

    def draw_robot(self, q_array):
        q_arr = np.asarray(q_array)

        if q_arr.ndim == 1:
            q_arr = np.expand_dims(q_arr, 0)

        colors = ["orange", "green", "blue"]
        for i in range(len(q_arr)):
            coordinates, rotation = self.robot.calc_x_y_z(q_arr[i])
            self.plot_positions(coordinates, rotation, colors[i])
            if i == 2:
                break
        self.set_ax()

        plt.draw()

    def plot_positions(self, coordinates, rotation, _color="orange"):  # gets the x,y,z values
        x = (coordinates[0] * self.robotscale).tolist()
        y = (coordinates[1] * self.robotscale).tolist()
        z = (coordinates[2] * self.robotscale).tolist()

        if self.plot_basis:
            for i in range(len(self.robot.joint_limits) + 1):
                #if(i != 0):
                #    if(x[i - 1] == x[i] and y[i -1] == y[i] and z[i-1] == z[i]):
                #        continue
                plot_basis(self.ax, R=rotation[i], p=[x[i], y[i], z[i]], s=self.coordscale)

        self.ax.plot(x, y, z, 'o-', markersize=8,
                     markerfacecolor="black", linewidth=3, color=_color, label='ground truth')

    def set_ax(self):  # ax panel set up
        self.ax.set_xlim3d(-self.axscale, self.axscale)
        self.ax.set_ylim3d(-self.axscale, self.axscale)
        self.ax.set_zlim3d(-self.axscale / 4, self.axscale)
        # Turn off tick labels
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])
        #self.ax.set_xlabel('X axis')
        #self.ax.set_ylabel('Y axis')
        #self.ax.set_zlabel('Z axis')

    def draw_trajectory_robot(self, trajectory, repeat=False):
        self.robot_states = []

        for n in range(len(trajectory)):
            coordinates, rotation = self.robot.calc_x_y_z(trajectory[n])
            self.robot_states.append([coordinates, rotation])

        self.set_ax()
        self.ani = animation.FuncAnimation(self.fig, self.update_trajectory_robot, frames=len(self.robot_states), interval=300, cache_frame_data=False, repeat=repeat, blit=True)
        plt.show()

    def update_trajectory_robot(self, idx):
        self.ax.clear()
        self.set_ax()
        
        coordinates = self.robot_states[idx][0]
        rotation = self.robot_states[idx][1]
        #os.write(1, str(framers))

        x = self.robot_states[idx][0][0] * self.robotscale
        y = self.robot_states[idx][0][1] * self.robotscale
        z = self.robot_states[idx][0][2] * self.robotscale
        self.xw.append(x[-1])
        self.yw.append(y[-1])
        self.zw.append(z[-1])
        #self.text.set_text(self.xw)

        self.plot_positions(coordinates, rotation)

        self.wframe = self.ax.plot_wireframe(np.array([self.xw]), np.array([self.yw]), np.array([self.zw]))
        if idx == len(self.robot_states) - 1:
            self.fig.canvas.draw_idle()
        # time.sleep(0.1)

class DataHandler(object):

    def __init__(self, robot, euler=False, verbose=False):

        self.euler = euler
        self.compute_extreme_positions = False
        self.split = 4
        self.robot = robot

        self.dmax = np.around(np.radians(np.max(self.robot.joint_limits)), decimals=4)
        self.dmin = np.around(np.radians(np.min(self.robot.joint_limits)), decimals=4)
        
        self.y_max = -float('inf')
        self.y_min = float('inf')
        self.x_max = -float('inf')
        self.x_min = float('inf')
        self.z_max = -float('inf')
        self.z_min = float('inf')
        self.rotatemin = -1
        self.rotatemax = 1
        
        self.verbose = verbose

    def get_roll_pitch_yaw(self, rotation):
        
        pitch = math.atan2(-rotation[6], math.sqrt(math.pow(rotation[0], 2) + math.pow(rotation[3], 2)))  # beta

        if pitch == np.pi or pitch == -np.pi:
            yaw = 0
            roll = (pitch / np.pi) * math.atan2(rotation[1], rotation[4])
        else:
            yaw = math.atan2(rotation[3], rotation[0])
            roll = math.atan2(rotation[7], rotation[8])

        return roll, pitch, yaw
    
    def calc_tcp(self, positions):
        tcp = []
        for i in range(len(positions)):
            frame = self.robot.buildDhTcpFrame(positions[i])
            frame = np.asarray(frame.flatten())
            frame = frame[0:12]

            # Aufrunden des Frames
            xyz = np.around(frame[3::4], decimals=3)
            rotation = np.around(np.asarray([frame[0:3], frame[4:7], frame[8:11]]).flatten(), decimals=9)
            for j in range(12):
                if (j != 3) and (j != 7) and (j != 11):
                    frame[j] = rotation[j - int(j / 4)]
                else:
                    frame[j] = xyz[j % 3]

            tcp.append(frame)

        return np.asarray(tcp)

    def calc_xyz_euler(self, positions):
        start1 = time.time()
        tcp = []
        tcp_time = 0
        for i in range(len(positions)):
            start = time.time()
            frame = self.robot.buildDhTcpFrame(positions[i])
            frame = np.asarray(frame.flatten())
            frame = frame[0:12]
            xyz_euler = np.zeros(6)
            end = time.time()
            tcp_time += end - start
            # Aufrunden des Frames
            xyz = np.around(frame[3::4], decimals=3)
            rotation = np.asarray([frame[0:3], frame[4:7], frame[8:11]]).flatten()

            roll, pitch, yaw = self.get_roll_pitch_yaw(rotation)

            xyz_euler[0:3] = xyz
            xyz_euler[3] = np.around(roll, decimals=6)
            xyz_euler[4] = np.around(pitch, decimals=6)
            xyz_euler[5] = np.around(yaw, decimals=6)

            tcp.append(xyz_euler)
        
        end2 = time.time()
        if self.verbose:
            print("DH Transformation: %.2f Sekunden" %tcp_time)
            print("Euler Winkel: %.2f Sekunden" %(end2 - start1 - tcp_time))
        return np.asarray(tcp)

    def get_extreme_positions(self):
        split = self.split
        split -= 1
        limits = self.robot.joint_limits
        extremes_robot = []
        for lim in limits:
            abs_ = np.sum(np.absolute(lim))
            steps = int(abs_ / split)
            missing = abs_ % split
            adding_point = int(split / 2)
            extreme = lim[1]
            extremes_joint = [extreme]
            for it in range(split):
                extreme += steps
                if it == adding_point:
                    extreme += missing
                extremes_joint.append(extreme)
            extremes_robot.append(np.array(extremes_joint))

        return np.array(np.meshgrid(*np.array(extremes_robot))).T.reshape(-1, len(limits))
    
    def generate_n_positions(self, n=100):
        degree_joint_pos = []
        extremes_computed = False

        if self.compute_extreme_positions and self.split ** len(self.robot.joint_limits) > n:
            print("Too few samples to generate. Either turn off generation of extreme positions or "
                  "generate more Data.\n", self.split ** len(self.robot.joint_limits), "extreme positions "
                  "must be generated but you only generate", n, "samples.")
            sys.exit(1)
        elif self.compute_extreme_positions:
            n -= self.split ** len(self.robot.joint_limits)
            extremes = self.get_extreme_positions()
            extremes_computed = True

        for joint_range in self.robot.joint_limits:
            degree_joint_pos.append(np.random.randint(joint_range[1], joint_range[0] + 1, n))

        positions = np.around(np.transpose(degree_joint_pos), decimals=4)

        if extremes_computed:
            positions = np.concatenate((positions, extremes))

        return positions

    def generate_noised_data(self, iterations=100, sigma=5):
        """
        Generates Input and Output Data for the neural network.
        Randomly generates angles according to robot limits and applys normal distribution to them for noise creation.
        Generated angles with and without noise are used in DH transformation to get tcp for both.
        """
        start = time.time()

        positions = self.generate_n_positions(iterations)

        noise = np.random.normal(0, sigma, len(self.robot.joint_limits) * iterations).astype(int).reshape(iterations, len(self.robot.joint_limits))
        noised = np.transpose(np.add(positions, noise))
        for i in range(len(self.robot.joint_limits)):
            np.clip(noised[i], self.robot.joint_limits[i][1], self.robot.joint_limits[i][0], noised[i], casting='unsafe')
        noised_pos = np.transpose(noised)

        noised_pos = np.around(np.radians(noised_pos), decimals=4)
        positions = np.around(np.radians(positions), decimals=4)

        if self.verbose:
            print("Generierung der Gelenkwinkel: %.2f Sekunden" %(time.time() - start))

        start = time.time()

        if not self.euler:
            tcp = self.calc_tcp(positions)
            noised_tcp = self.calc_tcp(noised_pos)
        else:
            tcp = self.calc_xyz_euler(positions)
            noised_tcp = self.calc_xyz_euler(noised_pos)

        if self.verbose:
            print("Generierung des TCPs: %.2f Sekunden" %(time.time() - start))

        relative_tcp = np.subtract(noised_tcp, tcp)

        return positions, tcp, noised_pos, relative_tcp

    def generate_data(self, iterations):

        start = time.time()

        positions = self.generate_n_positions(iterations)
        positions = np.around(np.radians(positions), decimals=4)

        if self.verbose:
            print("Generierung der Gelenkwinkel: %.2f Sekunden" %(time.time() - start))

        start = time.time()

        if not self.euler:
            tcp = self.calc_tcp(positions)
        else:
            tcp = self.calc_xyz_euler(positions)
        if self.verbose:
            print("Generierung des TCPs: %.2f Sekunden" %(time.time() - start))

        return positions, tcp

    def denormalize(self, data):
        '''
        Denormalizes joint angles
        '''
        tmp = copy.deepcopy(data)
        num_joints = tmp.shape[1]

        for i in range(num_joints):
            joint_max, joint_min = self.robot.joint_limits[i]
            tmp[:, i] = (((tmp[:, i] + 1) * (np.radians(joint_max) - np.radians(joint_min))) / 2) + np.radians(joint_min)

        return tmp

    # def normeuler(self, tcp):
    #     '''
    #     Normalizes XYZ and euler angles
    #     '''
    #     tmp = copy.deepcopy(tcp)
    #     xyz = (2 * (tmp[:, 0:3] - self.xyzmin) / (self.xyzmax - self.xyzmin)) - 1
    #     euler = (2 * (tmp[:, 3:6] - (-np.pi)) / (np.pi - (-np.pi))) - 1

    #     return np.concatenate((xyz, euler), axis=1)

    # def normxyz(self, tcp):
    #     '''
    #     Normalizes tcp
    #     '''
    #     tmp = copy.deepcopy(tcp)
    #     tmp[:, 3::4] = (2 * (tmp[:, 3::4] - self.xyzmin) / (self.xyzmax - self.xyzmin)) - 1
    #     return tmp
    
    def normeuler(self, tcp):
        '''
        Normalizes XYZ and euler angles
        '''
        tmp = copy.deepcopy(tcp)
        xyz = np.zeros(tmp[:, 0:3].shape)
        xyz[:, 0] = (2 * (tmp[:, 0] - self.x_min) / (self.x_max - self.x_min)) - 1
        xyz[:, 1] = (2 * (tmp[:, 1] - self.y_min) / (self.y_max - self.y_min)) - 1
        xyz[:, 2] = (2 * (tmp[:, 2] - self.z_min) / (self.z_max - self.z_min)) - 1
        
        euler = (2 * (tmp[:, 3:6] - (-np.pi)) / (np.pi - (-np.pi))) - 1

        return np.concatenate((xyz, euler), axis=1)

    def normxyz(self, tcp):
        '''
        Normalizes tcp
        '''
        xyz = copy.deepcopy(tcp)
        xyz[:, 3] = (2 * (xyz[:, 3] - self.x_min) / (self.x_max - self.x_min)) - 1
        xyz[:, 7] = (2 * (xyz[:, 7] - self.y_min) / (self.y_max - self.y_min)) - 1
        xyz[:, 11] = (2 * (xyz[:, 11] - self.z_min) / (self.z_max - self.z_min)) - 1

        return xyz

    def normalize_joint_angles(self, data):
        '''
        Normalizes joint angles
        '''
        tmp = copy.deepcopy(data)
        num_joints = tmp.shape[1]
        normalized_data = np.zeros(tmp.shape)

        for i in range(num_joints):
            joint_max, joint_min = self.robot.joint_limits[i]
            normalized_data[:, i] = ((2 * (tmp[:, i] - np.radians(joint_min)) / (np.radians(joint_max) - np.radians(joint_min))) - 1)

        return normalized_data

    def normalize_tcp(self, tpos):
        if not self.euler:
            tpos = self.normxyz(tpos)
        else:
            tpos = self.normeuler(tpos)

        return tpos

    def generate_noised(self, batch_size, sigma=60):
        begin = time.time()
        jpos, tpos, njpos, ntpos = self.generate_noised_data(batch_size, sigma)
        start = time.time()
        jpos = self.normalize_joint_angles(jpos)
        njpos = self.normalize_joint_angles(njpos)
        self.set_maxima(tpos)
        tpos = self.normalize_tcp(tpos)
        ntpos = self.normalize_tcp(ntpos)
        normalizing_time = time.time() - start
        if self.verbose:
            print("Normalisierung: %.2f Sekunden" %(normalizing_time))
            print("Gesamt:  %.2f Sekunden" %(time.time() - begin))

        return np.concatenate((jpos, tpos, ntpos), axis=1), njpos

    def generate(self, batch_size):
        begin = time.time()
        jpos, tpos = self.generate_data(batch_size)
        start = time.time()
        jpos = self.normalize_joint_angles(jpos)
        self.set_maxima(tpos)
        tpos = self.normalize_tcp(tpos)
        normalizing_time = time.time() - start
        if self.verbose:
            print("Normalisierung: %.2f Sekunden" %(normalizing_time))
            print("Gesamt:  %.2f Sekunden" %(time.time() - begin))

        return jpos, tpos

    def set_maxima(self, tcp):
        if not self.euler:
            x_s = tcp[:, 3].flatten()
            y_s = tcp[:, 7].flatten()
            z_s = tcp[:, 11].flatten()
        else:
            x_s = tcp[:, 0].flatten()
            y_s = tcp[:, 1].flatten()
            z_s = tcp[:, 2].flatten()

        xmax = np.amax(x_s)
        xmin = np.amin(x_s)
        ymax = np.amax(y_s)
        ymin = np.amin(y_s)
        zmax = np.amax(z_s)
        zmin = np.amin(z_s)

        if xmax > self.x_max: self.x_max = xmax
        if xmin < self.x_min: self.x_min = xmin
        if ymax > self.y_max: self.y_max = ymax
        if ymin < self.y_min: self.y_min = ymin
        if zmax > self.z_max: self.z_max = zmax
        if zmin < self.z_min: self.z_min = zmin
