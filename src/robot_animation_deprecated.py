import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display
from pytransform3d.rotations import *
from mpl_toolkits.mplot3d import proj3d
import copy
from src.robot import Robot

class RobotAnimation(object):

    def __init__(self, robot : Robot, basis=True):
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
        self.ax = self.fig.add_axes([-0.05, -0.05, 1.20, 1.20], projection='3d', autoscale_on=False)
        self.text = self.ax.text(0,0,self.axscale + self.axscale/10, s="", va="bottom", ha="left")

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
            self.goal = goal * self.robotscale
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

    def dh_animation(self, position=None):
        if position is None:
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
        plt.show()

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