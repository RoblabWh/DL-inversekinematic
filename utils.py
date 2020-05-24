import numpy as np
import matplotlib.pyplot as plt
import keras.utils
import time
import matplotlib.animation as animation


class Drawer():
    def __init__(self):
        '''variable definitions'''
        #         self.z = np.empty(5,dtype=float) #vertical coordinate
        #         self.x = np.empty(5,dtype=float) #x axis components
        #         self.y = np.empty(5,dtype=float) #y axis components

        self.coordinates_gt = [np.empty(5, dtype=float), np.empty(5, dtype=float), np.empty(5, dtype=float)]
        self.coordinates_netout = [np.empty(5, dtype=float), np.empty(5, dtype=float), np.empty(5, dtype=float)]

        self.xl, self.yl, self.zl = np.empty(100), np.empty(100), np.empty(100)
        self.wframe = None
        self.robot_states = []
        self.robot = None

        self.fig = plt.figure("Robot Simulator")
        self.ax = plt.axes([0.05, 0.2, 0.90, .75], projection='3d')

    def set_positions(self):  # gets the x,y,z values
        #         xs = self.x.tolist()
        #         ys = self.y.tolist()
        #         zs = self.z.tolist()
        x_gt = self.coordinates_gt[0].tolist()
        y_gt = self.coordinates_gt[1].tolist()
        z_gt = self.coordinates_gt[2].tolist()

        x_netout = self.coordinates_netout[0].tolist()
        y_netout = self.coordinates_netout[1].tolist()
        z_netout = self.coordinates_netout[2].tolist()

        self.ax.cla()
        self.ax.plot(x_gt, y_gt, z_gt, 'o-', markersize=10,
                     markerfacecolor="black", linewidth=6, color="orange", label='ground truth')

        self.ax.plot(x_netout, y_netout, z_netout, 'o-', markersize=10,
                     markerfacecolor="k", linewidth=6, color="green", label='output')

    def set_ax(self):  # ax panel set up
        self.ax.set_xlim3d(-400, 400)
        self.ax.set_ylim3d(-400, 400)
        self.ax.set_zlim3d(-400, 400)
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')

    def calc_x_y_z(self, gt, netout):  # calc x_y_z coordinates
        dh_gt = np.identity(4)
        dh_netout = np.identity(4)
        for i in range(5):
            tmp_dh_gt = dhIthFrame(gt[i] + dh_theta_values[i],
                                   dh_d_values[i],
                                   dh_a_values[i],
                                   dh_alpha_values[i])
            tmp_dh_netout = dhIthFrame(netout[i] + dh_theta_values[i],
                                       dh_d_values[i],
                                       dh_a_values[i],
                                       dh_alpha_values[i])

            dh_gt = np.matmul(dh_gt, tmp_dh_gt)
            dh_netout = np.matmul(dh_netout, tmp_dh_netout)

            xyz = dh_gt[:-1, -1]
            self.coordinates_gt[0][i] = xyz[0] * 1000
            self.coordinates_gt[1][i] = xyz[1] * 1000
            self.coordinates_gt[2][i] = xyz[2] * 1000

            xyz = dh_netout[:-1, -1]
            self.coordinates_netout[0][i] = xyz[0] * 1000
            self.coordinates_netout[1][i] = xyz[1] * 1000
            self.coordinates_netout[2][i] = xyz[2] * 1000

    def draw_trajectory_robot(self, netout):
        self.robot_states = []

        for n in range(len(netout)):
            dh_frame = np.identity(4)
            robot = []
            for i in range(5):
                tmp_dh = dhIthFrame(netout[n][i] + dh_theta_values[i],
                                    dh_d_values[i],
                                    dh_a_values[i],
                                    dh_alpha_values[i])

                dh_frame = dh_frame.dot(tmp_dh)
                xyz = np.asarray(dh_frame.flatten())
                xyz = xyz[0:12]
                xyz = xyz[3::4]
                robot.append(xyz)
            self.robot_states.append(robot)

        self.set_ax()

        self.ani = animation.FuncAnimation(self.fig, self.update_trajectory_robot, interval=400, cache_frame_data=False)
        plt.show()

    def draw_trajectory(self, netout):
        x, y, z = [], [], []

        for i in range(len(netout)):
            frame = buildDhTcpFrame(netout[i])
            frame = np.asarray(frame.flatten())
            frame = frame[0:12]
            xyz = frame[3::4]
            x.append(xyz[0] * 1000)
            y.append(xyz[1] * 1000)
            z.append(xyz[2] * 1000)
        self.xl = np.asarray([x])
        self.yl = np.asarray([y])
        self.zl = np.asarray([z])
        self.set_ax()

        self.ani = animation.FuncAnimation(self.fig, self.update_trajectory, interval=400, cache_frame_data=False)
        plt.show()

    def draw_robot(self, gt, netout):
        self.calc_x_y_z(gt, netout)
        self.set_positions()
        self.set_ax()

        plt.draw()

    def update_trajectory_robot(self, idx):
        if self.wframe:
            self.ax.collections.remove(self.wframe)
        if self.robot:
            self.ax.clear()
            self.set_ax()

        states = self.robot_states[:idx + 1]
        tcp = []
        for state in states:
            tcp.append(state[4])
        tcp = np.asarray(tcp) * 1000
        x, y, z = [], [], []
        for elm in tcp:
            x.append(elm[0])
            y.append(elm[1])
            z.append(elm[2])

        robot = np.asarray(states[-1]) * 1000

        rx, ry, rz = [], [], []
        for elm in robot:
            rx.append(elm[0])
            ry.append(elm[1])
            rz.append(elm[2])

        x = np.asarray([x])
        y = np.asarray([y])
        z = np.asarray([z])
        self.robot = self.ax.plot(rx, ry, rz, 'o-', markersize=10,
                     markerfacecolor="black", linewidth=6, color="orange", label='ground truth')
        self.wframe = self.ax.plot_wireframe(x, y, z)
        self.fig.canvas.draw()
        # time.sleep(0.1)

    def update_trajectory(self, idx):
        if self.wframe:
            self.ax.collections.remove(self.wframe)
        x = np.asarray([self.xl[0][:idx]])
        y = np.asarray([self.yl[0][:idx]])
        z = np.asarray([self.zl[0][:idx]])
        self.wframe = self.ax.plot_wireframe(x, y, z)
        self.fig.canvas.draw()
        # time.sleep(0.1)


pi_2 = np.pi / 2
dh_theta_values = np.array([0, -pi_2, 0, pi_2, pi_2])
dh_alpha_values = np.array([-pi_2, 0, 0, pi_2, 0])
dh_a_values = np.array([0.033, 0.155, 0.135, 0, 0])
dh_d_values = np.array([0.075, 0, 0, 0, 0.218])


def dhIthFrame(theta, d, a, alpha):
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


def buildDhTcpFrame(q_array):
    dh_frame = np.identity(4)

    for i in range(5):
        tmp_dh_ith = dhIthFrame(q_array[i] + dh_theta_values[i],
                                dh_d_values[i],
                                dh_a_values[i],
                                dh_alpha_values[i])
        dh_frame = dh_frame.dot(tmp_dh_ith)

    return dh_frame


class DataHandler(object):

    def __init__(self):
        # self.dmax = np.radians(165)
        # self.dmin = np.radians(-168)
        # self.dmax = 165.0
        # self.dmin = -168.0
        # Maximale und minimale Werte (based on robo freedom tests) */
        # self.a1 = [165, -168]
        # self.a2 = [85, -64]
        # self.a3 = [145, -141]
        # self.a4 = [101, -101]
        # self.a5 = [155, -161]

        self.dmax = np.around(np.radians(169), decimals=4)
        self.dmin = np.around(np.radians(-169), decimals=4)
        self.xyzmin = -0.54
        self.xyzmax = 0.59
        self.rotatemin = -1
        self.rotatemax = 1

        self.a1 = [169, -169]
        self.a2 = [90, -65]
        self.a3 = [146, -151]
        self.a4 = [102.5, -102.5]
        self.a5 = [167.5, -167.5]

    def calc_tcp(self, positions):
        tcp = []
        for i in range(len(positions)):
            frame = buildDhTcpFrame(positions[i])
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

    def generate_data(self, iterations):
        joint_limits = [self.a1, self.a2, self.a3, self.a4, self.a5]
        pos_arr = []
        for i in range(iterations):
            degree_joint_pos = []
            for joint_range in joint_limits:
                joint_val = np.random.randint(joint_range[1], joint_range[0] + 1)
                degree_joint_pos.append(joint_val)
            radians = np.radians(np.asarray(degree_joint_pos))
            pos_arr.append(radians)

        positions = np.around(np.asarray(pos_arr), decimals=4)

        tcp = self.calc_tcp(positions)

        return positions, tcp

    def generate_data_step(self, iterations=100):

        joint_limits = [self.a1, self.a2, self.a3, self.a4, self.a5]
        state = np.zeros(5)
        direction = np.ones(5)
        pos_arr = []

        for i in range(iterations):
            degree_joint_pos = []
            for j, joint_range in enumerate(joint_limits):
                # step = (2*np.random.randint(0,2)-1) * np.random.randint(1,6) * direction[j]
                step = np.random.randint(1, 6) * direction[j]
                if (step + state[j]) > joint_range[0] or (step + state[j]) < joint_range[1]:
                    direction[j] *= -1
                    step *= direction[j]
                state[j] += step
                degree_joint_pos.append(state[j])

            radians = np.radians(np.asarray(degree_joint_pos))
            pos_arr.append(radians)

        positions = np.around(np.asarray(pos_arr), decimals=4)

        # print("positions calculated")
        tcp = self.calc_tcp(positions)
        # print("tcp calculated")

        # positions, tcp = self.deleteDuplicate(tcp, positions)
        # print("duplicates erased")
        return positions, tcp

    def deleteDuplicate(self, tcp, pos):
        isDuplicate = []
        for i in range(len(tcp)):
            if (i % 10000) == 0:
                print("Durchlauf ", i)
            xyz = tcp[i][3::4]
            for j in range((i + 1), len(tcp)):
                xyz_ = tcp[j][3::4]
                if np.isclose(xyz[0], xyz_[0], rtol=1e-02):
                    if np.isclose(xyz[1], xyz_[1], rtol=1e-02):
                        if np.isclose(xyz[2], xyz_[2], rtol=1e-02):
                            print("Found Duplicate.. ", xyz[0], xyz_[0], xyz[1], xyz_[1], xyz[2], xyz_[2])
                # if np.isclose(xyz[0], xyz_[0], rtol=1e-03) and np.isclose(xyz[1], xyz_[1], rtol=1e-03) and np.isclose(xyz[2], xyz_[2], rtol=1e-03):
                #     print(i)
                #     isDuplicate.append(i)

        #     print(len(isDuplicate))
        new_tcp = np.delete(tcp, isDuplicate, 0)
        new_joint_pos = np.delete(pos, isDuplicate, 0)
        return new_joint_pos, new_tcp

    def makeUnique(self, tcp, pos):
        seen = set()
        for i in range(len(tcp) - 1, -1, -1):
            xyz = tcp[i][3::4]
            if xyz in seen:
                pass

    def denormalize(self, data):
        tmp = np.copy(data)

        for i, arr in enumerate(data):
            for j, value in enumerate(arr):
                tmp[i][j] = (((value + 1) * (self.dmax - self.dmin)) / 2) + self.dmin

        return tmp

    def norm(self, tcp):
        tmp = np.copy(tcp)
        for i, arr in enumerate(tcp):
            xyz = arr[3::4]
            frame = np.asarray([arr[0:3], arr[4:7], arr[8:11]]).flatten()

            for j, value in enumerate(xyz):
                xyz[j] = (2 * (value - self.xyzmin) / (self.xyzmax - self.xyzmin)) - 1

            for j, value in enumerate(frame):
                frame[j] = (2 * (value - self.rotatemin) / (self.rotatemax - self.rotatemin)) - 1

            for j in range(12):
                if (j != 3) and (j != 7) and (j != 11):
                    tmp[i][j] = frame[j - int(j / 4)]
                else:
                    tmp[i][j] = xyz[j % 3]
        return tmp

    def normd(self, tcp):
        tmp = np.copy(tcp)
        for i, arr in enumerate(tcp):
            xyz = arr[3::4]

            for j, value in enumerate(xyz):
                xyz[j] = (2 * (value - self.xyzmin) / (self.xyzmax - self.xyzmin)) - 1

            for j in range(12):
                if (j == 3) or (j == 7) or (j == 11):
                    tmp[i][j] = xyz[j % 3]
        return tmp

    def normalize(self, data):
        tmp = np.copy(data)
        for i, arr in enumerate(data):
            for j, value in enumerate(arr):
                tmp[i][j] = (2 * (value - self.dmin) / (self.dmax - self.dmin)) - 1
        return tmp

    def make(self, batch_size):
        while True:
            jpos, tpos = self.generate_data(batch_size)
            jpos = self.normalize(jpos)
            tpos = self.norm(tpos)
            # tpos = keras.utils.normalize(tcp, axis=-1, order=2)
            yield (tpos, jpos)

    def generate_step(self, batch_size):
        jpos, tpos = self.generate_data_step(batch_size)
        jpos = self.normalize(jpos)
        tpos = self.norm(tpos)
        # tpos = keras.utils.normalize(tcp, axis=-1, order=2)
        return jpos, tpos

    def generate(self, batch_size):
        jpos, tpos = self.generate_data(batch_size)
        jpos = self.normalize(jpos)
        tpos = self.normd(tpos)
        # tpos = keras.utils.normalize(tpos, axis=-1, order=2)
        return jpos, tpos

    def maxxyz(self, tcp):
        tcps = []
        for arr in tcp:
            xyz = np.array([arr[3], arr[7], arr[11]])
            tcps.append(xyz)
        tcps = np.asarray(tcps)
        max = np.amax(tcps)
        min = np.amin(tcps)
        print("xyz max: ", max)
        print("xyz min: ", min)

    def maxradian(self, tcp):
        frames = []
        for arr in tcp:
            frame = np.asarray([arr[0:3], arr[4:7], arr[8:11]]).flatten()
            frames.append(frame)
        frames = np.asarray(frames)
        max = np.amax(frames)
        min = np.amin(frames)
        print("max rad: ", max)
        print("min rad: ", min)
