import numpy as np
import matplotlib.pyplot as plt
import time
import math
from matplotlib import rc
import matplotlib.animation as animation
import copy
rc('animation', html='html5')


class ForwardKinematicsYouBot:
    def __init__(self):
        self.pi_2 = np.pi / 2
        self.dh_theta_values = np.array([0, -self.pi_2, 0, self.pi_2, self.pi_2])
        self.dh_alpha_values = np.array([-self.pi_2, 0, 0, self.pi_2, 0])
        self.dh_a_values = np.array([0.033, 0.155, 0.135, 0, 0])
        self.dh_d_values = np.array([0.075, 0, 0, 0, 0.218])
        self.onematrix = False

    def buildDhTcpFrame(self, q_array):
        dh_frame = np.identity(4)

        for i in range(5):
            tmp_dh_ith = self.dhIthFrame(q_array[i] + self.dh_theta_values[i],
                                        self.dh_d_values[i],
                                        self.dh_a_values[i],
                                        self.dh_alpha_values[i])
            dh_frame = dh_frame.dot(tmp_dh_ith)

        return dh_frame

    def dhIthFrame(self, theta, d, a, alpha):
        rot_theta = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                            [np.sin(theta), np.cos(theta), 0, 0], 
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        trans_d = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]])
        trans_a = np.array([[1, 0, 0, a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        rot_alpha = np.array([[1, 0, 0, 0],
                            [0, np.cos(alpha), -np.sin(alpha), 0],
                            [0, np.sin(alpha), np.cos(alpha), 0], [0, 0, 0, 1]])

        if not self.onematrix:
            dh_ith_frame = rot_theta.dot(trans_d).dot(trans_a).dot(rot_alpha)
        else:
        # somewhat slower?? why??
            dh_ith_frame = np.asarray([ [np.cos(theta), -np.sin(theta)*np.cos(alpha)  , np.sin(theta)* np.sin(alpha)  , a*np.cos(theta)],
                                        [np.sin(theta), np.cos(theta)*np.cos(alpha)   , -np.cos(theta)*np.sin(alpha)  , a*np.sin(theta)],
                                        [0            , np.sin(alpha)                 , np.cos(alpha)                 , d]              ,
                                        [0            , 0                             , 0                             , 1               ]])

        return dh_ith_frame    


class Drawer():
    def __init__(self):
        '''variable definitions'''
        #         self.z = np.empty(5,dtype=float) #vertical coordinate
        #         self.x = np.empty(5,dtype=float) #x axis components
        #         self.y = np.empty(5,dtype=float) #y axis components
        self.kinematic = ForwardKinematicsYouBot()
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
        self.ax.set_zlim3d(0, 800)
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')

    def calc_x_y_z(self, gt, netout):  # calc x_y_z coordinates
        dh_gt = np.identity(4)
        dh_netout = np.identity(4)
        for i in range(5):
            tmp_dh_gt = self.kinematic.dhIthFrame(gt[i] + self.kinematic.dh_theta_values[i],
                                   self.kinematic.dh_d_values[i],
                                   self.kinematic.dh_a_values[i],
                                   self.kinematic.dh_alpha_values[i])
            tmp_dh_netout = self.kinematic.dhIthFrame(netout[i] + self.kinematic.dh_theta_values[i],
                                       self.kinematic.dh_d_values[i],
                                       self.kinematic.dh_a_values[i],
                                       self.kinematic.dh_alpha_values[i])

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
                tmp_dh = self.kinematic.dhIthFrame(netout[n][i] + self.kinematic.dh_theta_values[i],
                                    self.kinematic.dh_d_values[i],
                                    self.kinematic.dh_a_values[i],
                                    self.kinematic.dh_alpha_values[i])

                dh_frame = dh_frame.dot(tmp_dh)
                xyz = np.asarray(dh_frame.flatten())
                xyz = xyz[0:12]
                xyz = xyz[3::4]
                print("DH Frame flatten: ", dh_frame.flatten())
                print("XYZ", xyz)
                robot.append(xyz)
            self.robot_states.append(robot)

        self.set_ax()

        self.ani = animation.FuncAnimation(self.fig, self.update_trajectory_robot, interval=400, cache_frame_data=True,blit=True, repeat=False)
        plt.show()

    def draw_trajectory(self, netout):
        x, y, z = [], [], []

        for i in range(len(netout)):
            frame = self.kinematic.buildDhTcpFrame(netout[i])
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


class DataHandler(object):

    def __init__(self, euler=False):
        self.euler = euler
        self.kinematic = ForwardKinematicsYouBot()

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

    def calc_tcp(self, positions, calc_time=False):
        '''
        Does the forward kinematic on positions array and returns the full rotation matrix.
        If calc_time is set to True the calculated process time is given back, too.
        '''
        start = time.time()
        tcp = []
        for i in range(len(positions)):
            frame = self.kinematic.buildDhTcpFrame(positions[i])
            frame = frame[:3,:].flatten()
            frame[::3] = np.around(frame[::3], decimals=9)
            frame[3:12:4] = np.around(frame[3:12:4], decimals=3)

            tcp.append(frame)
        end = time.time()
        if calc_time:
            return np.asarray(tcp), (end - start)
        return np.asarray(tcp)

    def get_roll_pitch_yaw(self, rotation):
        '''
        Computes Roll Pitch and Yaw from a Rotation Matrix
        '''
        
        pitch = math.atan2(-rotation[6], math.sqrt(math.pow(rotation[0], 2) + math.pow(rotation[3], 2)))  # beta

        if pitch == np.pi or pitch == -np.pi:
            yaw = 0
            roll = (pitch / np.pi) * math.atan2(rotation[1], rotation[4])
        else:
            yaw = math.atan2(rotation[3], rotation[0])
            roll = math.atan2(rotation[7], rotation[8])

        return roll, pitch, yaw

    def calc_xyz_euler(self, positions, calc_time=False):
        '''
        Calculates XYZ values and euler angles from an array of joint positions.
        If calc_time is True it also gives back the calculation time of the process.
        '''
        start1 = time.time()
        tcp = []
        tcp_time = 0
        for i in range(len(positions)):
            start = time.time()
            frame = self.kinematic.buildDhTcpFrame(positions[i])
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
        if calc_time:
            return np.asarray(tcp), tcp_time, (end2 - start1 - tcp_time)
        return np.asarray(tcp)

    def generate_rnd_jnt_pos(self, iterations=100, generate_noise=False, sigma=5):
        '''
        Randomly generates iteration-number joint positions. There is an option to
        generate a second vector of joint positions from the first one by setting
        generate_noise to True. This adds a gaussian distribution to the first generated
        joint positions. 
        '''
        joint_limits = [self.a1, self.a2, self.a3, self.a4, self.a5]
        degree_joint_pos = []
        for joint_range in joint_limits:
            joint_vals = np.random.randint(joint_range[1], joint_range[0] + 1, iterations)
            degree_joint_pos.append(joint_vals)

        degree_joint_pos = np.asarray(degree_joint_pos)
        positions = np.transpose(degree_joint_pos)
        positions = np.around(np.radians(positions), decimals=4)
        if not generate_noise:
            return positions
        else:
            noise = np.random.normal(0,sigma,5 * iterations).astype(int).reshape(5,iterations)
            noised = np.add(degree_joint_pos, noise)
            for i in range(5):
                np.clip(noised[i], joint_limits[i][1], joint_limits[i][0], noised[i])
            noised_pos = np.transpose(noised)
            noised_pos = np.around(np.radians(noised_pos), decimals=4)
            return positions, noised_pos

    def generate_data(self, iterations):
        '''
        Generates random joint angles and calculates TCP from them, either as Rotation Matrix
        or XYZ and Euler angles. Does not normalize the data. Returns computation time to the
        console for each step.
        '''
        joint_limits = [self.a1, self.a2, self.a3, self.a4, self.a5]

        degree_joint_pos = []
        for joint_range in joint_limits:
            degree_joint_pos.append(np.random.randint(joint_range[1], joint_range[0] + 1, iterations))

        positions = np.around(np.transpose(degree_joint_pos), decimals=4)

        if not self.euler:
            tcp = self.calc_tcp(positions)
        else:
            tcp = self.calc_xyz_euler(positions)

        return positions, tcp

    def denormalize(self, data):
        tmp = copy.deepcopy(data)

        for i, arr in enumerate(tmp):
            for j, value in enumerate(arr):
                tmp[i][j] = (((value + 1) * (self.dmax - self.dmin)) / 2) + self.dmin

        return tmp

    def normeuler(self, tcp):
        '''
        Normalizes XYZ position and euler angles between [-1, 1]
        '''
        tmp = copy.deepcopy(tcp)
        xyz = (2 * (tmp[:, 0:3] - self.xyzmin) / (self.xyzmax - self.xyzmin)) - 1
        euler = (2 * (tmp[:, 3:6] - (-np.pi)) / (np.pi - (-np.pi))) - 1

        return np.concatenate((xyz, euler), axis=1)

    def normxyz(self, tcp):
        '''
        Normalizes just XYZ position of a Rotation Matrix between [-1, 1]
        '''
        tmp = copy.deepcopy(tcp)
        tmp[:, 3::4] = (2 * (tmp[:, 3::4] - self.xyzmin) / (self.xyzmax - self.xyzmin)) - 1
        return tmp

    def normalize(self, data):
        '''
        Normalizes joint positions between [-1, 1]
        '''
        tmp = copy.deepcopy(data)
        return ((2 * (tmp - self.dmin) / (self.dmax - self.dmin)) - 1)

    def do_norm(self, tpos):
        '''
        Does normalization of a TCP according to the choosen TCP computation(either Rotation Matrix or Euler)
        '''
        if not self.euler:
            tpos = self.normxyz(tpos)
        else:
            tpos = self.normeuler(tpos)

        return tpos

    def generate(self, batch_size, noised=False, sigma=5, euler_known=False):
        '''
        Generates randomly batch_size number Joint Positions. Does Forward Kinematic on them
        and gets TCP either as Rotation Matrix or XYZ and Euler angles. Normalizes all values.
        '''
        begin = time.time()
        tcp_time = 0 
        euler_time = 0

        '''Generate random Joint Positions'''
        if noised:
            joint_pos, noised_pos = self.generate_rnd_jnt_pos(batch_size, noised, sigma)
        else:
            joint_pos = self.generate_rnd_jnt_pos(batch_size)

        '''Calculate TCP from Joint Positions'''
        if not self.euler:
            tcp, tcp_time = self.calc_tcp(joint_pos, calc_time=True)
            if noised:
                noised_tcp, t_ = self.calc_tcp(noised_pos, calc_time=True)
                tcp_time += t_
            print("DH Transformation: %.2f Sekunden" %tcp_time)
        else:
            tcp, tcp_time, euler_time = self.calc_xyz_euler(joint_pos, calc_time=True)
            if noised:
                noised_tcp, t_, e_ = self.calc_xyz_euler(noised_pos, calc_time=True)
                tcp_time += t_
                euler_time += e_
            print("DH Transformation: %.2f Sekunden" %tcp_time)
            print("Euler Winkel: %.2f Sekunden" %euler_time)

        '''Normalize Data'''
        start = time.time()
        jpos = self.normalize(joint_pos)
        tpos = self.do_norm(tcp)

        if noised:
            njpos = self.normalize(noised_pos)
            ntpos = self.do_norm(noised_tcp)
            normalizing_time = time.time() - start
            print("Normalisierung: %.2f Sekunden" %(normalizing_time))
            print("Gesamt:  %.2f Sekunden" %(time.time() - begin))
            return np.concatenate((jpos, tpos, ntpos), axis=1), njpos
        else:
            normalizing_time = time.time() - start
            print("Normalisierung: %.2f Sekunden" %(normalizing_time))
            print("Gesamt:  %.2f Sekunden" %(time.time() - begin))
            return jpos, tpos

    def get_tcp_from_dat_file(self, file):
        points = [[int(y) for y in i.strip().split()] for i in open(file).readlines()]
        max_ = -999
        min_ = 999
        for t in points:
            if t:
                ma = np.max(t)
                mi = np.min(t)
                if max_ < ma: 
                    max_ = ma
                if min_ > mi: 
                    min_ = np.min(t)

        tcp = []
        old_p = []
        for i, point in enumerate(points):
            if point:
                # x = point[0]
                # y = point[1]
                # x_ = 100 + (x_ + 1) * 20
                # y_ = 100 + (y_ + 1) * 20
                x = ((2 * (point[0] - min_) / (max_ - min_)) - 1)
                y = ((2 * (point[1] - min_) / (max_ - min_)) - 1)
                # x = (point[0] - 400) / (800)
                # y = (point[1] - 400) / (800)
                # x_ = x * (self.xyzmax - self.xyzmin) / 2 + self.xyzmin
                # y_ = y  * (self.xyzmax - self.xyzmin) / 2 + self.xyzmin
                # x = point[0]
                # y = point[1]
                x_ = ((x + 1) * (self.xyzmax - self.xyzmin) / 2 + self.xyzmin)
                y_ = ((y + 1) * (self.xyzmax - self.xyzmin) / 2 + self.xyzmin)
                # print(x_, y_)
                p = [x_, y_, 0.49]
                old_p = p
                tcp.append(p)
            else:
                old_p[2] = -0.05
                tcp.append(old_p)
            
        return np.array(tcp)

    '''Not used anymore'''

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

    def make(self, batch_size):
        while True:
            jpos, tpos = self.generate_data(batch_size)
            jpos = self.normalize(jpos)
            tpos = self.do_norm(tpos)

            yield (tpos, jpos)

    def generate_step(self, batch_size):
        jpos, tpos = self.generate_data_step(batch_size)
        jpos = self.normalize(jpos)
        tpos = self.do_norm(tpos)

        return jpos, tpos

    def generate_gauss(self, batch_size, sigma=5):
        begin = time.time()
        jpos, tpos, njpos, ntpos = self.generate_data_gauss(batch_size, sigma)
        start = time.time()
        jpos = self.normalize(jpos)
        njpos = self.normalize(njpos)
        tpos = self.do_norm(tpos)
        ntpos = self.do_norm(ntpos)
        normalizing_time = time.time() - start
        print("Normalisierung: %.2f Sekunden" %(normalizing_time))
        print("Gesamt:  %.2f Sekunden" %(time.time() - begin))

        return np.concatenate((jpos, tpos, ntpos), axis=1), njpos

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

        if not self.euler:
            tcp = self.calc_tcp(positions)
        else:
            tcp = self.calc_xyz_euler(positions)

        return positions, tcp

    def generate_data_gauss(self, iterations=100, sigma=5):
        '''
        Generates Input and Output Data for the neural network.
        Randomly generates angles according to robot limits and applys normal distribution to them for noise creation.
        Generated angles with and without noise are used in DH transformation to get tcp for both.
        '''
        start = time.time()
        joint_limits = [self.a1, self.a2, self.a3, self.a4, self.a5]
        degree_joint_pos = []
        for joint_range in joint_limits:
            joint_vals = np.random.randint(joint_range[1], joint_range[0] + 1, iterations)
            degree_joint_pos.append(joint_vals)

        degree_joint_pos = np.asarray(degree_joint_pos)

        noise = np.random.normal(0,sigma,5 * iterations).astype(int).reshape(5,iterations)
        noised = np.add(degree_joint_pos, noise)
        for i in range(5):
            np.clip(noised[i], joint_limits[i][1], joint_limits[i][0], noised[i])
        noised_pos = np.transpose(noised)
        positions = np.transpose(degree_joint_pos)

        noised_pos = np.around(np.radians(noised_pos), decimals=4)
        positions = np.around(np.radians(positions), decimals=4)

        print("Datengenerierung: %.2f Sekunden" %(time.time() - start))

        if not self.euler:
            tcp = self.calc_tcp(positions)
            noised_tcp = self.calc_tcp(noised_pos)
        else:
            tcp = self.calc_xyz_euler(positions)
            noised_tcp = self.calc_xyz_euler(noised_pos)

        return positions, tcp, noised_pos, noised_tcp

    # normiert frame xyz und rotationsmatrix anhand getesteter werte
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