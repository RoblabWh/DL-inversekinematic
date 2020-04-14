import numpy as np
import matplotlib.pyplot as plt
import keras.utils as tf

class Drawer():      
    def __init__(self):
        '''variable definitions'''
#         self.z = np.empty(5,dtype=float) #vertical coordinate
#         self.x = np.empty(5,dtype=float) #x axis components 
#         self.y = np.empty(5,dtype=float) #y axis components
        
        self.coordinates_gt = [np.empty(5,dtype=float), np.empty(5,dtype=float), np.empty(5,dtype=float)]
        self.coordinates_netout = [np.empty(5,dtype=float), np.empty(5,dtype=float), np.empty(5,dtype=float)]

  
        self.fig = plt.figure("Robot Simulator")
        self.ax = plt.axes([0.05, 0.2, 0.90, .75], projection='3d')

    def set_positions(self):#gets the x,y,z values
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
                     markerfacecolor="black", linewidth = 6, color="orange", label='ground truth')
        
        self.ax.plot(x_netout, y_netout, z_netout, 'o-', markersize=10, 
                     markerfacecolor="k", linewidth = 6, color="green", label='output')
        
    def set_ax(self):#ax panel set up
        self.ax.set_xlim3d(-400, 400)
        self.ax.set_ylim3d(-400, 400)
        self.ax.set_zlim3d(-400, 400)
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')
        
    def calc_x_y_z(self, gt, netout):#calc x_y_z coordinates
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
#             self.x[i] = xyz[0] * 1000
#             self.y[i] = xyz[1] * 1000
#             self.z[i] = xyz[2] * 1000
            
    def draw_robot(self, gt, netout):
        self.calc_x_y_z(gt, netout)
        self.set_positions()
        self.set_ax()

        plt.draw()

dh_theta_values = np.array([0, -np.pi / 2, 0, np.pi / 2, np.pi / 2])
dh_alpha_values = np.array([-np.pi / 2, 0, 0, np.pi / 2, 0])
dh_a_values = np.array([0.033, 0.155, 0.135, 0, 0])
dh_d_values = np.array([0.075, 0, 0, 0, 0.218])

def dhIthFrame(theta, d, a, alpha):
    
    rot_theta = np.matrix([ [np.cos(theta), -np.sin(theta), 0, 0], 
                            [np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0], 
                            [0, 0, 0, 1] ])
    
    trans_d = np.matrix([ [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1] ])
    trans_a = np.matrix([ [1, 0, 0, a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1] ])
    
    rot_alpha = np.matrix([ [1, 0, 0, 0], 
                            [0, np.cos(alpha), -np.sin(alpha), 0], 
                            [0, np.sin(alpha), np.cos(alpha), 0], [0, 0, 0, 1] ])
    
    dh_ith_frame = rot_theta * trans_d * trans_a * rot_alpha
    
    return dh_ith_frame;


def buildDhTcpFrame(q_array):
    dh_frame = np.identity(4)
      
    for i in range(5):
        tmp_dh_ith = dhIthFrame(q_array[i] + dh_theta_values[i], 
                                dh_d_values[i], 
                                dh_a_values[i], 
                                dh_alpha_values[i])
        dh_frame = np.matmul(dh_frame, tmp_dh_ith)
    
    return dh_frame


class Generator(object):
    
    def __init__(self):
        self.dmax = np.radians(165)
        self.dmin = np.radians(-168)
        
    def generate_data(self, iterations):
        #Maximale und minimale Werte (based on robo freedom tests) */
        a1 = [165, -168]
        a2 = [85, -64]
        a3 = [145, -141]
        a4 = [101, -101]
        a5 = [155, -161]
        joint_limits = [a1, a2, a3, a4, a5]
        pos_arr = []
        for i in range(iterations):
            degree_joint_pos = []
            for joint_range in joint_limits:
                joint_val = np.random.randint(joint_range[1], joint_range[0] + 1)
                degree_joint_pos.append(joint_val)
            degree_joint_pos = np.asarray(degree_joint_pos)
            radians = np.radians(degree_joint_pos)
    #         radians = degree_joint_pos
            pos_arr.append(radians)

        positions = np.asarray(pos_arr)

        tcp = []
        for i in range(iterations):
            frame = buildDhTcpFrame(positions[i])
            frame = np.asarray(frame.flatten())
            frame = frame[0:, :12]
            frame = np.squeeze(frame)
    #         xyz = frame[3::4]
            tcp.append(frame)

        tcp = np.asarray(tcp)

        return positions, tcp

    def isDuplicate(tcp):
        xyz = frame[3::4]
        for i in range(len(tcp)):
                xyz = tcp[i][3::4]
                for j in range((i+1), len(tcp)):
                        xyz_ = tcp[j][3::4]
        return False
    
    def denormalize(self, data):
    
        for i, arr in enumerate(data):
            for j, value in enumerate(arr):
                data[i][j] = (((value + 1) * (self.dmax - self.dmin)) / 2) + self.dmin
            
    def normalize(self, data):

        for i, arr in enumerate(data):
            for j, value in enumerate(arr):
                data[i][j] = (2 * (value - self.dmin) / (self.dmax - self.dmin)) - 1
                
    def make(self, batch_size):
        while True:
            jpos, tcp = self.generate_data(batch_size)
            self.normalize(jpos)
            tpos = tf.normalize(tcp, axis=-1, order=2)
            yield (tpos, jpos)

    def generate(self, batch_size):
        jpos, tcp = self.generate_data(batch_size)
        self.normalize(jpos)
        tpos = tf.normalize(tcp, axis=-1, order=2)
        return tpos, jpos