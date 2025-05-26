import numpy as np
import base64
import copy

import torch

class Robot():
    def __init__(self, dh_values=None, joint_limits=None, robot=None):
        #[dh_theta_values, dh_d_values, dh_a_values, dh_alpha_values]
        self.name = "Robot"
        if robot:
            if robot=="youbot":
                self.name = "YouBot"
                mokuba = np.array([
                    [0, 0, 0, 0, 0],  #Theta
                    [0.147, 0.0, 0.0, 0.0, 0.2175],  #d
                    [0.033, 0.155, 0.135, 0.0, 0.0],  # a
                    [np.pi/2, 0.0, 0.0, np.pi / 2, 0.0]  # alpha
                ])
                joint_limits = [
                    [169, -169],
                    [90, -65],
                    [146, -151],
                    [102.5, -102.5],
                    [167.5, -167.5]
                ]
            elif robot == "baxter":
                self.name = "Baxter"
                balon = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABI4XoUrkfRPwAAAAAAAAAAsp3vp8ZL1z8A \
                AAAAAAAAAFYOLbKd79c/AAAAAAAAAADsUbgehevRP0SLbOf7qbE/AAAAAAAAAABEi2zn+6mxPwAAAAAAAAAAexSuR+F6hD8AAAAAAAAAAAAAAAAAAAAAGC \
                1EVPsh+b8YLURU+yH5PxgtRFT7Ifm/GC1EVPsh+T8YLURU+yH5vxgtRFT7Ifk/AAAAAAAAAAA="
                berong = base64.decodebytes(balon)
                mokuba = np.frombuffer(berong, dtype=np.float64).reshape((4,7))

                joint_limits = [[180.0, -180.0] for i in range(7)]
                self.name = "Baxter"
            elif robot == "twoaxis":
                self.name = "TwoAxis"
                # Define a generic two axis manipulator
                mokuba = np.array([[-np.pi/2, -np.pi/2], #Theta
                                   [0, 0], #d
                                   [0.055, 0.155], #a
                                   [-np.pi/2, 0]]) #alpha
                joint_limits = [[169, -169], [110, -75]]
            elif robot == "threeaxis":
                self.name = "threeaxis"
                mokuba = np.array([
                    [0, 0.0, 0.05, -np.pi / 2],  # Joint 1
                    [0, 0.0, 0.16, 0.0],  # Joint 2
                    [0, 0.0, 0.12, 0.0]  # Joint 3
                ]).transpose()
                joint_limits = [
                    [169, -169],
                    [90, -90],
                    [130, -130]
                ]
            elif robot == "fouraxis":
                self.name = "fouraxis"
                mokuba = np.array([
                    [0, 0.0, 0.05, -np.pi / 2],  # Joint 1
                    [0, 0.0, 0.16, 0.0],  # Joint 2
                    [0, 0.0, 0.12, 0.0],  # Joint 3
                    [0, 0.0, 0.08, np.pi / 2]   # Joint 4
                ]).transpose()
                joint_limits = [
                    [169, -169],
                    [90, -90],
                    [130, -130],
                    [180, -180]
                ]
            elif robot == "fiveaxis":
                self.name = "fiveaxis"
                mokuba = np.array([
                    [0, 0, 0, 0, 0],  #Theta
                    [0.147, 0.0, 0.0, 0.0, 0.2175],  #d
                    [0.033, 0.155, 0.135, 0.0, 0.0],  # a
                    [np.pi/2, 0.0, 0.0, np.pi / 2, 0.0]  # alpha
                ])
                joint_limits = [
                    [169, -169],
                    [90, -65],
                    [146, -151],
                    [102.5, -102.5],
                    [167.5, -167.5]
                ]
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
        self.use_torch = False
        self.device = None
        
    def get_num_axis(self):
        return len(self.joint_limits)
        
    def set_torch(self, use_torch, device=None):
        self.use_torch = use_torch
        
        if isinstance(self.dh_values, np.ndarray) and self.use_torch:
            self.dh_values = torch.tensor(self.dh_values, dtype=torch.float32).to(device)
            self.device = device
            
        if isinstance(self.dh_values, torch.Tensor) and not self.use_torch:
            self.dh_values = self.dh_values.cpu().numpy()
            self.device = None

    def dhIthFrame(self, theta, d, a, alpha):
        if self.use_torch:
            cos = torch.cos
            sin = torch.sin
            stack = torch.stack
            zero_tensor = torch.zeros_like(theta)
            one_tensor = torch.ones_like(theta)
        else:
            cos = np.cos
            sin = np.sin
            stack = np.stack
            zero_tensor = np.zeros_like(theta)
            one_tensor = np.ones_like(theta)
            
        T_joint = stack([
            stack([cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)], axis=-1),
            stack([sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)], axis=-1),
            stack([zero_tensor, sin(alpha), cos(alpha), d], axis=-1),
            stack([zero_tensor, zero_tensor, zero_tensor, one_tensor], axis=-1),
        ], axis=-2)
        
        return T_joint

    def getdhIthFrame(self, start, idx):
        return self.dhIthFrame(start + self.dh_values[0][idx], self.dh_values[1][idx], self.dh_values[2][idx], self.dh_values[3][idx])


    def buildDhTcpFrame(self, q_array):
        
        batch_size = q_array.shape[0]
        if self.use_torch:
            eye = torch.eye(4, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            eye = np.eye(4)[np.newaxis, :, :].repeat(batch_size, axis=0)
        
        dh_frames = eye

        thetas = q_array + self.dh_values[0]
        if self.use_torch:
            ds = self.dh_values[1].unsqueeze(0).repeat(batch_size, 1)
            alphas = self.dh_values[3].unsqueeze(0).repeat(batch_size, 1)
            as_ = self.dh_values[2].unsqueeze(0).repeat(batch_size, 1)
        else:
            ds = np.tile(self.dh_values[1], (batch_size, 1))
            alphas = np.tile(self.dh_values[3], (batch_size, 1))
            as_ = np.tile(self.dh_values[2], (batch_size, 1))
            
        for i in range(len(self.joint_limits)):
            tmp_dh_ith = self.dhIthFrame(thetas[:, i], ds[:, i], as_[:, i], alphas[:, i])
            if self.use_torch:
                dh_frames = torch.bmm(dh_frames, tmp_dh_ith)
            else:
                dh_frames = np.einsum('bij,bjk->bik', dh_frames, tmp_dh_ith)
        
        return dh_frames

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