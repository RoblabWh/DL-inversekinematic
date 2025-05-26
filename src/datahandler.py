import numpy as np
import time
import math
import copy
import sys

import torch
import warnings
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
from src.robot import Robot

class DataHandler(object):

    def __init__(self, robot : Robot, euler=False, verbose=False):

        self.euler = euler
        self.compute_extreme_positions = True
        self.split = 4
        self.robot = robot

        self.dmax = np.around(np.radians(np.max(self.robot.joint_limits)), decimals=4)
        self.dmin = np.around(np.radians(np.min(self.robot.joint_limits)), decimals=4)
        
        self.normrange = "minus1to1"
        self.y_max = -float('inf')
        self.y_min = float('inf')
        self.x_max = -float('inf')
        self.x_min = float('inf')
        self.z_max = -float('inf')
        self.z_min = float('inf')
        self.rotatemin = -1
        self.rotatemax = 1
        
        self.verbose = verbose
        self.use_torch = False
        self.device = None
        
        self.datahandler_generated = False
        self.relative = False
        self.noised = False
        self.normalize = True
        self.init_maxima = False # Get's set to True when the first data is generated
        
    def set_torch(self, use_torch):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not use_torch:
            if isinstance(self.y_max, torch.Tensor):
                self.y_max = float(self.y_max.detach().cpu().numpy())
                self.y_min = float(self.y_min.detach().cpu().numpy())
                self.x_max = float(self.x_max.detach().cpu().numpy())
                self.x_min = float(self.x_min.detach().cpu().numpy())
                self.z_max = float(self.z_max.detach().cpu().numpy())
                self.z_min = float(self.z_min.detach().cpu().numpy())
                self.rotatemin = float(self.rotatemin.detach().cpu().numpy())
                self.rotatemax = float(self.rotatemax.detach().cpu().numpy())
        elif use_torch:
            if isinstance(self.y_max, float):
                self.y_max = torch.tensor(self.y_max, dtype=torch.float32, device=self.device)
                self.y_min = torch.tensor(self.y_min, dtype=torch.float32, device=self.device)
                self.x_max = torch.tensor(self.x_max, dtype=torch.float32, device=self.device)
                self.x_min = torch.tensor(self.x_min, dtype=torch.float32, device=self.device)
                self.z_max = torch.tensor(self.z_max, dtype=torch.float32, device=self.device)
                self.z_min = torch.tensor(self.z_min, dtype=torch.float32, device=self.device)
                self.rotatemin = torch.tensor(self.rotatemin, dtype=torch.float32, device=self.device)
                self.rotatemax = torch.tensor(self.rotatemax, dtype=torch.float32, device=self.device)
            
        self.use_torch = use_torch
        self.robot.set_torch(use_torch, device=self.device)

    def get_roll_pitch_yaw(self, rotation):
        if self.use_torch:
            sqrt = torch.sqrt
            where = torch.where
            atan2 = torch.atan2
            zeros_like = torch.zeros_like
            isclose = torch.isclose
            pi = torch.tensor(math.pi, device=rotation.device)
        else:
            sqrt = np.sqrt
            where = np.where
            atan2 = np.arctan2
            zeros_like = np.zeros_like
            isclose = np.isclose
            pi = np.pi
        
        sy = sqrt(rotation[:, 0, 0] ** 2 + rotation[:, 1, 0] ** 2)
    
        singular = sy < 1e-6

        roll = where(singular,
                        atan2(-rotation[:, 1, 2], rotation[:, 1, 1]),
                        atan2(rotation[:, 2, 1], rotation[:, 2, 2]))
        
        pitch = where(singular,
                            atan2(-rotation[:, 2, 0], sy),
                            atan2(-rotation[:, 2, 0], sy))
        
        yaw = where(singular,
                        zeros_like(roll),
                        atan2(rotation[:, 1, 0], rotation[:, 0, 0]))
        
        # TODO Maybe round here like this: np.around(roll, decimals=6), np.around(pitch, decimals=6), np.around(yaw, decimals=6)
        return roll, pitch, yaw

    
    def calc_tcp(self, positions):

        if self.use_torch:
            round = torch.round
            cat = torch.cat
            stack = torch.stack
            if not isinstance(positions, torch.Tensor):
                positions = torch.tensor(positions, dtype=torch.float32, device=self.device)
        else:
            round = np.around
            cat = np.concatenate
            stack = np.stack

        batch_size = positions.shape[0]
        frames = self.robot.buildDhTcpFrame(positions)
        frames_flat = frames.reshape(batch_size, -1)[:, :12]

        xyz = frames_flat[:, 3::4]

        rot_cols = [
            frames_flat[:, 0:3],
            frames_flat[:, 4:7],
            frames_flat[:, 8:11]
        ]

        rotation = round(stack(rot_cols, 1).reshape(batch_size, -1), decimals=9)

        tcp_cols = []
        for j in range(12):
            if (j != 3) and (j != 7) and (j != 11):
                #tcp[:, j] = rotation[:, j - int(j / 4)]
                tcp_cols.append(rotation[:, j - int(j / 4)].unsqueeze(1) if self.use_torch else np.expand_dims(rotation[:, j - int(j / 4)], axis=1))
            else:
                #tcp[:, j] = xyz[:, j % 3]
                tcp_cols.append(xyz[:, j % 3].unsqueeze(1) if self.use_torch else np.expand_dims(xyz[:, j % 3], axis=1))

        tcp = cat(tcp_cols, axis=1)

        return tcp

    def calc_xyz_euler(self, positions):
        if self.use_torch:
            stack = torch.stack
        else:
            stack = np.stack
        start1 = time.time()
        tcp_time = 0

        #batch_size = positions.shape[0]
        #tcp = np.zeros((batch_size, 6)) if not self.use_torch else torch.zeros((batch_size, 6), dtype=torch.float32, device=self.device)
        start = time.time()

        # Build the DH frames in one batch
        if self.use_torch and not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.float32, device=self.device)

        frames = self.robot.buildDhTcpFrame(positions)
        end = time.time()
        tcp_time += end - start

        # Extract xyz and euler angles
        xyz = frames[:, :3, 3]
        rotations = frames[:, :3, :3]

        roll, pitch, yaw = self.get_roll_pitch_yaw(rotations)

        tcp = stack((xyz[:, 0], xyz[:, 1], xyz[:, 2], roll, pitch, yaw), 1)

        # tcp[:, 0:3] = xyz
        # tcp[:, 3] = roll
        # tcp[:, 4] = pitch
        # tcp[:, 5] = yaw

        end2 = time.time()

        if self.verbose:
            print("DH Transformation: %.2f Sekunden" %tcp_time)
            print("Euler Winkel: %.2f Sekunden" %(end2 - start1 - tcp_time))

        return tcp

    def get_tcp(self, positions):
        if not self.euler:
            return self.calc_tcp(positions)
        else:
            return self.calc_xyz_euler(positions)

    def get_extreme_positions(self):
        joint_grids = []
        for joint_range in self.robot.joint_limits:
            joint_grids.append(np.linspace(joint_range[1], joint_range[0], self.split))
        
        joint_combinations = np.array(list(product(*joint_grids)))
        
        return joint_combinations
    
    def generate_n_positions(self, n=100):
        degree_joint_pos = []
        extremes_computed = False

        if self.compute_extreme_positions and self.split ** len(self.robot.joint_limits) > n:
            warnings.warn(f"Too few samples to generate extreme positions." 
                          f" Either turn off generation of extreme positions or generate more Data." 
                          f" {self.split ** len(self.robot.joint_limits)} extreme positions must be generated" 
                          f" but you only generate {n} samples.")
            print("Extreme Positions not computed!")
        elif self.compute_extreme_positions:
            n -= self.split ** len(self.robot.joint_limits)
            extremes = self.get_extreme_positions()
            extremes_computed = True

        for joint_range in self.robot.joint_limits:
            degree_joint_pos.append(np.random.randint(joint_range[1], joint_range[0] + 1, n))

        positions = np.transpose(degree_joint_pos)

        if extremes_computed:
            positions = np.concatenate((positions, extremes))

        return positions

    def generate_data(self, datapoints, sigma=60):
        """This function generates data for the robot simulation. 
        It generates the joint angles and the TCPs.

        Args:
            datapoints (int): Number of datapoints to generate.
            relative (bool, optional): Should the relative TCPs be returned. Defaults to False.
            noised (bool, optional): Should the target positions be noised versions of the GT. Defaults to False.
            sigma (int, optional): If noised used, determines the sigma of the normal distribution. Defaults to 60.

        Returns:
            _type_: _description_
        """

        start = time.time()

        positions = self.generate_n_positions(datapoints)
        positions = np.around(np.radians(positions), decimals=4)

        if self.noised:
            # Generate noise for each joint
            noise = np.random.normal(0, np.radians(sigma), positions.shape)
            noised_pos = positions + noise
            for i in range(len(self.robot.joint_limits)):
                noised_pos[:, i] = np.clip(noised_pos[:, i], np.radians(self.robot.joint_limits[i][1]), np.radians(self.robot.joint_limits[i][0]))
            positions_end = noised_pos
        elif self.relative:
            positions_end = self.generate_n_positions(datapoints)
            
        if self.noised or self.relative:
            positions_end = np.around(np.radians(positions_end), decimals=4)

        if self.verbose:
            print("Generierung der Gelenkwinkel: %.2f Sekunden" %(time.time() - start))

        start = time.time()

        tcp = self.get_tcp(positions)
        if self.noised or self.relative:
            tcp_end = self.get_tcp(positions_end)
        
        if self.verbose:
            print("Generierung des TCPs: %.2f Sekunden" %(time.time() - start))

        if self.relative:
            if self.use_torch:
                subtract = torch.subtract
            else:
                subtract = np.subtract
            tcp_end = subtract(tcp_end, tcp)
        
        if self.relative or self.noised:
            return positions, tcp, positions_end, tcp_end

        return positions, tcp
        
    def norm_joint(self, data):
        '''
        Normalizes joint angles
        '''
        
        if self.use_torch:
            deg2rad = torch.deg2rad
        else:
            deg2rad = np.deg2rad

        num_joints = self.robot.get_num_axis()
        out = []

        for i in range(num_joints):
            joint_max, joint_min = self.robot.joint_limits[i]
            max_rad = deg2rad(joint_max if not self.use_torch else torch.tensor(joint_max, dtype=torch.float32, device=data.device))
            min_rad = deg2rad(joint_min if not self.use_torch else torch.tensor(joint_min, dtype=torch.float32, device=data.device))
            if self.normrange == "minus1to1":
                joint = ((2 * (data[:, i] - min_rad) / (max_rad - min_rad)) - 1)
            else:
                joint = (data[:, i] - min_rad) / (max_rad - min_rad)

            out.append(joint.unsqueeze(1) if self.use_torch else np.expand_dims(joint, axis=1))

        return torch.cat(out, dim=1) if self.use_torch else np.concatenate(out, axis=1)

    def denorm_joint(self, data):
        '''
        Denormalizes joint angles
        '''
        if self.use_torch:
            deg2rad = torch.deg2rad
        else:
            deg2rad = np.deg2rad
            
        num_joints = self.robot.get_num_axis()
        out = []

        for i in range(num_joints):
            joint_max, joint_min = self.robot.joint_limits[i]
            max_rad = deg2rad(joint_max if not self.use_torch else torch.tensor(joint_max, dtype=torch.float32, device=data.device))
            min_rad = deg2rad(joint_min if not self.use_torch else torch.tensor(joint_min, dtype=torch.float32, device=data.device))
            if self.normrange == "minus1to1":
                joint = (((data[:, i] + 1) * (max_rad - min_rad)) / 2) + min_rad
            else:
                joint = (data[:, i] * (max_rad - min_rad)) + min_rad

            out.append(joint.unsqueeze(1) if self.use_torch else np.expand_dims(joint, axis=1))

        return torch.cat(out, dim=1) if self.use_torch else np.concatenate(out, axis=1)

    def set_maxima(self, tcp):
        if not self.init_maxima:
            if tcp.shape[0] < 5000:
                warnings.warn("Not enough data to find maxima. Please generate more data or turn off normalization.")
                return False
            if self.use_torch:
                amax = torch.amax
                amin = torch.amin
            else:
                amax = np.amax
                amin = np.amin
            
            if not self.euler:
                x_s = tcp[:, 3].flatten()
                y_s = tcp[:, 7].flatten()
                z_s = tcp[:, 11].flatten()
            else:
                x_s = tcp[:, 0].flatten()
                y_s = tcp[:, 1].flatten()
                z_s = tcp[:, 2].flatten()

            xmax = amax(x_s)
            xmin = amin(x_s)
            ymax = amax(y_s)
            ymin = amin(y_s)
            zmax = amax(z_s)
            zmin = amin(z_s)

            if xmax > self.x_max: self.x_max = xmax
            if xmin < self.x_min: self.x_min = xmin
            if ymax > self.y_max: self.y_max = ymax
            if ymin < self.y_min: self.y_min = ymin
            if zmax > self.z_max: self.z_max = zmax
            if zmin < self.z_min: self.z_min = zmin
            
            self.init_maxima = True
        return True
    
    def norm_euler(self, tcp):
        '''
        Normalizes XYZ and euler angles
        '''
        
        if self.use_torch:
            pi = torch.pi
            cat = torch.cat
            stack = torch.stack
        else:
            pi = np.pi
            cat = np.concatenate
            stack = np.stack
            
        if self.normrange == "minus1to1":
            x = (2 * (tcp[:, 0] - self.x_min) / (self.x_max - self.x_min)) - 1
            y = (2 * (tcp[:, 1] - self.y_min) / (self.y_max - self.y_min)) - 1
            z = (2 * (tcp[:, 2] - self.z_min) / (self.z_max - self.z_min)) - 1

            # Normalize Euler angles to [-1, 1] based on range [-π, π]
            euler = tcp[:, 3:6] / pi
        else:
            x = (tcp[:, 0] - self.x_min) / (self.x_max - self.x_min)
            y = (tcp[:, 1] - self.y_min) / (self.y_max - self.y_min)
            z = (tcp[:, 2] - self.z_min) / (self.z_max - self.z_min)

            # Normalize Euler angles to [0, 1] from [-π, π]
            euler = (tcp[:, 3:6] + pi) / (2 * pi)

        xyz = stack((x, y, z), 1)

        return cat((xyz, euler), axis=1)
    
    def norm_xyz(self, tcp):
        '''
        Normalizes tcp
        '''
        if self.use_torch:
            cat = torch.cat
        else:
            cat = np.concatenate

        if self.normrange == "minus1to1":
            x = (2 * (tcp[:, 3] - self.x_min) / (self.x_max - self.x_min)) - 1
            y = (2 * (tcp[:, 7] - self.y_min) / (self.y_max - self.y_min)) - 1
            z = (2 * (tcp[:, 11] - self.z_min) / (self.z_max - self.z_min)) - 1
        else:
            x = (tcp[:, 3] - self.x_min) / (self.x_max - self.x_min)
            y = (tcp[:, 7] - self.y_min) / (self.y_max - self.y_min)
            z = (tcp[:, 11] - self.z_min) / (self.z_max - self.z_min)

        tcp_ = [
            tcp[:, 0:3],
            x.unsqueeze(1) if self.use_torch else np.expand_dims(x, 1),
            tcp[:, 4:7],
            y.unsqueeze(1) if self.use_torch else np.expand_dims(y, 1),
            tcp[:, 8:11],
            z.unsqueeze(1) if self.use_torch else np.expand_dims(z, 1)
        ]

        return cat(tcp_, 1)

    def norm_tcp(self, tcp):
        if not self.euler:
            return self.norm_xyz(tcp)
        else:
            return self.norm_euler(tcp)
    
    def denorm_euler(self, tcp):
        '''
        Denormalizes XYZ and euler angles
        '''
        if self.use_torch:
            pi = torch.pi
            cat = torch.cat
            stack = torch.stack
        else:
            pi = np.pi
            cat = np.concatenate
            stack = np.stack

        if self.normrange == "minus1to1":
            x = (tcp[:, 0] + 1) * (self.x_max - self.x_min) / 2 + self.x_min
            y = (tcp[:, 1] + 1) * (self.y_max - self.y_min) / 2 + self.y_min
            z = (tcp[:, 2] + 1) * (self.z_max - self.z_min) / 2 + self.z_min

            euler = (tcp[:, 3:6] + 1) * (pi - (-pi)) / 2 + (-pi)
        else:
            x = tcp[:, 0] * (self.x_max - self.x_min) + self.x_min
            y = tcp[:, 1] * (self.y_max - self.y_min) + self.y_min
            z = tcp[:, 2] * (self.z_max - self.z_min) + self.z_min

            euler = tcp[:, 3:6] * (pi - (-pi)) + (-pi)

        xyz = stack((x, y, z), 1)

        return cat((xyz, euler), axis=1)
    
    def denorm_xyz(self, tcp):
        '''
        Denormalizes tcp
        '''
        if self.use_torch:
            cat = torch.cat
        else:
            cat = np.concatenate

        if self.normrange == "minus1to1":
            x = (tcp[:, 3] + 1) * (self.x_max - self.x_min) / 2 + self.x_min
            y = (tcp[:, 7] + 1) * (self.y_max - self.y_min) / 2 + self.y_min
            z = (tcp[:, 11] + 1) * (self.z_max - self.z_min) / 2 + self.z_min
        else:
            x = tcp[:, 3] * (self.x_max - self.x_min) + self.x_min
            y = tcp[:, 7] * (self.y_max - self.y_min) + self.y_min
            z = tcp[:, 11] * (self.z_max - self.z_min) + self.z_min

        tcp_ = [
            tcp[:, 0:3],
            x.unsqueeze(1) if self.use_torch else np.expand_dims(x, 1),
            tcp[:, 4:7],
            y.unsqueeze(1) if self.use_torch else np.expand_dims(y, 1),
            tcp[:, 8:11],
            z.unsqueeze(1) if self.use_torch else np.expand_dims(z, 1)
        ]

        return cat(tcp_, 1)
    
    def denorm_tcp(self, tcp):
        if self.euler:
            return self.denorm_euler(tcp)
        else:
            return self.denorm_xyz(tcp)
    
    def __call__(self, batch_size, relative=None, noised=None, sigma=60, normalize=None, concat=True):
        begin = time.time()
        
        self.relative = relative if relative is not None else self.relative
        self.noised = noised if noised is not None else self.noised
        self.normalize = normalize if normalize is not None else self.normalize
        
        # Generate data with or without noise and relative positions
        if self.relative or self.noised:
            jpos, tcp, njpos, ntcp = self.generate_data(batch_size, sigma)
        else:
            jpos, tcp = self.generate_data(batch_size)
            
        # Convert to torch tensors if needed
        if self.use_torch:
            jpos = torch.tensor(jpos, dtype=torch.float32, device=self.device)
            #tcp = torch.tensor(tcp, dtype=torch.float32, device=self.device)
            if self.relative or self.noised:
                njpos = torch.tensor(njpos, dtype=torch.float32, device=self.device)
                #ntcp = torch.tensor(ntcp, dtype=torch.float32, device=self.device)
            
        # Normalize data if needed
        if self.normalize:
            start = time.time()
            if self.relative or self.noised:
                jpos = self.norm_joint(jpos)
                njpos = self.norm_joint(njpos)
                if self.set_maxima(tcp):
                    tcp = self.norm_tcp(tcp)
                    ntcp = self.norm_tcp(ntcp)
            else:
                jpos = self.norm_joint(jpos)
                if self.set_maxima(tcp):
                    tcp = self.norm_tcp(tcp)
            normalizing_time = time.time() - start
            
        # Print time if verbose
        if self.verbose:
            if self.normalize:
                print("Normalisierung: %.2f Sekunden" %(normalizing_time))
            print("Gesamt:  %.2f Sekunden" %(time.time() - begin))
        
        # Return data in the correct format
        if self.relative or self.noised:
            if concat:
                if self.use_torch:
                    concat = torch.cat
                else:
                    concat = np.concatenate
                return concat((jpos, tcp, ntcp), axis=1), njpos
            else:
                return tcp, jpos, ntcp, njpos
        else:
            return tcp, jpos
        
    def get_input_shape(self):
        '''
        Calculates the input shape of the neural network. This depends on the euler flag, aswell as on the method used for
        supervised learning.
        '''
        if self.relative or self.noised:
            
            input_shape = 12 + self.robot.get_num_axis() if self.euler else 24 + self.robot.get_num_axis()
        else:
            input_shape = 6 if self.euler else 12
            
        return input_shape
    
    def get_output_shape(self):
        return self.robot.get_num_axis()
    
    def get_data_loaders(self, samples, batch_size, validation_split=0.05):
        inputs, targets = self(batch_size=samples, relative=self.relative, noised=self.noised, normalize=True, concat=True)
        
        # Split data into training and validation set
        val_size = int(samples * validation_split)
        train_size = samples - val_size
        batches = train_size // batch_size
        
        train_dataset = TensorDataset(inputs[:-val_size], targets[:-val_size])
        val_dataset = TensorDataset(inputs[-val_size:], targets[-val_size:])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        
        return train_loader, val_loader
