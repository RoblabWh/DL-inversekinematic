from src.datahandler import DataHandler
from src.robot import Robot
from src.trainer import Trainer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.block(x)

class IK_Solver(nn.Module):
    def __init__(self, euler=False):
        super(IK_Solver, self).__init__()

        in_features = datahandler.get_input_shape()
        out_features = datahandler.get_output_shape()
        
        neurons = [ 2760, 1040, 2910, 3390, 2690, 340, 710, 1170, 690 ]
        layers = []
        input_dim = in_features
        for out_dim in neurons:
            layers.append(nn.Linear(input_dim, out_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            input_dim = out_dim
        layers.append(nn.Linear(input_dim, out_features))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
if __name__ == "__main__":
    datahandler = DataHandler(robot=Robot(robot='youbot'))

    datahandler.euler = True
    datahandler.relative = False
    datahandler.noised = False
    datahandler.normrange = "zeroToOne"

    # Instantiate the model
    model = IK_Solver()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.MSELoss()

    model.to(device)
    model.train() # Set the model to train mode

    datahandler.set_torch(True)
    trainer = Trainer(model, optimizer, criterion, device)
    trainer.tcp_loss = False
    trainer(datahandler, samples=1000000, epochs=100, batch_size=250)
    
    # Evaluate
    datahandler.compute_extreme_positions = False
    datahandler.set_torch(True)
    model.eval()
    tcp, gt_joints = datahandler(2000)
    pred = model(tcp)
    # Denormalize joint positions and TCP and get it off the GPU and back to numpy arrays
    gt_pos = datahandler.denorm_joint(gt_joints).cpu().numpy()
    if trainer.tcp_loss:
        pred = datahandler.denorm_tcp(pred).cpu().detach().numpy()
    else:
        pred = datahandler.denorm_joint(pred).cpu().detach().numpy()
    datahandler.set_torch(False)

    # Calculate the error
    if trainer.tcp_loss:
        error = np.linalg.norm(pred - gt_pos, axis=1)
    else:
        error = np.linalg.norm(pred - gt_joints, axis=1)