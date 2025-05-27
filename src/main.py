from src.datahandler import DataHandler
from src.robot import Robot
from src.trainer import Trainer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna

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
    def __init__(self, in_shape, out_shape, neurons):
        super(IK_Solver, self).__init__()

        layers = []
        input_dim = in_shape
        for out_dim in neurons:
            layers.append(nn.Linear(input_dim, out_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            input_dim = out_dim
        layers.append(nn.Linear(input_dim, out_shape))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

def normal_training():
    datahandler = DataHandler(robot=Robot(robot='youbot'))

    datahandler.euler = False
    datahandler.relative = False
    datahandler.noised = False
    datahandler.normrange = "zeroToOne"
    tcp, gt_joints = datahandler(1000000)

    # Instantiate the model
    in_shape = datahandler.get_input_shape()
    out_shape = datahandler.get_output_shape()
    neurons = [2760, 1040, 2910, 3390, 2690, 340, 710, 1170, 690]
    model = IK_Solver(in_shape, out_shape, neurons)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.MSELoss()

    model.to(device)
    model.train()  # Set the model to train mode

    datahandler.set_torch(True)
    trainer = Trainer(model, optimizer, criterion, device)
    trainer.tcp_loss = True
    trainer(datahandler, samples=1000, epochs=1, batch_size=250)

    # Evaluate
    datahandler.compute_extreme_positions = False
    datahandler.set_torch(True)
    model.eval()
    tcp, gt_joints = datahandler(5000)
    pred = model(tcp)
    # Denormalize joint positions and TCP and get it off the GPU and back to numpy arrays
    gt_pos = datahandler.denorm_joint(gt_joints).cpu().numpy()
    pred = datahandler.denorm_joint(pred).cpu().detach().numpy()
    datahandler.set_torch(False)

    mae_per_joint = np.mean(np.abs(gt_pos - pred), axis=0)
    mae_total = np.mean(mae_per_joint)
    rmse_total = np.sqrt(np.mean((gt_pos - pred) ** 2))

def objective(trial):
    # Sample hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    num_layers = trial.suggest_int("num_layers", 3, 10)
    neurons = [
        trial.suggest_int(f"n_units_layer_{i}", 64, 4096)
        for i in range(num_layers)
    ]
    use_euler = trial.suggest_categorical('euler', [True, False])
    use_negativeToOne = trial.suggest_categorical('use_negativeToOne', [True, False])
    samples = trial.suggest_int('samples', 10000, 1000000)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    epochs = 20

    # Create data handler
    datahandler = DataHandler(robot=Robot(robot='fiveaxis'))
    datahandler.euler = use_euler
    datahandler.relative = False
    datahandler.noised = False
    datahandler.normrange = "zeroToOne" if not use_negativeToOne else "minus1to1"

    # Just used for initital maximum computation
    tcp, gt_joints = datahandler(1000000)

    # Instantiate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_shape = datahandler.get_input_shape()
    out_shape = datahandler.get_output_shape()
    model = IK_Solver(in_shape, out_shape, neurons).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()  # Set the model to train mode

    datahandler.set_torch(True)
    trainer = Trainer(model, optimizer, criterion, device)
    trainer.tcp_loss = False
    trainer(datahandler, samples=samples, epochs=epochs, batch_size=batch_size)

    # Evaluate
    datahandler.compute_extreme_positions = False
    datahandler.set_torch(True)
    model.eval()
    tcp, gt_joints = datahandler(100000)
    pred = model(tcp)
    # Denormalize joint positions and TCP and get it off the GPU and back to numpy arrays
    gt_pos = datahandler.denorm_joint(gt_joints).cpu().numpy()
    pred = datahandler.denorm_joint(pred).cpu().detach().numpy()
    datahandler.set_torch(False)

    mae_per_joint = np.mean(np.abs(gt_pos - pred), axis=0)
    mae_total = np.mean(mae_per_joint)
    rmse_total = np.sqrt(np.mean((gt_pos - pred) ** 2)) # Drop for now

    return mae_total

def data_test():
    # Create data handler
    datahandler = DataHandler(robot=Robot(robot='youbot'))
    use_negativeToOne = True
    datahandler.euler = False
    datahandler.relative = False
    datahandler.noised = True
    datahandler.normrange = "zeroToOne" if not use_negativeToOne else "minus1to1"

    # Just used for initital maximum computation
    tcp, gt_joints = datahandler(100)

    from src.robot_animation import RobotAnimation

    drawer = RobotAnimation(datahandler.robot)
    # Nummer des Datensatzes
    for t in range(30):
        drawer.draw_robot([gt_joints[t], tcp[t,6:11]])

if __name__ == "__main__":
    data_test()
    #normal_training()
    # study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=20)
    #
    # # Print results
    # print("Best MAE:", study.best_value)
    # print("Best hyperparameters:", study.best_params)