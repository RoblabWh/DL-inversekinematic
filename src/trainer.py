from src.datahandler import DataHandler
import torch
from tqdm import tqdm

class Trainer():
    
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.tcp_loss = False
        
    def __call__(self, datahandler : DataHandler, samples : int, epochs : int, batch_size : int, validation_split=0.05):
        if self.tcp_loss and (datahandler.relative or datahandler.noised):
            print("TCP Loss is currently only supported for non-relative data without noise.")
            return
                
        for epoch in range(epochs):
            train_loader, val_loader = datahandler.get_data_loaders(samples, batch_size, validation_split)
            self.model.train()
            train_loss = 0.0
            tqdm_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
            
            for batch in tqdm_iterator:
                inputs, target = batch
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                
                if self.tcp_loss:
                    # Denormalize the outputs and calculate TCP
                    denorm_outputs = datahandler.denorm_joint(outputs)
                    pred_tcp = datahandler.get_tcp(denorm_outputs)
                    norm_pred_tcp = datahandler.norm_tcp(pred_tcp)
                    outputs = norm_pred_tcp
                    target = inputs
                
                loss = self.criterion(outputs, target)
                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                tqdm_iterator.set_postfix(loss=loss.item())
                
            train_loss /= len(train_loader)
            
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, target in val_loader:
                    output = self.model(inputs)
                    val_loss += self.criterion(output, target).item()
                    
            val_loss /= len(val_loader)
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')