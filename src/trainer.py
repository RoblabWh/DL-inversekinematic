from src.datahandler import DataHandler
import torch
import torch.optim as optim
from tqdm import tqdm

class Trainer():

    def __init__(self, model, optimizer, criterion, device,
                 scheduler_type="plateau", scheduler_kwargs=None, grad_clip=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.tcp_loss = False
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.grad_clip = grad_clip

    def _create_scheduler(self, epochs, steps_per_epoch):
        """Create learning rate scheduler based on type."""
        if self.scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10,
                **self.scheduler_kwargs
            )
        elif self.scheduler_type == "cosine":
            T_0 = self.scheduler_kwargs.get("T_0", 10)
            T_mult = self.scheduler_kwargs.get("T_mult", 2)
            eta_min = self.scheduler_kwargs.get("eta_min", 1e-6)
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min,
            )
        elif self.scheduler_type == "onecycle":
            max_lr = self.scheduler_kwargs.get("max_lr", self.optimizer.defaults['lr'])
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=max_lr,
                total_steps=epochs * steps_per_epoch,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

    def __call__(self, datahandler : DataHandler, samples : int, epochs : int, batch_size : int, validation_split=0.05):
        steps_per_epoch = max(1, int(samples * (1 - validation_split)) // batch_size)
        scheduler = self._create_scheduler(epochs, steps_per_epoch)
        is_onecycle = self.scheduler_type == "onecycle"
        is_plateau = self.scheduler_type == "plateau"

        if self.tcp_loss and (datahandler.relative or datahandler.noised):
            print("TCP Loss is currently only supported for non-relative data without noise.")
            return

        for epoch in range(epochs):
            train_loader, val_loader = datahandler.get_data_loaders(samples, batch_size, validation_split)

            # Sync normalization bounds to model after first data generation
            if epoch == 0:
                datahandler.sync_normalization_to_model(self.model)

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
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                if is_onecycle:
                    scheduler.step()
                train_loss += loss.item()
                tqdm_iterator.set_postfix(loss=loss.item())

            train_loss /= len(train_loader)

            self.model.train(False)
            val_loss = 0
            with torch.no_grad():
                for inputs, target in val_loader:
                    output = self.model(inputs)

                    if self.tcp_loss:
                        # Use same loss computation as training
                        denorm_output = datahandler.denorm_joint(output)
                        pred_tcp = datahandler.get_tcp(denorm_output)
                        norm_pred_tcp = datahandler.norm_tcp(pred_tcp)
                        output = norm_pred_tcp
                        target = inputs

                    val_loss += self.criterion(output, target).item()

            val_loss /= len(val_loader)

            if is_plateau:
                scheduler.step(val_loss)
            elif not is_onecycle:
                scheduler.step(epoch)

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}')
