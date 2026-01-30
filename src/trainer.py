from src.datahandler import DataHandler
import torch
import torch.optim as optim
from tqdm import tqdm

class Trainer():

    def __init__(self, model, optimizer, criterion, device,
                 scheduler_type="plateau", scheduler_kwargs=None, grad_clip=None,
                 trackio_project="ik-dl", trackio_space_id=None,
                 compute_runtime_metrics=False, eval_interval=5, log_batch_loss=False,
                 run_name=None,
                 compute_llc=False, llc_interval=10, llc_num_chains=3, llc_num_draws=100,
                 llc_num_burnin_steps=200, llc_num_steps_bw_draws=5):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.tcp_loss = False
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.grad_clip = grad_clip
        # Trackio integration
        self.trackio_project = trackio_project
        self.trackio_space_id = trackio_space_id
        # Runtime metrics
        self.compute_runtime_metrics = compute_runtime_metrics
        self.eval_interval = eval_interval
        self.log_batch_loss = log_batch_loss
        # Checkpoint saving
        self.run_name = run_name
        # LLC (grokking detection)
        self.compute_llc = compute_llc
        self.llc_interval = llc_interval
        self.llc_num_chains = llc_num_chains
        self.llc_num_draws = llc_num_draws
        self.llc_num_burnin_steps = llc_num_burnin_steps
        self.llc_num_steps_bw_draws = llc_num_steps_bw_draws

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

    def __call__(self, datahandler : DataHandler, samples : int, epochs : int, batch_size : int, validation_split=0.05, plot_training=False):
        steps_per_epoch = max(1, int(samples * (1 - validation_split)) // batch_size)
        scheduler = self._create_scheduler(epochs, steps_per_epoch)
        is_onecycle = self.scheduler_type == "onecycle"
        is_plateau = self.scheduler_type == "plateau"

        if self.tcp_loss and (datahandler.relative or datahandler.noised):
            print("TCP Loss is currently only supported for non-relative data without noise.")
            return

        # Initialize Trackio if configured
        if self.trackio_project:
            import trackio

            # Build config automatically from available sources
            config = {}

            # Model architecture (if model has get_config)
            if hasattr(self.model, 'get_config'):
                config.update(self.model.get_config())

            # DataHandler settings
            config["robot"] = datahandler.robot.name
            config["euler"] = datahandler.euler
            config["normrange"] = datahandler.normrange.value
            config["relative"] = datahandler.relative
            config["noised"] = datahandler.noised

            # Training parameters
            config["samples"] = samples
            config["epochs"] = epochs
            config["batch_size"] = batch_size
            config["validation_split"] = validation_split

            # Trainer settings
            config["scheduler_type"] = self.scheduler_type
            config["tcp_loss"] = self.tcp_loss
            if self.grad_clip is not None:
                config["grad_clip"] = self.grad_clip

            trackio.init(
                project=self.trackio_project,
                config=config,
                space_id=self.trackio_space_id
            )

        # Determine effective run name for checkpoint saving
        # Use explicit run_name if set, otherwise use Trackio's auto-generated name
        if self.run_name is not None:
            effective_run_name = self.run_name
        elif self.trackio_project:
            from trackio.context_vars import current_run
            effective_run_name = current_run.get().name
        else:
            effective_run_name = None

        best_val_loss = float('inf')

        # Initialize LLC tracking
        llc_tracker = None
        if self.compute_llc:
            from src.grokking import LLCTracker
            llc_tracker = LLCTracker()

        for epoch in range(epochs):
            train_loader, val_loader = datahandler.get_data_loaders(samples, batch_size, validation_split)

            # Sync normalization bounds to model after first data generation
            if epoch == 0:
                datahandler.sync_normalization_to_model(self.model)

            self.model.train()
            train_loss = 0.0
            tqdm_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True, disable=not plot_training)

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

                if self.trackio_project and self.log_batch_loss:
                    trackio.log({"batch_loss": loss.item()})

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

            # Save checkpoint if this is the best model so far
            if effective_run_name is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                from src.models import save_checkpoint
                save_checkpoint(
                    self.model,
                    f"checkpoints/{effective_run_name}.pt",
                    robot_config={"name": datahandler.robot.name},
                    training_info={
                        "epoch": epoch + 1,
                        "val_loss": val_loss,
                        "train_loss": train_loss,
                        "samples": samples,
                    },
                )

            if is_plateau:
                scheduler.step(val_loss)
            elif not is_onecycle:
                scheduler.step(epoch)

            current_lr = self.optimizer.param_groups[0]['lr']
            if plot_training:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}')

            # Log epoch-level metrics to Trackio
            if self.trackio_project:
                trackio.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                })

            # Compute detailed runtime metrics periodically
            if self.compute_runtime_metrics and (epoch + 1) % self.eval_interval == 0:
                from src.metrics import compute_joint_metrics

                all_preds = []
                all_targets = []
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        outputs = self.model(inputs)
                        all_preds.append(datahandler.denorm_joint(outputs).cpu())
                        all_targets.append(datahandler.denorm_joint(targets).cpu())

                pred_joints = torch.cat(all_preds, dim=0).numpy()
                gt_joints = torch.cat(all_targets, dim=0).numpy()

                joint_metrics = compute_joint_metrics(gt_joints, pred_joints)

                if self.trackio_project:
                    trackio.log({
                        "mae_deg": joint_metrics["mae_total_deg"],
                        "rmse_deg": joint_metrics["rmse_total_deg"],
                        "success_rate_1deg": joint_metrics["success_rate_1deg"],
                        "success_rate_5deg": joint_metrics["success_rate_5deg"],
                    })

                if plot_training:
                    print(f"  → MAE: {joint_metrics['mae_total_deg']:.2f}°, Success<5°: {joint_metrics['success_rate_5deg']:.1%}")

            # Compute LLC at specified intervals
            if self.compute_llc and (epoch + 1) % self.llc_interval == 0:
                from src.grokking import estimate_llc

                llc_results = estimate_llc(
                    model=self.model,
                    loader=train_loader,  # Use train_loader for grokking detection
                    device=self.device,
                    criterion=self.criterion,
                    num_chains=self.llc_num_chains,
                    num_draws=self.llc_num_draws,
                    num_burnin_steps=self.llc_num_burnin_steps,
                    num_steps_bw_draws=self.llc_num_steps_bw_draws,
                )

                llc_metrics = llc_tracker.update(
                    llc_mean=llc_results["llc/mean"],
                    llc_std=llc_results["llc/std"],
                )

                if self.trackio_project:
                    llc_log = {
                        "llc/mean": llc_metrics["llc_mean"],
                        "llc/std": llc_metrics["llc_std"],
                    }
                    if "llc_change_rate" in llc_metrics:
                        llc_log["llc/change_rate"] = llc_metrics["llc_change_rate"]
                    trackio.log(llc_log)

                if plot_training:
                    print(f"  → LLC: {llc_metrics['llc_mean']:.4f} ± {llc_metrics['llc_std']:.4f}")

        # Finalize Trackio
        if self.trackio_project:
            trackio.finish()
