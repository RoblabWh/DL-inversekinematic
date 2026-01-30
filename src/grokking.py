"""LLC-based grokking detection using devinterp."""

import torch
from torch.utils.data import DataLoader
from typing import Optional


def _compute_init_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> float:
    """Compute initial loss for LLCEstimator."""
    was_training = model.training
    model.train(False)  # Set to evaluation mode
    total_loss = 0.0
    num_samples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # Use sum reduction to accumulate, then divide by total samples
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)
    model.train(was_training)  # Restore training mode
    return total_loss / num_samples


def estimate_llc(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    num_chains: int = 5,
    num_draws: int = 50,
    num_burnin_steps: int = 10,
    num_steps_bw_draws: int = 1,
) -> dict:
    """Estimate Local Learning Coefficient for current model state.

    Args:
        model: The neural network model.
        loader: DataLoader for computing loss.
        device: Device to run on.
        criterion: Loss function (must match training criterion).
        num_chains: Number of MCMC chains (more = better estimates, slower).
        num_draws: Samples per chain.
        num_burnin_steps: Steps to discard before sampling.
        num_steps_bw_draws: Steps between samples.

    Returns:
        Dictionary with LLC statistics.
    """
    import warnings
    warnings.filterwarnings("ignore", message="If you're setting a nbeta or temperature")

    from devinterp.slt.sampler import sample, LLCEstimator
    from devinterp.optim import SGLD
    from devinterp.utils import default_nbeta

    # Compute n*beta scaling
    nbeta = default_nbeta(loader)

    # Compute initial loss for LLCEstimator
    init_loss = _compute_init_loss(model, loader, device, criterion)

    # Create evaluate function for SGLD sampling
    def evaluate_fn(model, batch):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        return loss, outputs

    # Create LLC estimator callback
    llc_estimator = LLCEstimator(
        num_chains=num_chains,
        num_draws=num_draws,
        nbeta=nbeta,
        init_loss=init_loss,
    )

    # Run SGLD sampling
    sample(
        model=model,
        loader=loader,
        evaluate=evaluate_fn,
        callbacks=[llc_estimator],
        sampling_method=SGLD,
        optimizer_kwargs={"lr": 1e-5, "nbeta": nbeta},
        num_chains=num_chains,
        num_draws=num_draws,
        num_burnin_steps=num_burnin_steps,
        num_steps_bw_draws=num_steps_bw_draws,
        device=device,
        verbose=False,
    )

    return llc_estimator.get_results()


class LLCTracker:
    """Track LLC values over training epochs."""

    def __init__(self, window_size: int = 5):
        self.llc_history = []
        self.window_size = window_size

    def update(self, llc_mean: float, llc_std: float) -> dict:
        """Record LLC and compute derived metrics."""
        self.llc_history.append({"mean": llc_mean, "std": llc_std})

        metrics = {
            "llc_mean": llc_mean,
            "llc_std": llc_std,
        }

        # Compute trend if we have enough history
        if len(self.llc_history) >= self.window_size:
            recent = [h["mean"] for h in self.llc_history[-self.window_size:]]
            older = [h["mean"] for h in self.llc_history[-2*self.window_size:-self.window_size]]

            if older:
                # LLC change rate (negative = complexity decreasing = grokking signal)
                metrics["llc_change_rate"] = (sum(recent)/len(recent) - sum(older)/len(older))

        return metrics

    def reset(self):
        self.llc_history = []
