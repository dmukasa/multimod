"""Training loop for the ABMAP model."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import TargetNormalizer
from .metrics import compute_metrics


@dataclass
class EarlyStoppingConfig:
    patience: int = 5
    min_delta: float = 0.0


@dataclass
class SchedulerConfig:
    type: str = "cosine"
    min_lr: float = 1e-5
    t_max: int = 50


@dataclass
class TrainingHistory:
    epochs: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_metrics: List[Dict[str, float]] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        normalizer: TargetNormalizer,
        optimizer: torch.optim.Optimizer,
        *,
        device: torch.device,
        gradient_clip_val: float,
        scheduler_config: Optional[SchedulerConfig] = None,
        early_stopping: Optional[EarlyStoppingConfig] = None,
        save_dir: Optional[Path] = None,
    ) -> None:
        self.model = model.to(device)
        self.normalizer = normalizer
        self.optimizer = optimizer
        self.device = device
        self.gradient_clip_val = gradient_clip_val
        self.scheduler = self._build_scheduler(optimizer, scheduler_config)
        self.early_stopping = early_stopping or EarlyStoppingConfig()
        self.save_dir = save_dir
        self.history = TrainingHistory()
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_metric = float("-inf")
        self.epochs_without_improvement = 0

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer, config: Optional[SchedulerConfig]
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        if config is None:
            return None
        if config.type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(config.t_max, 1),
                eta_min=config.min_lr,
            )
        raise ValueError(f"Unsupported scheduler type: {config.type}")

    def _step_scheduler(self) -> None:
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                self.scheduler.step()
            else:
                self.scheduler.step()

    def train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(loader, desc="Training", leave=False):
            batch = {key: value.to(self.device) for key, value in batch.items()}
            targets = batch["target"].float()
            predictions = self.model(batch)
            loss = nn.functional.mse_loss(predictions, targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            self.optimizer.step()
            running_loss += loss.item() * targets.size(0)
        self._step_scheduler()
        return running_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        losses = []
        predictions_list = []
        targets_list = []
        for batch in tqdm(loader, desc="Validation", leave=False):
            batch = {key: value.to(self.device) for key, value in batch.items()}
            targets = batch["target"].float()
            predictions = self.model(batch)
            loss = nn.functional.mse_loss(predictions, targets)
            losses.append(loss.item() * targets.size(0))
            predictions_list.append(self.normalizer.decode(predictions.cpu()))
            targets_list.append(self.normalizer.decode(targets.cpu()))
        predictions_tensor = torch.cat(predictions_list)
        targets_tensor = torch.cat(targets_list)
        metrics = compute_metrics(predictions_tensor, targets_tensor)
        metrics["loss"] = sum(losses) / len(loader.dataset)
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, max_epochs: int) -> TrainingHistory:
        for epoch in range(1, max_epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            eval_metrics = self.evaluate(val_loader)

            self.history.epochs.append(epoch)
            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(eval_metrics["loss"])
            self.history.val_metrics.append(eval_metrics)

            spearman = eval_metrics.get("spearman", float("-inf"))
            if spearman > self.best_metric + self.early_stopping.min_delta:
                self.best_metric = spearman
                self.best_state = {key: value.cpu() for key, value in self.model.state_dict().items()}
                self.epochs_without_improvement = 0
                if self.save_dir is not None:
                    self.save_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(self.best_state, self.save_dir / "best_model.pt")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.early_stopping.patience:
                    break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.history

    def save_checkpoint(self, path: Path) -> None:
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "history": self.history.__dict__,
            "best_metric": self.best_metric,
        }
        torch.save(state, path)
