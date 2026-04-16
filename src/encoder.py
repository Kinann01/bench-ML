#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from constants import PAD_VALUE

class DilatedConvBlock(nn.Module):

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))

        if out.size(-1) != residual.size(-1):
            out = out[..., :residual.size(-1)]

        return out + residual


class TSEncoder(nn.Module):

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64,
                 repr_dim: int = 320, depth: int = 10, kernel_size: int = 3):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            DilatedConvBlock(hidden_dim, kernel_size, dilation=2 ** i)
            for i in range(depth)
        ])

        self.repr_projection = nn.Linear(hidden_dim, repr_dim)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        h = self.input_projection(x)

        if mask is not None:
            h = h * mask.unsqueeze(-1)

        h = h.transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        h = h.transpose(1, 2)

        return self.repr_projection(h)


# ---------------------------------------------------------------------------
#  Contrastive Loss
# ---------------------------------------------------------------------------

def hierarchical_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor,
                                  temporal_unit: int = 0,
                                  alpha: float = 0.5) -> torch.Tensor:

    loss = torch.tensor(0., device=z1.device)
    n_scales = 0

    while z1.size(1) > 1:
        if n_scales >= temporal_unit:
            loss += alpha * _instance_contrastive_loss(z1, z2)
            loss += (1 - alpha) * _temporal_contrastive_loss(z1, z2)
        n_scales += 1

        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)

    if z1.size(1) == 1:
        if n_scales >= temporal_unit:
            loss += alpha * _instance_contrastive_loss(z1, z2)
            loss += (1 - alpha) * _temporal_contrastive_loss(z1, z2)
        n_scales += 1

    return loss / max(n_scales, 1)


def _instance_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:

    batch_size, T, D = z1.shape
    if batch_size <= 1:
        return torch.tensor(0., device=z1.device)

    loss = torch.tensor(0., device=z1.device)
    for t in range(T):
        sim = torch.mm(z1[:, t, :], z2[:, t, :].T)
        loss -= F.log_softmax(sim, dim=1).diag().mean()

    return loss / T


def _temporal_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:

    batch_size, T, D = z1.shape
    if T <= 1:
        return torch.tensor(0., device=z1.device)

    loss = torch.tensor(0., device=z1.device)
    for i in range(batch_size):
        sim = torch.mm(z1[i], z2[i].T)
        loss -= F.log_softmax(sim, dim=1).diag().mean()

    return loss / batch_size


# ---------------------------------------------------------------------------
#  Data Augmentation
# ---------------------------------------------------------------------------

def timestamp_masking(x: torch.Tensor, mask_ratio: float = 0.5) -> torch.Tensor:

    batch, T, F_ = x.shape
    keep_mask = (torch.rand(batch, T, device=x.device) > mask_ratio).float()
    return x * keep_mask.unsqueeze(-1)


def random_cropping(x: torch.Tensor, lengths: torch.Tensor):

    batch, T, F_ = x.shape
    min_len = 2

    min_real_len = max(int(lengths.min().item()), min_len)
    crop_len = max(min_len, int(min_real_len * 0.7))

    if crop_len >= min_real_len:
        crop_len = max(min_len, min_real_len - 1)

    max_start = max(0, int(min_real_len - crop_len))

    start1 = np.random.randint(0, max_start + 1)
    start2 = np.random.randint(0, max_start + 1)

    crop1 = x[:, start1:start1 + crop_len, :]
    crop2 = x[:, start2:start2 + crop_len, :]

    return crop1, crop2, crop_len


# ---------------------------------------------------------------------------
#  TS2Vec Model
# ---------------------------------------------------------------------------

class TS2Vec(nn.Module):

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64,
                 repr_dim: int = 320, depth: int = 10,
                 lr: float = 0.001, mask_ratio: float = 0.5,
                 weight_decay: float = 1e-4, eta_min: float = 1e-6):

        super().__init__()

        self.repr_dim = repr_dim
        self.mask_ratio = mask_ratio
        self.eta_min = eta_min
        self.device = torch.device(
            'mps' if torch.backends.mps.is_available()
            else 'cuda' if torch.cuda.is_available()
            else 'cpu'
        )

        self.encoder = TSEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            repr_dim=repr_dim,
            depth=depth,
        )

        self.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=eta_min)
        self.history = {'train_loss': []}

        n_params = sum(p.numel() for p in self.parameters())
        print(f"TS2Vec: repr_dim={repr_dim}, depth={depth}, "
              f"hidden_dim={hidden_dim}, params={n_params:,}, device={self.device}")

    def fit(self, padded_runs: np.ndarray, epochs: int = 100,
            batch_size: int = 32, plot_every: int = 5,
            plot_dir: str = "plots_ts2vec"):

        from tqdm import tqdm

        data = torch.tensor(padded_runs, dtype=torch.float32)
        lengths = self._compute_lengths(data)
        data[data == PAD_VALUE] = 0.0

        n_samples = len(data)
        print(f"Training: {n_samples} samples, seq_len={data.size(1)}, epochs={epochs}")

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=self.eta_min)

        plot_path = Path(plot_dir)
        if plot_every > 0:
            plot_path.mkdir(exist_ok=True)

        for epoch in range(epochs):
            self.train()
            epoch_losses = []
            perm = torch.randperm(n_samples)

            pbar = tqdm(range(0, n_samples, batch_size),
                        desc=f"Epoch {epoch+1}/{epochs}")

            for start in pbar:
                idx = perm[start:start + batch_size]
                x_batch = data[idx].to(self.device)
                l_batch = lengths[idx]

                crop1, crop2, _ = random_cropping(x_batch, l_batch)
                crop1_masked = timestamp_masking(crop1, self.mask_ratio)
                crop2_masked = timestamp_masking(crop2, self.mask_ratio)

                z1 = self.encoder(crop1_masked)
                z2 = self.encoder(crop2_masked)

                loss = hierarchical_contrastive_loss(z1, z2)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            self.scheduler.step()

            epoch_loss = np.mean(epoch_losses)
            self.history['train_loss'].append(epoch_loss)
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} — loss: {epoch_loss:.6f}, lr: {lr:.2e}")

            if plot_every > 0 and (epoch + 1) % plot_every == 0:
                self._save_loss_curve(plot_path)

        if plot_every > 0:
            self._save_loss_curve(plot_path)

        return self.history

    @torch.no_grad()
    def encode(self, padded_runs: np.ndarray, batch_size: int = 64) -> np.ndarray:

        self.eval()
        data = torch.tensor(padded_runs, dtype=torch.float32)
        lengths = self._compute_lengths(data)
        data[data == PAD_VALUE] = 0.0

        max_len = data.size(1)
        positions = torch.arange(max_len).unsqueeze(0)
        padding_mask = (positions < lengths.unsqueeze(1)).float()

        all_embeddings = []

        for i in range(0, len(data), batch_size):
            x = data[i:i+batch_size].to(self.device)
            m = padding_mask[i:i+batch_size].to(self.device)

            r = self.encoder(x, mask=m)

            r[m.unsqueeze(-1).expand_as(r) == 0] = float('-inf')
            embedding = r.max(dim=1).values

            all_embeddings.append(embedding.cpu())

        return torch.cat(all_embeddings, dim=0).numpy()

    def save(self, path: str):
        torch.save({
            'encoder_state': self.encoder.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history,
            'repr_dim': self.repr_dim,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(checkpoint['encoder_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.history = checkpoint.get('history', {'train_loss': []})
        print(f"Model loaded from {path}")

    def _compute_lengths(self, data: torch.Tensor) -> torch.Tensor:
        if data.dim() == 3:
            x_flat = data.squeeze(-1)
        else:
            x_flat = data
        real_mask = (x_flat != PAD_VALUE)
        positions = torch.arange(x_flat.size(1), device=x_flat.device).unsqueeze(0)
        masked_positions = real_mask.long() * positions
        lengths = masked_positions.max(dim=1).values + 1
        return lengths.clamp(min=2)

    def _save_loss_curve(self, plot_dir: Path):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.history['train_loss'], linewidth=1.5, color='#2196F3')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('TS2Vec Training Progress')
        if len(self.history['train_loss']) > 2:
            ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(plot_dir / "loss_curve.png", dpi=100)
        plt.close(fig)
