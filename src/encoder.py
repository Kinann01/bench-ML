"""
TS2Vec: Universal Time Series Representation Learning with Contrastive Learning.

This module implements the TS2Vec framework for learning universal representations
of time series data. Instead of reconstruction (like an autoencoder), TS2Vec uses
hierarchical temporal contrastive learning to train an encoder that maps any time
series to a meaningful embedding vector.

Reference: Yue et al., "TS2Vec: Towards Universal Representation of Time Series", 2022.

Architecture:
    Input → Projection FC → [Dilated Conv Residual Block × depth] → Representations
    
    Augmentation: timestamp masking + random cropping
    Loss: hierarchical temporal contrastive loss at multiple pooling scales
    
    At inference: max-pool over time dimension → single instance-level embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

PAD_VALUE = -999.0


# ---------------------------------------------------------------------------
#  Building blocks
# ---------------------------------------------------------------------------

class DilatedConvBlock(nn.Module):
    """
    Single residual block with two 1D dilated convolutions.
    
    Conv1d(dilation=d) → ReLU → Conv1d(dilation=d) → ReLU + skip
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2  # causal-ish padding

        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, time)"""
        residual = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))

        # Trim if conv produced extra timesteps due to padding
        if out.size(-1) != residual.size(-1):
            out = out[..., :residual.size(-1)]

        return out + residual


class TSEncoder(nn.Module):
    """
    Dilated CNN encoder: projects input to hidden dim, then applies 
    `depth` dilated residual blocks with exponentially increasing dilation.
    
    The receptive field grows as 2^depth, so depth=10 → receptive field 
    covers ~1024 timesteps, sufficient for sequences up to 890.
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64,
                 repr_dim: int = 320, depth: int = 10, kernel_size: int = 3):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            DilatedConvBlock(hidden_dim, kernel_size, dilation=2 ** i)
            for i in range(depth)
        ])

        self.repr_projection = nn.Linear(hidden_dim, repr_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, time, input_dim) — the time series
            mask: (batch, time) — 1 for real timesteps, 0 for padding/masked
            
        Returns:
            repr: (batch, time, repr_dim) — per-timestamp representations
        """
        # Project input: (batch, time, input_dim) → (batch, time, hidden_dim)
        h = self.input_projection(x)

        # Apply mask if provided (zero out padded/masked positions)
        if mask is not None:
            h = h * mask.unsqueeze(-1)

        # Conv1d expects (batch, channels, time)
        h = h.transpose(1, 2)

        for block in self.blocks:
            h = block(h)

        # Back to (batch, time, hidden_dim)
        h = h.transpose(1, 2)

        # Final projection to representation space
        repr = self.repr_projection(h)
        return repr


# ---------------------------------------------------------------------------
#  Contrastive Loss
# ---------------------------------------------------------------------------

def hierarchical_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor,
                                  temporal_unit: int = 0, alpha: float = 0.5) -> torch.Tensor:
    """
    Hierarchical temporal contrastive loss.
    
    Applied at multiple scales: original resolution, then progressively 
    max-pooled by 2. At each scale, compute instance-wise and temporal 
    contrastive losses.
    
    Args:
        z1, z2: (batch, time, repr_dim) — representations from two augmented views
        temporal_unit: minimum temporal unit (skip pooling below this)
        alpha: weight balance between temporal and instance contrastive loss
        
    Returns:
        Scalar loss
    """
    loss = torch.tensor(0., device=z1.device)
    n_scales = 0
    
    while z1.size(1) > 1:
        if n_scales >= temporal_unit:
            loss += alpha * _instance_contrastive_loss(z1, z2)
            loss += (1 - alpha) * _temporal_contrastive_loss(z1, z2)
        n_scales += 1
        
        # Max-pool by factor 2 along time dimension
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
    
    # Final single-timestep level
    if z1.size(1) == 1:
        if n_scales >= temporal_unit:
            loss += alpha * _instance_contrastive_loss(z1, z2)
            loss += (1 - alpha) * _temporal_contrastive_loss(z1, z2)
        n_scales += 1
    
    return loss / max(n_scales, 1)


def _instance_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Instance-level contrastive loss.
    
    For each timestamp t: z1[i, t] and z2[i, t] are positive pairs (same instance),
    z1[i, t] and z2[j, t] for j != i are negative pairs (different instances).
    """
    batch_size, T, D = z1.shape
    if batch_size <= 1:
        return torch.tensor(0., device=z1.device)
    
    # Average over timestamps
    loss = torch.tensor(0., device=z1.device)
    
    for t in range(T):
        # z1_t, z2_t: (batch, D)
        z1_t = z1[:, t, :]
        z2_t = z2[:, t, :]
        
        # Similarity matrix between z1_t and z2_t: (batch, batch)
        sim = torch.mm(z1_t, z2_t.T)  # cosine-like (unnormalized)
        
        # Positive pairs are on the diagonal
        # InfoNCE-style: -log(exp(sim[i,i]) / sum_j(exp(sim[i,j])))
        log_softmax = F.log_softmax(sim, dim=1)
        loss -= log_softmax.diag().mean()
    
    return loss / T


def _temporal_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Temporal contrastive loss.
    
    For each instance i: z1[i, t] and z2[i, t] are positive pairs (same timestamp),
    z1[i, t] and z2[i, s] for s != t are negative pairs (different timestamps).
    """
    batch_size, T, D = z1.shape
    if T <= 1:
        return torch.tensor(0., device=z1.device)
    
    # Average over instances
    loss = torch.tensor(0., device=z1.device)
    
    for i in range(batch_size):
        # z1_i, z2_i: (T, D)
        z1_i = z1[i]
        z2_i = z2[i]
        
        # Similarity matrix between timestamps: (T, T)
        sim = torch.mm(z1_i, z2_i.T)
        
        # Positive pairs are on the diagonal (same timestamp, same instance)
        log_softmax = F.log_softmax(sim, dim=1)
        loss -= log_softmax.diag().mean()
    
    return loss / batch_size


# ---------------------------------------------------------------------------
#  Data augmentation
# ---------------------------------------------------------------------------

def timestamp_masking(x: torch.Tensor, mask_ratio: float = 0.5) -> torch.Tensor:
    """
    Randomly mask timestamps by setting them to zero.
    
    Args:
        x: (batch, time, features)
        mask_ratio: fraction of timestamps to mask
        
    Returns:
        x_masked: (batch, time, features) with some timesteps zeroed
    """
    batch, T, F_ = x.shape
    # Create random mask: 1 = keep, 0 = mask
    keep_mask = (torch.rand(batch, T, device=x.device) > mask_ratio).float()
    return x * keep_mask.unsqueeze(-1)


def random_cropping(x: torch.Tensor, lengths: torch.Tensor):
    """
    Random crop: select an overlapping region from two different crops.
    
    Returns two cropped views of x that share an overlapping segment.
    
    Args:
        x: (batch, time, features) 
        lengths: (batch,) — real sequence lengths (positions beyond are padding)
        
    Returns:
        crop1, crop2: two cropped views, both (batch, crop_len, features)
    """
    batch, T, F_ = x.shape
    
    # Minimum crop length: at least 2 timesteps
    min_len = 2
    
    # For each sample, crop within its real data portion
    # Use the minimum real length in the batch to determine crop boundaries
    min_real_len = max(lengths.min().item(), min_len)
    
    # Crop length: random between min_len and min_real_len
    crop_len = max(min_len, int(min_real_len * 0.7))  # 70% of shortest sequence
    
    if crop_len >= min_real_len:
        crop_len = max(min_len, min_real_len - 1)
    
    # Random start positions for two overlapping crops
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
    """
    TS2Vec: contrastive learning framework for time series representations.
    
    Trains a dilated CNN encoder using hierarchical temporal contrastive loss
    with timestamp masking and random cropping augmentations.
    
    Usage:
        model = TS2Vec(input_dim=1)
        model.fit(X_train, epochs=50)
        embeddings = model.encode(X_new)  # (n_samples, repr_dim)
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        repr_dim: int = 320,
        depth: int = 10,
        lr: float = 0.001,
        mask_ratio: float = 0.5,
    ):
        super().__init__()

        self.repr_dim = repr_dim
        self.mask_ratio = mask_ratio
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
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        self.history = {'train_loss': []}

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"TS2Vec built: repr_dim={repr_dim}, depth={depth}, "
                    f"hidden_dim={hidden_dim}, params={n_params:,}, device={self.device}")

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        plot_every: int = 5,
        plot_dir: str = "plots_ts2vec",
    ):
        """
        Train the TS2Vec encoder.
        
        Args:
            X: Training data, shape (n_samples, max_len, n_features).
                Padded with PAD_VALUE for variable-length sequences.
            epochs: Number of training epochs
            batch_size: Batch size
            plot_every: Save loss curve every N epochs
            plot_dir: Directory for plots
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Pre-compute sequence lengths from PAD_VALUE
        logger.info("Pre-computing sequence lengths...")
        lengths = self._compute_lengths(X_tensor)
        
        # Replace PAD_VALUE with 0 for the encoder
        X_tensor[X_tensor == PAD_VALUE] = 0.0

        # Create padding mask: 1 = real data, 0 = padding
        max_len = X_tensor.size(1)
        positions = torch.arange(max_len).unsqueeze(0)  # (1, max_len)
        padding_mask = (positions < lengths.unsqueeze(1)).float()  # (n_samples, max_len)

        n_samples = len(X_tensor)
        logger.info(f"Training TS2Vec: {n_samples} samples, max_len={max_len}")

        # Update cosine annealing schedule
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        # Setup plot directory
        plot_path = Path(plot_dir)
        if plot_every > 0:
            plot_path.mkdir(exist_ok=True)

        for epoch in range(epochs):
            self.train()
            epoch_losses = []

            # Shuffle
            perm = torch.randperm(n_samples)
            
            pbar = tqdm(range(0, n_samples, batch_size),
                       desc=f"Epoch {epoch+1}/{epochs}")
            
            for start in pbar:
                idx = perm[start:start + batch_size]
                x_batch = X_tensor[idx].to(self.device)         # (B, T, F)
                l_batch = lengths[idx]                           # (B,)
                m_batch = padding_mask[idx].to(self.device)      # (B, T)

                # Augmentation: random cropping → two overlapping views
                crop1, crop2, crop_len = random_cropping(x_batch, l_batch)
                
                # Augmentation: timestamp masking on each crop
                crop1_masked = timestamp_masking(crop1, self.mask_ratio)
                crop2_masked = timestamp_masking(crop2, self.mask_ratio)

                # Encode both augmented views
                z1 = self.encoder(crop1_masked)  # (B, crop_len, repr_dim)
                z2 = self.encoder(crop2_masked)  # (B, crop_len, repr_dim)

                # Hierarchical contrastive loss
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
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1}/{epochs} — loss: {epoch_loss:.6f}, lr: {current_lr:.2e}")

            # Save loss curve periodically
            if plot_every > 0 and (epoch + 1) % plot_every == 0:
                self._save_loss_curve(plot_path)

        # Final save
        if plot_every > 0:
            self._save_loss_curve(Path(plot_dir))

        return self.history

    @torch.no_grad()
    def encode(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """
        Encode time series into instance-level embedding vectors.
        
        For each sequence, encodes all timestamps then max-pools over time
        to produce a single repr_dim-dimensional vector.
        
        Args:
            X: Shape (n_samples, max_len, n_features), padded with PAD_VALUE
            batch_size: Processing batch size
            
        Returns:
            embeddings: Shape (n_samples, repr_dim)
        """
        self.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Compute lengths and create mask
        lengths = self._compute_lengths(X_tensor)
        X_tensor[X_tensor == PAD_VALUE] = 0.0
        
        max_len = X_tensor.size(1)
        positions = torch.arange(max_len).unsqueeze(0)
        padding_mask = (positions < lengths.unsqueeze(1)).float()
        
        all_embeddings = []
        
        for i in range(0, len(X_tensor), batch_size):
            x = X_tensor[i:i+batch_size].to(self.device)
            m = padding_mask[i:i+batch_size].to(self.device)
            
            # Get per-timestamp representations
            repr = self.encoder(x, mask=m)  # (B, T, repr_dim)
            
            # Mask out padding positions before pooling
            repr = repr * m.unsqueeze(-1)
            
            # Max-pool over time → instance-level embedding
            # Set padded positions to -inf so they don't win the max
            repr_masked = repr.clone()
            repr_masked[m.unsqueeze(-1).expand_as(repr_masked) == 0] = float('-inf')
            embedding = repr_masked.max(dim=1).values  # (B, repr_dim)
            
            all_embeddings.append(embedding.cpu())
        
        return torch.cat(all_embeddings, dim=0).numpy()

    def save(self, path: str):
        """Save model state."""
        torch.save({
            'encoder_state': self.encoder.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history,
            'repr_dim': self.repr_dim,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(checkpoint['encoder_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.history = checkpoint.get('history', {'train_loss': []})
        logger.info(f"Model loaded from {path}")

    def _compute_lengths(self, X: torch.Tensor) -> torch.Tensor:
        """Compute real sequence lengths from padded data (vectorized)."""
        if X.dim() == 3:
            x_flat = X.squeeze(-1)
        else:
            x_flat = X
        real_mask = (x_flat != PAD_VALUE)
        positions = torch.arange(x_flat.size(1), device=x_flat.device).unsqueeze(0)
        masked_positions = real_mask.long() * positions
        lengths = masked_positions.max(dim=1).values + 1
        return lengths.clamp(min=2)  # min 2 for cropping

    def _save_loss_curve(self, plot_dir: Path):
        """Save the loss curve."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.history['train_loss'], label='Contrastive Loss', linewidth=1.5, color='#2196F3')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('TS2Vec Training Progress')
        ax.legend()
        if len(self.history['train_loss']) > 2:
            ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(plot_dir / "loss_curve.png", dpi=100)
        plt.close(fig)
        logger.info(f"Saved loss curve: {plot_dir}/loss_curve.png")
