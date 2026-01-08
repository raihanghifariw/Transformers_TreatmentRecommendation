"""
Research-Grade Offline RL Pipeline for Clinical Treatment Recommendation
=========================================================================
Three-Stage Training Paradigm:
  Stage 1: Temporal Variational Autoencoder (Temporal-VAE)
  Stage 2: Decision Transformer for Offline Policy Learning
  Stage 3: Conservative Policy Fine-tuning (Transformer SAC + BC/CQL)

Author: Clinical AI Research Lab
Target Venues: MLHC / NeurIPS / JAMIA
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from typing import Tuple, Optional, Dict, List

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# UTILITY MODULES
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding for fixed-length sequences."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        return self.dropout(x)


# =============================================================================
# STAGE 1: TEMPORAL VARIATIONAL AUTOENCODER (Temporal-VAE)
# =============================================================================

class TemporalEncoder(nn.Module):
    """
    Temporal encoder using GRU + Attention for sequence encoding.
    Outputs mean and log-variance for latent distribution.
    """
    
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Bidirectional GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Temporal attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Latent distribution parameters
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sequence [batch, seq_len, input_dim]
            mask: Optional attention mask [batch, seq_len]
        Returns:
            mu: Mean of latent distribution [batch, latent_dim]
            logvar: Log variance of latent distribution [batch, latent_dim]
        """
        # Input projection
        h = self.input_proj(x)
        
        # GRU encoding
        gru_out, _ = self.gru(h)  # [batch, seq_len, hidden_dim * 2]
        
        # Self-attention with residual
        attn_out, _ = self.attention(gru_out, gru_out, gru_out, key_padding_mask=mask)
        h = self.attn_norm(gru_out + attn_out)
        
        # Global pooling (use last timestep or mean pooling)
        h_pooled = h[:, -1, :]  # [batch, hidden_dim * 2]
        
        # Latent parameters
        mu = self.fc_mu(h_pooled)
        logvar = self.fc_logvar(h_pooled)
        
        return mu, logvar


class TemporalDecoder(nn.Module):
    """
    Temporal decoder using GRU for sequence reconstruction.
    """
    
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Latent to hidden projection
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # GRU decoder
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Args:
            z: Latent vector [batch, latent_dim]
            seq_len: Target sequence length
        Returns:
            Reconstructed sequence [batch, seq_len, output_dim]
        """
        # Project latent to hidden
        h = self.latent_proj(z)  # [batch, hidden_dim]
        
        # Repeat for sequence
        h = h.unsqueeze(1).repeat(1, seq_len, 1)  # [batch, seq_len, hidden_dim]
        
        # GRU decoding
        out, _ = self.gru(h)  # [batch, seq_len, hidden_dim]
        
        # Output projection
        recon = self.output_proj(out)  # [batch, seq_len, output_dim]
        
        return recon


class TemporalVAE(nn.Module):
    """
    Temporal Variational Autoencoder for patient trajectory representation learning.
    
    Learns a compact latent representation z_t that captures temporal dynamics
    of (state, action) sequences.
    """
    
    def __init__(
        self,
        state_dim: int = 37,
        action_dim: int = 2,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        input_dim = state_dim + action_dim
        
        # Encoder
        self.encoder = TemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout
        )
        
        # Decoder
        self.decoder = TemporalDecoder(
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, states: torch.Tensor, actions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode state-action sequences to latent space.
        
        Args:
            states: [batch, seq_len, state_dim]
            actions: [batch, seq_len, action_dim]
            mask: Optional mask for padding
            
        Returns:
            z: Latent representation [batch, latent_dim]
            mu: Mean [batch, latent_dim]
            logvar: Log variance [batch, latent_dim]
        """
        # Concatenate state and action
        x = torch.cat([states, actions], dim=-1)
        
        # Encode
        mu, logvar = self.encoder(x, mask)
        
        # Sample z
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar
    
    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode latent to sequence."""
        return self.decoder(z, seq_len)
    
    def forward(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with reconstruction.
        
        Returns:
            Dictionary containing: z, mu, logvar, recon_states, recon_actions
        """
        seq_len = states.size(1)
        
        # Encode
        z, mu, logvar = self.encode(states, actions, mask)
        
        # Decode
        recon = self.decode(z, seq_len)
        
        # Split reconstruction
        recon_states = recon[:, :, :self.state_dim]
        recon_actions = recon[:, :, self.state_dim:]
        
        return {
            'z': z,
            'mu': mu,
            'logvar': logvar,
            'recon_states': recon_states,
            'recon_actions': recon_actions
        }
    
    def get_latent(self, states: torch.Tensor, actions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get latent representation (inference mode, no gradient)."""
        with torch.no_grad():
            z, _, _ = self.encode(states, actions, mask)
        return z


class TemporalVAETrainer:
    """
    Trainer for Temporal VAE with KL annealing.
    """
    
    def __init__(
        self,
        model: TemporalVAE,
        lr: float = 1e-4,
        kl_weight_max: float = 1.0,
        kl_anneal_epochs: int = 50
    ):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        
        self.kl_weight_max = kl_weight_max
        self.kl_anneal_epochs = kl_anneal_epochs
        self.current_epoch = 0
        
    def kl_weight(self) -> float:
        """KL weight with linear annealing."""
        return min(1.0, self.current_epoch / self.kl_anneal_epochs) * self.kl_weight_max
    
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute VAE loss = Reconstruction + KL divergence.
        """
        outputs = self.model(states, actions, mask)
        
        # Reconstruction loss (MSE)
        recon_loss_states = F.mse_loss(outputs['recon_states'], states, reduction='mean')
        recon_loss_actions = F.mse_loss(outputs['recon_actions'], actions, reduction='mean')
        recon_loss = recon_loss_states + recon_loss_actions
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp())
        
        # Total loss with annealing
        kl_w = self.kl_weight()
        total_loss = recon_loss + kl_w * kl_loss
        
        metrics = {
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'kl_weight': kl_w,
            'total_loss': total_loss.item()
        }
        
        return total_loss, metrics
    
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Single training step for a batch of data."""
        self.model.train()
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        
        self.optimizer.zero_grad()
        loss, metrics = self.compute_loss(states, actions, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'total': metrics['total_loss'],
            'recon': metrics['recon_loss'],
            'kl': metrics['kl_loss']
        }
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch using a dataloader."""
        self.model.train()
        total_metrics = {'recon_loss': 0, 'kl_loss': 0, 'total_loss': 0}
        num_batches = 0
        
        for batch in dataloader:
            states, actions = batch['states'].to(self.device), batch['actions'].to(self.device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(self.device)
            
            self.optimizer.zero_grad()
            loss, metrics = self.compute_loss(states, actions, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            for k in total_metrics:
                total_metrics[k] += metrics.get(k, 0)
            num_batches += 1
        
        self.scheduler.step()
        self.current_epoch += 1
        
        return {k: v / num_batches for k, v in total_metrics.items()}


# =============================================================================
# STAGE 2: DECISION TRANSFORMER
# =============================================================================

class DecisionTransformer(nn.Module):
    """
    Decision Transformer for offline policy learning via sequence modeling.
    
    Predicts actions conditioned on return-to-go (RTG) and latent states.
    Tokens: (RTG_1, z_1, a_1, RTG_2, z_2, a_2, ...)
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        action_dim: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        max_episode_len: int = 100,
        dropout: float = 0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.max_episode_len = max_episode_len
        
        # Token embeddings
        self.rtg_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.state_embedding = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.action_embedding = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # Learned positional encoding (for timesteps, not token positions)
        self.timestep_embedding = nn.Embedding(max_episode_len, d_model)
        
        # Token type embedding (RTG=0, State=1, Action=2)
        self.token_type_embedding = nn.Embedding(3, d_model)
        
        # Causal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.action_head_mean = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, action_dim)
        )
        self.action_head_logstd = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, action_dim)
        )
        
        # Optional value head for auxiliary supervision
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal attention mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(device)
    
    def forward(
        self,
        rtgs: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            rtgs: Return-to-go [batch, seq_len, 1]
            states: Latent states [batch, seq_len, latent_dim]
            actions: Actions [batch, seq_len, action_dim]
            timesteps: Timestep indices [batch, seq_len]
            attention_mask: Padding mask [batch, seq_len]
            
        Returns:
            Dictionary with action predictions and value estimates
        """
        batch_size, seq_len = states.shape[:2]
        
        # Embed each token type
        rtg_tokens = self.rtg_embedding(rtgs)  # [batch, seq_len, d_model]
        state_tokens = self.state_embedding(states)
        action_tokens = self.action_embedding(actions)
        
        # Add timestep embeddings
        time_emb = self.timestep_embedding(timesteps.clamp(0, self.max_episode_len - 1))
        rtg_tokens = rtg_tokens + time_emb
        state_tokens = state_tokens + time_emb
        action_tokens = action_tokens + time_emb
        
        # Add token type embeddings
        token_types = torch.arange(3, device=device).unsqueeze(0).unsqueeze(0)
        type_emb = self.token_type_embedding(token_types)  # [1, 1, 3, d_model]
        
        rtg_tokens = rtg_tokens + type_emb[:, :, 0, :]
        state_tokens = state_tokens + type_emb[:, :, 1, :]
        action_tokens = action_tokens + type_emb[:, :, 2, :]
        
        # Interleave tokens: (rtg_1, s_1, a_1, rtg_2, s_2, a_2, ...)
        # Shape: [batch, seq_len * 3, d_model]
        tokens = torch.stack([rtg_tokens, state_tokens, action_tokens], dim=2)
        tokens = tokens.view(batch_size, seq_len * 3, self.d_model)
        
        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len * 3)
        
        # Extend attention mask for interleaved tokens
        if attention_mask is not None:
            # Repeat mask for each token type
            extended_mask = attention_mask.unsqueeze(-1).repeat(1, 1, 3).view(batch_size, -1)
            extended_mask = ~extended_mask.bool()  # True = masked position
        else:
            extended_mask = None
        
        # Transformer forward
        hidden = self.transformer(tokens, mask=causal_mask, src_key_padding_mask=extended_mask)
        
        # Extract state positions for action prediction (indices 1, 4, 7, ...)
        state_positions = torch.arange(1, seq_len * 3, 3, device=device)
        state_hidden = hidden[:, state_positions, :]  # [batch, seq_len, d_model]
        
        # Action prediction (Gaussian)
        action_mean = self.action_head_mean(state_hidden)
        action_logstd = self.action_head_logstd(state_hidden).clamp(-5, 2)
        
        # Value prediction from RTG positions
        rtg_positions = torch.arange(0, seq_len * 3, 3, device=device)
        rtg_hidden = hidden[:, rtg_positions, :]
        values = self.value_head(rtg_hidden)
        
        return {
            'action_mean': action_mean,
            'action_logstd': action_logstd,
            'values': values,
            'hidden': state_hidden
        }
    
    def get_action(
        self,
        rtg: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Get action for the last timestep (inference).
        """
        outputs = self.forward(rtg, states, actions, timesteps)
        # Return last action
        return outputs['action_mean'][:, -1, :]
    
    def sample_action(
        self,
        rtg: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action with Gaussian policy.
        """
        outputs = self.forward(rtg, states, actions, timesteps)
        mean = outputs['action_mean'][:, -1, :]
        logstd = outputs['action_logstd'][:, -1, :]
        std = logstd.exp()
        
        if deterministic:
            action = mean
            log_prob = torch.zeros(mean.shape[0], 1, device=device)
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Apply tanh squashing
        action_squashed = torch.tanh(action)
        log_prob = log_prob - torch.log(1 - action_squashed.pow(2) + 1e-7).sum(dim=-1, keepdim=True)
        
        return action_squashed, log_prob


class DecisionTransformerTrainer:
    """
    Trainer for Decision Transformer with supervised learning.
    """
    
    def __init__(
        self,
        model: DecisionTransformer,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0
    ):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.grad_clip = grad_clip
        
    def compute_loss(
        self,
        rtgs: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        target_actions: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for Decision Transformer.
        
        Action prediction loss + optional value auxiliary loss.
        """
        outputs = self.model(rtgs, states, actions, timesteps, mask)
        
        # Action prediction loss (MSE for simplicity, can use Gaussian NLL)
        action_pred = outputs['action_mean']
        action_loss = F.mse_loss(action_pred, target_actions, reduction='mean')
        
        # Value auxiliary loss (if returns provided)
        value_loss = torch.tensor(0.0, device=self.device)
        if returns is not None:
            values = outputs['values']
            value_loss = F.mse_loss(values.squeeze(-1), returns, reduction='mean')
        
        total_loss = action_loss + 0.5 * value_loss
        
        metrics = {
            'action_loss': action_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, metrics
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_metrics = {'action_loss': 0, 'value_loss': 0, 'total_loss': 0}
        num_batches = 0
        
        for batch in dataloader:
            rtgs = batch['rtgs'].to(self.device)
            states = batch['states'].to(self.device)
            actions = batch['actions'].to(self.device)
            timesteps = batch['timesteps'].to(self.device)
            target_actions = batch['target_actions'].to(self.device)
            returns = batch.get('returns')
            if returns is not None:
                returns = returns.to(self.device)
            mask = batch.get('mask')
            if mask is not None:
                mask = mask.to(self.device)
            
            self.optimizer.zero_grad()
            loss, metrics = self.compute_loss(
                rtgs, states, actions, timesteps, target_actions, returns, mask
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            
            for k in total_metrics:
                total_metrics[k] += metrics.get(k, 0)
            num_batches += 1
        
        self.scheduler.step()
        
        return {k: v / num_batches for k, v in total_metrics.items()}
    
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rtgs: torch.Tensor,
        timesteps: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Single training step for a batch of data.
        
        Args:
            states: [batch, seq_len, state_dim]
            actions: [batch, seq_len, action_dim] - target actions
            rtgs: [batch, seq_len, 1] - returns-to-go
            timesteps: [batch, seq_len]
            mask: [batch, seq_len] - attention mask
            
        Returns:
            Loss value as float
        """
        self.model.train()
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rtgs = rtgs.to(self.device)
        timesteps = timesteps.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        
        # For DT, we shift actions for autoregressive prediction
        # Input actions are previous actions (shifted right)
        # Target actions are current actions
        input_actions = torch.zeros_like(actions)
        input_actions[:, 1:, :] = actions[:, :-1, :]  # Shift right
        target_actions = actions
        
        self.optimizer.zero_grad()
        loss, metrics = self.compute_loss(
            rtgs, states, input_actions, timesteps, target_actions, mask=mask
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        
        return metrics['total_loss']


# =============================================================================
# STAGE 3: TRANSFORMER SAC WITH BEHAVIOR CLONING / CQL
# =============================================================================

class TransformerActor(nn.Module):
    """
    Transformer-based actor for SAC with continuous action space.
    Uses frozen latent states from Temporal-VAE.
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        action_dim: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.mean_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, action_dim)
        )
        self.logstd_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: Latent state [batch, latent_dim] or [batch, seq_len, latent_dim]
        Returns:
            mean, std: Action distribution parameters
        """
        if state.dim() == 2:
            state = state.unsqueeze(1)  # [batch, 1, latent_dim]
        
        x = self.input_embedding(state)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # Use last timestep output
        x = x[:, -1, :]
        
        mean = self.mean_head(x)
        logstd = self.logstd_head(x).clamp(-5, 2)
        std = logstd.exp()
        
        return mean, std
    
    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from Gaussian policy with tanh squashing.
        """
        mean, std = self.forward(state)
        
        if deterministic:
            z = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            z = dist.rsample()
        
        action = torch.tanh(z)
        
        # Log probability with tanh correction
        log_prob = torch.distributions.Normal(mean, std).log_prob(z)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob, mean, std


class TransformerCritic(nn.Module):
    """
    Transformer-based critic for Q-value estimation.
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        action_dim: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        input_dim = latent_dim + action_dim
        
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.q_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q-value."""
        if state.dim() == 2:
            state = state.unsqueeze(1)
            action = action.unsqueeze(1)
        
        x = torch.cat([state, action], dim=-1)
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        
        return self.q_head(x)


class ConservativeSACAgent:
    """
    Transformer-based SAC Agent with Conservative Q-Learning (CQL) and Behavior Cloning.
    
    Implements offline RL with strong regularization to prevent distribution shift.
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        action_dim: int = 2,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        bc_weight: float = 0.5,
        cql_weight: float = 1.0,
        cql_temp: float = 1.0,
        cql_num_samples: int = 10,
        use_cql: bool = True,
        use_bc: bool = True,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.bc_weight = bc_weight
        self.cql_weight = cql_weight
        self.cql_temp = cql_temp
        self.cql_num_samples = cql_num_samples
        self.use_cql = use_cql
        self.use_bc = use_bc
        self.action_dim = action_dim
        
        # Actor
        self.actor = TransformerActor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        ).to(self.device)
        
        # Twin Critics
        self.critic_1 = TransformerCritic(
            latent_dim=latent_dim,
            action_dim=action_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        ).to(self.device)
        
        self.critic_2 = TransformerCritic(
            latent_dim=latent_dim,
            action_dim=action_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        ).to(self.device)
        
        # Target critics
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)
        
        # Optimizers
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=3e-4, weight_decay=1e-5)
        self.critic_1_optimizer = optim.AdamW(self.critic_1.parameters(), lr=3e-4, weight_decay=1e-5)
        self.critic_2_optimizer = optim.AdamW(self.critic_2.parameters(), lr=3e-4, weight_decay=1e-5)
        
        # Schedulers
        self.actor_scheduler = ReduceLROnPlateau(self.actor_optimizer, mode='min', factor=0.5, patience=10)
        self.critic_1_scheduler = ReduceLROnPlateau(self.critic_1_optimizer, mode='min', factor=0.5, patience=10)
        self.critic_2_scheduler = ReduceLROnPlateau(self.critic_2_optimizer, mode='min', factor=0.5, patience=10)
        
        # Automatic entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=3e-4)
        self.alpha = alpha
        
    def load_from_decision_transformer(self, dt_model: DecisionTransformer):
        """
        Initialize actor from pretrained Decision Transformer weights.
        Transfer relevant layers for warm-starting.
        """
        # Map state embedding
        with torch.no_grad():
            # Copy embedding weights (adjust dimensions if needed)
            dt_state_embed = dt_model.state_embedding[0].weight.data
            actor_embed = self.actor.input_embedding[0]
            min_dim = min(dt_state_embed.shape[1], actor_embed.weight.shape[1])
            actor_embed.weight.data[:, :min_dim] = dt_state_embed[:actor_embed.weight.shape[0], :min_dim]
            
            # Copy action head weights
            dt_action_mean = dt_model.action_head_mean[-1].weight.data
            actor_mean = self.actor.mean_head[-1]
            actor_mean.weight.data = dt_action_mean[:actor_mean.weight.shape[0], :actor_mean.weight.shape[1]]
            
        print("Loaded weights from Decision Transformer to Actor")
        
    def compute_cql_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        q1_pred: torch.Tensor,
        q2_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Conservative Q-Learning regularization.
        
        CQL penalizes Q-values for actions not in the dataset.
        """
        batch_size = state.shape[0]
        
        # Sample random actions
        random_actions = torch.FloatTensor(batch_size * self.cql_num_samples, self.action_dim).uniform_(-1, 1).to(self.device)
        state_repeated = state.unsqueeze(1).repeat(1, self.cql_num_samples, 1).view(-1, state.shape[-1])
        
        # Q-values for random actions
        q1_random = self.critic_1(state_repeated, random_actions).view(batch_size, self.cql_num_samples)
        q2_random = self.critic_2(state_repeated, random_actions).view(batch_size, self.cql_num_samples)
        
        # Sample actions from current policy
        with torch.no_grad():
            policy_actions, policy_log_prob, _, _ = self.actor.sample(state_repeated)
        
        q1_policy = self.critic_1(state_repeated, policy_actions).view(batch_size, self.cql_num_samples)
        q2_policy = self.critic_2(state_repeated, policy_actions).view(batch_size, self.cql_num_samples)
        
        # CQL loss: logsumexp(Q(s, a)) - Q(s, a_data)
        q1_concat = torch.cat([q1_random, q1_policy], dim=1)
        q2_concat = torch.cat([q2_random, q2_policy], dim=1)
        
        cql_loss_1 = torch.logsumexp(q1_concat / self.cql_temp, dim=1).mean() * self.cql_temp - q1_pred.mean()
        cql_loss_2 = torch.logsumexp(q2_concat / self.cql_temp, dim=1).mean() * self.cql_temp - q2_pred.mean()
        
        return (cql_loss_1 + cql_loss_2) / 2
    
    def train_step(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        behavior_action: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Single training step with SAC + BC + CQL.
        """
        # Ensure proper shapes
        if reward.dim() == 1:
            reward = reward.unsqueeze(1)
        if done.dim() == 1:
            done = done.unsqueeze(1)
        
        # ============ Critic Update ============
        with torch.no_grad():
            next_action, next_log_prob, _, _ = self.actor.sample(next_state)
            target_q1 = self.target_critic_1(next_state, next_action)
            target_q2 = self.target_critic_2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            q_target = reward + (1 - done) * self.gamma * target_q
        
        # Current Q estimates
        q1_pred = self.critic_1(state, action)
        q2_pred = self.critic_2(state, action)
        
        # TD loss
        critic_1_loss = F.mse_loss(q1_pred, q_target)
        critic_2_loss = F.mse_loss(q2_pred, q_target)
        
        # CQL regularization
        cql_loss = torch.tensor(0.0, device=self.device)
        if self.use_cql:
            cql_loss = self.compute_cql_loss(state, action, q1_pred, q2_pred)
            critic_1_loss = critic_1_loss + self.cql_weight * cql_loss
            critic_2_loss = critic_2_loss + self.cql_weight * cql_loss
        
        # Update critics
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1.0)
        self.critic_1_optimizer.step()
        
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1.0)
        self.critic_2_optimizer.step()
        
        # ============ Actor Update ============
        new_action, log_prob, mean, _ = self.actor.sample(state)
        q1_new = self.critic_1(state, new_action)
        q2_new = self.critic_2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        # SAC actor loss
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        # Behavior cloning regularization
        bc_loss = torch.tensor(0.0, device=self.device)
        if self.use_bc and behavior_action is not None:
            bc_loss = F.mse_loss(mean, behavior_action)
            actor_loss = actor_loss + self.bc_weight * bc_loss
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # ============ Entropy Tuning ============
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()
        
        # ============ Target Update ============
        self.soft_update()
        
        return {
            'critic_1_loss': critic_1_loss.item(),
            'critic_2_loss': critic_2_loss.item(),
            'actor_loss': actor_loss.item(),
            'bc_loss': bc_loss.item() if isinstance(bc_loss, torch.Tensor) else bc_loss,
            'cql_loss': cql_loss.item() if isinstance(cql_loss, torch.Tensor) else cql_loss,
            'alpha': self.alpha,
            'q_mean': q_new.mean().item()
        }
    
    def train(self, batches, epoch: int) -> List[float]:
        """
        Train for one epoch (compatible with original interface).
        """
        (state, next_state, action, _, reward, done, bloc_num, _) = batches
        state = state.clone().detach().float().to(self.device)
        next_state = next_state.clone().detach().float().to(self.device)
        action = action.clone().detach().float().to(self.device)
        reward = reward.clone().detach().float().to(self.device)
        done = done.clone().detach().float().to(self.device)
        bloc_num = torch.tensor(bloc_num).long().to(self.device)
        
        batch_size = 128
        uids = torch.unique(bloc_num)
        num_batches = len(uids) // batch_size
        
        record_loss = []
        
        for batch_idx in range(num_batches + 1):
            batch_uids = uids[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_mask = torch.isin(bloc_num, batch_uids)
            
            if batch_mask.sum() == 0:
                continue
            
            batch_state = state[batch_mask]
            batch_next_state = next_state[batch_mask]
            batch_action = action[batch_mask]
            batch_reward = reward[batch_mask]
            batch_done = done[batch_mask]
            
            metrics = self.train_step(
                batch_state, batch_next_state, batch_action,
                batch_reward, batch_done, behavior_action=batch_action
            )
            
            avg_loss = (metrics['critic_1_loss'] + metrics['critic_2_loss'] + metrics['actor_loss']) / 3
            record_loss.append(avg_loss)
            
            if batch_idx % 25 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {avg_loss:.4f}, "
                      f"Alpha: {self.alpha:.4f}, Q: {metrics['q_mean']:.4f}")
        
        return record_loss
    
    def soft_update(self):
        """Soft update target networks."""
        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """Get action for inference."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        
        with torch.no_grad():
            action, _, _, _ = self.actor.sample(state, deterministic=deterministic)
        
        return action.cpu().numpy()


# =============================================================================
# UNIFIED CLINICAL RL AGENT (THREE-STAGE PIPELINE)
# =============================================================================

class ClinicalRLAgent:
    """
    Unified Clinical RL Agent implementing the three-stage training paradigm.
    
    Stage 1: Temporal-VAE for representation learning
    Stage 2: Decision Transformer for offline policy pretraining
    Stage 3: Conservative SAC + BC for fine-tuning
    """
    
    def __init__(
        self,
        state_dim: int = 37,
        action_dim: int = 2,
        latent_dim: int = 32,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        seq_len: int = 10,
        bc_weight: float = 0.5,
        cql_weight: float = 1.0
    ):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        
        # Stage 1: Temporal VAE
        self.temporal_vae = TemporalVAE(
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=d_model,
            nhead=nhead
        ).to(self.device)
        self.vae_trainer = None
        self.vae_frozen = False
        
        # Stage 2: Decision Transformer
        self.decision_transformer = DecisionTransformer(
            latent_dim=latent_dim,
            action_dim=action_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        ).to(self.device)
        self.dt_trainer = None
        self.dt_frozen = False
        
        # Stage 3: Conservative SAC
        self.sac_agent = ConservativeSACAgent(
            latent_dim=latent_dim,
            action_dim=action_dim,
            bc_weight=bc_weight,
            cql_weight=cql_weight,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )
        
        # For backward compatibility
        self.actor = self.sac_agent.actor
        self.critic_1 = self.sac_agent.critic_1
        self.critic_2 = self.sac_agent.critic_2
        
    def init_stage1_trainer(self, lr: float = 1e-4):
        """Initialize Stage 1 (VAE) trainer."""
        self.vae_trainer = TemporalVAETrainer(self.temporal_vae, lr=lr)
        
    def init_stage2_trainer(self, lr: float = 1e-4):
        """Initialize Stage 2 (DT) trainer."""
        self.dt_trainer = DecisionTransformerTrainer(self.decision_transformer, lr=lr)
    
    def freeze_vae(self):
        """Freeze Temporal VAE after Stage 1 training."""
        for param in self.temporal_vae.parameters():
            param.requires_grad = False
        self.temporal_vae.eval()
        self.vae_frozen = True
        print("Temporal VAE frozen")
        
    def freeze_decision_transformer(self):
        """Freeze Decision Transformer after Stage 2 training."""
        for param in self.decision_transformer.parameters():
            param.requires_grad = False
        self.decision_transformer.eval()
        self.dt_frozen = True
        print("Decision Transformer frozen")
    
    def encode_states(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Encode raw states to latent using frozen VAE."""
        if not self.vae_frozen:
            print("Warning: VAE not frozen, encoding in eval mode")
            self.temporal_vae.eval()
        
        with torch.no_grad():
            # Handle single timestep by creating sequence
            if states.dim() == 2:
                states = states.unsqueeze(1)
                actions = actions.unsqueeze(1)
            z = self.temporal_vae.get_latent(states, actions)
        
        return z
    
    def init_sac_from_dt(self):
        """Initialize SAC actor from pretrained Decision Transformer."""
        self.sac_agent.load_from_decision_transformer(self.decision_transformer)
    
    def train(self, batches, epoch: int) -> List[float]:
        """
        Train Stage 3 (SAC) with latent states.
        Compatible with original training interface.
        """
        return self.sac_agent.train(batches, epoch)
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """Get action using full pipeline."""
        return self.sac_agent.get_action(state, deterministic)
    
    def save(self, path: str):
        """Save all model components."""
        torch.save({
            'temporal_vae': self.temporal_vae.state_dict(),
            'decision_transformer': self.decision_transformer.state_dict(),
            'sac_actor': self.sac_agent.actor.state_dict(),
            'sac_critic_1': self.sac_agent.critic_1.state_dict(),
            'sac_critic_2': self.sac_agent.critic_2.state_dict(),
            'vae_frozen': self.vae_frozen,
            'dt_frozen': self.dt_frozen
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load all model components."""
        checkpoint = torch.load(path, map_location=self.device)
        self.temporal_vae.load_state_dict(checkpoint['temporal_vae'])
        self.decision_transformer.load_state_dict(checkpoint['decision_transformer'])
        self.sac_agent.actor.load_state_dict(checkpoint['sac_actor'])
        self.sac_agent.critic_1.load_state_dict(checkpoint['sac_critic_1'])
        self.sac_agent.critic_2.load_state_dict(checkpoint['sac_critic_2'])
        self.vae_frozen = checkpoint.get('vae_frozen', False)
        self.dt_frozen = checkpoint.get('dt_frozen', False)
        
        if self.vae_frozen:
            self.freeze_vae()
        if self.dt_frozen:
            self.freeze_decision_transformer()
        
        print(f"Model loaded from {path}")


# =============================================================================
# BACKWARD COMPATIBILITY: Simple Transformer Agent
# =============================================================================

class TransformerAgent(ConservativeSACAgent):
    """
    Backward-compatible TransformerAgent wrapping ConservativeSACAgent.
    For users who want a simple interface without the three-stage pipeline.
    """
    
    def __init__(
        self,
        state_dim: int = 37,
        action_dim: int = 2,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256
    ):
        # Use state_dim as latent_dim for backward compatibility
        super().__init__(
            latent_dim=state_dim,
            action_dim=action_dim,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            bc_weight=0.3,  # Default BC weight
            cql_weight=0.5,  # Default CQL weight
            use_cql=True,
            use_bc=True,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )
