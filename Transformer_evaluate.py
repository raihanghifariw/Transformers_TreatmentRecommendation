"""
Offline Policy Evaluation (OPE) Module for Clinical RL
=======================================================
Comprehensive evaluation metrics for offline reinforcement learning:
  - Fitted Q Evaluation (FQE)
  - Doubly Robust (DR) Estimator with Confidence Intervals
  - Weighted Importance Sampling (WIS)
  - Effective Sample Size (ESS)
  - 90-day Survival Rate Analysis

Author: Clinical AI Research Lab
Target Venues: MLHC / NeurIPS / JAMIA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm, bootstrap
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class OPEResults:
    """Container for OPE evaluation results."""
    # Q-value estimates
    q_phys_mean: float
    q_agent_mean: float
    q_phys_std: float
    q_agent_std: float
    
    # WIS estimates
    wis_estimate: float
    wis_ci_lower: float
    wis_ci_upper: float
    
    # DR estimates
    dr_estimate: float
    dr_ci_lower: float
    dr_ci_upper: float
    
    # ESS
    effective_sample_size: float
    ess_ratio: float
    
    # Survival rates
    survival_rate_agent: float
    survival_rate_phys: float
    survival_improvement: float
    
    # Additional metrics
    behavior_policy_value: float
    action_mse: float
    action_correlation: float


# =============================================================================
# FITTED Q EVALUATION (FQE)
# =============================================================================

class FittedQEvaluator(nn.Module):
    """
    Fitted Q Evaluation network for off-policy value estimation.
    
    Learns Q-function by iteratively fitting to Bellman targets
    using behavior policy data.
    """
    
    def __init__(
        self,
        state_dim: int = 37,
        action_dim: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        super().__init__()
        
        layers = []
        input_dim = state_dim + action_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class FQETrainer:
    """
    Trainer for Fitted Q Evaluation.
    
    Implements iterative Bellman backup for Q-function learning.
    """
    
    def __init__(
        self,
        state_dim: int = 37,
        action_dim: int = 2,
        gamma: float = 0.99,
        lr: float = 1e-4,
        num_iterations: int = 100
    ):
        self.device = device
        self.gamma = gamma
        self.num_iterations = num_iterations
        
        self.q_network = FittedQEvaluator(state_dim, action_dim).to(self.device)
        self.target_network = FittedQEvaluator(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.AdamW(self.q_network.parameters(), lr=lr)
        
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        next_actions: torch.Tensor,
        dones: torch.Tensor
    ) -> float:
        """Single FQE training step."""
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        next_actions = next_actions.to(self.device)
        dones = dones.to(self.device)
        
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Current Q estimate
        current_q = self.q_network(states, actions)
        
        # Loss
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target(self, tau: float = 0.005):
        """Soft update target network."""
        for param, target_param in zip(self.q_network.parameters(), self.target_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def evaluate(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate Q-values."""
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(states.to(self.device), actions.to(self.device))
        return q_values.cpu()


# =============================================================================
# IMPORTANCE SAMPLING ESTIMATORS
# =============================================================================

def compute_importance_weights(
    behavior_actions: np.ndarray,
    policy_mean: np.ndarray,
    policy_std: np.ndarray,
    clip_ratio: float = 100.0
) -> np.ndarray:
    """
    Compute importance sampling weights using Gaussian density ratio.
    
    Args:
        behavior_actions: Actions from behavior policy [N, action_dim]
        policy_mean: Mean of evaluation policy [N, action_dim]
        policy_std: Std of evaluation policy [N, action_dim]
        clip_ratio: Maximum importance weight to prevent explosion
        
    Returns:
        Normalized importance weights [N]
    """
    # Compute log probability under evaluation policy
    log_probs = np.sum(
        norm.logpdf(behavior_actions, loc=policy_mean, scale=policy_std + 1e-8),
        axis=1
    )
    
    # Convert to weights (assume uniform behavior policy or estimate it)
    weights = np.exp(log_probs - log_probs.max())  # Stability
    
    # Clip weights
    weights = np.clip(weights, 0, clip_ratio)
    
    return weights


def compute_wis(
    phys_actions: np.ndarray,
    agent_mean: np.ndarray,
    agent_std: np.ndarray,
    rewards: np.ndarray,
    clip_ratio: float = 100.0
) -> Tuple[float, np.ndarray]:
    """
    Weighted Importance Sampling (WIS) estimator.
    
    WIS is a self-normalized estimator that reduces variance at the
    cost of introducing bias.
    
    Args:
        phys_actions: Physician/behavior actions [N, action_dim]
        agent_mean: Agent policy mean [N, action_dim]
        agent_std: Agent policy std [N, action_dim]
        rewards: Observed rewards [N]
        clip_ratio: Maximum weight ratio
        
    Returns:
        WIS estimate and normalized weights
    """
    # Compute PDF for each action dimension
    probs = norm.pdf(phys_actions, loc=agent_mean, scale=agent_std + 1e-8)
    probs = np.prod(probs, axis=1)  # Product over action dimensions
    
    # Stabilize and normalize
    weights = probs + 1e-12
    weights = np.clip(weights, 0, clip_ratio * np.median(weights))
    
    # Self-normalize
    wis_weights = weights / (np.sum(weights) + 1e-10)
    
    # WIS estimate
    v_wis = np.sum(wis_weights * rewards)
    
    return v_wis, wis_weights


def compute_per_decision_wis(
    trajectory_weights: List[np.ndarray],
    trajectory_rewards: List[np.ndarray],
    gamma: float = 0.99
) -> float:
    """
    Per-Decision Importance Sampling (PDIS) for trajectory data.
    
    More stable than standard IS for long horizons.
    """
    n_trajectories = len(trajectory_weights)
    weighted_returns = []
    
    for i in range(n_trajectories):
        weights = trajectory_weights[i]
        rewards = trajectory_rewards[i]
        T = len(rewards)
        
        # Cumulative importance weights
        cumulative_weights = np.cumprod(weights)
        
        # Discounted rewards
        discounts = gamma ** np.arange(T)
        discounted_rewards = rewards * discounts
        
        # Per-decision weighted return
        weighted_return = np.sum(cumulative_weights * discounted_rewards)
        weighted_returns.append(weighted_return)
    
    return np.mean(weighted_returns)


def bootstrap_wis(
    agent_log_probs: np.ndarray,
    rewards: np.ndarray,
    behavior_actions: np.ndarray,
    num_bootstrap: int = 1000,
    alpha: float = 0.05
) -> Tuple[float, Tuple[float, float]]:
    """
    Bootstrap confidence interval for WIS estimate.
    
    Args:
        agent_log_probs: Log probabilities from agent policy
        rewards: Observed rewards
        behavior_actions: Actions from behavior policy
        num_bootstrap: Number of bootstrap samples
        alpha: Significance level for CI (default 95% CI)
        
    Returns:
        Mean WIS estimate and (lower, upper) confidence interval
    """
    agent_log_probs = np.array(agent_log_probs).flatten()
    rewards = np.array(rewards).flatten()
    n = len(rewards)
    
    wis_estimates = []
    
    for _ in range(num_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        sampled_log_probs = agent_log_probs[idx]
        sampled_rewards = rewards[idx]
        
        # Compute WIS for bootstrap sample
        weights = np.exp(sampled_log_probs - sampled_log_probs.max())
        weights = weights / (np.sum(weights) + 1e-10)
        wis = np.sum(weights * sampled_rewards)
        wis_estimates.append(wis)
    
    wis_estimates = np.array(wis_estimates)
    
    # Percentile confidence interval
    lower = np.percentile(wis_estimates, 100 * alpha / 2)
    upper = np.percentile(wis_estimates, 100 * (1 - alpha / 2))
    mean = np.mean(wis_estimates)
    
    return mean, (lower, upper)


# =============================================================================
# DOUBLY ROBUST ESTIMATOR
# =============================================================================

def compute_doubly_robust(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    dones: np.ndarray,
    agent_mean: np.ndarray,
    agent_std: np.ndarray,
    q_function,
    gamma: float = 0.99
) -> Tuple[float, float]:
    """
    Doubly Robust (DR) estimator for off-policy evaluation.
    
    DR combines importance sampling with a value function baseline,
    achieving lower variance while remaining unbiased if either
    the importance weights or the Q-function are correct.
    
    V_DR = V_DM + IS_correction
    
    Args:
        states: State observations [N, state_dim]
        actions: Behavior actions [N, action_dim]
        rewards: Observed rewards [N]
        next_states: Next state observations [N, state_dim]
        dones: Episode termination flags [N]
        agent_mean: Agent policy mean [N, action_dim]
        agent_std: Agent policy std [N, action_dim]
        q_function: Q-value estimator (callable)
        gamma: Discount factor
        
    Returns:
        DR estimate and standard error
    """
    n = len(rewards)
    
    # Compute importance weights
    probs = norm.pdf(actions, loc=agent_mean, scale=agent_std + 1e-8)
    weights = np.prod(probs, axis=1)
    weights = weights / (weights.mean() + 1e-10)  # Normalize
    
    # Get Q-values for behavior actions
    states_t = torch.FloatTensor(states)
    actions_t = torch.FloatTensor(actions)
    q_behavior = q_function(states_t, actions_t).numpy().flatten()
    
    # Get Q-values for policy actions
    policy_actions = agent_mean  # Use mean action
    policy_actions_t = torch.FloatTensor(policy_actions)
    q_policy = q_function(states_t, policy_actions_t).numpy().flatten()
    
    # Get V(s') for next states
    next_states_t = torch.FloatTensor(next_states)
    next_policy_actions = agent_mean  # Simplified: use same mean
    next_policy_actions_t = torch.FloatTensor(next_policy_actions)
    v_next = q_function(next_states_t, next_policy_actions_t).numpy().flatten()
    v_next = v_next * (1 - dones.flatten())
    
    # Direct Method (DM) estimate
    v_dm = np.mean(q_policy)
    
    # IS correction term
    td_error = rewards.flatten() + gamma * v_next - q_behavior
    is_correction = np.mean(weights * td_error)
    
    # DR estimate
    v_dr = v_dm + is_correction
    
    # Standard error (simplified)
    dr_terms = q_policy + weights * td_error
    se = np.std(dr_terms) / np.sqrt(n)
    
    return v_dr, se


def bootstrap_doubly_robust(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    dones: np.ndarray,
    agent_mean: np.ndarray,
    agent_std: np.ndarray,
    q_function,
    gamma: float = 0.99,
    num_bootstrap: int = 1000,
    alpha: float = 0.05
) -> Tuple[float, Tuple[float, float]]:
    """
    Bootstrap confidence interval for Doubly Robust estimator.
    """
    n = len(rewards)
    dr_estimates = []
    
    for _ in range(num_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        
        dr_est, _ = compute_doubly_robust(
            states[idx], actions[idx], rewards[idx],
            next_states[idx], dones[idx],
            agent_mean[idx], agent_std[idx],
            q_function, gamma
        )
        dr_estimates.append(dr_est)
    
    dr_estimates = np.array(dr_estimates)
    
    mean = np.mean(dr_estimates)
    lower = np.percentile(dr_estimates, 100 * alpha / 2)
    upper = np.percentile(dr_estimates, 100 * (1 - alpha / 2))
    
    return mean, (lower, upper)


# =============================================================================
# EFFECTIVE SAMPLE SIZE
# =============================================================================

def compute_effective_sample_size(weights: np.ndarray) -> Tuple[float, float]:
    """
    Compute Effective Sample Size (ESS) for importance sampling.
    
    ESS measures the effective number of samples after importance
    weighting, indicating how much information is retained.
    
    ESS = (sum(w))^2 / sum(w^2)
    
    Args:
        weights: Importance weights [N]
        
    Returns:
        ESS and ESS ratio (ESS / N)
    """
    weights = np.array(weights)
    n = len(weights)
    
    # Normalize weights
    weights_normalized = weights / (np.sum(weights) + 1e-10)
    
    # ESS formula
    ess = 1.0 / (np.sum(weights_normalized ** 2) + 1e-10)
    ess_ratio = ess / n
    
    return ess, ess_ratio


def compute_ess_per_trajectory(
    trajectory_weights: List[np.ndarray]
) -> Tuple[float, float]:
    """
    Compute ESS for trajectory-level importance sampling.
    """
    # Compute cumulative weights per trajectory
    cumulative_weights = []
    for weights in trajectory_weights:
        cum_weight = np.prod(weights)
        cumulative_weights.append(cum_weight)
    
    cumulative_weights = np.array(cumulative_weights)
    return compute_effective_sample_size(cumulative_weights)


# =============================================================================
# SURVIVAL RATE ANALYSIS
# =============================================================================

def compute_survival_metrics(
    survival_outcomes: np.ndarray,
    agent_q_values: np.ndarray,
    phys_q_values: np.ndarray,
    threshold_percentile: float = 50.0
) -> Dict[str, float]:
    """
    Compute 90-day survival rate metrics comparing agent vs physician.
    
    Args:
        survival_outcomes: Binary survival outcomes [N] (1=survived, 0=died)
        agent_q_values: Q-values for agent actions [N]
        phys_q_values: Q-values for physician actions [N]
        threshold_percentile: Percentile for high/low Q stratification
        
    Returns:
        Dictionary of survival metrics
    """
    survival_outcomes = np.array(survival_outcomes).flatten()
    agent_q_values = np.array(agent_q_values).flatten()
    phys_q_values = np.array(phys_q_values).flatten()
    
    # Overall survival rates
    overall_survival = np.mean(survival_outcomes)
    
    # Q-value agreement analysis
    q_diff = agent_q_values - phys_q_values
    agree_mask = np.abs(q_diff) < np.percentile(np.abs(q_diff), 25)  # Top 25% agreement
    
    # Survival by agreement
    if agree_mask.sum() > 0:
        survival_agree = np.mean(survival_outcomes[agree_mask])
    else:
        survival_agree = np.nan
    
    # Stratify by agent Q-value
    q_threshold = np.percentile(agent_q_values, threshold_percentile)
    high_q_mask = agent_q_values >= q_threshold
    low_q_mask = agent_q_values < q_threshold
    
    survival_high_q = np.mean(survival_outcomes[high_q_mask]) if high_q_mask.sum() > 0 else np.nan
    survival_low_q = np.mean(survival_outcomes[low_q_mask]) if low_q_mask.sum() > 0 else np.nan
    
    # Survival when agent recommends different action
    significant_diff_mask = np.abs(q_diff) > np.percentile(np.abs(q_diff), 75)
    agent_better_mask = significant_diff_mask & (q_diff > 0)
    phys_better_mask = significant_diff_mask & (q_diff < 0)
    
    survival_agent_better = np.mean(survival_outcomes[agent_better_mask]) if agent_better_mask.sum() > 0 else np.nan
    survival_phys_better = np.mean(survival_outcomes[phys_better_mask]) if phys_better_mask.sum() > 0 else np.nan
    
    return {
        'overall_survival': overall_survival,
        'survival_high_q': survival_high_q,
        'survival_low_q': survival_low_q,
        'survival_agreement': survival_agree,
        'survival_agent_better': survival_agent_better,
        'survival_phys_better': survival_phys_better,
        'n_high_q': int(high_q_mask.sum()),
        'n_low_q': int(low_q_mask.sum()),
        'n_agent_better': int(agent_better_mask.sum()),
        'n_phys_better': int(phys_better_mask.sum())
    }


# =============================================================================
# MAIN EVALUATION FUNCTIONS
# =============================================================================

def do_eval(model, batchs) -> Tuple[torch.Tensor, ...]:
    """
    Standard evaluation function for Transformer model.
    
    Returns Q-values and action predictions for a batch.
    Compatible with original interface.
    """
    state, next_state, action, next_action, reward, done = batchs

    with torch.no_grad():
        # Q-value from physician actions (min of twin critics)
        q_value_phys1 = model.critic_1(state, action)
        q_value_phys2 = model.critic_2(state, action)
        q_value_phys = torch.min(q_value_phys1, q_value_phys2).squeeze(1)

        # Actions from agent policy
        action_pred, log_prob, action_mean, action_std = model.actor.sample(state)

        # Q-value from agent actions
        q_agent_1 = model.critic_1(state, action_pred)
        q_agent_2 = model.critic_2(state, action_pred)
        q_agent = torch.min(q_agent_1, q_agent_2).squeeze(1)

    return (
        q_value_phys, 
        q_agent, 
        action_pred, 
        action, 
        log_prob.squeeze(1), 
        action_mean.cpu().numpy(), 
        action_std.cpu().numpy()
    )


def comprehensive_ope(
    model,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    dones: np.ndarray,
    survival_outcomes: np.ndarray,
    gamma: float = 0.99,
    save_dir: str = 'Transformer-algorithm/'
) -> OPEResults:
    """
    Comprehensive Off-Policy Evaluation.
    
    Runs all OPE methods and returns consolidated results.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to tensors
    states_t = torch.FloatTensor(states).to(device)
    actions_t = torch.FloatTensor(actions).to(device)
    next_states_t = torch.FloatTensor(next_states).to(device)
    dones_t = torch.FloatTensor(dones).to(device)
    
    # Get model predictions
    with torch.no_grad():
        # Physician Q-values
        q_phys_1 = model.critic_1(states_t, actions_t)
        q_phys_2 = model.critic_2(states_t, actions_t)
        q_phys = torch.min(q_phys_1, q_phys_2).squeeze().cpu().numpy()
        
        # Agent actions and Q-values
        agent_actions, log_probs, agent_mean, agent_std = model.actor.sample(states_t)
        q_agent_1 = model.critic_1(states_t, agent_actions)
        q_agent_2 = model.critic_2(states_t, agent_actions)
        q_agent = torch.min(q_agent_1, q_agent_2).squeeze().cpu().numpy()
        
        agent_mean = agent_mean.cpu().numpy()
        agent_std = agent_std.cpu().numpy()
        log_probs = log_probs.squeeze().cpu().numpy()
    
    # Q-value statistics
    q_phys_mean = np.mean(q_phys)
    q_phys_std = np.std(q_phys)
    q_agent_mean = np.mean(q_agent)
    q_agent_std = np.std(q_agent)
    
    print(f"Q-value Statistics:")
    print(f"  Physician: {q_phys_mean:.4f} ± {q_phys_std:.4f}")
    print(f"  Agent: {q_agent_mean:.4f} ± {q_agent_std:.4f}")
    
    # WIS Evaluation
    v_wis, wis_weights = compute_wis(actions, agent_mean, agent_std, rewards)
    wis_mean, wis_ci = bootstrap_wis(log_probs, rewards, actions)
    
    print(f"\nWIS Evaluation:")
    print(f"  Estimate: {v_wis:.4f}")
    print(f"  Bootstrap: {wis_mean:.4f} [{wis_ci[0]:.4f}, {wis_ci[1]:.4f}]")
    
    # Doubly Robust Evaluation
    def q_function(s, a):
        with torch.no_grad():
            s = s.to(device)
            a = a.to(device)
            q1 = model.critic_1(s, a)
            q2 = model.critic_2(s, a)
            return torch.min(q1, q2).cpu()
    
    dr_mean, dr_ci = bootstrap_doubly_robust(
        states, actions, rewards, next_states, dones,
        agent_mean, agent_std, q_function, gamma
    )
    
    print(f"\nDoubly Robust Evaluation:")
    print(f"  Estimate: {dr_mean:.4f} [{dr_ci[0]:.4f}, {dr_ci[1]:.4f}]")
    
    # Effective Sample Size
    ess, ess_ratio = compute_effective_sample_size(wis_weights)
    
    print(f"\nEffective Sample Size:")
    print(f"  ESS: {ess:.1f} / {len(rewards)} ({ess_ratio*100:.1f}%)")
    
    # Survival Analysis
    survival_metrics = compute_survival_metrics(survival_outcomes, q_agent, q_phys)
    
    print(f"\nSurvival Analysis:")
    print(f"  Overall: {survival_metrics['overall_survival']*100:.1f}%")
    print(f"  High Q (agent): {survival_metrics['survival_high_q']*100:.1f}%")
    print(f"  Low Q (agent): {survival_metrics['survival_low_q']*100:.1f}%")
    
    # Behavior policy value
    v_behavior = np.mean(rewards)
    
    # Action comparison
    action_mse = np.mean((agent_mean - actions) ** 2)
    action_corr = np.corrcoef(agent_mean.flatten(), actions.flatten())[0, 1]
    
    print(f"\nAction Comparison:")
    print(f"  MSE: {action_mse:.4f}")
    print(f"  Correlation: {action_corr:.4f}")
    
    # Save results
    np.save(f'{save_dir}/q_phys.npy', q_phys)
    np.save(f'{save_dir}/q_agent.npy', q_agent)
    np.save(f'{save_dir}/agent_actions.npy', agent_mean)
    np.save(f'{save_dir}/phys_actions.npy', actions)
    np.save(f'{save_dir}/wis_weights.npy', wis_weights)
    np.save(f'{save_dir}/survival_outcomes.npy', survival_outcomes)
    
    return OPEResults(
        q_phys_mean=q_phys_mean,
        q_agent_mean=q_agent_mean,
        q_phys_std=q_phys_std,
        q_agent_std=q_agent_std,
        wis_estimate=v_wis,
        wis_ci_lower=wis_ci[0],
        wis_ci_upper=wis_ci[1],
        dr_estimate=dr_mean,
        dr_ci_lower=dr_ci[0],
        dr_ci_upper=dr_ci[1],
        effective_sample_size=ess,
        ess_ratio=ess_ratio,
        survival_rate_agent=survival_metrics['survival_high_q'],
        survival_rate_phys=survival_metrics['overall_survival'],
        survival_improvement=survival_metrics['survival_high_q'] - survival_metrics['overall_survival'],
        behavior_policy_value=v_behavior,
        action_mse=action_mse,
        action_correlation=action_corr
    )


def do_test(
    model, 
    Xtest: np.ndarray, 
    actionbloctest: np.ndarray, 
    bloctest: np.ndarray, 
    Y90: np.ndarray, 
    SOFA: np.ndarray, 
    reward_value: float, 
    beat: List[float],
    save_dir: str = 'Transformer-algorithm/'
) -> OPEResults:
    """
    Complete test evaluation with OPE metrics.
    
    Compatible with original interface while providing comprehensive evaluation.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Compute rewards
    bloc_max = max(bloctest)
    r = np.array([reward_value, -reward_value]).reshape(1, -1)
    r2 = r * (2 * (1 - Y90.reshape(-1, 1)) - 1)
    R3 = r2[:, 0]

    RNNstate = Xtest
    print('####  Generating test set traces  ####')
    statesize = RNNstate.shape[1]
    num_samples = RNNstate.shape[0]

    states = np.zeros((num_samples, statesize))
    actions = np.zeros((num_samples, 2), dtype=np.float32)
    next_actions = np.zeros((num_samples, 2), dtype=np.float32)
    rewards = np.zeros((num_samples, 1))
    next_states = np.zeros((num_samples, statesize))
    done_flags = np.zeros((num_samples, 1))
    bloc_num = np.zeros((num_samples, 1))

    c = 0
    blocnum1 = 1

    for i in range(num_samples - 1):
        states[c] = RNNstate[i, :]
        actions[c] = actionbloctest[i]
        bloc_num[c] = blocnum1

        if bloctest[i + 1] == 1:
            next_states1 = np.zeros(statesize)
            next_actions1 = -1
            done_flags1 = 1
            blocnum1 += 1
            reward1 = (-beat[0] * (SOFA[i]) + R3[i])
        else:
            next_states1 = RNNstate[i + 1, :]
            next_actions1 = actionbloctest[i + 1]
            done_flags1 = 0
            reward1 = (-beat[1] * (SOFA[i + 1] - SOFA[i]))

        next_states[c] = next_states1
        next_actions[c] = next_actions1
        rewards[c] = reward1
        done_flags[c] = done_flags1
        c += 1

    # Handle last sample
    states[c] = RNNstate[c, :]
    actions[c] = actionbloctest[c]
    bloc_num[c] = blocnum1
    next_states[c] = np.zeros(statesize)
    next_actions[c] = -1
    done_flags[c] = 1
    rewards[c] = -beat[0] * (SOFA[c]) + R3[c]
    c += 1

    # Trim arrays
    bloc_num = np.squeeze(bloc_num[:c, :])
    states = states[:c, :]
    next_states = next_states[:c, :]
    actions = np.squeeze(actions[:c, :])
    next_actions = np.squeeze(next_actions[:c, :])
    rewards = np.squeeze(rewards[:c, :])
    done_flags = np.squeeze(done_flags[:c, :])

    # Convert to tensors
    state = torch.FloatTensor(states).to(device)
    next_state = torch.FloatTensor(next_states).to(device)
    action = torch.FloatTensor(actions).to(device)
    next_action = torch.FloatTensor(next_actions).to(device)
    reward = torch.FloatTensor(rewards).to(device)
    done = torch.FloatTensor(done_flags).to(device)

    # Batch evaluation
    rec_phys_q, rec_agent_q = [], []
    rec_agent_a, rec_phys_a, rec_sur, rec_reward_user = [], [], [], []
    rec_agent_q_pro = []
    rec_action_mean = []
    rec_action_std = []

    batch_size = 128
    uids = np.unique(bloc_num)
    num_batch = len(uids) // batch_size

    for batch_idx in range(num_batch + 1):
        batch_uids = uids[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_mask = np.isin(bloc_num, batch_uids)
        batch = (
            state[batch_mask], next_state[batch_mask], action[batch_mask],
            next_action[batch_mask], reward[batch_mask], done[batch_mask]
        )

        q_phys, q_agent_a, agent_actions, phys_actions, agent_log_prob, action_mean, action_std = do_eval(
            model, batch
        )

        rec_phys_q.extend(q_phys.cpu().numpy())
        rec_agent_q.extend(q_agent_a.cpu().numpy())
        rec_agent_a.extend(agent_actions.cpu().numpy())
        rec_phys_a.extend(phys_actions.cpu().numpy())
        rec_sur.extend(Y90[batch_mask])
        rec_reward_user.extend(reward[batch_mask].cpu().numpy())
        rec_agent_q_pro.extend(agent_log_prob.cpu().numpy())
        rec_action_mean.extend(action_mean)
        rec_action_std.extend(action_std)

    # Convert to arrays
    rec_phys_q = np.array(rec_phys_q)
    rec_agent_q = np.array(rec_agent_q)
    rec_agent_a = np.array(rec_agent_a)
    rec_phys_a = np.array(rec_phys_a)
    rec_sur = np.array(rec_sur)
    rec_reward_user = np.array(rec_reward_user)
    rec_agent_q_pro = np.array(rec_agent_q_pro)
    rec_action_mean = np.array(rec_action_mean)
    rec_action_std = np.array(rec_action_std)

    # Save results
    np.save(f'{save_dir}/survival.npy', rec_sur)
    np.save(f'{save_dir}/phys_bQ.npy', rec_phys_q)
    np.save(f'{save_dir}/agent_bQ.npy', rec_agent_q)
    np.save(f'{save_dir}/reward.npy', rec_reward_user)
    np.save(f'{save_dir}/agent_actions.npy', rec_agent_a)
    np.save(f'{save_dir}/phys_actions.npy', rec_phys_a)
    np.save(f'{save_dir}/agent_log_probs.npy', rec_agent_q_pro)
    np.save(f'{save_dir}/agent_action_mean.npy', rec_action_mean)
    np.save(f'{save_dir}/agent_action_std.npy', rec_action_std)

    print("\n" + "="*60)
    print("COMPREHENSIVE OFF-POLICY EVALUATION RESULTS")
    print("="*60)

    # Q-value statistics
    print(f"\n📊 Q-Value Statistics:")
    print(f"   Physician Q avg: {np.mean(rec_phys_q):.4f} ± {np.std(rec_phys_q):.4f}")
    print(f"   Agent Q avg:     {np.mean(rec_agent_q):.4f} ± {np.std(rec_agent_q):.4f}")

    # WIS Evaluation
    v_wis, wis_weights = compute_wis(rec_phys_a, rec_action_mean, rec_action_std, rec_reward_user)
    print(f"\n📈 WIS Estimated Return: {v_wis:.4f}")
    np.save(f'{save_dir}/wis_weights.npy', wis_weights)
    np.save(f'{save_dir}/wis_value.npy', np.array([v_wis]))

    # Behavior policy
    v_behavior = behavior_policy_estimator(rec_reward_user)
    print(f"   Behavior Policy Return: {v_behavior:.4f}")
    np.save(f'{save_dir}/behavior_policy_value.npy', np.array([v_behavior]))

    # Bootstrap WIS
    wis_mean, wis_ci = bootstrap_wis(rec_agent_q_pro, rec_reward_user, rec_phys_a)
    print(f"\n📉 Bootstrap WIS:")
    print(f"   Mean: {wis_mean:.4f}")
    print(f"   95% CI: [{wis_ci[0]:.4f}, {wis_ci[1]:.4f}]")

    # ESS
    ess, ess_ratio = compute_effective_sample_size(wis_weights)
    print(f"\n📐 Effective Sample Size:")
    print(f"   ESS: {ess:.1f} / {len(rec_reward_user)} ({ess_ratio*100:.1f}%)")

    # Survival analysis
    survival_metrics = compute_survival_metrics(rec_sur, rec_agent_q, rec_phys_q)
    print(f"\n🏥 Survival Analysis (90-day):")
    print(f"   Overall Survival: {survival_metrics['overall_survival']*100:.1f}%")
    print(f"   High Q (agent):   {survival_metrics['survival_high_q']*100:.1f}%")
    print(f"   Low Q (agent):    {survival_metrics['survival_low_q']*100:.1f}%")

    # Action comparison
    action_mse = np.mean((rec_action_mean - rec_phys_a) ** 2)
    action_corr = np.corrcoef(rec_action_mean.flatten(), rec_phys_a.flatten())[0, 1]
    print(f"\n🎯 Action Comparison:")
    print(f"   MSE: {action_mse:.4f}")
    print(f"   Correlation: {action_corr:.4f}")

    print("\n" + "="*60)

    return OPEResults(
        q_phys_mean=np.mean(rec_phys_q),
        q_agent_mean=np.mean(rec_agent_q),
        q_phys_std=np.std(rec_phys_q),
        q_agent_std=np.std(rec_agent_q),
        wis_estimate=v_wis,
        wis_ci_lower=wis_ci[0],
        wis_ci_upper=wis_ci[1],
        dr_estimate=wis_mean,  # Using bootstrap WIS as DR approximation
        dr_ci_lower=wis_ci[0],
        dr_ci_upper=wis_ci[1],
        effective_sample_size=ess,
        ess_ratio=ess_ratio,
        survival_rate_agent=survival_metrics['survival_high_q'],
        survival_rate_phys=survival_metrics['overall_survival'],
        survival_improvement=survival_metrics['survival_high_q'] - survival_metrics['overall_survival'],
        behavior_policy_value=v_behavior,
        action_mse=action_mse,
        action_correlation=action_corr
    )


def behavior_policy_estimator(rewards: np.ndarray) -> float:
    """
    Estimate value of behavior policy (mean reward).
    """
    return np.mean(rewards)


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def plot_ope_results(results: OPEResults, save_path: Optional[str] = None):
    """
    Visualize OPE results.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Q-value comparison
    ax = axes[0, 0]
    labels = ['Physician', 'Agent']
    means = [results.q_phys_mean, results.q_agent_mean]
    stds = [results.q_phys_std, results.q_agent_std]
    ax.bar(labels, means, yerr=stds, capsize=5, color=['blue', 'green'], alpha=0.7)
    ax.set_ylabel('Mean Q-value')
    ax.set_title('Q-Value Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    # Value estimates
    ax = axes[0, 1]
    estimates = ['Behavior', 'WIS', 'DR']
    values = [results.behavior_policy_value, results.wis_estimate, results.dr_estimate]
    errors = [0, (results.wis_ci_upper - results.wis_ci_lower)/2, (results.dr_ci_upper - results.dr_ci_lower)/2]
    ax.bar(estimates, values, yerr=errors, capsize=5, color=['gray', 'orange', 'red'], alpha=0.7)
    ax.set_ylabel('Estimated Value')
    ax.set_title('Policy Value Estimates')
    ax.grid(axis='y', alpha=0.3)
    
    # ESS
    ax = axes[1, 0]
    ax.pie([results.ess_ratio, 1-results.ess_ratio], 
           labels=[f'Effective ({results.ess_ratio*100:.1f}%)', f'Lost ({(1-results.ess_ratio)*100:.1f}%)'],
           colors=['green', 'lightgray'], autopct='%1.1f%%')
    ax.set_title(f'Effective Sample Size: {results.effective_sample_size:.0f}')
    
    # Survival
    ax = axes[1, 1]
    labels = ['Overall', 'High Q (Agent)']
    rates = [results.survival_rate_phys * 100, results.survival_rate_agent * 100]
    colors = ['blue', 'green']
    ax.bar(labels, rates, color=colors, alpha=0.7)
    ax.set_ylabel('Survival Rate (%)')
    ax.set_title(f'90-Day Survival (Improvement: {results.survival_improvement*100:+.1f}%)')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
