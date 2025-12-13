"""
Covariance-Weighted Attention Mechanism

Implements attention where neighbor messages are weighted by inverse uncertainty,
enabling principled information fusion in multi-agent settings.
"""

import numpy as np
from typing import List, Tuple, Optional


class CovarianceAttention:
    """
    Covariance-weighted multi-head attention for cooperative localization.
    
    Drones share state estimates and covariance matrices. The attention
    mechanism down-weights neighbors with high uncertainty (large covariance trace).
    
    Parameters
    ----------
    embed_dim : int
        Dimension of state embeddings
    num_heads : int
        Number of attention heads
    lambda_cov : float
        Weight for covariance penalty in attention scores
        
    Example
    -------
    >>> attn = CovarianceAttention(embed_dim=128, num_heads=4, lambda_cov=0.1)
    >>> h_self = np.random.randn(128)
    >>> neighbors = [(np.random.randn(128), np.eye(6) * 0.5) for _ in range(3)]
    >>> h_fused, weights = attn.forward(h_self, neighbors)
    """
    
    def __init__(
        self, 
        embed_dim: int = 128, 
        num_heads: int = 4,
        lambda_cov: float = 0.1
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.lambda_cov = lambda_cov
        
        # Initialize projection matrices (in practice, these would be learned)
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.02
        
    def compute_attention_weights(
        self,
        query: np.ndarray,
        keys: List[np.ndarray],
        covariances: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute attention weights with covariance penalty.
        
        Parameters
        ----------
        query : np.ndarray
            Query vector from self state, shape (embed_dim,)
        keys : List[np.ndarray]
            Key vectors from neighbors, each shape (embed_dim,)
        covariances : List[np.ndarray]
            Covariance matrices from neighbors, each shape (6, 6)
            
        Returns
        -------
        np.ndarray
            Normalized attention weights, shape (n_neighbors,)
        """
        if len(keys) == 0:
            return np.array([])
            
        q = query @ self.W_q
        
        scores = []
        for key, cov in zip(keys, covariances):
            k = key @ self.W_k
            # Scaled dot-product attention with covariance penalty
            score = np.dot(q, k) / np.sqrt(self.embed_dim)
            score -= self.lambda_cov * np.trace(cov)
            scores.append(score)
            
        scores = np.array(scores)
        # Softmax normalization
        exp_scores = np.exp(scores - np.max(scores))
        weights = exp_scores / (np.sum(exp_scores) + 1e-8)
        
        return weights
    
    def forward(
        self,
        h_self: np.ndarray,
        neighbors: List[Tuple[np.ndarray, np.ndarray]],
        self_cov: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply covariance-weighted attention.
        
        Parameters
        ----------
        h_self : np.ndarray
            Self state embedding, shape (embed_dim,)
        neighbors : List[Tuple[np.ndarray, np.ndarray]]
            List of (embedding, covariance) tuples for each neighbor
        self_cov : np.ndarray, optional
            Self covariance for mixing coefficient
            
        Returns
        -------
        h_fused : np.ndarray
            Fused state embedding
        weights : np.ndarray
            Attention weights used
        """
        if len(neighbors) == 0:
            return h_self, np.array([])
            
        keys = [n[0] for n in neighbors]
        covs = [n[1] for n in neighbors]
        
        weights = self.compute_attention_weights(h_self, keys, covs)
        
        # Compute weighted sum of value projections
        values = [n[0] @ self.W_v for n in neighbors]
        neighbor_contribution = sum(w * v for w, v in zip(weights, values))
        
        # Mix self and neighbor information
        if self_cov is not None:
            beta = 1.0 / (1.0 + np.trace(self_cov))  # More self-weight when uncertain
        else:
            beta = 0.5
            
        h_fused = (1 - beta) * h_self + beta * neighbor_contribution
        
        return h_fused, weights
    
    def __repr__(self) -> str:
        return (
            f"CovarianceAttention(embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, lambda_cov={self.lambda_cov})"
        )
