# Simplified self-attention implementation for educational purposes
import torch
import torch.nn.functional as F

def simple_self_attention(embeddings, d_k=64):
    """
    Simplified single-head self-attention.

    Args:
        embeddings: Tensor of shape (seq_len, d_model) — token embeddings
        d_k: Dimension of query/key projections
    """
    seq_len, d_model = embeddings.shape

    # Learnable projection matrices (normally these are nn.Linear layers)
    W_Q = torch.randn(d_model, d_k)
    W_K = torch.randn(d_model, d_k)
    W_V = torch.randn(d_model, d_k)

    # Project embeddings into Q, K, V spaces
    Q = embeddings @ W_Q   # (seq_len, d_k)
    K = embeddings @ W_K   # (seq_len, d_k)
    V = embeddings @ W_V   # (seq_len, d_k)

    # Compute attention scores: QK^T / sqrt(d_k)
    scores = (Q @ K.T) / (d_k ** 0.5)   # (seq_len, seq_len)

    # Apply softmax to get attention weights (probabilities)
    attention_weights = F.softmax(scores, dim=-1)   # (seq_len, seq_len)

    # Weighted sum of values
    output = attention_weights @ V   # (seq_len, d_k)

    return output, attention_weights

# Example: 4 tokens, each with 8-dimensional embeddings
torch.manual_seed(42)
token_embeddings = torch.randn(4, 8)
output, weights = simple_self_attention(token_embeddings, d_k=4)

print("Attention weights (each row sums to 1.0):")
print(weights.detach().numpy().round(3))
print(f"\nRow sums: {weights.sum(dim=-1).detach().numpy().round(3)}")