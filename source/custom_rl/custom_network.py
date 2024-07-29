import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=12, num_heads=4, ff_dim=48, sequence_length=10):
        super(TransformerEncoder, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.sequence_length = sequence_length

        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)

        # Feedforward network
        self.ff_network = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, input_dim)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        # Create causal mask
        self.register_buffer("causal_mask", self._generate_square_subsequent_mask(sequence_length))

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        batch_size, seq_len, _ = x.size()

        # Ensure the input matches our expected sequence length
        assert seq_len == self.sequence_length, f"Expected sequence length {self.sequence_length}, but got {seq_len}"

        # Reshape input for self-attention
        x = x.transpose(0, 1)  # (sequence_length, batch_size, input_dim)

        # Self-attention with causal mask
        attn_output, _ = self.self_attention(x, x, x, attn_mask=self.causal_mask)

        # Add & Norm
        x = self.norm1(x + attn_output)

        # Feedforward
        ff_output = self.ff_network(x)

        # Add & Norm
        x = self.norm2(x + ff_output)

        # Reshape output
        output = x.transpose(0, 1)  # (batch_size, sequence_length, input_dim)

        return output


# # Create an instance of the model
# model = TransformerEncoder()

# # Example input tensor (batch_size x sequence_length x input_dim)
# input_tensor = torch.randn(1, 10, 12)

# # Forward pass
# output = model(input_tensor)

# print(f"Input shape: {input_tensor.shape}")
# print(f"Output shape: {output.shape}")
# print(f"Output: {output}")