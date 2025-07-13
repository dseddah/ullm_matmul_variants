import torch
import torch.nn as nn

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, hidden_size, seq_len, num_layers, num_heads):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(seq_len, hidden_size)
#        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=0.1)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, start_pos=0):
        seq_len_actual, batch_size = x.shape
        positions = torch.arange(0, seq_len_actual, device=x.device).unsqueeze(1).expand(seq_len_actual, batch_size)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        x = x * (self.token_embedding.embedding_dim ** 0.5)

        # transformer expects [seq_len, batch_size, hidden]
        x = self.transformer(x)
        return self.lm_head(x)


# Optional: drop-in simple model for debugging
class SanityLM(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.token_embedding(x)
        return self.lm_head(x)
